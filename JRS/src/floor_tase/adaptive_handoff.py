"""Budgeted nominal screening and paired stochastic racing for quality handoff.

Only frozen public observations enter this module. The shared physical world,
measurement pipeline and admission gate are unchanged. A nominal forecast is
a cheap screening model; final contenders face full public-assumption cascade
forecasts. Neither forecast is a calibrated probability or an oracle.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

from .handoff import HandoffFrontier, HandoffOptions
from .model import Observation
from .planners import _akey, _dist, _pack
from .prediction import public_belief_environment


@dataclass(frozen=True)
class AdaptiveOptions:
    max_candidates: int = 6
    finalists: int = 2
    min_samples: int = 2
    max_samples: int = 4
    max_rollout_events: int = 2000
    diagnostic_policy: str = "adaptive"
    protect_handoffs: bool = True
    fast_dispatch: bool = True
    adaptive_scope: bool = True
    nominal_weight: float = .5

    def __post_init__(self):
        if not 1 <= self.finalists <= self.max_candidates:
            raise ValueError("finalists must be within candidate budget")
        if not 1 <= self.min_samples <= self.max_samples:
            raise ValueError("invalid sample budget")
        if self.max_rollout_events < 1 or not 0 <= self.nominal_weight <= 1:
            raise ValueError("invalid prediction budget/weight")
        if self.diagnostic_policy not in ("adaptive", "always", "off"):
            raise ValueError("invalid diagnostic policy")


class AdaptiveFrontier(HandoffFrontier):
    def __init__(self, options):
        self.adaptive = options
        super().__init__(HandoffOptions(diagnostic_checks=options.diagnostic_policy != "off",
                                       adaptive_scope=options.adaptive_scope,
                                       max_candidates=options.max_candidates))

    def features(self, obs, allowed):
        rows = super().features(obs, allowed)
        jobs = [j for j in obs.jobs if j.status in ("pending", "running")]
        running = {a.target_id:a.finish for a in obs.active if a.kind == "repair"}
        kept = []
        for row in rows:
            diagnostic = set(row["diagnostic_cells"])
            if diagnostic:
                blockers = [j for j in jobs if diagnostic & set(j.local_impact)]
                clear_at = max((running.get(j.id, min((max(obs.time, r.busy_until)
                                + _dist(r.position, j.position)/obs.travel_speed+j.duration
                                for r in obs.robots if r.skill == j.skill), default=obs.horizon))
                                for j in blockers), default=obs.time)
                lead = min(max(0., clear_at-obs.time), obs.remaining_budget)
                # A public, dimensioned screening heuristic: expected repair
                # work plus usable lead, compared with robot/measurement time.
                benefit = obs.assumed_collateral_probability * (len(diagnostic)*obs.repair_duration+lead)
                row["diagnostic_benefit_proxy"] = benefit
                row["diagnostic_cost_proxy"] = row["cost"]
                if (self.adaptive.diagnostic_policy == "adaptive" and
                        not row["acceptance_cells"] and benefit < row["cost"]):
                    continue
            if row["action"].kind == "successor":
                target = next(t for t in obs.tasks if t.id == row["action"].target_id)
                # Potential expanded interference is a ranking cost, never a
                # new permission rule or a reason to globally block execution.
                potential = sum(bool(set(target.requires) & set(j.expanded_impact)) for j in jobs)
                row["future_interference"] = potential
                row["redo_proxy"] = potential * obs.assumed_collateral_probability * row["duration"]
            kept.append(row)
        return kept

    def candidates(self, obs, allowed):
        rows = self.features(obs, allowed)
        result, seen = [], set()
        def add(ordered):
            actions = _pack([r["action"] for r in ordered], obs, allowed)
            key = tuple(sorted(_akey(a) for a in actions))
            if actions and key not in seen:
                seen.add(key)
                result.append((actions, {"actions":[self.describe(r) for r in ordered if r["action"] in actions]}))
        balanced = sorted(rows, key=lambda r:self._rank(r,"balanced"))
        add(balanced)
        add(sorted(rows, key=lambda r:self._rank(r,"repair")))
        add(sorted(rows, key=lambda r:self._rank(r,"deadline")))
        if self.adaptive.protect_handoffs:
            protected=[r for r in balanced if not (r.get("future_interference",0)
                       and r["slack"] > r["duration"]+obs.repair_duration)]
            add(protected)
        # Expose scope alternatives before filling remaining slots with generic
        # action anchors. Unrelated early handoffs otherwise consume the cap.
        repairs = sorted((r for r in rows if r["action"].kind == "repair"),
                         key=lambda r:(r["cost"]/max(.05,r["gain"]),_akey(r["action"])))
        for anchor in repairs:
            add([anchor]+[r for r in balanced if r is not anchor])
            if len(result) >= self.adaptive.max_candidates: break
        if len(result) < self.adaptive.max_candidates:
            add(sorted(rows, key=lambda r:self._rank(r,"diagnose")))
        if (obs.active or obs.pending_packets) and len(result) < self.adaptive.max_candidates:
            result.append(((),{"actions":[],"wait":"in-flight action or packet"}))
        return tuple(result[:self.adaptive.max_candidates]) or (((),{"actions":[]}),)


class AdaptiveHandoffPlanner:
    def __init__(self, seed=0, options=None):
        self.seed=seed
        self.options=options or AdaptiveOptions()
        self.frontier=AdaptiveFrontier(self.options)
        self.name="handoff_adaptive"
        self.stats={"calls":0,"master_candidates":0,"rollout_worlds":0,"rollout_events":0,
                    "fast_dispatches":0,"race_extra_samples":0,"total_wall_s":0.,
                    "public_information_only":True,"exact_mission_solver":False}
        self.last_diagnostics={}

    def diagnostics(self):
        return dict(self.stats,options=asdict(self.options))

    def _seed(self, state, sample):
        return int(hashlib.sha256(f"adaptive:{self.seed}:{state}:{sample}".encode()).hexdigest()[:15],16)

    def _forecast(self, obs, actions, sample_seed, mode):
        env=public_belief_environment(obs,sample_seed,mode)
        env.step_joint(actions)
        count=1
        while not env.done and count < self.options.max_rollout_events:
            current=env.observe()
            env.step_joint(self.frontier.select(current,env.allowed_actions()))
            count+=1
        if not env.done:
            raise RuntimeError("Adaptive forecast exceeded event budget; no truncated outcome accepted")
        self.stats["rollout_worlds"]+=1
        self.stats["rollout_events"]+=count
        r=env.result()
        return {k:float(r[k]) for k in ("quality_success","J","qualified_tasks","time","successor_revocations")}

    @staticmethod
    def _mean(rows,key):
        return sum(r[key] for r in rows)/len(rows)

    def _score(self, nominal, risks):
        w=self.options.nominal_weight
        return (-(w*nominal["quality_success"]+(1-w)*self._mean(risks,"quality_success")),
                w*nominal["J"]+(1-w)*self._mean(risks,"J"))

    @staticmethod
    def _separated(left,right):
        # Paired uncertainty screen; not a statistical superiority test. At
        # least two pairs are needed; zero sample variance is insufficient to
        # certify identical/noiseless outcomes from a small Monte Carlo set.
        if len(left) < 2:return False
        d=[a["J"]-b["J"] for a,b in zip(left,right)]
        mean=sum(d)/len(d)
        variance=sum((x-mean)**2 for x in d)/(len(d)-1)
        return variance > 0 and abs(mean) > 2*math.sqrt(variance/len(d))

    def select(self,obs,allowed_actions):
        start=time.perf_counter();self.stats["calls"]+=1
        if not isinstance(obs,Observation):raise TypeError("public Observation required")
        rows=self.frontier.features(obs,allowed_actions)
        candidates=self.frontier.candidates(obs,allowed_actions)
        self.stats["master_candidates"]+=len(candidates)
        reason="model_race"
        # Reconsider recovery choices at repair/diagnostic frontiers; routine
        # acceptance and legal handoffs use the declared continuation directly.
        routine=not any(r["action"].kind == "repair" or r["diagnostic_cells"]
                       or (self.options.protect_handoffs and r.get("future_interference",0)) for r in rows)
        if len(candidates)==1 or (self.options.fast_dispatch and routine):
            actions=self.frontier.select(obs,allowed_actions)
            self.stats["fast_dispatches"]+=1
            self.last_diagnostics={"method":self.name,"reason":"single_or_routine_dispatch",
                                   "candidate_count":len(candidates),"predictions":[]}
        else:
            state=hashlib.sha256(json.dumps(asdict(obs),sort_keys=True,separators=(",",":")).encode()).hexdigest()
            nominal=[self._forecast(obs,a,self._seed(state,0),"optimistic") for a,_ in candidates]
            order=sorted(range(len(candidates)),key=lambda i:(-nominal[i]["quality_success"],nominal[i]["J"],i))
            finalists=order[:self.options.finalists]
            risks={i:[] for i in finalists}
            for n in range(self.options.min_samples):
                seed=self._seed(state,n+1)
                for i in finalists:risks[i].append(self._forecast(obs,candidates[i][0],seed,"cascade"))
            samples=self.options.min_samples
            while samples < self.options.max_samples and len(finalists)>1:
                ranked=sorted(finalists,key=lambda i:(self._score(nominal[i],risks[i]),i))
                if self._separated(risks[ranked[0]],risks[ranked[1]]):break
                seed=self._seed(state,samples+1)
                for i in finalists:risks[i].append(self._forecast(obs,candidates[i][0],seed,"cascade"))
                samples+=1;self.stats["race_extra_samples"]+=1
            index=min(finalists,key=lambda i:(self._score(nominal[i],risks[i]),i))
            actions=candidates[index][0]
            self.last_diagnostics={"method":self.name,"reason":reason,"selected_index":index,
                                   "candidate_count":len(candidates),"finalists":finalists,
                                   "risk_samples":samples,"nominal_weight":self.options.nominal_weight,
                                   "nominal_predictions":nominal,"risk_predictions":risks,
                                   "selected":candidates[index][1],
                                   "selection_objective":"model-mixture completion; mean unconditional J",
                                   "continuation":"adaptive_frontier_same_options"}
        self.stats["total_wall_s"]+=time.perf_counter()-start
        return tuple(actions)
