"""Fixed-budget adaptations of the legacy public optimistic candidate search.

These are internal controls, not author methods. Scope selection, public
forecast, cover solver and delivery guard are retained. Candidate count is
deterministic; the cover-node cap applies separately at each checkpoint.
Equal candidate caps are NOT equal CPU or equal state expansion budgets.
"""
from __future__ import annotations
import itertools
import math
from .planners import PublicPlanner, _pack


class FixedBudgetPlanner(PublicPlanner):
    def __init__(self, name="jrs", seed=0, max_candidates=32, cover_nodes=2000):
        if name not in ("jrs", "random_rollout"):
            raise ValueError("fixed legacy method must be jrs or random_rollout")
        super().__init__(name, seed, time_limit_s=math.inf)
        if max_candidates < 1 or cover_nodes < 1:
            raise ValueError("positive deterministic budgets required")
        self.max_candidates=int(max_candidates)
        self.cover_nodes=int(cover_nodes)
        self.stats.update(candidate_cap=self.max_candidates, cover_nodes_per_checkpoint=self.cover_nodes,
                          wall_clock_truncation=False, internal_baseline=True)

    def _templates(self, obs):
        ids=tuple(sorted(j.id for j in obs.jobs if j.status=="pending"))
        # Both controls always retain the three deterministic anchor candidates.
        anchor=[(ids,"local"),(ids,"expanded"),((),"local")]
        if self.name=="jrs":
            stream=itertools.chain(anchor,
                (((x,), mode) for x in ids for mode in ("local","expanded")),
                ((sub,"local") for k in (2,3) for sub in itertools.combinations(ids[:7],k)))
            return list(itertools.islice(dict.fromkeys(stream),self.max_candidates))
        candidates=list(dict.fromkeys(anchor))
        seen=set(candidates)
        # Fixed generation budget handles the finite small-state candidate set.
        for _ in range(self.max_candidates*20):
            if len(candidates)>=self.max_candidates:break
            order=list(ids);self.rng.shuffle(order)
            candidate=(tuple(order[:self.rng.randrange(len(ids)+1)]),
                       self.rng.choice(("local","expanded")))
            if candidate not in seen:
                seen.add(candidate);candidates.append(candidate)
        return candidates[:self.max_candidates]

    def _joint(self,obs,allowed,start):
        best=None;chosen=();evaluated=0;forecasts=[]
        for prefix,scope in self._templates(obs):
            result=self._forecast(obs,prefix,scope,math.inf,self.cover_nodes)
            evaluated+=1;self.stats["search_nodes"]+=result.nodes
            self.stats["cover_timeout"]+=sum(x=="feasible_timeout" for x in result.cover_statuses)
            forecasts.append({"prefix":prefix,"scope":scope,"score":result.score,
                              "first":tuple((a.kind,a.robot_id,a.target_id,a.mode) for a in result.first)})
            if not math.isfinite(result.score[0]):
                self.stats["forecast_infeasible"]+=1;continue
            first=_pack(result.first,obs,allowed)
            if not first and not obs.active and not obs.pending_packets:continue
            if best is None or result.score<best.score:
                best=result;chosen=first
        self.stats["master_candidates"]+=evaluated
        self.last_diagnostics={"candidates":evaluated,"templates":forecasts,
                               "prediction_is_not_quality_guarantee":True,
                               "forecast_model":"legacy_optimistic"}
        if best is not None:
            self.last_diagnostics.update(surrogate_score=best.score,predicted_finish=best.predicted_finish)
        if not chosen and not obs.active and not obs.pending_packets:
            chosen=self._rule(obs,allowed)
        return chosen
