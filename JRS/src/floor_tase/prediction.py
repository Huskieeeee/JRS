"""Public-belief generative predictions, with explicitly conditional semantics.

The predictor never takes an actual RecoveryEnv, Scenario, latent quality, true
radius, physical revision, or physical random seed. It instantiates NEW synthetic
worlds from Observation and declared public assumptions. It preserves in-flight
actions and delivery metadata. Modes share synthetic random draws and exactly
the same first-action candidates and reference continuation.

These are conditional forecasts: execute candidate once, then use robust_local.
They are NOT values of a repeatedly deployed learned/window policy. The scalar
quality belief and unobserved historical validity remain model assumptions, not
camera calibration or safety guarantees. See approximation_flags in each row.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import statistics
from .model import (Action, ActiveAction, Cell, CellState, InspectionView, JobState,
    Observation, PendingInspection, RepairJob, Robot, RobotState, Scenario,
    ScenarioConfig, SuccessorTask, TaskState, keyed_uniform)
from .environment import RecoveryEnv, InvalidAction

MODES = ("optimistic", "collateral", "cascade")


def observation_from_dict(d):
    d = dict(d)
    constructors = {"robots": RobotState, "cells": CellState, "jobs": JobState,
        "views": InspectionView, "tasks": TaskState, "active": ActiveAction,
        "pending_inspections": PendingInspection}
    tuple_fields = {"position", "targets", "local_impact", "expanded_impact",
                    "expanded_targets", "covers", "requires", "cells", "captured_versions", "authorization_versions", "authorization_capture_times"}
    for key, cls in constructors.items():
        rows=[]
        for source in d.get(key, ()):
            row=dict(source)
            for name in tuple_fields & row.keys():
                row[name]=tuple(tuple(v) if isinstance(v,list) else v for v in row[name])
            rows.append(cls(**row))
        d[key]=tuple(rows)
    return Observation(**d)


def _sample_quality(obs, cell, sample_seed):
    if cell.certified:
        lower = max(0., cell.lower if cell.lower is not None else 0.)
        upper = min(obs.quality_threshold, cell.upper if cell.upper is not None else obs.quality_threshold)
        return lower+(upper-lower)*keyed_uniform(sample_seed,"belief_quality",cell.id)
    if cell.status == "invalid":
        unrepaired = {q for j in obs.jobs if j.status in ("pending", "running", "exhausted") for q in j.targets}
        if cell.id not in unrepaired:
            # A finished (possibly expanded) repair leaves target evidence
            # invalid until recertification. Old lower bounds predate that
            # intervention and are not current-quality measurements.
            failed = keyed_uniform(sample_seed, "belief_repair_unverified", cell.id) < obs.assumed_repair_failure_prob
            return .78 if failed else .16
        lower=max(obs.quality_threshold+.01, cell.lower or .68)
        return lower+(max(lower,.82)-lower)*keyed_uniform(sample_seed,"belief_quality",cell.id)
    # A revoked observation cannot be treated as a measurement of current quality.
    bad=keyed_uniform(sample_seed,"belief_uncertain",cell.id)<obs.assumed_collateral_probability
    return .74 if bad else .16


class _PublicBeliefEnv(RecoveryEnv):
    def _repair_complete(self,s):
        cfg=self.cfg
        # A one-generation model recovers discovered damage but predicts no
        # additional damage/failures caused by its descendants. Cascade mode
        # uses the same transition rule for every discovered repair.
        jid=s["action"].target_id
        first=(jid in self._initial_attempts and s["attempt"]==self._initial_attempts[jid]+1)
        # In-flight repairs already incremented attempts before this snapshot.
        first=first or (jid in self._initial_running and s["attempt"]==self._initial_attempts[jid])
        if self._prediction_mode == "collateral" and not first:
            self.cfg=replace(cfg,collateral_probability=0.,repair_failure_prob=0.)
        try:
            super()._repair_complete(s)
        finally:
            self.cfg=cfg


def public_belief_environment(obs: Observation, sample_seed: int, mode="cascade"):
    """Build a synthetic event state using only public input.

    Private attributes BELOW belong to this newly-created synthetic environment;
    no live simulator reference is accepted or copied. Nominal prior assumption:
    already-certified/completed work has no hidden historical safety violation.
    """
    if not isinstance(obs, Observation):
        raise TypeError("Prediction accepts Observation only, never a real environment")
    if mode not in MODES: raise ValueError(mode)
    if obs.pending_packets != len(obs.pending_inspections):
        raise ValueError("Incomplete pending inspection metadata; cannot reconstruct this snapshot")
    if any(a.finish < obs.time-1e-9 for a in obs.active):
        raise ValueError("Snapshot contains a completed active action")
    if any(p.delivery_time < obs.time-1e-9 for p in obs.pending_inspections):
        raise ValueError("Snapshot contains an overdue undelivered packet")
    collateral=obs.assumed_collateral_probability if mode!="optimistic" else 0.
    failure=obs.assumed_repair_failure_prob if mode=="cascade" else 0.
    cfg=ScenarioConfig(horizon=obs.horizon,n_cells=len(obs.cells),
        n_regions=max(c.region for c in obs.cells)+1,
        impact_radius=obs.assumed_radius,assumed_radius=obs.assumed_radius,
        sensor_error=obs.assumed_sensor_error,assumed_sensor_error=obs.assumed_sensor_error,
        measurement_delay=obs.measurement_delay,travel_speed=obs.travel_speed,
        repair_duration=obs.repair_duration,max_repairs_per_cell=obs.max_repairs_per_cell,
        inspection_repeat=obs.inspection_repeat,collateral_probability=collateral,
        repair_failure_prob=failure,assumed_collateral_probability=obs.assumed_collateral_probability,
        assumed_repair_failure_prob=obs.assumed_repair_failure_prob,
        shared_passage=obs.shared_passage,evidence_policy=obs.evidence_policy,
        quality_threshold=obs.quality_threshold,freshness=obs.freshness)
    cells=tuple(Cell(c.id,*c.position,c.region,c.skill) for c in obs.cells)
    jobs=tuple(RepairJob(j.id,j.skill,j.targets,j.local_impact,j.expanded_impact,j.duration,
        j.expanded_duration,j.expanded_targets,j.position) for j in obs.jobs)
    tasks=tuple(SuccessorTask(t.id,t.skill,t.requires,t.duration,t.deadline,t.position) for t in obs.tasks)
    quality=tuple((c.id,_sample_quality(obs,c,sample_seed)) for c in obs.cells)
    scenario=Scenario("public-belief",sample_seed,cfg,cells,
        tuple(Robot(r.id,r.skill,r.position) for r in obs.robots),jobs,obs.views,tasks,quality,
        measurement_seed=sample_seed,description="Synthetic public belief; never a physical world copy")
    env=_PublicBeliefEnv(scenario)
    env._prediction_mode=mode
    env._initial_attempts={j.id:j.attempts for j in obs.jobs if j.status in ("pending","running")}
    env._initial_running={j.id for j in obs.jobs if j.status=="running"}
    env.time=obs.time;env._terminated=obs.done;env._termination_reason="snapshot_done" if obs.done else None
    env._events=[];env._trace=[];env._event_index=0;env._active={}
    env._stats["travel"]=obs.accrued_travel;env._stats["repairs"]=obs.accrued_repairs
    for c in obs.cells:
        env._ledger[c.id].update(status=c.status,version=c.version,mean=c.mean,upper=c.upper,
            lower=c.lower,last_observed=c.last_observed,batch_id=c.batch_id,samples=c.samples,
            physical_revision=0)
    for j in obs.jobs:
        env._jobs[j.id].update(status=j.status,attempts=j.attempts)
    for t in obs.tasks:
        env._tasks[t.id].update(status=t.status,
            completion_revisions={c:0 for c in t.requires} if t.status=="done" else None)
    for r in obs.robots:
        env._robots[r.id].update(position=r.position,busy_until=r.busy_until)
    jobs_by_id={j.id:j for j in obs.jobs};tasks_by_id={t.id:t for t in obs.tasks}
    views_by_id={v.id:v for v in obs.views}
    # Restore original public enqueue order, including delivery/authorization
    # ties. A local priority convention would change the actual event contract.
    def restore_event(at,order,kind,payload):
        if order is None or order <= 0:
            raise ValueError("Snapshot lacks public queued-event order metadata")
        if any(e[1]==order for e in env._events):
            raise ValueError("Snapshot has duplicate queued-event order metadata")
        env._events.append((float(at),order,kind,payload))
        env._event_index=max(env._event_index,order)
    for i,p in enumerate(obs.active):
        action=Action(p.kind,p.robot_id,p.target_id,p.mode)
        target=(jobs_by_id[p.target_id] if p.kind=="repair" else
                views_by_id[p.target_id] if p.kind=="inspect" else tasks_by_id[p.target_id])
        targets=(target.expanded_targets or target.targets) if p.kind=="repair" and p.mode=="expanded" else (
                 target.targets if p.kind=="repair" else target.covers if p.kind=="inspect" else target.requires)
        key=f"restored{i}"
        payload={"public":p,"action":action,"targets":targets,"position":target.position,
                 "distance":0.,"key":key}
        if p.kind=="repair":payload["attempt"]=target.attempts
        if p.kind=="successor":
            if target.status=="running":
                if set(dict(p.authorization_versions)) != set(targets):
                    raise ValueError("Running successor lacks public authorization-time version metadata")
                payload["authorized_revisions"]={c:0 for c in targets}
                payload["authorized_declared_versions"]=dict(p.authorization_versions)
                if set(dict(p.authorization_capture_times)) != set(targets):
                    raise ValueError("Running successor lacks authorization capture timestamps")
                payload["authorized_capture_times"]=dict(p.authorization_capture_times)
            elif p.arrival>obs.time+1e-9:restore_event(p.arrival,p.authorization_order,"authorize",key)
            else:payload["denied"]=True
        env._active[key]=payload
        restore_event(p.finish,p.completion_order,"complete",key)
    for packet in obs.pending_inspections:
        records=[]
        for cid,version in packet.captured_versions:
            noise=cfg.sensor_error*(2*keyed_uniform(sample_seed,"pending_measurement",packet.batch_id,cid)-1)
            records.append({"cell":cid,"version":version,"mean":min(1.,max(0.,env._quality[cid]+noise)),
                            "physical_revision":0,"samples":obs.inspection_repeat})
        restore_event(packet.delivery_time,packet.queue_order,"deliver",{"batch":packet.batch_id,"view":packet.view_id,
            "captured_at":packet.captured_at,"records":records})
        try:env._capture_ordinal[packet.view_id]=max(env._capture_ordinal.get(packet.view_id,0),int(packet.batch_id.rsplit(":",1)[1]))
        except (ValueError,IndexError):pass
    import heapq
    heapq.heapify(env._events)
    return env


def one_step_repair_predictions(obs, candidate, mode="cascade"):
    """Bernoulli predictions for observable completion-event audit targets.

    `collateral_event` predicts the simulator's damage draw, including an already
    bad cell; it is deliberately NOT called a newly discovered defect.
    """
    jobs={j.id:j for j in obs.jobs};rows=[]
    for action in candidate:
        if action.kind!="repair":continue
        j=jobs[action.target_id]
        targets=set(j.expanded_targets or j.targets) if action.mode=="expanded" else set(j.targets)
        impact=set(j.expanded_impact if action.mode=="expanded" else j.local_impact)
        for cid in sorted(impact):
            event="repair_failure" if cid in targets else "collateral_event"
            p=(obs.assumed_repair_failure_prob if mode=="cascade" else 0.) if cid in targets else (
                obs.assumed_collateral_probability if mode!="optimistic" else 0.)
            rows.append({"job":j.id,"mode":action.mode,"attempt":j.attempts+1,
                         "cell":cid,"event":event,"probability":p})
    return rows


@dataclass(frozen=True)
class Prediction:
    candidate_id: str
    mode: str
    samples: int
    completion_probability: float
    mean_J: float
    se_J: float
    mean_finish: float
    mean_repairs: float
    mean_inspections: float
    mean_discovered_repairs: float
    rollout_steps: int
    continuation: str="robust_local_after_single_intervention"
    approximation_flags: tuple=("scalar_discrete_uncertain_quality_prior", "assumed_radius_and_noise_are_model_inputs",
        "current_certificates_assumed_physically_valid", "pending_readouts_sampled_not_observed")

    def to_dict(self):return asdict(self)


class PublicPredictionModel:
    def __init__(self,mode="cascade",samples=8,seed=0,max_events=2000):
        if mode not in MODES:raise ValueError(mode)
        if samples<1 or max_events<1:raise ValueError("positive samples and events required")
        self.mode=mode;self.samples=samples;self.seed=seed;self.max_events=max_events

    def evaluate(self,obs,candidates):
        from .planners import PublicPlanner
        # A mode/candidate-independent seed gives common synthetic random worlds.
        digest=hashlib.sha256(json.dumps(asdict(obs),sort_keys=True,default=list).encode()).hexdigest()
        rows=[]
        for candidate in candidates:
            candidate=tuple(candidate)
            candidate_id=hashlib.sha256(json.dumps([asdict(a) for a in candidate],sort_keys=True).encode()).hexdigest()[:16]
            outcomes=[];step_count=0
            for sample in range(self.samples):
                sample_seed=int(hashlib.sha256(f"{self.seed}:{digest}:{sample}".encode()).hexdigest()[:15],16)
                env=public_belief_environment(obs,sample_seed,self.mode)
                planner=PublicPlanner("robust_local",seed=self.seed)
                # The reference policy remembers already-outstanding captures.
                for p in obs.pending_inspections:
                    planner._pending_inspections[p.view_id]=(p.delivery_time,dict(p.captured_versions))
                for a in obs.active:
                    if a.kind=="inspect":
                        planner._pending_inspections[a.target_id]=(a.finish+obs.measurement_delay,
                            {cid:next(c.version for c in obs.cells if c.id==cid) for cid in a.cells})
                if not env.done:
                    env.step_joint(candidate)
                steps=1
                while not env.done and steps<self.max_events:
                    current=env.observe()
                    actions=planner.select(current,env.allowed_actions())
                    env.step_joint(actions);steps+=1
                if not env.done:
                    raise RuntimeError("prediction event cap reached; no silently truncated return")
                result=env.result()
                result["new_jobs"]=len(set(env._jobs)-{j.id for j in obs.jobs})
                outcomes.append(result);step_count+=steps
            values=[r["J"] for r in outcomes]
            mean=lambda key:sum(r[key] for r in outcomes)/len(outcomes)
            rows.append(Prediction(candidate_id,self.mode,self.samples,mean("quality_success"),
                mean("J"),statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.,
                mean("time"),mean("repairs"),mean("inspections"),mean("new_jobs"),step_count))
        return rows
