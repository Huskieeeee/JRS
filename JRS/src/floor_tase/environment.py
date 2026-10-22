"""Concurrent, auditable recovery-frontier simulator.

World state and keyed random streams are private to the environment. Algorithms
receive only frozen Observation and Action values. Permission checks use the
public ledger; true quality/revisions are used only by evaluation and audit.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import heapq
import math
from copy import deepcopy
from .model import (Scenario, Action, Observation, CellState, RobotState, JobState,
                    TaskState, ActiveAction, RepairJob, PendingInspection, keyed_uniform, _footprint)


class InvalidAction(ValueError):
    pass


class RecoveryEnv:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.cfg = scenario.config
        self.time = 0.0
        self._geometry = {c.id:c for c in scenario.cells}
        self._quality = dict(scenario.initial_quality) or {c.id:.16 for c in scenario.cells}
        self._physical_revision = {c.id:0 for c in scenario.cells}
        targets={x for j in scenario.jobs for x in j.targets}
        self._ledger={c.id:{"status":"invalid" if c.id in targets else "valid", "version":0,
                      "mean":None if c.id in targets else self._quality[c.id],
                      "upper":None if c.id in targets else min(1,self._quality[c.id]+self.cfg.assumed_sensor_error),
                      "lower":None if c.id in targets else max(0,self._quality[c.id]-self.cfg.assumed_sensor_error),
                      "last_observed":None if c.id in targets else 0.0,
                      "batch_id":None if c.id in targets else "initial-certificate",
                      "samples":0 if c.id in targets else 1,
                      "physical_revision":0} for c in scenario.cells}
        self._robots={r.id:{"skill":r.skill,"position":r.start,"busy_until":0.0} for r in scenario.robots}
        self._jobs={j.id:{"job":j,"status":"pending","attempts":0} for j in scenario.jobs}
        self._tasks={t.id:{"task":t,"status":"pending","releases":0,"completion_revisions":None} for t in scenario.tasks}
        self._views={v.id:v for v in scenario.views}
        self._active={}
        self._events=[]
        self._event_index=0
        self._action_index=0
        self._capture_ordinal={}
        self._trace=[]
        self._terminated=False
        self._termination_reason=None
        self._stats={"inspections":0,"inspection_cell_observations":0,"duplicate_inspections":0,
                     "repairs":0,"repair_cells":0,"travel":0.0,"robot_busy_time":0.0,
                     "authorization_attempts":0,"authorizations":0,"false_releases":0,
                     "quality_false_releases":0,"stale_authorizations":0,"stale_packets_rejected":0,
                     "invalidated_evidence":0,"uncertain_evidence":0,"preserved_evidence":0,
                     "repeated_invalidations":0,"successor_revocations":0,"action_rejections":0,
                     "deadline_misses":0,"repair_cap_exhaustions":0,"measurement_packets":0}
        self._log("initial",scenario_id=scenario.id,cell_count=len(self._geometry),robot_count=len(self._robots),
                  tasks=len(self._tasks),evidence_policy=self.cfg.evidence_policy,
                  quality_threshold=self.cfg.quality_threshold,freshness=self.cfg.freshness)
        self._update_done()

    @property
    def done(self): return self._terminated

    def _log(self, event, **kwargs):
        self._trace.append({"seq":len(self._trace),"time":self.time,"event":event,**kwargs})

    def get_trace(self): return deepcopy(self._trace)

    def _push(self,time,kind,payload):
        self._event_index+=1
        heapq.heappush(self._events,(float(time),self._event_index,kind,payload))
        return self._event_index

    def _certified(self, cid):
        e=self._ledger[cid]
        return (e["status"]=="valid" and e["upper"] is not None
                and e["upper"]<=self.cfg.quality_threshold and e["last_observed"] is not None
                and self.time-e["last_observed"]<=self.cfg.freshness)

    def _expire(self):
        for cid,e in self._ledger.items():
            if e["status"]=="valid" and e["last_observed"] is not None and self.time-e["last_observed"]>self.cfg.freshness:
                # Age invalidates this certificate, not the physical epoch.
                # A newer in-flight capture of the same unchanged surface may
                # still arrive fresh and must retain its declared version.
                e["status"]="uncertain"
                self._log("evidence_expired",cell=cid,version=e["version"])

    def observe(self):
        self._expire()
        cells=tuple(CellState(c.id,c.position,c.region,c.skill,self._ledger[c.id]["status"],
                              self._ledger[c.id]["version"],self._certified(c.id),self._ledger[c.id]["mean"],
                              self._ledger[c.id]["upper"],self._ledger[c.id]["last_observed"],
                              self._ledger[c.id]["lower"],self._ledger[c.id]["batch_id"],self._ledger[c.id]["samples"])
                    for c in self.scenario.cells)
        robots=tuple(RobotState(rid,r["skill"],r["position"],r["busy_until"]) for rid,r in self._robots.items())
        jobs=tuple(JobState(j.id,j.skill,j.targets,j.local_impact,j.expanded_impact,j.duration,j.expanded_duration,
                            j.expanded_targets,j.position,d["status"],d["attempts"])
                   for d in self._jobs.values() for j in (d["job"],))
        tasks=tuple(TaskState(t.id,t.skill,t.requires,t.duration,t.deadline,t.position,d["status"])
                    for d in self._tasks.values() for t in (d["task"],))
        return Observation(self.time,self.cfg.horizon,robots,cells,jobs,tuple(self._views.values()),tasks,
                           tuple(x["public"] for x in self._active.values()),max(0,self.cfg.horizon-self.time),
                           self.cfg.quality_threshold,self.cfg.assumed_sensor_error,self.cfg.travel_speed,
                           self.cfg.measurement_delay,self.cfg.evidence_policy,
                           sum(1 for e in self._events if e[2]=="deliver"),self.done,self.cfg.freshness,
                           assumed_radius=self.cfg.assumed_radius,
                           assumed_collateral_probability=self.cfg.assumed_collateral_probability,
                           assumed_repair_failure_prob=self.cfg.assumed_repair_failure_prob,
                           shared_passage=self.cfg.shared_passage,repair_duration=self.cfg.repair_duration,
                           max_repairs_per_cell=self.cfg.max_repairs_per_cell,
                           inspection_repeat=self.cfg.inspection_repeat,
                           pending_inspections=tuple(PendingInspection(p["view"],p["batch"],p["captured_at"],when,
                               tuple((r["cell"],r["version"]) for r in p["records"]),order)
                               for when,order,kind,p in self._events if kind=="deliver"),
                           accrued_travel=self._stats["travel"],accrued_repairs=self._stats["repairs"])

    snapshot=observe

    def _motion(self,robot_id,position):
        start=self._robots[robot_id]["position"]
        distance=abs(start[0]-position[0])+abs(start[1]-position[1])
        travel=distance/self.cfg.travel_speed
        # A shared narrow passage divides the floor. It is occupied only in transit.
        max_x=max(c.x for c in self.scenario.cells)
        divider=max_x/2
        crossing=(start[0]<=divider<position[0]) or (position[0]<=divider<start[0])
        return distance,travel,"main_passage" if self.cfg.shared_passage and crossing and travel>0 else None

    def _spec(self,a):
        if a.robot_id not in self._robots: raise InvalidAction("unknown robot")
        r=self._robots[a.robot_id]
        if self.done or r["busy_until"]>self.time+1e-9: raise InvalidAction("robot unavailable")
        if a.kind=="repair":
            if a.target_id not in self._jobs: raise InvalidAction("unknown repair")
            d=self._jobs[a.target_id];j=d["job"]
            if d["status"]!="pending": raise InvalidAction("repair not pending")
            if r["skill"]!=j.skill: raise InvalidAction("repair skill mismatch")
            if a.mode not in ("local","expanded"): raise InvalidAction("unknown repair scope")
            if d["attempts"]>=self.cfg.max_repairs_per_cell: raise InvalidAction("repair attempt limit")
            target=j.targets if a.mode=="local" else (j.expanded_targets or j.targets)
            impact=j.local_impact if a.mode=="local" else j.expanded_impact
            duration=j.duration if a.mode=="local" else j.expanded_duration
            position=j.position
        elif a.kind=="inspect":
            if a.target_id not in self._views: raise InvalidAction("unknown inspection view")
            v=self._views[a.target_id]
            if r["skill"]!=v.skill: raise InvalidAction("inspection skill mismatch")
            target=v.covers;impact=v.covers;duration=v.duration;position=v.position
            # Useless inspections are not masked: repetitive policies remain diagnosable.
        elif a.kind=="successor":
            if a.target_id not in self._tasks: raise InvalidAction("unknown task")
            d=self._tasks[a.target_id];t=d["task"]
            if d["status"]!="pending": raise InvalidAction("successor not pending")
            if r["skill"]!=t.skill: raise InvalidAction("successor skill mismatch")
            if not all(self._certified(c) for c in t.requires): raise InvalidAction("missing current certificate")
            target=t.requires;impact=t.requires;duration=t.duration;position=t.position
        else: raise InvalidAction("unknown action kind")
        distance,travel,channel=self._motion(a.robot_id,position)
        arrival=self.time+travel;finish=arrival+duration
        if finish>self.cfg.horizon+1e-9: raise InvalidAction("action exceeds horizon")
        if a.kind=="successor" and finish>self._tasks[a.target_id]["task"].deadline+1e-9:
            raise InvalidAction("action misses task deadline")
        p=ActiveAction(a.kind,a.robot_id,a.target_id,a.mode,self.time,arrival,finish,tuple(impact),channel)
        return {"public":p,"action":a,"targets":tuple(target),"position":position,"distance":distance}

    @staticmethod
    def _overlap(s1,e1,s2,e2): return max(s1,s2)<min(e1,e2)-1e-9

    def _conflict(self,x,y):
        a=x["public"];b=y["public"]
        if a.robot_id==b.robot_id: return True
        if a.channel and a.channel==b.channel and self._overlap(a.start,a.arrival,b.start,b.arrival): return True
        if set(a.cells)&set(b.cells) and (a.kind!="inspect" or b.kind!="inspect"):
            # All affected cells stay reserved from dispatch, including travel.
            return self._overlap(a.start,a.finish,b.start,b.finish)
        return False

    def allowed_actions(self):
        self._expire()
        if self.done: return ()
        candidates=[]
        for rid,r in self._robots.items():
            if r["busy_until"]>self.time+1e-9: continue
            candidates.extend(Action("repair",rid,jid,m) for jid,d in self._jobs.items()
                              if d["status"]=="pending" and d["job"].skill==r["skill"] for m in ("local","expanded"))
            candidates.extend(Action("inspect",rid,v.id) for v in self._views.values() if v.skill==r["skill"])
            candidates.extend(Action("successor",rid,t.id) for d in self._tasks.values() for t in (d["task"],)
                              if d["status"]=="pending" and t.skill==r["skill"])
        valid=[]
        for a in candidates:
            try: s=self._spec(a)
            except InvalidAction: continue
            if not any(self._conflict(s,x) for x in self._active.values()): valid.append(a)
        return tuple(valid)

    def _invalidate(self,s):
        a=s["action"];targets=set(s["targets"]);public=set(s["public"].cells)
        if self.cfg.evidence_policy=="whole_reset": affected=set(self._ledger)
        elif self.cfg.evidence_policy=="optimistic_local": affected=targets
        else: affected=public
        changed=[]
        for cid,e in self._ledger.items():
            if cid in affected:
                if e["status"]=="valid": self._stats["invalidated_evidence"]+=1
                elif e["version"]>0: self._stats["repeated_invalidations"]+=1
                e["status"]="invalid" if cid in targets else "uncertain"
                e["version"]+=1
                changed.append((cid,e["version"],e["status"]))
                if e["status"]=="uncertain": self._stats["uncertain_evidence"]+=1
            elif e["status"]=="valid": self._stats["preserved_evidence"]+=1
        for tid,d in self._tasks.items():
            if d["status"]=="done" and affected.intersection(d["task"].requires):
                d["status"]="pending";d["completion_revisions"]=None
                self._stats["successor_revocations"]+=1
                self._log("successor_revoked",task=tid,cause=a.target_id)
        self._log("evidence_invalidation",job=a.target_id,mode=a.mode,policy=self.cfg.evidence_policy,
                  affected=sorted(affected),targets=sorted(targets),versions=changed)

    def _dispatch(self,s):
        a=s["action"];p=s["public"]
        self._action_index+=1;key=f"a{self._action_index}"
        s["key"]=key
        if a.kind=="repair":
            self._jobs[a.target_id]["status"]="running"
            self._jobs[a.target_id]["attempts"]+=1
            s["attempt"]=self._jobs[a.target_id]["attempts"]
            self._invalidate(s)
            self._stats["repairs"]+=1;self._stats["repair_cells"]+=len(s["targets"])
        elif a.kind=="inspect":
            self._stats["inspections"]+=1
            if all(self._certified(c) for c in s["targets"]): self._stats["duplicate_inspections"]+=1
        else:
            # A certificate must still be valid at arrival (travel is not permission).
            self._tasks[a.target_id]["status"]="travelling"
            authorization_order=self._push(p.arrival,"authorize",key)
            p=replace(p,authorization_order=authorization_order)
        self._active[key]=s
        self._robots[a.robot_id]["busy_until"]=p.finish
        self._stats["travel"]+=s["distance"]
        self._stats["robot_busy_time"]+=p.finish-p.start
        completion_order=self._push(p.finish,"complete",key)
        p=replace(p,completion_order=completion_order)
        s["public"]=p
        self._log("action_start",action_id=key,**asdict(p),targets=list(s["targets"]),distance=s["distance"])

    def step_joint(self,actions):
        """Atomically validate a joint dispatch; advance to the earliest event.

        Conflicting actions reject the entire set, leaving physical state intact.
        Empty dispatch is an event wait. When no event remains it ends at H.
        """
        if self.done: return self.observe()
        if isinstance(actions,Action): actions=(actions,)
        actions=tuple(actions)
        try:
            specs=[self._spec(a) for a in actions]
            for i,s in enumerate(specs):
                if any(self._conflict(s,x) for x in list(self._active.values())+specs[:i]):
                    raise InvalidAction("joint resource or intervention conflict")
        except InvalidAction as e:
            self._stats["action_rejections"]+=1
            self._log("action_rejected",reason=str(e))
            raise
        for s in specs: self._dispatch(s)
        return self.advance()

    def _authorize(self,key):
        if key not in self._active: return
        s=self._active[key];a=s["action"];d=self._tasks[a.target_id]
        self._stats["authorization_attempts"]+=1
        cells=d["task"].requires
        permitted=all(self._certified(c) for c in cells)
        if not permitted:
            d["status"]="pending";s["denied"]=True
            self._log("authorization_denied",task=a.target_id,reason="certificate invalid or expired at arrival")
            return
        stale=[c for c in cells if self._ledger[c]["physical_revision"]!=self._physical_revision[c]]
        bad=[c for c in cells if self._quality[c]>self.cfg.quality_threshold]
        self._stats["authorizations"]+=1
        self._stats["false_releases"]+=int(bool(stale or bad))
        self._stats["quality_false_releases"]+=int(bool(bad))
        self._stats["stale_authorizations"]+=int(bool(stale))
        d["status"]="running";d["releases"]+=1
        s["authorized_revisions"]={c:self._physical_revision[c] for c in cells}
        s["authorized_declared_versions"]={c:self._ledger[c]["version"] for c in cells}
        s["authorized_capture_times"]={c:self._ledger[c]["last_observed"] for c in cells}
        s["public"]=replace(s["public"],
            authorization_versions=tuple(s["authorized_declared_versions"].items()),
            authorization_capture_times=tuple(s["authorized_capture_times"].items()))
        self._log("authorization",task=a.target_id,robot=a.robot_id,
                  certificates={c:{"version":self._ledger[c]["version"],"upper":self._ledger[c]["upper"],
                                    "timestamp":self._ledger[c]["last_observed"],"batch":self._ledger[c]["batch_id"]} for c in cells},
                  evaluator_stale_cells=stale,evaluator_bad_cells=bad,permitted=True)

    def _repair_complete(self,s):
        a=s["action"];targets=s["targets"];attempt=s["attempt"]
        # The radius is an uncertain physical upper bound, not a planner input.
        radius=self.cfg.impact_radius
        actual=_footprint(self.scenario.cells,targets,radius)
        affected_bad=[];failed_targets=[]
        for cid in actual:
            self._physical_revision[cid]+=1
            if cid in targets:
                fail=keyed_uniform(self.scenario.seed,"repair_fail",a.target_id,attempt,cid)<self.cfg.repair_failure_prob
                if fail:failed_targets.append(cid)
                self._quality[cid]=.78 if fail else .12+.06*keyed_uniform(self.scenario.seed,"repair_finish",a.target_id,attempt,cid)
            elif keyed_uniform(self.scenario.seed,"collateral",a.target_id,attempt,cid)<self.cfg.collateral_probability:
                self._quality[cid]=.68+.12*keyed_uniform(self.scenario.seed,"damage_level",a.target_id,attempt,cid)
                affected_bad.append(cid)
        for jid,d in self._jobs.items():
            if set(d["job"].targets).issubset(targets) and d["status"] in {"pending","running"}: d["status"]="done"
        self._log("repair_complete",job=a.target_id,mode=a.mode,attempt=attempt,
                  evaluator_actual_impact=list(actual),evaluator_bad_collateral=affected_bad,
                  evaluator_failed_targets=failed_targets,
                  declared_impact=list(s["public"].cells))

    def _capture(self,s):
        a=s["action"];view=self._views[a.target_id]
        ordinal=self._capture_ordinal.get(view.id,0)+1;self._capture_ordinal[view.id]=ordinal
        batch=f"{view.id}:{ordinal}"
        mseed=self.scenario.measurement_seed if self.scenario.measurement_seed is not None else self.scenario.seed
        records=[]
        for cid in view.covers:
            samples=[min(1,max(0,self._quality[cid]+self.cfg.sensor_bias+self.cfg.sensor_error*(2*keyed_uniform(mseed,"measurement",view.id,ordinal,cid,k)-1)))
                     for k in range(self.cfg.inspection_repeat)]
            mean=sum(samples)/len(samples)
            records.append({"cell":cid,"version":self._ledger[cid]["version"],"mean":mean,
                            "physical_revision":self._physical_revision[cid],"samples":len(samples)})
        packet={"batch":batch,"view":view.id,"captured_at":self.time,"records":records}
        self._stats["inspection_cell_observations"]+=len(records)
        self._stats["measurement_packets"]+=1
        self._push(self.time+self.cfg.measurement_delay,"deliver",packet)
        self._log("inspection_capture",batch=batch,view=view.id,cells=list(view.covers),
                  versions={r["cell"]:r["version"] for r in records},delivery=self.time+self.cfg.measurement_delay)

    def _ensure_repair(self,cid):
        if any(cid in d["job"].targets and d["status"] in {"pending","running"} for d in self._jobs.values()): return
        c=self._geometry[cid];jid="j_"+cid
        if jid in self._jobs:
            d=self._jobs[jid]
            if d["attempts"]>=self.cfg.max_repairs_per_cell:
                d["status"]="exhausted";self._stats["repair_cap_exhaustions"]+=1
                return
            d["status"]="pending"
        else:
            expanded=tuple(x.id for x in self.scenario.cells if x.skill==c.skill and max(abs(x.x-c.x),abs(x.y-c.y))<=2)
            j=RepairJob(jid,c.skill,(cid,),_footprint(self.scenario.cells,(cid,),self.cfg.assumed_radius),
                        _footprint(self.scenario.cells,expanded,self.cfg.assumed_radius),self.cfg.repair_duration,
                        self.cfg.repair_duration*(1+.35*max(1,len(expanded)-1)),expanded,c.position)
            self._jobs[jid]={"job":j,"status":"pending","attempts":0}
        self._log("repair_discovered",job=jid,cell=cid)

    def _deliver(self,p):
        accepted=[];rejected=[]
        for r in p["records"]:
            cid=r["cell"];e=self._ledger[cid]
            if (r["version"]!=e["version"] or self.time-p["captured_at"]>self.cfg.freshness
                    or (e["last_observed"] is not None and p["captured_at"]<e["last_observed"]-1e-9)):
                rejected.append(cid);self._stats["stale_packets_rejected"]+=1
                continue
            e.update(mean=r["mean"],upper=min(1,r["mean"]+self.cfg.assumed_sensor_error),
                     lower=max(0,r["mean"]-self.cfg.assumed_sensor_error),last_observed=p["captured_at"],
                     batch_id=p["batch"],samples=r["samples"],physical_revision=r["physical_revision"])
            if e["upper"]<=self.cfg.quality_threshold: e["status"]="valid"
            elif e["lower"]>self.cfg.quality_threshold:
                e["status"]="invalid";self._ensure_repair(cid)
            else: e["status"]="uncertain"
            accepted.append(cid)
        self._log("inspection_delivery",batch=p["batch"],captured_at=p["captured_at"],accepted=accepted,rejected=rejected,
                  certificates={c:{"status":self._ledger[c]["status"],"version":self._ledger[c]["version"],
                                   "upper":self._ledger[c]["upper"],"timestamp":self._ledger[c]["last_observed"]} for c in accepted})

    def _complete(self,key):
        s=self._active.pop(key,None)
        if s is None:return
        a=s["action"]
        self._robots[a.robot_id]["position"]=s["position"]
        self._robots[a.robot_id]["busy_until"]=self.time
        if a.kind=="repair":self._repair_complete(s)
        elif a.kind=="inspect":self._capture(s)
        elif not s.get("denied"):
            d=self._tasks[a.target_id]
            if "authorized_revisions" not in s:
                raise RuntimeError("successor completed without authorization event")
            intervention_free=all(s["authorized_declared_versions"][c]==self._ledger[c]["version"] for c in d["task"].requires)
            certificate_fresh=all(self.time-s["authorized_capture_times"][c]<=self.cfg.freshness+1e-9
                                  for c in d["task"].requires)
            current=intervention_free and certificate_fresh
            d["status"]="done" if current else "pending"
            d["completion_revisions"]=dict(s["authorized_revisions"]) if current else None
            self._log("successor_complete",task=a.target_id,intervention_free=intervention_free,
                      authorization_certificates_fresh_at_finish=certificate_fresh,
                      completion_accepted=current)
        self._log("action_complete",action_id=key,kind=a.kind,robot_id=a.robot_id,target_id=a.target_id)

    def _update_done(self):
        if self._terminated:return
        if all(d["status"]=="done" for d in self._tasks.values()) and not self._active:
            self._terminated=True;self._termination_reason="complete"
        elif self.time>=self.cfg.horizon-1e-9:
            self._terminated=True;self._termination_reason="horizon"
        else:
            for tid,d in self._tasks.items():
                if d["status"]=="pending" and self.time+d["task"].duration>d["task"].deadline+1e-9:
                    d["status"]="missed";self._stats["deadline_misses"]+=1
                    self._log("deadline_missed",task=tid,deadline=d["task"].deadline)
            if not self._active and all(d["status"] in {"done","missed"} for d in self._tasks.values()):
                self._terminated=True;self._termination_reason="deadline"
        if self._terminated:self._log("termination",reason=self._termination_reason)

    def advance(self):
        if self.done:return self.observe()
        if not self._events:
            # A policy that deliberately idles does not receive cheap early failure.
            self.time=self.cfg.horizon;self._update_done();return self.observe()
        nexttime=self._events[0][0]
        if nexttime>self.cfg.horizon:
            self.time=self.cfg.horizon;self._update_done();return self.observe()
        self.time=nexttime;self._expire()
        while self._events and self._events[0][0]<=self.time+1e-9:
            _,_,kind,payload=heapq.heappop(self._events)
            if kind=="complete":self._complete(payload)
            elif kind=="deliver":self._deliver(payload)
            elif kind=="authorize":self._authorize(payload)
        self._update_done()
        return self.observe()

    def result(self):
        completed=sum(d["status"]=="done" for d in self._tasks.values())
        qualified=0
        for d in self._tasks.values():
            if d["status"]!="done":continue
            if all(self._quality[c]<=self.cfg.quality_threshold and d["completion_revisions"] is not None
                   and d["completion_revisions"][c]==self._physical_revision[c] for c in d["task"].requires): qualified+=1
        success=completed==len(self._tasks)
        quality_success=success and qualified==len(self._tasks) and self._stats["false_releases"]==0
        charged=self.time if quality_success else self.cfg.horizon
        unfinished=1-qualified/max(1,len(self._tasks))
        cost=charged/self.cfg.horizon+.05*self._stats["travel"]/max(1,30*len(self._geometry))+.02*self._stats["repairs"]/max(1,len(self._geometry))+4*unfinished
        result={"scenario_id":self.scenario.id,"seed":self.scenario.seed,"time":self.time,"charged_time":charged,
                "horizon":self.cfg.horizon,"cost":cost,"J":cost,"success":success,"quality_success":quality_success,
                "qualified_completion":quality_success,"completed_tasks":completed,"qualified_tasks":qualified,
                "total_tasks":len(self._tasks),"unfinished_fraction":unfinished,"done":self.done,
                "termination_reason":self._termination_reason,"refused_or_unfinished":not quality_success,
                "timeout":self._termination_reason=="horizon","evidence_policy":self.cfg.evidence_policy,
                "waiting_time":max(0,len(self._robots)*self.time-self._stats["robot_busy_time"]),
                "safe":self._stats["false_releases"]==0,**self._stats}
        result["audit"]=audit_trace(self._trace)
        return result


def audit_trace(trace):
    """Independent structural trace audit, separate from evaluator truth flags.

    A pass verifies dispatch/permission bookkeeping, not simulator realism or
    any deployment-level safety guarantee. True safety statistics are separate.
    """
    errors=[];starts={};completed=set();versions={};deliveries={};authorizations=0
    authorization_by_task={}
    robot_intervals={};channel_intervals=[]
    initial=next((e for e in trace if e["event"]=="initial"),{})
    threshold=initial.get("quality_threshold")
    freshness=initial.get("freshness")
    for e in trace:
        if e["event"]=="action_start":
            aid=e["action_id"]
            if aid in starts:errors.append("duplicate action id")
            starts[aid]=e
            robot_intervals.setdefault(e["robot_id"],[]).append((e["start"],e["finish"],aid))
            if e.get("channel"):channel_intervals.append((e["start"],e["arrival"],aid))
        elif e["event"]=="action_complete":
            if e["action_id"] not in starts:errors.append("complete without start")
            completed.add(e["action_id"])
        elif e["event"]=="evidence_invalidation":
            for cid,v,status in e["versions"]:versions[cid]=v
        elif e["event"]=="evidence_expired":versions[e["cell"]]=e["version"]
        elif e["event"]=="inspection_delivery":
            for cid,c in e["certificates"].items():
                if c["version"]!=versions.get(cid,0):errors.append("accepted stale declared version")
                deliveries[cid]=c
        elif e["event"]=="authorization":
            authorization_by_task[e["task"]]=e["certificates"]
            authorizations+=1
            for cid,c in e["certificates"].items():
                if c["version"]!=versions.get(cid,0):errors.append("authorization uses obsolete declared version")
                if threshold is not None and (c["upper"] is None or c["upper"]>threshold):
                    errors.append("authorization lacks passing quality bound")
                if c["timestamp"] is None or c["timestamp"]>e["time"]+1e-9:
                    errors.append("authorization has invalid capture timestamp")
                elif freshness is not None and e["time"]-c["timestamp"]>freshness+1e-9:
                    errors.append("authorization certificate expired")
        elif e["event"]=="successor_complete" and e.get("completion_accepted"):
            certificates=authorization_by_task.get(e["task"])
            if certificates is None:
                errors.append("accepted completion lacks prior authorization")
            else:
                for cid,c in certificates.items():
                    if c["version"]!=versions.get(cid,0):
                        errors.append("accepted completion crosses an intervention version")
                    if c["timestamp"] is None or (freshness is not None and
                            e["time"]-c["timestamp"]>freshness+1e-9):
                        errors.append("accepted completion uses expired authorization certificate")
    for intervals in list(robot_intervals.values())+[channel_intervals]:
        intervals.sort()
        for i,x in enumerate(intervals):
            if any(max(x[0],y[0])<min(x[1],y[1])-1e-9 for y in intervals[:i]):errors.append("resource interval overlap")
    return {"pass":not errors,"errors":errors,"events":len(trace),"actions":len(starts),
            "completed_actions":len(completed),"authorizations":authorizations,
            "scope":"declared versions, authorization bounds/freshness, and resource bookkeeping; truth outcomes separate"}
