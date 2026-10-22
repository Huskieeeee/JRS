"""Schedulers using only Observation and advertised legal actions.

JRS enumerates a restricted two-checkpoint intervention master. Each candidate
chooses a repair prefix and local/expanded scope, then solves a public weighted
inspection-cover subproblem at each checkpoint. A public deterministic event
forecast includes shared robot occupancy, travel, measurement delay, evidence
revocation and successor redo/deadlines. Its first feasible event is executed;
the model is rebuilt after every real event. This is a receding-horizon
heuristic, not an optimal POMDP or a learned policy. All outcomes in the forecast
are optimistic public predictions; no simulator state is copied or sampled.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools, math, random, time
from .model import Action
from .optimizer import CoverOption,solve_cover

METHODS=("qa_sequential","robust_local","whole_reset","overlap_batch",
         "independent","random_rollout","jrs")

def _dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
def _akey(a):return (a.kind,a.robot_id,a.target_id,a.mode)
def _impact(job,mode):return set(job.expanded_impact if mode=="expanded" else job.local_impact)
def _fixes(job,mode):return set(job.expanded_targets if mode=="expanded" and job.expanded_targets else job.targets)

def _compatible(action,chosen,obs):
    if any(a.robot_id==action.robot_id for a in chosen):return False
    jobs={j.id:j for j in obs.jobs};views={v.id:v for v in obs.views};tasks={t.id:t for t in obs.tasks}
    def footprint(a):
        if a.kind=="repair":return _impact(jobs[a.target_id],a.mode)
        if a.kind=="inspect":return set(views[a.target_id].covers)
        return set(tasks[a.target_id].requires)
    cells=footprint(action)
    robots={r.id:r for r in obs.robots}
    divider=max(c.position[0] for c in obs.cells)/2
    def crossing(a):
        start=robots[a.robot_id].position
        target=(jobs[a.target_id].position if a.kind=="repair" else
                views[a.target_id].position if a.kind=="inspect" else tasks[a.target_id].position)
        return ((start[0]<=divider<target[0]) or (target[0]<=divider<start[0])) and _dist(start,target)>0
    # Both actions depart now, so crossing intervals overlap whenever both
    # cross the single public passage. Single-action active conflicts were
    # already checked by the common advertised legal-action interface.
    if obs.shared_passage and crossing(action) and any(crossing(a) for a in chosen):return False
    return not any((a.kind!="inspect" or action.kind!="inspect") and cells&footprint(a) for a in chosen)

def _pack(ordered,obs,allowed):
    legal={_akey(a) for a in allowed};chosen=[]
    for a in ordered:
        if _akey(a) in legal and _compatible(a,chosen,obs):chosen.append(a)
    return tuple(chosen)

@dataclass
class Forecast:
    score: tuple
    first: tuple
    nodes: int
    cover_statuses: tuple
    predicted_finish: float
    predicted_misses: int
    inspection_work: float
    candidates: int=1

class PublicPlanner:
    def __init__(self,name,seed=0,time_limit_s=.05):
        if name not in METHODS+("jrs_no_batch","jrs_no_joint","jrs_no_uncertainty","jrs_no_reuse"):
            raise ValueError(f"Unknown planner {name}; choose {METHODS}")
        self.name=name;self.rng=random.Random(seed);self.time_limit_s=time_limit_s
        self.stats={"calls":0,"search_nodes":0,"total_wall_s":0.,"cover_timeout":0,
                    "master_candidates":0,"master_truncated":0,"forecast_infeasible":0,
                    "public_information_only":True,"exact_mission_solver":False}
        self.last_diagnostics={};self._immediate_job=None;self._overlap_ids=set();self._pending_inspections={}

    def diagnostics(self):return dict(self.stats)

    def select(self,obs,allowed_actions):
        start=time.perf_counter();self.stats["calls"]+=1
        # Common delivery-aware guard for every comparator. A robot remembers
        # its own outstanding capture and public delivery time; no hidden data
        # or oracle visibility is used. Versions changed by a repair release the
        # guard because the delayed packet can no longer certify those cells.
        versions={c.id:c.version for c in obs.cells}
        self._pending_inspections={vid:(until,old) for vid,(until,old) in self._pending_inspections.items()
                                   if until>obs.time+1e-9 and all(versions.get(cid)==v for cid,v in old.items())}
        allowed=tuple(a for a in allowed_actions if not(a.kind=="inspect" and a.target_id in self._pending_inspections))
        if not allowed:
            self.stats["total_wall_s"]+=time.perf_counter()-start
            return ()
        if self.name.startswith("jrs") or self.name=="random_rollout":
            chosen=self._joint(obs,allowed,start)
        else:chosen=self._rule(obs,allowed)
        views={v.id:v for v in obs.views};robots={r.id:r for r in obs.robots}
        for action in chosen:
            if action.kind=="inspect":
                view=views[action.target_id];robot=robots[action.robot_id]
                until=obs.time+_dist(robot.position,view.position)/obs.travel_speed+view.duration+obs.measurement_delay
                self._pending_inspections[view.id]=(until,{cid:versions[cid] for cid in view.covers})
        self.stats["total_wall_s"]+=time.perf_counter()-start
        return tuple(chosen)

    def _rule(self,obs,allowed):
        jobs={j.id:j for j in obs.jobs};views={v.id:v for v in obs.views}
        robots={r.id:r for r in obs.robots};tasks={t.id:t for t in obs.tasks}
        uncert={c.id for c in obs.cells if not c.certified}
        defects=set().union(*(set(j.targets) for j in obs.jobs if j.status in ("pending","running")))
        def score(a):
            robot=robots[a.robot_id]
            if a.kind=="successor":
                t=tasks[a.target_id];return (0,t.deadline,_dist(robot.position,t.position),a.target_id)
            if a.kind=="repair":
                j=jobs[a.target_id]
                priority=1
                if self.name=="qa_sequential" and self._immediate_job is not None and a.target_id!=self._immediate_job:
                    prior=jobs.get(self._immediate_job)
                    if prior and prior.status=="done" and uncert&set(prior.local_impact):priority=4
                if self.name=="overlap_batch" and self._overlap_ids and a.target_id not in self._overlap_ids:priority=4
                return (priority,j.duration+_dist(robot.position,j.position)/obs.travel_speed,
                        a.mode!="local",a.target_id)
            v=views[a.target_id];gain=len(set(v.covers)&(uncert-defects))
            if not gain:return (9,0,0,a.target_id)
            priority=2
            if self.name=="qa_sequential" and self._immediate_job is not None:
                prior=jobs.get(self._immediate_job)
                if prior and prior.status=="done" and set(v.covers)&set(prior.local_impact)&uncert:priority=.5
            if self.name=="independent":priority=.75
            if self.name=="overlap_batch" and self._overlap_ids and all(jobs[x].status=="done" for x in self._overlap_ids if x in jobs):priority=.5
            return (priority,(v.duration+_dist(robot.position,v.position)/obs.travel_speed)/gain,0,a.target_id)
        ranked=sorted(allowed,key=score)
        chosen=_pack([a for a in ranked if score(a)[0]<9],obs,allowed)
        if self.name=="qa_sequential":
            repairs=[a for a in chosen if a.kind=="repair"]
            if repairs:
                self._immediate_job=repairs[0].target_id
                # Immediate baseline commits to one intervention at a time.
                chosen=tuple(a for a in chosen if a.kind!="repair" or a==repairs[0])
        if self.name=="overlap_batch":
            if self._overlap_ids and all(jobs[x].status=="done" for x in self._overlap_ids if x in jobs):
                impact=set().union(*(set(jobs[x].local_impact) for x in self._overlap_ids if x in jobs))
                if not impact&uncert:self._overlap_ids=set()
            if not self._overlap_ids:
                repair=next((a for a in chosen if a.kind=="repair"),None)
                if repair:
                    anchor=set(jobs[repair.target_id].local_impact)
                    self._overlap_ids={j.id for j in obs.jobs if j.status=="pending" and anchor&set(j.local_impact)}
        return chosen

    def _forecast(self,obs,prefix,scope,cover_budget,cover_nodes=10000):
        """Two-checkpoint public surrogate; whole-reset respected here too."""
        jobs={j.id:j for j in obs.jobs if j.status=="pending"};tasks={t.id:t for t in obs.tasks if t.status in ("pending","done")}
        robots={r.id:r for r in obs.robots};clock={r.id:max(obs.time,r.busy_until) for r in obs.robots}
        pos={r.id:r.position for r in obs.robots};valid={c.id for c in obs.cells if c.certified}
        cell_ids={c.id for c in obs.cells};done={t.id for t in obs.tasks if t.status=="done"}
        remaining=set(jobs);events=[];space_intervals=[];travel_intervals=[];nodes=0;statuses=[];work=0.;misses=0
        all_jobs={j.id:j for j in obs.jobs};all_views={v.id:v for v in obs.views};all_tasks={t.id:t for t in obs.tasks}
        for active in obs.active:
            space_intervals.append((active.start,active.finish,set(active.cells),active.kind))
            if obs.shared_passage and active.channel and active.arrival>obs.time:travel_intervals.append((active.start,active.arrival))
            target=(all_jobs.get(active.target_id) if active.kind=="repair" else
                    all_views.get(active.target_id) if active.kind=="inspect" else all_tasks.get(active.target_id))
            if target is not None:pos[active.robot_id]=target.position
        stage_time=obs.time
        def assign(kind,oid,skill,position,duration,release,mode="local",affected=()):
            choices=[(max(clock[r.id],release)+_dist(pos[r.id],position)/obs.travel_speed,r.id)
                     for r in obs.robots if r.skill==skill]
            if not choices:return math.inf
            arrival,rid=min(choices);start=max(clock[rid],release);end=arrival+duration
            divider=max(c.position[0] for c in obs.cells)/2
            crossing=((pos[rid][0]<=divider<position[0]) or (position[0]<=divider<pos[rid][0]))
            travel=arrival-start
            if kind=="inspect":affected=set(all_views[oid].covers)
            elif kind=="successor":affected=set(all_tasks[oid].requires)
            changed=True
            while changed:
                changed=False
                if obs.shared_passage and crossing and travel>0:
                    for ta,tb in travel_intervals:
                        if start<tb-1e-9 and start+travel>ta+1e-9:
                            start=tb;changed=True
                end=start+travel+duration
                for ta,tb,foot,other_kind in space_intervals:
                    if set(affected)&foot and (kind!="inspect" or other_kind!="inspect") and start<tb-1e-9 and end>ta+1e-9:
                        start=tb;changed=True;end=start+travel+duration
            arrival=start+travel;end=arrival+duration
            space_intervals.append((start,end,set(affected),kind))
            if obs.shared_passage and crossing and travel>0:travel_intervals.append((start,arrival))
            events.append((start,Action(kind,rid,oid,mode)));clock[rid]=end;pos[rid]=position
            return end
        def repair_group(ids,release):
            nonlocal valid,done,work
            finishes=[release]
            for jid in ids:
                if jid not in remaining:continue
                j=jobs[jid];mode=scope if prefix and jid==prefix[0] else "local"
                fixed=_fixes(j,mode);affected=_impact(j,mode)
                if obs.evidence_policy=="whole_reset":affected=cell_ids
                duration=j.expanded_duration if mode=="expanded" else j.duration
                finishes.append(assign("repair",jid,j.skill,j.position,duration,release,mode,affected));work+=duration
                valid-=affected
                done-={tid for tid in done if set(tasks[tid].requires)&affected}
                remaining.difference_update(x for x in list(remaining) if set(jobs[x].targets)<=fixed)
            return max(finishes)
        def checkpoint(release):
            nonlocal valid,done,nodes,work,misses
            defects=set().union(*(set(jobs[x].targets) for x in remaining)) if remaining else set()
            needs=set().union(*(set(t.requires) for tid,t in tasks.items() if tid not in done)) if tasks else set()
            required=(needs-valid)-defects
            options=[]
            for v in obs.views:
                peers=[r for r in obs.robots if r.skill==v.skill]
                if peers:
                    # Fixed view cost is a travel-aware static cover surrogate.
                    travel=min(_dist(pos[r.id],v.position)/obs.travel_speed for r in peers)
                    options.append(CoverOption(v.id,frozenset(v.covers),v.duration+travel))
            cover=solve_cover(required,options,time_limit_s=cover_budget,max_nodes=cover_nodes)
            nodes+=cover.nodes;statuses.append(cover.status)
            if not cover.feasible:return math.inf
            by_id={v.id:v for v in obs.views};finishes=[release]
            for vid in cover.selected:
                v=by_id[vid];finishes.append(assign("inspect",vid,v.skill,v.position,v.duration,release)+obs.measurement_delay)
                valid|=set(v.covers)-defects;work+=v.duration
            inspection_end=max(finishes)
            # A handoff may be repeated if a later intervention revokes it.
            for t in sorted(tasks.values(),key=lambda x:(x.deadline,x.id)):
                if t.id not in done and set(t.requires)<=valid:
                    end=assign("successor",t.id,t.skill,t.position,t.duration,inspection_end)
                    if end>t.deadline+1e-9:misses+=1
                    done.add(t.id);work+=t.duration
            return inspection_end
        if self.name=="jrs_no_batch":
            first=stage_time
            order=tuple(prefix)+tuple(x for x in sorted(remaining) if x not in prefix)
            for jid in order:
                first=repair_group((jid,),first)
                first=checkpoint(first)
            second=checkpoint(first) if not order else first
        else:
            first=repair_group(prefix,stage_time) if prefix else stage_time
            first=checkpoint(first)
            second=repair_group(sorted(remaining),first)
            second=checkpoint(second)
        finish=max(clock.values(),default=obs.time)
        unfinished=len(tasks)-len(done)
        budget_fail=int(finish>obs.horizon)
        score=(unfinished+misses+budget_fail,finish+0.15*work,work)
        if not math.isfinite(second):score=(math.inf,math.inf,math.inf)
        immediate=tuple(a for t,a in sorted(events,key=lambda x:(x[0],_akey(x[1]))) if t<=obs.time+1e-9)
        return Forecast(score,immediate,nodes,tuple(statuses),finish,misses,work)

    def _joint(self,obs,allowed,start):
        pending=sorted((j for j in obs.jobs if j.status=="pending"),key=lambda j:j.id)
        # New samples cannot change certification until delayed packets arrive;
        # unrelated productive work is still considered by the forecast.
        ids=tuple(j.id for j in pending);candidates=[]
        if self.name=="jrs_no_batch":
            candidates=[((x,),"local") for x in ids] or [((),"local")]
        elif self.name=="random_rollout":
            for _ in range(min(40,2**min(len(ids),5)+1)):
                shuffled=list(ids);self.rng.shuffle(shuffled);n=self.rng.randrange(len(ids)+1)
                candidates.append((tuple(shuffled[:n]),self.rng.choice(("local","expanded"))))
        else:
            # Always compare all-repair batching against recertifying immediately,
            # then singleton checkpoints and bounded-size overlap prefixes.
            candidates=[(ids,"local"),(ids,"expanded"),((),"local")]
            candidates += [((x,),m) for x in ids for m in ("local","expanded")]
            for size in (2,3):
                for subset in itertools.combinations(ids[:7],size):
                    candidates.append((subset,"local"))
        best=None;chosen=();seen=set();evaluated=0
        for prefix,scope in candidates:
            key=(prefix,scope)
            if key in seen:continue
            seen.add(key)
            if evaluated and time.perf_counter()-start>=self.time_limit_s:
                self.stats["master_truncated"]+=1;break
            remaining=max(.0001,self.time_limit_s-(time.perf_counter()-start))
            result=self._forecast(obs,prefix,scope,min(.003,remaining/2))
            evaluated+=1;self.stats["search_nodes"]+=result.nodes
            self.stats["cover_timeout"]+=sum(x=="feasible_timeout" for x in result.cover_statuses)
            if not math.isfinite(result.score[0]):self.stats["forecast_infeasible"]+=1;continue
            first=_pack(result.first,obs,allowed)
            if not first and not obs.active and not obs.pending_packets:continue
            if best is None or result.score<best.score:
                best=result;chosen=first
        self.stats["master_candidates"]+=evaluated
        if best is not None:
            self.last_diagnostics={"surrogate_score":best.score,"cover_statuses":best.cover_statuses,
                                   "predicted_finish":best.predicted_finish,"candidates":evaluated,
                                   "prediction_is_not_quality_guarantee":True}
        if not chosen and not obs.active and not obs.pending_packets:
            chosen=self._rule(obs,allowed)
        if self.name=="jrs_no_joint" and chosen:
            # Serialization ablation removes simultaneous team dispatch; it is
            # not a claim of independently learned/optimized robot controllers.
            chosen=(chosen[0],)
        return chosen


def make_planner(name="jrs",seed=0,time_limit_s=.05):
    """Mechanism ablations require runner-side env changes as documented.

    jrs_no_uncertainty -> evidence_policy='optimistic_local';
    jrs_no_reuse / whole_reset -> evidence_policy='whole_reset'.
    The planner does not silently change world or hidden quality conditions.
    """
    return PublicPlanner(name,seed,time_limit_s)
