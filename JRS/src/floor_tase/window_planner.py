"""Public-only receding-horizon certificate-window scheduler.

This is an executable scheduling heuristic, not an exact stochastic optimizer.
It enumerates task orders and repair-scope vectors. For each, every *known*
intervention affecting a final handoff is scheduled before that handoff's
certificate. Inspections have capture, delivery and freshness-through-completion constraints; valid
unaffected observations can be reused. Robot, declared-footprint and passage
reservations match the public event abstraction. Unknown future collateral faults
are intentionally left to the separately tested public prediction model.

Cuts reject only candidate deterministic completion claims. They never mask a
legal environmental action or prove a stochastic world infeasible.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,itertools,json,math,random,time
from .model import Action
from .planners import PublicPlanner,_dist,_impact,_fixes,_akey,_pack
from .certificate_windows import energetic_overload


@dataclass(frozen=True)
class WindowPlan:
    first: tuple
    events: tuple
    score: tuple
    predicted_finish: float
    predicted_qualified: int
    inspections: int
    reused: int
    nodes: int
    windows: tuple
    rejected: bool=False
    rejection: str|None=None


class _Budget(Exception):pass


class WindowPlanner:
    def __init__(self,seed=0,max_candidates=32,max_nodes=6000,use_cuts=True,coordinate_windows=True):
        if max_candidates<1 or max_nodes<1:raise ValueError('Positive deterministic budgets required')
        self.seed=seed;self.max_candidates=max_candidates;self.max_nodes=max_nodes;self.use_cuts=use_cuts
        self.coordinate_windows=coordinate_windows
        self.name=('certificate_window_staged' if not coordinate_windows else 'certificate_window' if use_cuts else 'certificate_window_no_cuts')
        self._pending_inspections={};self._reference=PublicPlanner('robust_local',seed)
        self.stats={'calls':0,'master_candidates':0,'search_nodes':0,'capacity_cuts':0,
                    'deadline_cuts':0,'budget_exhaustions':0,'fallbacks':0,'total_wall_s':0.,
                    'public_information_only':True,'exact_mission_solver':False}
        self.last_diagnostics={}

    def diagnostics(self):return dict(self.stats)

    def _guard(self,obs,allowed):
        versions={c.id:c.version for c in obs.cells}
        self._pending_inspections={vid:(until,old,capture) for vid,(until,old,capture) in self._pending_inspections.items()
            if until>obs.time+1e-9 and all(versions.get(cid)==v for cid,v in old.items())}
        return tuple(a for a in allowed if not(a.kind=='inspect' and a.target_id in self._pending_inspections))

    def _templates(self,obs):
        tasks=tuple(t.id for t in obs.tasks if t.status in ('pending','done','travelling','running'))
        bytask={t.id:t for t in obs.tasks};pending=tuple(sorted(j.id for j in obs.jobs if j.status=='pending'))
        jobs={j.id:j for j in obs.jobs};cells={c.id:c for c in obs.cells}
        def affected_score(t):
            return sum(bool(set(t.requires)&set(j.local_impact)) for j in jobs.values() if j.status=='pending')
        orders=[tuple(sorted(tasks,key=lambda t:(bytask[t].deadline,t))),
                tuple(sorted(tasks,key=lambda t:(affected_score(bytask[t]),bytask[t].deadline,t))),
                tuple(sorted(tasks,key=lambda t:(not all(cells[c].certified for c in bytask[t].requires),
                                                 bytask[t].deadline,t))),
                tuple(sorted(tasks,key=lambda t:(bytask[t].deadline,-affected_score(bytask[t]),t)))]
        modes=[tuple('local' for _ in pending),tuple('expanded' for _ in pending)]
        for jid in pending:
            modes.append(tuple('expanded' if j==jid else 'local' for j in pending))
        # A deterministic state-local seed makes no-cuts and cuts see the same
        # candidates even when prior calls examined different internal nodes.
        key='|'.join([str(self.seed),str(obs.time)]+[f'{c.id}:{c.version}:{c.status}' for c in obs.cells])
        rng=random.Random(int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),'big'))
        for _ in range(self.max_candidates):
            order=list(tasks);rng.shuffle(order);orders.append(tuple(order))
            modes.append(tuple(rng.choice(('local','expanded')) for _ in pending))
        emitted=set();count=0
        # Interleave modes across task orders, keeping every known job eligible
        # for expansion (not only the first sorted repair).
        for idx in range(max(len(orders),len(modes))):
            for offset in range(4):
                order=orders[(idx+offset)%len(orders)] if orders else ()
                mode=modes[idx%len(modes)] if modes else ()
                template=(order,tuple(zip(pending,mode)),offset%2)
                if template in emitted:continue
                emitted.add(template);yield template;count+=1
                if count>=self.max_candidates:return

    def _forecast(self,obs,template,limit):
        taskorder,modepairs,repair_order=template
        cells={c.id:c for c in obs.cells};jobs={j.id:j for j in obs.jobs};views={v.id:v for v in obs.views}
        tasks={t.id:t for t in obs.tasks};robots={r.id:r for r in obs.robots};modes=dict(modepairs)
        clocks={r.id:max(obs.time,r.busy_until) for r in obs.robots};pos={r.id:r.position for r in obs.robots}
        spaces=[];passages=[];events=[];nodes=0;work=0.;travel=0.;inspection_count=0;reused=0;windows=[]
        selected=[];fixed=set()
        order=sorted(modes,key=lambda j:(jobs[j].duration,j)) if repair_order==0 else sorted(modes,key=lambda j:(-len(_impact(jobs[j],modes[j])),j))
        for jid in order:
            j=jobs[jid]
            if set(j.targets)<=fixed:continue
            selected.append(jid);fixed|=_fixes(j,modes[jid])
        active_fixes=set().union(*(_fixes(jobs[a.target_id],a.mode) for a in obs.active
                                   if a.kind=='repair' and a.target_id in jobs)) if obs.active else set()
        unfixable_known={c.id for c in obs.cells if c.status=='invalid' and c.id not in fixed|active_fixes}
        allcells=set(cells)
        impacts={j:(allcells if obs.evidence_policy=='whole_reset' else _impact(jobs[j],modes[j])) for j in selected}
        affected=set().union(*impacts.values()) if impacts else set()
        captures={c.id:c.last_observed for c in obs.cells if c.certified and c.last_observed is not None and c.id not in affected}
        delivered={c:obs.time for c in captures};last_release={c:obs.time for c in cells}
        qualified={t.id for t in obs.tasks if t.status=='done' and not(set(t.requires)&affected)}
        def active_certificate_supports_finish(action):
            if action.kind!='successor' or set(action.cells)&affected:return False
            if action.authorization_versions:
                versions=dict(action.authorization_versions)
                captured=dict(action.authorization_capture_times)
                # Authorization pins the particular observation used, not the
                # latest observation subsequently delivered to the same cell.
                return all(q in versions and q in captured and cells[q].version==versions[q]
                           and action.finish-captured[q]<=obs.freshness+1e-9 for q in action.cells)
            # A travelling action has not pinned authorization yet. Its current
            # public certificates must conservatively support completion; a
            # previously denied action has status pending and cannot qualify.
            return tasks[action.target_id].status=='travelling' and all(
                cells[q].certified and cells[q].last_observed is not None
                and action.finish-cells[q].last_observed<=obs.freshness+1e-9 for q in action.cells)
        active_successors={a.target_id for a in obs.active if active_certificate_supports_finish(a)}
        qualified|=active_successors
        repair_ends={};completed=set();divider=max(c.position[0] for c in obs.cells)/2
        shared_passage=getattr(obs,'shared_passage',True)
        def tick(n=1):
            nonlocal nodes
            if nodes+n>limit:raise _Budget
            nodes+=n
        for active in obs.active:
            spaces.append((active.start,active.finish,set(active.cells),active.kind))
            if active.channel and active.arrival>obs.time:passages.append((active.start,active.arrival))
            item=jobs.get(active.target_id) if active.kind=='repair' else views.get(active.target_id) if active.kind=='inspect' else tasks.get(active.target_id)
            if item:pos[active.robot_id]=item.position
            if active.kind=='repair':
                for cid in active.cells:last_release[cid]=max(last_release[cid],active.finish)
            if active.kind=='inspect':
                for cid in active.cells:
                    if cid not in affected and cells[cid].status!='invalid':
                        captures[cid]=active.finish;delivered[cid]=active.finish+obs.measurement_delay
        for vid,(until,old,capture) in self._pending_inspections.items():
            for cid in old:
                if cid not in affected and cells[cid].status!='invalid':
                    captures[cid]=capture;delivered[cid]=until
        def slot(kind,skill,position,duration,release,footprint):
            best=None
            for rid,r in robots.items():
                if r.skill!=skill:continue
                tick();start=max(clocks[rid],release);distance=_dist(pos[rid],position);motion=distance/obs.travel_speed
                crossing=shared_passage and ((pos[rid][0]<=divider<position[0]) or (position[0]<=divider<pos[rid][0])) and motion>0
                for _ in range(len(spaces)+len(passages)+2):
                    prior=start
                    if crossing:
                        for a,b in passages:
                            if start<b-1e-9 and start+motion>a+1e-9:start=max(start,b)
                    for a,b,other,otype in spaces:
                        if set(footprint)&other and (kind!='inspect' or otype!='inspect') and start<b-1e-9 and start+motion+duration>a+1e-9:
                            start=max(start,b)
                    if abs(start-prior)<1e-9:break
                candidate=(start+motion+duration,start,rid,start+motion,distance,crossing)
                if best is None or candidate<best:best=candidate
            return best
        def commit(kind,oid,skill,position,duration,release,footprint,mode='local'):
            nonlocal work,travel
            value=slot(kind,skill,position,duration,release,footprint)
            if value is None:return None
            end,start,rid,arrival,distance,crossing=value
            clocks[rid]=end;pos[rid]=position;spaces.append((start,end,set(footprint),kind))
            if crossing:passages.append((start,arrival))
            action=Action(kind,rid,oid,mode);events.append({'start':start,'arrival':arrival,'finish':end,'action':action})
            work+=duration;travel+=distance
            return events[-1]
        def ensure_repair(jid):
            if jid in completed:return
            j=jobs[jid];mode=modes[jid];duration=j.expanded_duration if mode=='expanded' else j.duration
            event=commit('repair',jid,j.skill,j.position,duration,obs.time,impacts[jid],mode)
            completed.add(jid);repair_ends[jid]=event['finish'] if event else math.inf
            for q in impacts[jid]:
                last_release[q]=max(last_release[q],repair_ends[jid]);captures.pop(q,None);delivered.pop(q,None)
        def cut_task(task):
            # Earliest robot arrival uses its actual current planned location and
            # clock; ignoring spatial/passage conflicts is an admissible bound.
            earliest=min((max(clocks[r.id],obs.time)+_dist(pos[r.id],task.position)/obs.travel_speed
                          for r in obs.robots if r.skill==task.skill),default=math.inf)
            if earliest+task.duration>min(task.deadline,obs.horizon)+1e-9:return 'deadline'
            needed=[q for q in task.requires if q not in captures or earliest-captures[q]>obs.freshness+1e-9]
            if not needed:return None
            # Each selected task check must fit before latest handoff dispatch.
            # A fractional 1/max-view-coverage charging gives a valid lower
            # bound, but processing energy here is implemented on singleton
            # mandatory views only: no unjustified sum across shared images.
            mandatory={}
            for q in needed:
                legal=[v for v in obs.views if q in v.covers]
                if not legal:return 'uncovered'
                capture_lb=min(max(last_release[q],obs.time)+v.duration for v in legal)
                if capture_lb+obs.measurement_delay+task.duration>min(task.deadline,obs.horizon)+1e-9:return 'deadline'
                if len(legal)==1:
                    mandatory.setdefault(legal[0].id,{'view':legal[0],'bound':set()})['bound'].add(q)
            if mandatory:
                acts={vid:(d['view'].duration,tuple(r.id for r in obs.robots if r.skill==d['view'].skill),d['view'].position,'inspect',vid) for vid,d in mandatory.items()}
                earliest_checks={vid:max(last_release[q] for q in d['bound']) for vid,d in mandatory.items()}
                latest_checks={vid:min(task.deadline,obs.horizon)-task.duration-obs.measurement_delay-d['view'].duration for vid,d in mandatory.items()}
                if energetic_overload(acts,earliest_checks,latest_checks):return 'capacity'
            return None
        try:
            misses=0;staged_release=obs.time
            if not self.coordinate_windows:
                for jid in selected:ensure_repair(jid)
                staged_release=max([obs.time]+list(repair_ends.values())+[a.finish for a in obs.active if a.kind=='repair'])
            for tid in taskorder:
                task=tasks[tid]
                if tid in qualified:continue
                # A camera cannot repair a publicly known defect. Optimistic
                # outcomes apply only after a selected/active feasible repair.
                if set(task.requires)&unfixable_known:
                    misses+=1;continue
                # No later selected intervention may invalidate this final task.
                relevant=[j for j in selected if impacts[j]&set(task.requires)]
                for jid in relevant:ensure_repair(jid)
                # This snapshot is candidate-local. An unsuccessful handoff
                # attempt must not consume resources in the partial plan.
                checkpoint=(dict(clocks),dict(pos),list(spaces),list(passages),list(events),
                            dict(captures),dict(delivered),work,travel,inspection_count,list(windows))
                if self.use_cuts:
                    reason=cut_task(task)
                    if reason:
                        self.stats['capacity_cuts' if reason=='capacity' else 'deadline_cuts']+=1
                        misses+=1;continue
                lower=max([staged_release]+[last_release[q] for q in task.requires])
                accepted=False
                for attempt in range(2*len(task.requires)+3):
                    desired=slot('successor',task.skill,task.position,task.duration,lower,task.requires)
                    if desired is None:break
                    task_end,dispatch,_,arrival,_,_=desired
                    needed={q for q in task.requires if q not in captures or delivered[q]>dispatch+1e-9 or task_end-captures[q]>obs.freshness+1e-9}
                    if not needed:
                        event=commit('successor',tid,task.skill,task.position,task.duration,lower,task.requires)
                        if event['finish']<=min(task.deadline,obs.horizon)+1e-9:
                            qualified.add(tid);accepted=True
                            reused+=sum(1 for q in task.requires if cells[q].certified and q not in affected and captures[q]==cells[q].last_observed)
                        windows.append({'task':tid,'dispatch':event['start'],'arrival':event['arrival'],'finish':event['finish'],
                                        'certificates':{q:{'capture':captures[q],'delivered':delivered[q],
                                                          'inspection_release_bound':last_release[q],
                                                          'old_certificate_reused':cells[q].certified and q not in affected and captures[q]==cells[q].last_observed,
                                                          'fresh_until':captures[q]+obs.freshness} for q in task.requires}})
                        break
                    pending={q for q in needed if q in captures and delivered[q]>dispatch+1e-9 and delivered[q]-captures[q]<=obs.freshness+1e-9}
                    if pending:
                        lower=max(lower,max(delivered[q] for q in pending));continue
                    options=[]
                    for view in obs.views:
                        gain=set(view.covers)&needed
                        if not gain:continue
                        release=max([staged_release]+[last_release[q] for q in gain])
                        # Delay camera service when another robot is unavailable;
                        # capture must stay fresh until the worker finishes (authorization pins its capture time).
                        release=max(release,task_end-obs.freshness-view.duration)
                        candidate=slot('inspect',view.skill,view.position,view.duration,release,view.covers)
                        if candidate:options.append(((candidate[0]+obs.measurement_delay)/len(gain),candidate[0],view.id,release))
                    if not options:break
                    _,_,vid,release=min(options);view=views[vid]
                    event=commit('inspect',vid,view.skill,view.position,view.duration,release,view.covers)
                    inspection_count+=1
                    for q in view.covers:
                        # An early image of a different unfinished quality unit
                        # cannot certify it before its last known intervention.
                        if all(j in completed for j in selected if q in impacts[j]) and event['start']>=last_release[q]-1e-9:
                            captures[q]=event['finish'];delivered[q]=event['finish']+obs.measurement_delay
                    lower=max(lower,event['finish']+obs.measurement_delay)
                if not accepted:
                    misses+=1
                    (clocks,pos,spaces,passages,events,captures,delivered,work,travel,inspection_count,windows)=checkpoint
            # Repairs whose cells have no pending successor still remain part of
            # the public recovery plan (normally every cell has a successor).
            for jid in selected:ensure_repair(jid)
        except _Budget:
            return WindowPlan((),(),(math.inf,math.inf),math.inf,len(qualified),inspection_count,reused,nodes,(),True,'node_budget')
        finish=max(clocks.values(),default=obs.time);fraction=1-len(qualified)/max(1,len(tasks))
        complete=fraction<=1e-12 and finish<=obs.horizon+1e-9
        charged=finish if complete else obs.horizon
        predicted_J=charged/obs.horizon+.05*(travel+getattr(obs,'accrued_travel',0.))/(30*max(1,len(cells)))+.02*(len(completed)+getattr(obs,'accrued_repairs',0))/max(1,len(cells))+4*fraction
        first=tuple(e['action'] for e in sorted(events,key=lambda e:(e['start'],_akey(e['action']))) if e['start']<=obs.time+1e-9)
        return WindowPlan(first,tuple(events),(len(tasks)-len(qualified),predicted_J),finish,len(qualified),inspection_count,reused,nodes,tuple(windows))

    def _plans(self,obs,allowed):
        evaluated=[];budget=self.max_nodes
        for template in self._templates(obs):
            if budget<=0:self.stats['budget_exhaustions']+=1;break
            plan=self._forecast(obs,template,budget);budget-=plan.nodes
            self.stats['search_nodes']+=plan.nodes;self.stats['master_candidates']+=1
            if plan.rejected:
                self.stats['budget_exhaustions']+=1;continue
            first=_pack(plan.first,obs,allowed)
            if not first and not obs.active and not obs.pending_packets:continue
            evaluated.append((plan,first,template))
        evaluated.sort(key=lambda row:(row[0].score,tuple(_akey(a) for a in row[1])))
        return evaluated

    def candidate_actions(self,obs,allowed_actions):
        """Same deterministic shortlist for external public-model reranking."""
        allowed=self._guard(obs,allowed_actions);plans=self._plans(obs,allowed);seen=set();result=[]
        for plan,first,template in plans:
            key=tuple(_akey(a) for a in first)
            if key not in seen:seen.add(key);result.append(first)
        return tuple(result)

    def select(self,obs,allowed_actions):
        begin=time.perf_counter();self.stats['calls']+=1;allowed=self._guard(obs,allowed_actions)
        if not allowed:return ()
        plans=self._plans(obs,allowed)
        if plans:
            plan,chosen,template=plans[0]
            self.last_diagnostics={'predicted_score':plan.score,'predicted_finish':plan.predicted_finish,
                'predicted_qualified':plan.predicted_qualified,'planned_inspections':plan.inspections,
                'reused_certificates':plan.reused,'candidate_count':len(plans),'selected_template':template,
                'certificate_windows':plan.windows,'nodes':sum(p.nodes for p,_,_ in plans),
                'prediction_is_not_quality_guarantee':True,'use_cuts':self.use_cuts,'coordinate_windows':self.coordinate_windows,
                'first':tuple(asdict(a) for a in chosen)}
        else:chosen=()
        if not chosen and not obs.active and not obs.pending_packets:
            chosen=self._reference._rule(obs,allowed);self.stats['fallbacks']+=1
        self.record_dispatch(obs,chosen)
        self.stats['total_wall_s']+=time.perf_counter()-begin
        return tuple(chosen)

    def record_dispatch(self,obs,chosen):
        """Call after externally choosing a candidate to retain delivery guard."""
        views={v.id:v for v in obs.views};robots={r.id:r for r in obs.robots};versions={c.id:c.version for c in obs.cells}
        for a in chosen:
            if a.kind=='inspect':
                view=views[a.target_id];capture=obs.time+_dist(robots[a.robot_id].position,view.position)/obs.travel_speed+view.duration
                self._pending_inspections[view.id]=(capture+obs.measurement_delay,{q:versions[q] for q in view.covers},capture)
