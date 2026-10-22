"""v1.5: baseline-anchored GPU evaluation with short policy continuation.

The learned model replaces online synthetic rollouts. The original event
environment remains the execution and quality evaluator for every method.
"""
from dataclasses import asdict, dataclass, replace
import time
import numpy as np
from .model import Observation
from .planners import _akey, _dist, _fixes, _impact
from .evidence_recovery import EvidenceOptions, EvidenceRecoveryPlanner, EvidenceFrontier
from .surrogate import candidate_features


def task_finish_bounds(obs, actions):
    """Optimistic repair -> reinspection -> delivery -> successor time estimates.

    Views and repairs for distinct cells may overlap in this estimate. It is a
    screening estimate, not a schedulability certificate or an exact solver.
    """
    cells={c.id:c for c in obs.cells};jobs={j.id:j for j in obs.jobs}
    views={v.id:v for v in obs.views};robots={r.id:r for r in obs.robots}
    tasks={t.id:t for t in obs.tasks};now=obs.time
    available={r.id:max(now,r.busy_until) for r in obs.robots}
    positions={r.id:r.position for r in obs.robots}
    damaged=set();fixed={};release={q:now for q in cells};checks=[];finishing={}
    for a in obs.active:
        if a.kind=='repair':
            for q in a.cells:release[q]=max(release[q],a.finish);damaged.add(q)
    for a in actions:
        target=(jobs if a.kind=='repair' else views if a.kind=='inspect' else tasks)[a.target_id]
        dur=target.expanded_duration if a.kind=='repair' and a.mode=='expanded' else target.duration
        end=available[a.robot_id]+_dist(positions[a.robot_id],target.position)/obs.travel_speed+dur
        available[a.robot_id]=end;positions[a.robot_id]=target.position
        if a.kind=='repair':
            impact=set(cells) if obs.evidence_policy=='whole_reset' else _impact(target,a.mode)
            for q in impact:release[q]=max(release[q],end)
            damaged.update(impact)
            for q in _fixes(target,a.mode):fixed[q]=end
        elif a.kind=='inspect':checks.append((target,end))
        else:finishing[a.target_id]=end
    for j in obs.jobs:
        if j.status!='pending':continue
        for q in j.targets:
            if q in fixed:continue
            end=min((available[r.id]+_dist(positions[r.id],j.position)/obs.travel_speed+j.duration
                     for r in obs.robots if r.skill==j.skill),default=float('inf'))
            release[q]=max(release[q],end);damaged.add(q)
    ready={}
    for q,c in cells.items():
        if c.certified and q not in damaged:ready[q]=now;continue
        # Transport metadata only; never inspect an undelivered quality value.
        alternatives=[]
        if q not in damaged:
            for p in obs.pending_inspections:
                if (q,c.version) in p.captured_versions and p.delivery_time-p.captured_at<=obs.freshness:
                    alternatives.append(max(now,p.delivery_time))
        for v,end in checks:
            if q in v.covers and end>=release[q]:alternatives.append(end+obs.measurement_delay)
        duration=min((v.duration for v in obs.views if q in v.covers),default=float('inf'))
        alternatives.append(release[q]+duration+obs.measurement_delay)
        ready[q]=min(alternatives)
    bounds={}
    for t in obs.tasks:
        if t.status=='done' and not set(t.requires)&damaged:continue
        if t.status not in ('pending','done'):continue
        if t.id in finishing and not set(t.requires)&damaged:
            bounds[t.id]=finishing[t.id];continue
        certificate=max((ready[q] for q in t.requires),default=now)
        bounds[t.id]=min((max(certificate,available[r.id]+_dist(positions[r.id],t.position)/obs.travel_speed)+t.duration
                          for r in obs.robots if r.skill==t.skill),default=float('inf'))
    return bounds


class ChainFrontier(EvidenceFrontier):
    def protected(self,obs,rows):
        blocked=super().protected(obs,rows)
        if not blocked:return blocked
        from .evidence_recovery import useful_arrivals
        arrivals=useful_arrivals(obs);tasks={t.id:t for t in obs.tasks}
        for row in rows:
            a=row['action'];key=_akey(a)
            if key not in blocked:continue
            delay=max((arrivals[q]-obs.time for q in row['impact'] if q in arrivals),default=0.)
            bounds=task_finish_bounds(obs,(a,))
            if any(end+delay>min(tasks[tid].deadline,obs.horizon)+1e-9
                   for tid,end in bounds.items() if set(tasks[tid].requires)&set(row['impact'])):
                blocked.remove(key)
        return blocked


@dataclass(frozen=True)
class ReleaseOptions:
    model_path:str=''
    model_sha256:str=''
    device:str='cpu'
    candidates:int=8
    provider_candidates:int=32
    provider_cover_nodes:int=2000
    chain_guard:bool=True
    commit_events:int=2
    gain_gate:bool=True
    completion_margin:float=.05
    cost_margin:float=.10
    completion_tolerance:float=.02
    ensemble_penalty:float=1.

    def __post_init__(self):
        if not self.model_path or not self.model_sha256:raise ValueError('Frozen ensemble required')
        if self.candidates<2 or self.commit_events<0:raise ValueError('Invalid candidate/continuation budget')
        if min(self.completion_margin,self.cost_margin,self.completion_tolerance,self.ensemble_penalty)<0:
            raise ValueError('Nonnegative gate settings required')


class ReleasePlanner(EvidenceRecoveryPlanner):
    def __init__(self,seed=0,options=None):
        from .ensemble import ReturnEnsemble
        o=options or ReleaseOptions()
        super().__init__(seed,EvidenceOptions(candidates=o.candidates,samples=4,
            provider_candidates=o.provider_candidates,provider_cover_nodes=o.provider_cover_nodes))
        self.release_options=o
        if o.chain_guard:self.evidence=ChainFrontier(self.frontier.options,max_wait=2.,slack_fraction=.25)
        self.return_model=ReturnEnsemble(o.model_path,o.model_sha256,o.device)
        self.name='jrs_release';self.phase_policy=None;self.remaining=0;self.phase_state=None
        self.stats.update(model_batches=0,model_rows=0,inference_wall_seconds=0.,phase_steps=0,
                          phase_interruptions=0,baseline_choices=0,interventions=0)

    def candidates(self, obs, allowed):
        # A provider may legitimately return no action even without in-flight
        # events. The shared proposal builder filters that empty action; this
        # planner still needs the actual provider action as its gain reference.
        result = super().candidates(obs, allowed)
        if any('provider' in meta['sources'] for _, meta in result):
            return result
        from .planners import _pack
        actions = tuple(_pack(self.continuation_action(obs, allowed, 'provider'), obs, allowed))
        key = tuple(sorted(_akey(a) for a in actions))
        for acts, meta in result:
            if tuple(sorted(_akey(a) for a in acts)) == key:
                meta['sources'].append('provider')
                return result
        return [(actions, {'sources': ['provider']})] + result[:self.options.candidates - 1]

    @staticmethod
    def signature(obs):
        return ({j.id for j in obs.jobs}, {t.id for t in obs.tasks if t.status=='done'},
                {c.id:c.batch_id for c in obs.cells})

    def diagnostics(self):
        return dict(super().diagnostics(),release_options=asdict(self.release_options),
                    model=self.return_model.runtime_info(),online_rollouts=0)

    def select(self,obs,allowed):
        if not isinstance(obs,Observation):raise TypeError('Public Observation required')
        started=time.perf_counter();self.stats['calls']+=1
        current=self.signature(obs)
        interrupted=False
        if self.remaining and self.phase_state:
            oldjobs,olddone,oldbatches=self.phase_state
            interrupted=bool(current[0]-oldjobs or olddone-current[1] or current[2]!=oldbatches)
            if interrupted:self.remaining=0;self.stats['phase_interruptions']+=1
        if self.remaining:
            actions=self.continuation_action(obs,allowed,self.phase_policy)
            if self.release_options.chain_guard and self.phase_policy!='provider':
                base=self.continuation_action(obs,allowed,'provider')
                deadlines={t.id:min(t.deadline,obs.horizon) for t in obs.tasks}
                def late_tasks(acts):
                    return {tid for tid,end in task_finish_bounds(obs,acts).items() if end>deadlines[tid]+1e-9}
                if late_tasks(actions)-late_tasks(base):
                    self.remaining=0;interrupted=True;self.stats['phase_interruptions']+=1
            if self.remaining:
                self.remaining-=1;self.phase_state=current;self.stats['phase_steps']+=1
                self.stats['total_wall_s']+=time.perf_counter()-started
                self.last_diagnostics={'method':self.name,'phase_step':True,'continuation':self.phase_policy,
                    'remaining_phase_events':self.remaining,'device':self.release_options.device,'model_evaluated':False}
                return actions
        candidates=self.candidates(obs,allowed);self.stats['master_candidates']+=len(candidates)
        pairs=[(i,p) for i in range(len(candidates)) for p in ('provider','evidence')]
        baseline=next(k for k,(i,p) in enumerate(pairs) if 'provider' in candidates[i][1]['sources'] and p=='provider')
        features=[candidate_features(obs,candidates[i][0],p) for i,p in pairs]
        tick=time.perf_counter();pred=self.return_model.predict(features)
        self.stats['inference_wall_seconds']+=time.perf_counter()-tick
        self.stats['model_batches']+=1;self.stats['model_rows']+=len(features)
        means=pred.mean(axis=0);o=self.release_options
        tasks={t.id:t for t in obs.tasks}
        bounds=[task_finish_bounds(obs,a) for a,_ in candidates]
        late=[{tid for tid,end in b.items() if end>min(tasks[tid].deadline,obs.horizon)+1e-9} for b in bounds]
        base_late=late[pairs[baseline][0]]
        accepted=[];diagnostics=[]
        for k,(i,policy) in enumerate(pairs):
            dp=pred[:,k,0]-pred[:,baseline,0]
            dj=pred[:,baseline,1]-pred[:,k,1]
            lower_p=float(dp.mean()-o.ensemble_penalty*dp.std())
            lower_j=float(dj.mean()-o.ensemble_penalty*dj.std())
            gate=(lower_p>=o.completion_margin or
                  (lower_p>=-o.completion_tolerance and lower_j>=o.cost_margin))
            chain_ok=not(late[i]-base_late)
            ok=k==baseline or ((gate or not o.gain_gate) and (chain_ok or not o.chain_guard))
            if ok:accepted.append(k)
            diagnostics.append({'pair_index':k,'candidate_index':i,'continuation':policy,
                'predicted_completion':float(means[k,0]),'predicted_J':float(means[k,1]),
                'completion_gain_adjusted':lower_p,'cost_gain_adjusted':lower_j,
                'newly_late_tasks':sorted(late[i]-base_late),'accepted':ok})
        best=min(accepted,key=lambda k:(-means[k,0],means[k,1],k))
        i,self.phase_policy=pairs[best];actions,meta=candidates[i]
        self.remaining=o.commit_events;self.phase_state=current
        self.stats['baseline_choices']+=int(best==baseline);self.stats['interventions']+=int(best!=baseline)
        self.stats['provider_selected']+=int('provider' in meta['sources'])
        self.stats['total_wall_s']+=time.perf_counter()-started
        self.last_diagnostics={'method':self.name,'phase_step':False,'phase_interrupted':interrupted,
            'selected_index':i,'selected_pair':best,'baseline_pair':baseline,'selected':meta,
            'selected_continuation':self.phase_policy,'device':o.device,'model_evaluated':True,
            'candidate_count':len(candidates),'pair_count':len(pairs),'predictions':diagnostics,
            'candidates':[{'actions':[asdict(a) for a in acts],**m} for acts,m in candidates],
            'gain_gate_is_heuristic':True,'model_sha256':o.model_sha256}
        return actions
