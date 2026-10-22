"""JRS-RG: recovery-chain candidates and batched stochastic DAG evaluation.

Nodes represent movement, repair, camera capture, packet delivery and handoff.
The public duration approximation samples bounded repair retries and one-round
collateral recovery reserves. Physical execution uses the unchanged environment.
"""
from dataclasses import dataclass,asdict,replace
import hashlib,json,time
import numpy as np
from .model import Observation,Action
from .planners import _akey,_pack,_dist,_impact,_fixes

@dataclass(frozen=True)
class GraphOptions:
    device:str='cpu'
    candidates:int=16
    samples:int=64
    joint:bool=True
    stochastic:bool=True
    continuation:str='adaptive'
    batch:bool=True
    temporal:bool=True
    parallel:bool=True
    random_candidates:bool=False
    phase_events:int=2
    def __post_init__(self):
        if self.candidates<2 or self.samples<1 or self.phase_events<0:raise ValueError('Invalid planning budget')
        if self.continuation not in ('adaptive','fixed','none'):raise ValueError('Unknown continuation')
        if self.device!='cpu' and not self.device.startswith('cuda'):raise ValueError('Use cpu or cuda[:index]')

def recovery_hyperedges(obs):
    return [dict(task=t.id,deadline=min(t.deadline,obs.horizon),requires=list(t.requires),
                 repairs=[j.id for j in obs.jobs if j.status=='pending' and set(t.requires)&set(j.local_impact)],
                 views=[v.id for v in obs.views if set(t.requires)&set(v.covers)])
            for t in sorted(obs.tasks,key=lambda t:(t.deadline,t.id)) if t.status!='missed']

def templates(obs,opt,seed):
    jobs={j.id:j for j in obs.jobs if j.status=='pending'};edges=recovery_hyperedges(obs)
    deadline={j:min((e['deadline'] for e in edges if j in e['repairs']),default=obs.horizon) for j in jobs}
    order=tuple(sorted(jobs,key=lambda j:(deadline[j],jobs[j].duration,j)))
    if not order:return [((),'local')]
    if not opt.batch:return [(tuple((j,) for j in order),'local'),(tuple((j,) for j in reversed(order)),'local')]
    candidates=[]
    def add(groups,mode='local'):
        seen=set();normalized=[]
        for g in groups:
            g=tuple(j for j in g if j in jobs and j not in seen)
            if g:normalized.append(g);seen.update(g)
            elif not normalized:normalized.append(())
        rest=tuple(j for j in order if j not in seen)
        if rest:normalized.append(rest)
        key=(tuple(normalized),mode)
        if key not in candidates:candidates.append(key)
    if opt.random_candidates:
        rng=np.random.default_rng(seed)
        for _ in range(opt.candidates*20):
            ids=list(order);rng.shuffle(ids);groups=[];start=0
            for k in range(1,len(ids)):
                if rng.random()<.5:groups.append(ids[start:k]);start=k
            groups.append(ids[start:]);add(groups,'expanded' if rng.random()<.5 else 'local')
            if len(candidates)>=opt.candidates:break
        return candidates
    add([order]);add([(),order]);add([(j,) for j in order]);add([order],'expanded')
    for edge in edges:
        group=tuple(j for j in order if j in edge['repairs'])
        if group:add([group]);add([group],'expanded')
    for j in order:add([(j,)]);add([(j,)],'expanded')
    remaining=set(order);components=[]
    while remaining:
        group={min(remaining)}
        while True:
            footprint=set().union(*(_impact(jobs[j],'local') for j in group))
            more={j for j in remaining if footprint&_impact(jobs[j],'local')}
            if more<=group:break
            group|=more
        remaining-=group;components.append(tuple(j for j in order if j in group))
    add(sorted(components,key=lambda g:min(deadline[j] for j in g)))
    if len(candidates)>opt.candidates:candidates=[candidates[i] for i in np.linspace(0,len(candidates)-1,opt.candidates,dtype=int)]
    return candidates

class GraphCompiler:
    def __init__(self,obs,opt):
        self.o=obs;self.opt=opt;self.cells={c.id:c for c in obs.cells};self.cellids=sorted(self.cells)
        self.jobs={j.id:j for j in obs.jobs};self.views={v.id:v for v in obs.views};self.tasks={t.id:t for t in obs.tasks}
    def compile(self,template):
        o=self.o;groups,scope=template;groups=groups or ((),)
        nodes=[];nominal=[]
        def node(kind,key,duration,deps=(),release=None,action=None):
            index=len(nodes);ds=sorted(set(deps))
            nodes.append(dict(kind=kind,key=key,duration=float(duration),release=o.time if release is None else float(release),
                              deps=ds,quality_deps=ds,action=action,repair_prob=0.,repair_limit=1,
                              risk=[0.]*len(self.cells),service=0.,reserve_unit=0.))
            nominal.append(max(nodes[-1]['release'],max((nominal[d] for d in ds),default=o.time))+duration)
            return index
        node('root','root',0.)
        robot_last={r.id:0 for r in o.robots};pos={r.id:r.position for r in o.robots}
        robot_release={r.id:r.busy_until for r in o.robots};footprints=[];passage=[]
        latest={q:0 for q in self.cells};valid={q for q,c in self.cells.items() if c.certified}
        cert={q:(0,float(c.last_observed if c.last_observed is not None else o.time)) for q,c in self.cells.items() if c.certified}
        final={t.id:0 for t in o.tasks if t.status=='done'};remaining={j.id for j in o.jobs if j.status=='pending'}
        hazard={q:0 for q in self.cells};travel_total=0.;repair_travel=0.;repair_work=0.;revocations=0
        def assign(kind,target,foot,dependencies=(),duration=None,mode='local',risk=None):
            nonlocal travel_total,repair_travel
            choices=[]
            for r in o.robots:
                if r.skill!=target.skill:continue
                distance=_dist(pos[r.id],target.position);travel=distance/o.travel_speed
                ds=set(dependencies)|{robot_last[r.id]}
                for k,area,otherkind in footprints:
                    if set(foot)&area and (kind!='inspect' or otherkind!='inspect'):ds.add(k)
                divider=max(c.position[0] for c in o.cells)/2
                cross=((pos[r.id][0]<=divider<target.position[0]) or (target.position[0]<=divider<pos[r.id][0])) and distance>0
                if o.shared_passage and cross:ds.update(passage)
                if not self.opt.parallel:ds.update(range(1,len(nodes)))
                dur=target.duration if duration is None else duration
                start=max(robot_release[r.id],max((nominal[d] for d in ds),default=o.time))
                choices.append((start+travel+dur,r.id,ds,distance,travel,cross,dur,start))
            if not choices:return -1
            end,rid,ds,distance,travel,cross,dur,start=min(choices,key=lambda x:(x[0],x[1]))
            move=node('move',f'move:{len(nodes)}:{rid}',travel,ds,robot_release[rid]);nodes[move]['quality_deps']=[]
            occurrence=sum(n['kind']==kind and n.get('target')==target.id for n in nodes)
            i=node(kind,f'{kind}:{target.id}:{occurrence}',dur,set(ds)|{move},robot_release[rid],Action(kind,rid,target.id,mode))
            nodes[i].update(target=target.id,service=dur if kind=='repair' else 0.,planned_start=start,
                            quality_deps=list(dependencies),risk=list(risk or [0.]*len(self.cells)))
            if kind=='repair':
                repair_travel+=distance
                nodes[i]['repair_prob']=o.assumed_repair_failure_prob if self.opt.stochastic else 0.
                nodes[i]['repair_limit']=max(0,o.max_repairs_per_cell-target.attempts)
                # Failed attempts require diagnosis before a new repair can begin.
                nodes[i]['retry_cycle']=dur+min((v.duration for v in o.views if set(v.covers)&set(target.targets)),default=0.)+o.measurement_delay
            robot_last[rid]=i;pos[rid]=target.position;travel_total+=distance
            footprints.append((i,set(foot),kind))
            if o.shared_passage and cross:passage.append(move)
            return i
        for a in o.active:
            i=node('active',f'active:{a.robot_id}:{a.target_id}',0.,release=a.finish);robot_last[a.robot_id]=i
            target=(self.jobs if a.kind=='repair' else self.views if a.kind=='inspect' else self.tasks).get(a.target_id)
            if target:pos[a.robot_id]=target.position
            footprints.append((i,set(a.cells),a.kind))
            if a.channel:
                passage.append(node('active_move',f'active_move:{a.robot_id}',0.,release=max(o.time,a.arrival)))
            if a.kind=='repair':
                for q in a.cells:latest[q]=i;valid.discard(q);cert.pop(q,None)
                if target:
                    for q in set(a.cells)-_fixes(target,a.mode):hazard[q]+=1
            elif a.kind=='inspect' and target:
                delivery=node('delivery',f'active_delivery:{target.id}',o.measurement_delay,[i])
                known={q for j in o.jobs if j.status in ('pending','running') for q in j.targets}
                for q in set(target.covers)-known:valid.add(q);cert[q]=(i,0.);latest[q]=delivery
            elif a.kind=='successor':final[a.target_id]=i
        known={q for j in o.jobs if j.status in ('pending','running') for q in j.targets}
        for packet in o.pending_inspections:
            qs=[q for q,v in packet.captured_versions if q in self.cells and self.cells[q].version==v
                and packet.delivery_time-packet.captured_at<=o.freshness and q not in known]
            if qs:
                i=node('delivery',f'packet:{packet.batch_id}',0.,release=max(o.time,packet.delivery_time))
                for q in qs:valid.add(q);cert[q]=(0,packet.captured_at);latest[q]=i
        for gi,group in enumerate(groups):
            for jid in group:
                if jid not in remaining:continue
                j=self.jobs[jid];mode=scope if gi==0 else 'local';fixes=_fixes(j,mode)
                impact=set(self.cells) if o.evidence_policy=='whole_reset' else _impact(j,mode)
                i=assign('repair',j,impact,[latest[q] for q in impact],j.expanded_duration if mode=='expanded' else j.duration,mode)
                if i<0:continue
                repair_work+=nodes[i]['service'];valid-=impact
                for q in impact:latest[q]=i;cert.pop(q,None)
                for q in fixes:hazard[q]=0
                for q in impact-fixes:hazard[q]+=1
                for tid in list(final):
                    if set(self.tasks[tid].requires)&impact:final.pop(tid);revocations+=1
                remaining-={k for k in remaining if set(self.jobs[k].targets)<=fixes}
            defects={q for k in remaining for q in self.jobs[k].targets}
            eligible=sorted([t for t in o.tasks if t.status!='missed' and t.id not in final and not set(t.requires)&defects],key=lambda t:(t.deadline,t.id))
            needs={q for t in eligible for q in t.requires if q not in valid}
            for t in eligible:
                earliest=max([o.time]+[nominal[latest[q]] for q in t.requires])+t.duration
                for q in t.requires:
                    if q in cert:
                        cap,absolute=cert[q];capture=nominal[cap]+absolute-(o.time if cap==0 else 0.)
                        if earliest>capture+o.freshness:needs.add(q)
            def dispatch_ready():
                for t in eligible:
                    if t.id in final or not set(t.requires)<=valid or set(t.requires)&needs:continue
                    i=assign('successor',t,t.requires,[latest[q] for q in t.requires])
                    if i>=0:final[t.id]=i;nodes[i]['certificates']=[cert[q] for q in t.requires]
            dispatch_ready()
            while needs:
                choices=[]
                for v in o.views:
                    covered=set(v.covers)&needs;peers=[r for r in o.robots if r.skill==v.skill]
                    if not covered or not peers:continue
                    cost=v.duration+min(_dist(pos[r.id],v.position)/o.travel_speed for r in peers)
                    urgency=sum(1/max(1.,min((t.deadline-o.time for t in eligible if q in t.requires),default=o.horizon)) for q in covered)
                    choices.append((cost/max(urgency,1e-9),v.id,v,covered))
                if not choices:break
                _,_,v,covered=min(choices,key=lambda x:x[:2])
                risk=[1-(1-o.assumed_collateral_probability)**hazard[q] if q in covered and self.opt.stochastic else 0. for q in self.cellids]
                i=assign('inspect',v,v.covers,[latest[q] for q in v.covers],risk=risk)
                if i<0:break
                nodes[i]['reserve_unit']=o.repair_duration+v.duration+o.measurement_delay
                delivery=node('delivery',f'delivery:{i}',o.measurement_delay,[i])
                for q in set(v.covers)-defects:valid.add(q);cert[q]=(i,0.);latest[q]=delivery;hazard[q]=0
                needs-=set(v.covers);dispatch_ready()
            dispatch_ready()
        return dict(nodes=nodes,task_nodes=[final.get(t.id,-1) for t in o.tasks],deadlines=[min(t.deadline,o.horizon) for t in o.tasks],
                    travel=travel_total,repair_work=repair_work,revocations=revocations,repair_objective=repair_work+.01*repair_travel,
                    horizon=o.horizon,time=o.time,freshness=o.freshness,n_cells=len(o.cells),
                    accrued_travel=o.accrued_travel,accrued_repairs=o.accrued_repairs)

class BatchedCompletion:
    def __init__(self,device='cpu'):
        self.device=device;self.torch=None
        if device!='cpu':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
            self.torch=torch
    def evaluate(self,plans,samples,seed,temporal=True):
        C=len(plans);N=max(len(p['nodes']) for p in plans);Q=max(p['n_cells'] for p in plans)
        keys=sorted({n['key'] for p in plans for n in p['nodes']});keyidx={k:i for i,k in enumerate(keys)}
        rng=np.random.default_rng(seed);u=rng.uniform(1e-6,1-1e-6,(samples,len(keys))).astype('float32')
        cell_u=rng.uniform(0,1,(samples,len(keys),Q)).astype('float32')
        base=np.zeros((C,N),np.float32);release=base.copy();prob=base.copy();cycle=base.copy();limits=np.ones_like(base);reserve=base.copy()
        risks=np.zeros((C,N,Q),np.float32);uniform=np.full((C,samples,N),.5,np.float32);damage=np.ones((C,samples,N,Q),np.float32)
        K=max(1,max(len(n['deps']) for p in plans for n in p['nodes']));deps=np.zeros((C,N,K),np.int64)
        for c,p in enumerate(plans):
            for i,n in enumerate(p['nodes']):
                base[c,i]=n['duration'];release[c,i]=n['release'];prob[c,i]=n['repair_prob'];cycle[c,i]=n.get('retry_cycle',0.)
                limits[c,i]=n['repair_limit'];reserve[c,i]=n['reserve_unit'];risks[c,i,:len(n['risk'])]=n['risk']
                uniform[c,:,i]=u[:,keyidx[n['key']]];damage[c,:,i,:]=cell_u[:,keyidx[n['key']],:]
                ds=n['deps'] or [0]
                if not temporal and n['kind'] not in ('delivery','root','active','active_move'):ds=[0]
                deps[c,i,:len(ds)]=ds
        if self.torch is None:
            retries=np.where(prob[:,None,:]>0,np.floor(np.log(uniform)/np.log(np.clip(prob[:,None,:],1e-8,1-1e-6))),0.)
            exhausted=(cycle[:,None,:]>0)&(retries>=limits[:,None,:])
            duration=base[:,None,:]+np.minimum(retries,np.maximum(0,limits[:,None,:]-1))*cycle[:,None,:]
            duration+=(damage<risks[:,None,:,:]).sum(-1)*reserve[:,None,:]
            ends=np.zeros((C,samples,N),np.float32)
            for i in range(N):
                pred=np.take_along_axis(ends,deps[:,None,i,:].repeat(samples,axis=1),axis=2).max(2)
                ends[:,:,i]=np.maximum(release[:,None,i],pred)+duration[:,:,i]
        else:
            t=self.torch;cv=lambda x:t.as_tensor(x,device=self.device)
            b,r,pr,cy,lim,res,rk,un,dm,dp=map(cv,(base,release,prob,cycle,limits,reserve,risks,uniform,damage,deps))
            with t.inference_mode():
                retries=t.where(pr[:,None,:]>0,t.floor(t.log(un)/t.log(pr[:,None,:].clamp(1e-8,1-1e-6))),0.)
                exhausted=(cy[:,None,:]>0)&(retries>=lim[:,None,:])
                duration=b[:,None,:]+t.minimum(retries,(lim[:,None,:]-1).clamp_min(0))*cy[:,None,:]
                duration+=(dm<rk[:,None,:,:]).sum(-1)*res[:,None,:]
                ends=t.zeros((C,samples,N),device=self.device)
                for i in range(N):
                    pred=t.gather(ends,2,dp[:,None,i,:].expand(-1,samples,-1)).amax(2)
                    ends[:,:,i]=t.maximum(r[:,None,i],pred)+duration[:,:,i]
                ends=ends.cpu().numpy();exhausted=exhausted.cpu().numpy()
        values=[]
        for c,p in enumerate(plans):
            ancestors=[]
            for i,n in enumerate(p['nodes']):ancestors.append({i}|set().union(*(ancestors[d] for d in n['quality_deps'] if d<i)))
            done=[]
            for index,deadline in zip(p['task_nodes'],p['deadlines']):
                if index<0:done.append(np.zeros(samples,bool));continue
                yes=(ends[c,:,index]<=deadline+1e-6)&~exhausted[c][:,sorted(ancestors[index])].any(1)
                for cap,absolute in p['nodes'][index].get('certificates',[]):
                    capture=ends[c,:,cap]+absolute-(p['time'] if cap==0 else 0.)
                    yes &= ends[c,:,index]-capture<=p['freshness']+1e-6
                done.append(yes)
            complete=np.stack(done,1) if done else np.ones((samples,0),bool)
            success=complete.all(1);fraction=complete.mean(1) if done else np.ones(samples)
            finish=ends[c,:,:len(p['nodes'])].max(1)
            repair_count=sum(n['kind']=='repair' for n in p['nodes'])+p['accrued_repairs']
            cost=np.where(success,finish,p['horizon'])/p['horizon']+4*(1-fraction)
            cost+=.05*(p['travel']+p['accrued_travel'])/max(1,30*p['n_cells'])+.02*repair_count/max(1,p['n_cells'])
            values.append(dict(completion=float(success.mean()),qualified_fraction=float(fraction.mean()),cost=float(cost.mean()),finish_p90=float(np.quantile(finish,.9))))
        return values

class RecoveryGraphPlanner:
    def __init__(self,seed=0,options=None):
        self.seed=seed;self.options=options or GraphOptions();self.evaluator=BatchedCompletion(self.options.device)
        self.name='jrs_rg';self.phase=None;self.remaining=0;self.previous=None;self.last_diagnostics={}
        self.stats=dict(calls=0,searches=0,candidates=0,model_batches=0,phase_steps=0,phase_interruptions=0,total_wall_s=0.)
    def diagnostics(self):return dict(self.stats,options=asdict(self.options),baseline_policy_calls=0,gpu_scope='sampled DAG duration propagation')
    @staticmethod
    def state(obs):
        return dict(jobs={j.id:(j.status,j.attempts) for j in obs.jobs},done={t.id for t in obs.tasks if t.status=='done'},
                    cells={c.id:(c.version,c.status,c.batch_id) for c in obs.cells})
    def executable_front(self,plan,obs,allowed):
        pending={p.view_id for p in obs.pending_inspections if p.delivery_time>obs.time and
                 all(next(c.version for c in obs.cells if c.id==q)==v for q,v in p.captured_versions)}
        front=[n['action'] for n in plan['nodes'] if n['action'] is not None and n.get('planned_start',float('inf'))<=obs.time+1e-8]
        return _pack([a for a in front if not(a.kind=='inspect' and a.target_id in pending)],obs,allowed)
    def select(self,obs,allowed):
        if not isinstance(obs,Observation):raise TypeError('Public Observation required')
        start=time.perf_counter();self.stats['calls']+=1
        if not allowed:
            self.last_diagnostics=dict(method=self.name,no_legal_action=True)
            self.stats['total_wall_s']+=time.perf_counter()-start
            return ()
        current=self.state(obs);reasons=[]
        if self.previous and self.remaining and self.options.continuation=='adaptive':
            old=self.previous
            if set(current['jobs'])-set(old['jobs']):reasons.append('new_repair')
            if old['done']-current['done']:reasons.append('handoff_revoked')
            if any(status=='pending' and old['jobs'].get(j,(None,None))[0]=='running' for j,(status,attempt) in current['jobs'].items()):reasons.append('repair_failed')
            if any(t.status=='pending' and t.deadline-obs.time-t.duration<=obs.repair_duration+obs.measurement_delay for t in obs.tasks):reasons.append('deadline_pressure')
        if reasons:self.remaining=0;self.stats['phase_interruptions']+=1
        if self.remaining and self.phase:
            acts=self.executable_front(GraphCompiler(obs,self.options).compile(self.phase),obs,allowed)
            if acts:
                self.remaining-=1;self.previous=current;self.stats['phase_steps']+=1
                self.last_diagnostics=dict(method=self.name,phase_step=True,remaining=self.remaining)
                self.stats['total_wall_s']+=time.perf_counter()-start
                return acts if self.options.parallel else acts[:1]
        ts=templates(obs,self.options,self.seed+self.stats['searches']);compiler=GraphCompiler(obs,self.options)
        plans=[compiler.compile(t) for t in ts]
        state=json.dumps(asdict(obs),sort_keys=True).encode();seed=int.from_bytes(hashlib.blake2b(state+str(self.seed).encode(),digest_size=4).digest(),'big')
        values=self.evaluator.evaluate(plans,self.options.samples,seed,self.options.temporal)
        order=sorted(range(len(plans)),key=lambda k:(-values[k]['completion'],-values[k]['qualified_fraction'],values[k]['cost'],values[k]['finish_p90'],k)
                     if self.options.joint else (plans[k]['repair_objective'],k))
        chosen=order[0];acts=()
        for k in order:
            candidate=self.executable_front(plans[k],obs,allowed)
            if candidate:chosen=k;acts=candidate;break
        if not self.options.parallel:acts=acts[:1]
        self.phase=ts[chosen];self.remaining=self.options.phase_events if self.options.continuation!='none' else 0;self.previous=current
        self.stats['searches']+=1;self.stats['candidates']+=len(plans);self.stats['model_batches']+=int(self.options.device!='cpu')
        self.stats['total_wall_s']+=time.perf_counter()-start
        self.last_diagnostics=dict(method=self.name,phase_step=False,interrupt_reasons=reasons,selected=chosen,candidate_count=len(plans),
            predictions=values,templates=[dict(groups=g,scope=m) for g,m in ts],hyperedges=recovery_hyperedges(obs),
            selected_nodes=len(plans[chosen]['nodes']),selected_revocations=plans[chosen]['revocations'],baseline_policy_calls=0,device=self.options.device)
        return acts
