"""Finite deterministic certificate-window solver with safe energetic pruning.

NOT the FLOOR simulator, NOT a bound on its stochastic/parallel missions,
and NOT a stochastic mission optimum or quality guarantee.

Enumerates repair covers, partitions of final certificate demands, views,
robot assignments and within-robot orders. A standard difference-constraint
oracle checks each structure. All inspected cells are assumed good after the
selected deterministic repairs. New faults, sensing failure, shared corridors
and physical space conflicts are absent. Travel is Manhattan and resources are
nonpreemptive robots; inspection captures at its end and may be shared by tasks.
Certificates remain fresh through successor completion, matching the current
environment's pinned authorization-capture contract; this is stronger than checking
freshness only at initial authorization. For successor dispatch s and finish f,
a new capture c must satisfy c+delay<=s and f-c<=freshness.
"""
from dataclasses import dataclass,asdict
import itertools,math,time,json
from typing import Tuple,FrozenSet

@dataclass(frozen=True)
class Repair:
    id: str
    fixes: FrozenSet[str]
    affects: FrozenSet[str]
    duration: float
    robots: Tuple[str,...]
    position: tuple=(0.,0.)

@dataclass(frozen=True)
class View:
    id: str
    covers: FrozenSet[str]
    duration: float
    robots: Tuple[str,...]
    delay: float=0.
    position: tuple=(0.,0.)

@dataclass(frozen=True)
class Task:
    id: str
    requires: FrozenSet[str]
    duration: float
    robots: Tuple[str,...]
    deadline: float
    release: float=0.
    freshness: float=1000.
    position: tuple=(0.,0.)

@dataclass(frozen=True)
class Problem:
    defects: FrozenSet[str]
    repairs: tuple
    views: tuple
    tasks: tuple
    robot_starts: tuple
    horizon: float
    speed: float=1.
    # Already valid observations may be reused only if no selected repair affects q.
    existing_certificates: tuple=()  # (quality-unit ID, capture time)

@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    weight: float
    reason: str

@dataclass(frozen=True)
class STNResult:
    feasible: bool
    starts: dict
    conflict: tuple

@dataclass(frozen=True)
class ExactResult:
    status: str
    objective: float|None
    inspection_count: int|None
    schedule: tuple
    oracle_calls: int
    wall_seconds: float
    scope: str="finite_deterministic_certificate_window_model"
    algorithm: str="exhaustive_structures_plus_standard_difference_constraints"
    global_environment_bound: bool=False
    nodes: int=0
    capacity_cuts: int=0
    bound_cuts: int=0
    reuse_structures: int=0
    use_cuts: bool=True


def stn(nodes,edges):
    """Standard longest-path difference constraints x[v]>=x[u]+w.

    Origin Z is fixed to zero. Edges Z->action establish nonnegative starts;
    deadline edges action->Z are upper constraints. A positive cycle witnesses
    infeasibility. The returned edge list is a witness, not a new algorithm.
    """
    nodes=tuple(nodes);edges=tuple(edges);dist={n:0. for n in nodes};prev={}
    changed=None
    for iteration in range(len(nodes)):
        changed=None
        for edge in edges:
            candidate=dist[edge.u]+edge.weight
            if candidate>dist[edge.v]+1e-9:
                dist[edge.v]=candidate;prev[edge.v]=edge;changed=edge.v
        if changed is None:
            if abs(dist['Z'])>1e-8:
                raise AssertionError('Anchor changed without positive cycle')
            return STNResult(True,dist,())
    node=changed
    for _ in nodes:
        if node not in prev:break
        node=prev[node].u
    seen={};cycle=[]
    while node in prev and node not in seen:
        seen[node]=len(cycle);edge=prev[node];cycle.append(edge);node=edge.u
    cycle=cycle[seen[node]:] if node in seen else cycle
    return STNResult(False,{},tuple(cycle))


def partitions(items):
    """Each set partition once; tiny finite demand sets only."""
    items=tuple(items)
    if not items:
        yield ();return
    first,*rest=items
    for partition in partitions(rest):
        yield ((first,),)+partition
        for i in range(len(partition)):
            yield partition[:i]+((first,)+partition[i],)+partition[i+1:]


def _distance(p,q,speed):return (abs(p[0]-q[0])+abs(p[1]-q[1]))/speed


def _latest_starts(nodes, edges):
    """Upper bounds from longest paths to fixed origin; +inf if unbounded."""
    distance={n:-math.inf for n in nodes};distance['Z']=0.
    for _ in range(len(nodes)-1):
        changed=False
        for e in edges:
            if distance[e.v]>-math.inf and distance[e.v]+e.weight>distance[e.u]+1e-9:
                distance[e.u]=distance[e.v]+e.weight;changed=True
        if not changed:break
    return {n:-distance[n] for n in nodes}


def energetic_overload(actions, earliest, latest):
    """Admissible processing-energy cut, deliberately ignoring travel.

    Action i has duration p_i, release lower bound ES_i, completion upper bound
    LS_i+p_i and eligible robot set R_i. For interval [a,b], mandatory processing
    is max(0,p_i-max(0,a-ES_i)-max(0,LS_i+p_i-b)). For any tested robot subset R,
    only actions with R_i subset R contribute. If their sum exceeds |R|(b-a),
    no eligible assignment/order can be feasible. Testing fewer subsets weakens
    pruning but cannot invalidate it. This is standard energetic reasoning
    applied to certificate-induced temporal bounds, not a new theorem.
    """
    pools={frozenset(v[1]) for v in actions.values()}
    pools.add(frozenset(r for v in actions.values() for r in v[1]))
    for pool in sorted(pools,key=lambda r:(len(r),tuple(sorted(r)))):
        selected=[i for i,v in actions.items() if set(v[1])<=pool]
        endpoints=sorted({earliest[i] for i in selected} |
                         {latest[i]+actions[i][0] for i in selected if math.isfinite(latest[i])})
        for a,b in itertools.combinations(endpoints,2):
            if b<=a:continue
            energy=sum(max(0.,actions[i][0]-max(0.,a-earliest[i])-
                           max(0.,latest[i]+actions[i][0]-b)) for i in selected)
            if energy>len(pool)*(b-a)+1e-9:
                return {'robots':tuple(sorted(pool)),'interval':(a,b),'energy_lb':energy,
                        'capacity':len(pool)*(b-a),'actions':tuple(selected)}
    return None


def solve(problem,time_limit_s=math.inf,max_oracle_calls=100000,
          max_nodes=200000,use_cuts=True):
    """Enumerate finite structures; exact only when fully enumerated.

    The deterministic node budget counts repair sets, certificate choices,
    view/partition structures, assignments and resource orders. Time is optional
    and not used by default. Timeout is NEVER an infeasibility proof. Existing
    observations are optional per demand; they are excluded after any selected
    affecting intervention. Objective: makespan, then inspection count.
    """
    start=time.perf_counter();calls=0;nodes_count=0;best=None;best_schedule=();timed_out=False
    capacity_cuts=bound_cuts=reuse_structures=0
    if problem.speed<=0:raise ValueError('Positive speed required')
    if any(a.duration<=0 for a in problem.repairs+problem.views+problem.tasks):
        raise ValueError('Strictly positive durations required')
    robot_starts=dict(problem.robot_starts)
    if len(robot_starts)!=len(problem.robot_starts):raise ValueError('Unique robot IDs required')
    ids=[a.id for a in problem.repairs+problem.tasks]
    if len(set(ids))!=len(ids) or any(a=='Z' or a.startswith('check_') for a in ids):
        raise ValueError('Unique nonreserved repair/task IDs required')
    if len({v.id for v in problem.views})!=len(problem.views):raise ValueError('Unique view IDs required')
    if any(v.delay<0 for v in problem.views) or any(t.freshness<0 for t in problem.tasks):
        raise ValueError('Nonnegative sensing delay and evidence age required')
    if any(not a.robots or not set(a.robots)<=set(robot_starts) for a in problem.repairs+problem.views+problem.tasks):
        raise ValueError('Action robot qualifications must be nonempty and known')
    if any(not r.fixes<=r.affects for r in problem.repairs):
        raise ValueError('Repair fixes must be included in declared affected units')
    existing=dict(problem.existing_certificates)
    if set(existing)&set(problem.defects):
        raise ValueError('A known initial defect cannot have an initially valid certificate')
    if any(not math.isfinite(capture) or capture>0 for capture in existing.values()):
        raise ValueError('Existing certificate capture must be finite and at/before origin')
    if len(existing)!=len(problem.existing_certificates):raise ValueError('One existing certificate per quality unit')
    demands=tuple((q,t.id) for t in problem.tasks for q in sorted(t.requires))
    task_by_id={t.id:t for t in problem.tasks}
    def tick():
        nonlocal nodes_count
        if nodes_count>=max_nodes or calls>=max_oracle_calls or time.perf_counter()-start>=time_limit_s:
            raise TimeoutError
        nodes_count+=1
    try:
        for n in range(len(problem.repairs)+1):
            for chosen_repairs in itertools.combinations(problem.repairs,n):
                tick()
                fixed=set().union(*(r.fixes for r in chosen_repairs)) if chosen_repairs else set()
                if not problem.defects<=fixed:continue
                affected=set().union(*(r.affects for r in chosen_repairs)) if chosen_repairs else set()
                reusable=tuple(d for d in demands if d[0] in existing and d[0] not in affected)
                for reuse_flags in itertools.product((True,False),repeat=len(reusable)):
                    tick();reuse={d for d,f in zip(reusable,reuse_flags) if f}
                    if reuse:reuse_structures+=1
                    remaining=tuple(d for d in demands if d not in reuse)
                    for partition in partitions(remaining):
                        tick()
                        compatible=[tuple(v for v in problem.views if {q for q,_ in group}<=v.covers)
                                    for group in partition]
                        if any(not vv for vv in compatible):continue
                        for views in itertools.product(*compatible):
                            tick()
                            actions={r.id:(r.duration,r.robots,r.position,'repair',r.id) for r in chosen_repairs}
                            for t in problem.tasks:actions[t.id]=(t.duration,t.robots,t.position,'successor',t.id)
                            edges=[]
                            for q,tid in reuse:
                                edges.append(Edge(tid,'Z',task_by_id[tid].duration-existing[q]-task_by_id[tid].freshness,'old_certificate_age'))
                            for i,(group,view) in enumerate(zip(partition,views)):
                                vid=f'check_{i}_{view.id}'
                                actions[vid]=(view.duration,view.robots,view.position,'inspect',view.id)
                                requirements={q for q,_ in group}
                                for r in chosen_repairs:
                                    if r.affects&requirements:
                                        edges.append(Edge(r.id,vid,r.duration,'last_intervention'))
                                for q,tid in group:
                                    t=task_by_id[tid]
                                    edges.append(Edge(vid,tid,view.duration+view.delay,f'delivery:{q}'))
                                    edges.append(Edge(tid,vid,t.duration-t.freshness-view.duration,f'freshness:{q}'))
                            nodes=('Z',)+tuple(actions)
                            for aid,(duration,_,_,_,_) in actions.items():
                                edges.append(Edge('Z',aid,0.,'nonnegative'))
                                edges.append(Edge(aid,'Z',duration-problem.horizon,'horizon'))
                            for t in problem.tasks:
                                edges.append(Edge('Z',t.id,t.release,'release'))
                                edges.append(Edge(t.id,'Z',t.duration-t.deadline,'deadline'))
                            calls+=1;relaxed=stn(nodes,edges)
                            if not relaxed.feasible:continue
                            if use_cuts:
                                lower=max((relaxed.starts[i]+a[0] for i,a in actions.items()),default=0.)
                                if best is not None and (lower,len(views))>=best:
                                    bound_cuts+=1;continue
                                cut=energetic_overload(actions,relaxed.starts,_latest_starts(nodes,edges))
                                if cut is not None:capacity_cuts+=1;continue
                            action_ids=tuple(actions)
                            for assigned in itertools.product(*(actions[a][1] for a in action_ids)):
                                tick();groups={r:[] for r in robot_starts};mapping=dict(zip(action_ids,assigned))
                                for aid,rid in mapping.items():groups[rid].append(aid)
                                robots=tuple(groups)
                                # Materializing only each robot's tiny permutation set.
                                orders=[tuple(itertools.permutations(groups[r])) for r in robots]
                                for order_set in itertools.product(*orders):
                                    tick();resource=[]
                                    for rid,order in zip(robots,order_set):
                                        if order:
                                            resource.append(Edge('Z',order[0],_distance(robot_starts[rid],actions[order[0]][2],problem.speed),'initial_travel'))
                                        for a,b in zip(order,order[1:]):
                                            gap=actions[a][0]+_distance(actions[a][2],actions[b][2],problem.speed)
                                            resource.append(Edge(a,b,gap,f'robot_order:{rid}'))
                                    calls+=1;answer=stn(nodes,edges+resource)
                                    if not answer.feasible:continue
                                    makespan=max((answer.starts[a]+actions[a][0] for a in actions),default=0.)
                                    objective=(makespan,len(views))
                                    if best is None or objective<best:
                                        best=objective
                                        best_schedule=tuple(sorted(({'id':a,'kind':actions[a][3],
                                           'source':actions[a][4],'robot':mapping[a],
                                           'start':answer.starts[a],'end':answer.starts[a]+actions[a][0],
                                           'witnesses':list(partition[int(a.split('_')[1])]) if a.startswith('check_') else [],
                                           'reused':sorted(q for q,tid in reuse if tid==a)}
                                           for a in actions),key=lambda row:(row['start'],row['id'])))
    except TimeoutError:timed_out=True
    status=('feasible_budget' if best is not None else 'unknown_budget') if timed_out else ('optimal' if best is not None else 'infeasible')
    return ExactResult(status,best[0] if best else None,best[1] if best else None,best_schedule,calls,
                       time.perf_counter()-start,nodes=nodes_count,capacity_cuts=capacity_cuts,
                       bound_cuts=bound_cuts,reuse_structures=reuse_structures,use_cuts=use_cuts,
                       algorithm='finite_structure_enumeration_STN_energetic_cuts')


def structural_cases():
    # Shared camera and worker here: the batching example is deliberately serial;
    # the other examples below test multiple robots and certificate timing.
    serial=(('r',(0.,0.)),)
    overlap=Problem(frozenset('ab'),(
        Repair('ra',frozenset('a'),frozenset('ab'),2,('r',)),
        Repair('rb',frozenset('b'),frozenset('ab'),2,('r',))),
        (View('v',frozenset('ab'),1,('r',)),),
        (Task('t',frozenset('ab'),1,('r',),6),),serial,6)
    early=Problem(frozenset('ab'),(
        Repair('ra',frozenset('a'),frozenset('a'),1,('r',)),
        Repair('rb',frozenset('b'),frozenset('b'),4,('r',))),
        (View('va',frozenset('a'),1,('r',)),View('vb',frozenset('b'),1,('r',))),
        (Task('ta',frozenset('a'),1,('r',),3),Task('tb',frozenset('b'),1,('r',),10)),serial,10)
    # One quality unit, two consumers, separated release/deadline windows:
    # freshness prevents sharing despite identical image coverage.
    stale=Problem(frozenset(),(),(View('v',frozenset('a'),1,('camera',)),),
       (Task('early',frozenset('a'),1,('worker',),3,release=2,freshness=2),
        Task('late',frozenset('a'),1,('worker',),11,release=10,freshness=2)),
       (('camera',(0.,0.)),('worker',(0.,0.))),11)
    # Different last interventions, same view. Early release forces a split.
    window=Problem(frozenset('ab'),(
       Repair('ra',frozenset('a'),frozenset('a'),1,('repair_a',)),
       Repair('rb',frozenset('b'),frozenset('b'),8,('repair_b',))),
       (View('v',frozenset('ab'),1,('camera',)),),
       (Task('ta',frozenset('a'),1,('worker',),3),Task('tb',frozenset('b'),1,('worker',),11)),
       tuple((r,(0.,0.)) for r in ('repair_a','repair_b','camera','worker')),11)
    return {'overlap_batch':overlap,'independent_early_handoff':early,
            'freshness_forces_two_checks':stale,'last_intervention_forces_split':window}

if __name__=='__main__':
    from pathlib import Path
    out=Path.cwd()/'certificate_window_examples';out.mkdir(exist_ok=True)
    rows=[]
    for name,problem in structural_cases().items():
        result=solve(problem,time_limit_s=10.,max_oracle_calls=100000)
        rows.append({'case':name,**asdict(result)})
    (out/'exact_cases.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps([{k:r[k] for k in ('case','status','objective','inspection_count','oracle_calls')} for r in rows],indent=2))
