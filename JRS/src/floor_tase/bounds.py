"""Exact *static serial* references and explicitly conditional lower bounds.

These references never receive simulator private state. Serial optimum is not
an optimum or lower bound for the parallel stochastic environment. ``unknown``
means exhausted search budget, not a proof of infeasibility.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq,math,time
from .optimizer import CoverOption,solve_cover

@dataclass(frozen=True)
class StaticRepair:
    id: str
    fixes: frozenset
    invalidates: frozenset
    duration: float

@dataclass(frozen=True)
class StaticView:
    id: str
    covers: frozenset
    duration: float

@dataclass(frozen=True)
class StaticSuccessor:
    id: str
    requires: frozenset
    duration: float
    deadline: float = math.inf

@dataclass(frozen=True)
class StaticResult:
    status: str
    completion_time: float
    lower_bound: float
    sequence: tuple
    nodes: int
    elapsed_s: float
    scope: str = "deterministic_single_resource_surrogate"


def solve_static_serial(defects, initially_certified, repairs, views, successors,
                        horizon=math.inf,time_limit_s=1.,max_nodes=100000):
    """Dijkstra search over repaired cells, valid evidence, and successors.

    Repairs invalidate earlier evidence. A view certifies only already-good
    cells. A successor can run only after every still-possible intervention
    affecting its requirements is completed or rendered unnecessary. This
    explicit commitment condition prevents a later planned intervention from
    invalidating a completed successor. All actions share one resource here.
    Under these finite deterministic assumptions, completed search is exact.
    """
    start=time.perf_counter();repairs=tuple(repairs);views=tuple(views);successors=tuple(successors)
    defects=frozenset(defects);initially_certified=frozenset(initially_certified)
    if any(x.duration<=0 for x in repairs+views+successors):
        raise ValueError("Static action durations must be strictly positive")
    for rep in repairs:
        if not rep.fixes <= defects:raise ValueError("Repair fixes must name mandatory defect IDs")
    initial=(defects,initially_certified,0)
    counter=0;queue=[(0.,counter,initial,())];best={initial:0.};nodes=0
    all_tasks=(1<<len(successors))-1
    while queue:
        if time.perf_counter()-start>=time_limit_s or nodes>=max_nodes:
            return StaticResult("unknown_timeout",math.inf,queue[0][0],(),nodes,time.perf_counter()-start)
        t,_,state,path=heapq.heappop(queue)
        if t>best.get(state,math.inf):continue
        remaining,valid,done=state;nodes+=1
        if not remaining and done==all_tasks:
            return StaticResult("optimal",t,t,path,nodes,time.perf_counter()-start)
        candidates=[]
        pending=[r for r in repairs if r.fixes & remaining]
        for rep in pending:
            candidates.append((rep.duration,(remaining-rep.fixes,valid-rep.invalidates,done),("repair",rep.id)))
        for view in views:
            available=view.covers-remaining
            if available-valid:
                candidates.append((view.duration,(remaining,valid|available,done),("inspect",view.id)))
        for i,task in enumerate(successors):
            if done&(1<<i):continue
            if task.requires<=valid and not any(r.invalidates&task.requires for r in pending):
                if t+task.duration<=task.deadline+1e-12:
                    candidates.append((task.duration,(remaining,valid,done|(1<<i)),("successor",task.id)))
        for duration,newstate,action in candidates:
            newt=t+duration
            if newt>horizon+1e-12 or newt>=best.get(newstate,math.inf)-1e-12:continue
            best[newstate]=newt;counter+=1
            heapq.heappush(queue,(newt,counter,newstate,path+(action,)))
    return StaticResult("infeasible",math.inf,math.inf,(),nodes,time.perf_counter()-start)


def optimistic_work_bound(defects, repairs, required_evidence, views,
                          successor_durations, robot_count, time_limit_s=.05):
    """Optimistic mandatory-work bound under explicitly supplied obligations.

    Requires: all supplied defect repairs, evidence cells and successor actions
    are mandatory; each physical action occupies one robot; costs are true
    lower durations; a single action cannot count simultaneously as repair and
    inspection. Ignores travel, precedence, repeat samples, conflicts and
    release dates. No environmental feasibility conclusion follows by itself.
    """
    if robot_count<1:raise ValueError("robot_count must be positive")
    repair=solve_cover(defects,[CoverOption(r.id,r.fixes,r.duration) for r in repairs],time_limit_s=time_limit_s)
    inspect=solve_cover(required_evidence,[CoverOption(v.id,v.covers,v.duration) for v in views],time_limit_s=time_limit_s)
    lower=(repair.lower_bound+inspect.lower_bound+sum(successor_durations))/robot_count
    return {"lower_bound":lower,"repair_status":repair.status,"inspection_status":inspect.status,
            "scope":"conditional_mandatory_work_bound","feasibility_claim":False}
