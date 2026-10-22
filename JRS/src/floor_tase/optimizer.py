"""Public-evidence recertification subproblem.

This module solves *weighted set cover with fixed, nonnegative view costs*.
The exact status is local to this static surrogate: stochastic observations,
view-to-view routing and the complete construction problem are not solved
optimally.  A time limit never constitutes an infeasibility certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import time
from typing import Iterable, Hashable

@dataclass(frozen=True)
class CoverOption:
    id: Hashable
    covers: frozenset
    cost: float

@dataclass(frozen=True)
class CoverResult:
    selected: tuple
    cost: float
    lower_bound: float
    status: str
    nodes: int
    elapsed_s: float
    required_count: int

    @property
    def feasible(self):
        return math.isfinite(self.cost)

    @property
    def exact(self):
        return self.status == "optimal"


def solve_cover(required: Iterable, options: Iterable[CoverOption],
                time_limit_s: float = 0.05, max_nodes: int = 100000,
                exact_max_cells: int = 22) -> CoverResult:
    """A* branch-and-bound set cover with an initial greedy incumbent.

    The lower bound is admissible for the supplied fixed-cost cover problem.
    It is NOT a lower bound on stochastic mission completion time. A timeout
    can retain a feasible cover and is returned as ``feasible_timeout``.
    """
    start = time.perf_counter()
    universe = tuple(sorted(set(required), key=repr))
    if not universe:
        return CoverResult((), 0., 0., "optimal", 0, time.perf_counter()-start, 0)
    indices = {x:i for i,x in enumerate(universe)}
    full = (1 << len(universe))-1
    prepared=[]
    for option in options:
        if not math.isfinite(option.cost) or option.cost < 0:
            raise ValueError("Set-cover costs must be finite and nonnegative")
        mask=sum(1 << indices[x] for x in set(option.covers) if x in indices)
        if mask:
            prepared.append((option.id, mask, float(option.cost)))
    total=0
    for _,mask,_ in prepared:
        total |= mask
    if total != full:
        return CoverResult((), math.inf, math.inf, "infeasible", 0,
                           time.perf_counter()-start,len(universe))
    # Safe dominance removes equal/more expensive subset views.
    kept=[]
    for i,(oid,mask,cost) in enumerate(prepared):
        dominated=any(i != j and (mask|m2)==m2 and c2 <= cost and
                      (c2 < cost or m2 != mask or j < i)
                      for j,(_,m2,c2) in enumerate(prepared))
        if not dominated:
            kept.append((oid,mask,cost))
    prepared=kept

    def bound(mask):
        missing=full & ~mask
        if not missing:return 0.
        count=missing.bit_count()
        ratio=min(cost/(m&missing).bit_count() for _,m,cost in prepared if m&missing)
        per_cell=max(min(cost for _,m,cost in prepared if m&(1<<bit))
                     for bit in range(len(universe)) if missing&(1<<bit))
        return max(ratio*count, per_cell)

    # Greedy produces an honest feasible incumbent even with a zero search budget.
    mask=0;greedy=[];upper=0.
    while mask != full:
        oid,m,cost=min((x for x in prepared if x[1]&~mask),
                      key=lambda x:(x[2]/(x[1]&~mask).bit_count(), x[2],repr(x[0])))
        greedy.append(oid);upper+=cost;mask|=m
    lower=bound(0)
    if abs(upper-lower)<1e-12:
        return CoverResult(tuple(greedy),upper,upper,"optimal",0,time.perf_counter()-start,len(universe))
    if len(universe)>exact_max_cells:
        return CoverResult(tuple(greedy),upper,lower,"heuristic_feasible",0,
                           time.perf_counter()-start,len(universe))
    queue=[(lower,0.,0,())];best={0:0.};nodes=0
    incumbent=tuple(greedy)
    while queue:
        if nodes>=max_nodes or time.perf_counter()-start>=time_limit_s:
            return CoverResult(incumbent,upper,min(upper,queue[0][0]),
                               "feasible_timeout",nodes,time.perf_counter()-start,len(universe))
        estimate,cost,mask,path=heapq.heappop(queue)
        if estimate>=upper-1e-12:
            return CoverResult(incumbent,upper,upper,"optimal",nodes,time.perf_counter()-start,len(universe))
        if cost>best.get(mask,math.inf)+1e-12:continue
        nodes+=1
        # Every cover includes a view covering this uncovered element. Branching
        # on those views is complete; inspection order is immaterial here.
        missing=full&~mask;first=missing & -missing
        for oid,m,increment in prepared:
            if not(m&first):continue
            newmask=mask|m;newcost=cost+increment
            if newcost>=best.get(newmask,math.inf)-1e-12:continue
            best[newmask]=newcost
            if newmask==full:
                if newcost<upper:
                    upper=newcost;incumbent=path+(oid,)
                continue
            estimate=newcost+bound(newmask)
            if estimate<upper-1e-12:
                heapq.heappush(queue,(estimate,newcost,newmask,path+(oid,)))
    return CoverResult(incumbent,upper,upper,"optimal",nodes,time.perf_counter()-start,len(universe))


def coverage_union_cost(sets: Iterable[Iterable], options: Iterable[CoverOption],
                        **kwargs):
    """Joint and separate cover costs; shared views need only be purchased once."""
    sets=tuple(frozenset(x) for x in sets);options=tuple(options)
    joint=solve_cover(set().union(*sets) if sets else (),options,**kwargs)
    separate=tuple(solve_cover(s,options,**kwargs) for s in sets)
    return joint,separate
