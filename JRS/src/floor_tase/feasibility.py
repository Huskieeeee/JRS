"""Same-world feasibility *witness* search for tiny diagnostic instances.

This evaluator-only routine deep-copies the actual environment. It is not a
deployment baseline and never changes noise, effects, horizon, or random seed.
Finding a replayed quality-success trajectory proves existence only for that
realized world. No unsuccessful outcome is labelled ``infeasible``: branch,
depth and wall-clock truncation, as well as event-dispatch discretization,
prevent an exhaustive continuous-time infeasibility proof.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from copy import deepcopy
import hashlib
import heapq
import itertools
import pickle
import time
from .model import Action
from .environment import InvalidAction


def environment_fingerprint(env):
    """Identity of the complete starting simulator state, including hidden RNG keys.

    Pickle is only generated locally for hashing. No untrusted pickle is loaded.
    This diagnostic digest is not passed to any deployment planner.
    """
    return hashlib.sha256(pickle.dumps(env, protocol=5)).hexdigest()


@dataclass(frozen=True)
class FeasibilityResult:
    status: str
    witness: tuple
    nodes: int
    generated: int
    elapsed_s: float
    initial_fingerprint: str
    terminal_result: dict | None
    limits_hit: tuple
    replay_verified: bool = False
    scope: str = "same_realized_world_bounded_witness_search"
    deployment_method: bool = False
    proves_infeasibility: bool = False
    witness_source: str | None = None

    def to_dict(self):
        d=asdict(self)
        d['witness']=[[asdict(a) for a in step] for step in self.witness]
        return d


def replay_witness(initial_env, witness):
    """Execute a witness from an unchanged copy; raises on an illegal action."""
    env=deepcopy(initial_env)
    for actions in witness:
        if env.done: break
        if actions and isinstance(actions[0],dict):
            actions=tuple(Action(**x) for x in actions)
        env.step_joint(tuple(actions))
    return env.result()


def _branches(env, max_joint_actions, max_branches, deadline):
    allowed=sorted(env.allowed_actions(),key=lambda a:
                   ({'successor':0,'repair':1,'inspect':2}[a.kind],a.target_id,a.robot_id,a.mode))
    seen=set();count=0
    # Always retain the event-wait branch, even when joint branching is capped.
    trial=deepcopy(env);trial.step_joint(())
    yield (),trial
    # Larger joint sets are tested first to retain concurrent opportunities.
    # All pairwise physical/resource constraints are still checked by the real env.
    for k in range(min(max_joint_actions,len(allowed)),0,-1):
        for actions in itertools.combinations(allowed,k):
            if time.perf_counter()>=deadline:
                yield None,'time'
                return
            if len({a.robot_id for a in actions})<k:continue
            key=tuple((a.kind,a.robot_id,a.target_id,a.mode) for a in actions)
            if key in seen:continue
            seen.add(key)
            trial=deepcopy(env)
            try:trial.step_joint(actions)
            except InvalidAction:continue
            if count>=max_branches:
                yield None,'branches'
                return
            yield actions,trial
            count+=1


def find_feasible_witness(initial_env, *, time_limit_s=2.0, max_nodes=2000,
                          max_depth=80, max_joint_actions=4,
                          max_branches_per_state=64,
                          warm_start_policies=('jrs','robust_local','qa_sequential')):
    """Search legal action sequences using the *unchanged actual* simulator.

    Search priority uses evaluator outcomes and is deliberately privileged.
    The returned policy must never enter fair public-information comparisons.
    Noise remains whatever it was in the input instance; keyed random outcomes
    are those of the same simulator world, not a noise-free replacement world.
    A bounded search failure is ``unknown_*`` even if its queue empties.
    """
    if min(time_limit_s,max_nodes,max_depth)>=0 and max_joint_actions>=1 and max_branches_per_state>=1:
        pass
    else:raise ValueError('nonnegative search budgets and positive branch limits required')
    start=time.perf_counter();fingerprint=environment_fingerprint(initial_env)
    root=deepcopy(initial_env);nodes=0;generated=0;counter=0;limits=set()

    def outcome(status,path=(),result=None,replayed=False,source=None):
        assert environment_fingerprint(initial_env)==fingerprint, 'diagnostic mutated its input'
        return FeasibilityResult(status,path,nodes,generated,time.perf_counter()-start,
                                 fingerprint,result,tuple(sorted(limits)),replayed,
                                 witness_source=source)

    def priority(env,depth):
        # Hidden evaluator fields can guide an oracle existence search; they
        # do not enter any public planner and confer no deployable capability.
        result=env.result()
        obs=env.observe()
        pending=sum(j.status in ('pending','running') for j in obs.jobs)
        uncertified=sum(not c.certified for c in obs.cells)
        return (-result['qualified_tasks'],-result['completed_tasks'],pending,uncertified,env.time,depth)

    # Incumbents are reachable trajectories of existing public planners. They
    # accelerate a positive feasibility certificate but confer no optimality.
    # Report the originating planner so an incumbent is not sold as a new gain.
    if warm_start_policies:
        from .planners import make_planner
        for name in warm_start_policies:
            warm=deepcopy(initial_env);path=()
            planner=make_planner(name,seed=0,time_limit_s=min(.005,max(0,time_limit_s)))
            for _ in range(max_depth):
                if warm.done:break
                if nodes>=max_nodes:
                    limits.add('nodes');return outcome('unknown_budget')
                if time.perf_counter()-start>=time_limit_s:
                    limits.add('time');return outcome('unknown_timeout')
                actions=tuple(planner.select(warm.observe(),warm.allowed_actions()))
                try:warm.step_joint(actions)
                except InvalidAction:break
                path+=(actions,);nodes+=1;generated+=1
                if warm.result()['false_releases']>0:break
            if warm.done and warm.result()['quality_success']:
                replayed=replay_witness(initial_env,path)
                if not replayed['quality_success']:
                    raise RuntimeError('warm-start witness failed same-world replay')
                return outcome('feasible_witness',path,replayed,True,'warm_start:'+name)
    queue=[(priority(root,0),counter,root,())]
    while queue:
        if nodes>=max_nodes:
            limits.add('nodes');return outcome('unknown_budget')
        if time.perf_counter()-start>=time_limit_s:
            limits.add('time');return outcome('unknown_timeout')
        _,_,env,path=heapq.heappop(queue);nodes+=1
        result=env.result()
        if env.done and result['quality_success']:
            replayed=replay_witness(initial_env,path)
            if not replayed['quality_success']:
                raise RuntimeError('same-world witness failed independent replay')
            return outcome('feasible_witness',path,replayed,True,'bounded_branch_search')
        if env.done or result['false_releases']>0:continue
        if len(path)>=max_depth:
            limits.add('depth');continue
        for actions,trial in _branches(env,max_joint_actions,max_branches_per_state,start+time_limit_s):
            if actions is None:
                limits.add(trial)
                if trial=='time':return outcome('unknown_timeout')
                break
            if time.perf_counter()-start>=time_limit_s:
                limits.add('time');return outcome('unknown_timeout')
            generated+=1
            # Every queued state is reachable in the same real simulator model.
            if trial.result()['false_releases']>0:continue
            counter+=1
            heapq.heappush(queue,(priority(trial,len(path)+1),counter,trial,path+(actions,)))
    return outcome('unknown_no_witness')
