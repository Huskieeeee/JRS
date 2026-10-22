"""Joint rework / recertification candidate portfolio, public observations only.

All candidates receive identical synthetic seeds and model scenarios. No
optimistic pre-screen or small-sample significance-based early elimination.
The retained baseline is a candidate, not a guarantee of baseline dominance.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time

from .model import Observation
from .handoff import HandoffFrontier, HandoffOptions
from .deterministic_legacy import FixedBudgetPlanner
from .planners import _akey, _pack
from .prediction import public_belief_environment


def dependency_graph(obs):
    """Public job-scope -> affected cells -> recertification/task obligations."""
    edges = []
    allcells = {c.id for c in obs.cells}
    active = {a.target_id: set(a.cells) for a in obs.active if a.kind == 'repair'}
    for job in obs.jobs:
        if job.status not in ('pending', 'running'):
            continue
        for mode in ('local', 'expanded'):
            impact = (allcells if obs.evidence_policy == 'whole_reset' else
                      active.get(job.id, set(job.local_impact if mode == 'local' else job.expanded_impact)))
            edges.append({'job': job.id, 'scope': mode, 'impact': sorted(impact),
                          'recertify': sorted(c.id for c in obs.cells if c.id in impact and c.certified),
                          'affected_tasks': sorted(t.id for t in obs.tasks if set(t.requires) & impact),
                          'redo_tasks': sorted(t.id for t in obs.tasks if t.status == 'done' and set(t.requires) & impact)})
    return edges


class ArchitecturePlanner:
    """Executable internal architecture controls; not reproductions of papers."""
    def __init__(self, name='arch_dispatch'):
        if name not in ('arch_dispatch', 'arch_staged'):
            raise ValueError(name)
        self.name = name
        self.frontier = HandoffFrontier(HandoffOptions())
        self.stats = {'calls': 0, 'total_wall_s': 0., 'public_information_only': True}
        self.last_diagnostics = {}

    def diagnostics(self):
        return dict(self.stats, internal_architecture_control=True)

    def select(self, obs, allowed):
        if not isinstance(obs, Observation):
            raise TypeError('public Observation required')
        start = time.perf_counter()
        rows = self.frontier.features(obs, allowed)
        stage = 'dynamic_dispatch'
        if self.name == 'arch_staged':
            if any(j.status in ('pending', 'running') for j in obs.jobs):
                rows = [r for r in rows if r['action'].kind == 'repair']
                stage = 'repair'
            elif any(not c.certified for c in obs.cells) or obs.pending_packets:
                rows = [r for r in rows if r['action'].kind == 'inspect']
                stage = 'recertify'
            else:
                stage = 'dispatch'
        rows.sort(key=lambda r: (r['slack'], r['duration'], _akey(r['action'])))
        chosen = _pack([r['action'] for r in rows], obs, allowed)
        self.stats['calls'] += 1
        self.stats['total_wall_s'] += time.perf_counter()-start
        self.last_diagnostics = {'architecture': self.name, 'stage': stage}
        return chosen


@dataclass(frozen=True)
class JointOptions:
    candidates: int = 8
    samples: int = 4
    max_events: int = 2000
    retain_provider: bool = True
    provider: str = 'fixed_jrs'
    provider_candidates: int = 32
    provider_cover_nodes: int = 2000
    adaptive_scope: bool = True
    diagnostic_checks: bool = True
    local_dependencies: bool = True
    robust_models: bool = True
    tail_weight: float = .25
    tail_alpha: float = .8
    collateral_margin: float = .10
    failure_margin: float = .05
    continuation: str = 'handoff'

    def __post_init__(self):
        if min(self.candidates, self.samples, self.max_events, self.provider_candidates, self.provider_cover_nodes) < 1:
            raise ValueError('positive budgets required')
        if not 0 <= self.tail_alpha < 1 or self.tail_weight < 0:
            raise ValueError('invalid tail risk parameters')
        if min(self.collateral_margin, self.failure_margin) < 0:
            raise ValueError('nonnegative model margins required')
        if self.provider not in ('fixed_jrs', 'arch_dispatch') or self.continuation not in ('handoff', 'dispatch'):
            raise ValueError('unknown provider/continuation')


def empirical_cvar(values, alpha):
    """Exact upper-tail mean of equally weighted samples, fractional boundary."""
    values = sorted(values, reverse=True)
    mass = (1-alpha)*len(values)
    whole = int(math.floor(mass))
    fraction = mass-whole
    return (sum(values[:whole]) + (fraction*values[whole] if fraction > 1e-12 else 0.))/mass


class JointRecoveryPlanner:
    def __init__(self, seed=0, options=None):
        self.seed = seed
        self.options = options or JointOptions()
        self.name = 'jrs_joint'
        o = self.options
        self.frontier = HandoffFrontier(HandoffOptions(diagnostic_checks=o.diagnostic_checks,
            local_dependencies=o.local_dependencies, adaptive_scope=o.adaptive_scope, max_candidates=o.candidates))
        self.dispatch = ArchitecturePlanner()
        self.provider = (FixedBudgetPlanner('jrs', seed, o.provider_candidates, o.provider_cover_nodes)
                         if o.provider == 'fixed_jrs' else ArchitecturePlanner())
        self.stats = {'calls': 0, 'master_candidates': 0, 'rollout_worlds': 0,
                      'rollout_events': 0, 'total_wall_s': 0., 'provider_selected': 0,
                      'public_information_only': True, 'exact_mission_solver': False}
        self.last_diagnostics = {}

    def diagnostics(self):
        return dict(self.stats, options=asdict(self.options), provider=self.provider.diagnostics())

    def candidates(self, obs, allowed):
        o = self.options
        rows = self.frontier.features(obs, allowed)
        balanced = sorted(rows, key=lambda r: self.frontier._rank(r, 'balanced'))
        # Enforce ablation scope and diagnosis restrictions on provider proposals too.
        filtered = tuple(allowed) if (o.adaptive_scope and o.diagnostic_checks and o.local_dependencies) else tuple(r['action'] for r in rows)
        result, seen = [], {}
        def add(actions, source):
            actions = _pack(actions, obs, filtered)
            if not actions and not (obs.active or obs.pending_packets):
                return
            key = tuple(sorted(_akey(a) for a in actions))
            if key in seen:
                result[seen[key]][1]['sources'].append(source)
            else:
                seen[key] = len(result)
                result.append((tuple(actions), {'sources': [source]}))
        if o.retain_provider:
            # A proposal is speculative: never retain pending-inspection state
            # from a proposal that was not actually dispatched.
            if o.provider == 'fixed_jrs':
                self.provider = FixedBudgetPlanner('jrs', self.seed, o.provider_candidates, o.provider_cover_nodes)
                versions = {c.id:c.version for c in obs.cells}
                for packet in obs.pending_inspections:
                    self.provider._pending_inspections[packet.view_id] = (packet.delivery_time, dict(packet.captured_versions))
                for action in obs.active:
                    if action.kind == 'inspect':
                        self.provider._pending_inspections[action.target_id] = (action.finish+obs.measurement_delay, {c:versions[c] for c in action.cells})
            add(self.provider.select(obs, filtered), 'provider')
        add([r['action'] for r in balanced], 'balanced')
        for mode in ('local', 'expanded'):
            repair = [r for r in balanced if r['action'].kind == 'repair' and r['action'].mode == mode]
            rest = [r for r in balanced if r['action'].kind != 'repair']
            if repair:
                add([r['action'] for r in repair+rest], 'repair_'+mode)
        for profile in ('diagnose', 'deadline'):
            add([r['action'] for r in sorted(rows, key=lambda r:self.frontier._rank(r, profile))], profile)
        # Explicitly expose delaying diagnosis, and allowing in-flight work to finish.
        add([r['action'] for r in balanced if r['role'] != 'diagnostic'], 'defer_diagnostic')
        if obs.active or obs.pending_packets:
            add((), 'wait_inflight')
        if not result:
            result = [((), {'sources': ['no_productive_action']})]
        return result[:o.candidates]

    def models(self, obs):
        if not self.options.robust_models:
            return [('central', obs)]
        result = []
        for label, factor in [('low', -1), ('central', 0), ('high', 1)]:
            result.append((label, replace(obs,
                assumed_collateral_probability=min(1.,max(0.,obs.assumed_collateral_probability+factor*self.options.collateral_margin)),
                assumed_repair_failure_prob=min(1.,max(0.,obs.assumed_repair_failure_prob+factor*self.options.failure_margin)))))
        return result

    def forecast(self, obs, actions, seed):
        env = public_belief_environment(obs, seed, 'cascade')
        env.step_joint(actions)
        events = 1
        while not env.done and events < self.options.max_events:
            current = env.observe()
            continuation = self.frontier if self.options.continuation == 'handoff' else self.dispatch
            env.step_joint(continuation.select(current, env.allowed_actions()))
            events += 1
        if not env.done:
            raise RuntimeError('Joint forecast event budget exceeded')
        self.stats['rollout_worlds'] += 1
        self.stats['rollout_events'] += events
        result = env.result()
        return {k: float(result[k]) for k in ('quality_success', 'J', 'qualified_tasks', 'successor_revocations', 'inspections', 'time')}

    def score(self, predictions):
        groups = list(predictions.values())
        completion = min(sum(x['quality_success'] for x in g)/len(g) for g in groups)
        outcomes = [x for g in groups for x in g]
        costs = [x['J'] for x in outcomes]
        mean_j = sum(costs)/len(costs)
        tail = empirical_cvar(costs, self.options.tail_alpha)
        return (-completion, mean_j + self.options.tail_weight*tail)

    def select(self, obs, allowed):
        if not isinstance(obs, Observation):
            raise TypeError('public Observation required')
        start = time.perf_counter()
        self.stats['calls'] += 1
        candidates = self.candidates(obs, allowed)
        self.stats['master_candidates'] += len(candidates)
        predictions = []
        if len(candidates) > 1:
            state = hashlib.sha256(json.dumps(asdict(obs),sort_keys=True).encode()).hexdigest()
            seeds = [int(hashlib.sha256(f'joint:{self.seed}:{state}:{i}'.encode()).hexdigest()[:15],16)
                     for i in range(self.options.samples)]
            models = self.models(obs)
            for actions, meta in candidates:
                by_model = {label: [self.forecast(current, actions, seed) for seed in seeds]
                            for label, current in models}
                predictions.append({'sources':meta['sources'], 'by_model':by_model,
                                    'score':self.score(by_model)})
            index = min(range(len(candidates)), key=lambda i:(predictions[i]['score'],i))
        else:
            index = 0
        chosen, meta = candidates[index]
        self.stats['provider_selected'] += int('provider' in meta['sources'])
        self.stats['total_wall_s'] += time.perf_counter()-start
        self.last_diagnostics = {'method':self.name, 'selected_index':index,
            'selected':meta, 'candidate_count':len(candidates),
            'candidates':[{'actions':[asdict(a) for a in acts], **m} for acts,m in candidates],
            'predictions':predictions, 'dependencies':dependency_graph(obs),
            'selection_objective':'max worst-model empirical completion; min mean J + weighted CVaR J',
            'all_candidates_evaluated':True, 'continuation':self.options.continuation,
            'empirical_scores_not_calibrated_probabilities':True}
        return chosen
