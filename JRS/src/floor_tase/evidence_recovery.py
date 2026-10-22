"""v1.4: evidence-aware candidates and a portfolio of continuation policies.

Only public observations enter the planner. Optional learned screening reduces
CPU rollouts; it never supplies the final quality gate or reads physical truth.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import time

from .model import Observation
from .handoff import HandoffFrontier
from .joint_recovery import JointOptions, JointRecoveryPlanner, dependency_graph
from .deterministic_legacy import FixedBudgetPlanner
from .planners import _akey, _pack
from .prediction import public_belief_environment


def useful_arrivals(obs):
    """Cell -> next fresh, version-valid *captured* packet delivery.

    Uncaptured inspections are deliberately not protected: their version is
    not yet fixed. Already-known defects do not require waiting for diagnosis.
    """
    cells = {c.id: c for c in obs.cells}
    known = {q for j in obs.jobs if j.status in ('pending', 'running') for q in j.targets}
    needed = {q for t in obs.tasks if t.status == 'pending' for q in t.requires}
    arrivals = {}
    for packet in obs.pending_inspections:
        if not obs.time < packet.delivery_time <= obs.horizon:
            continue
        if packet.delivery_time - packet.captured_at > obs.freshness + 1e-9:
            continue
        for q, version in packet.captured_versions:
            c = cells.get(q)
            if c is not None and c.version == version and not c.certified and q not in known and q in needed:
                arrivals[q] = min(arrivals.get(q, float('inf')), packet.delivery_time)
    return arrivals


class EvidenceFrontier(HandoffFrontier):
    def __init__(self, options=None, max_wait=2., slack_fraction=.25, enabled=True):
        super().__init__(options)
        self.max_wait, self.slack_fraction, self.enabled = max_wait, slack_fraction, enabled

    def protected(self, obs, rows):
        incoming = useful_arrivals(obs) if self.enabled else {}
        blocked = set()
        for row in rows:
            a = row['action']
            if a.kind != 'repair':
                continue
            impacted = set(row['impact']) & incoming.keys()
            # Waiting is bounded by local slack and requires modeled damage risk.
            # Keep it as a candidate, never a blanket rule on the whole team.
            if impacted and obs.assumed_collateral_probability > 0:
                delay = max(incoming[q] - obs.time for q in impacted)
                impacted_deadline = min((t.deadline for t in obs.tasks
                    if t.status == 'pending' and set(t.requires) & set(row['impact'])), default=obs.horizon)
                slack = min(row['slack'], impacted_deadline-obs.time-row['duration'])
                if delay <= self.max_wait and delay <= self.slack_fraction * max(0., slack):
                    blocked.add(_akey(a))
        return blocked

    def select(self, obs, allowed):
        rows = self.features(obs, allowed)
        blocked = self.protected(obs, rows)
        ordered = sorted((r for r in rows if _akey(r['action']) not in blocked),
                         key=lambda r: self._rank(r, 'balanced'))
        return _pack([r['action'] for r in ordered], obs, allowed)


@dataclass(frozen=True)
class EvidenceOptions:
    candidates: int = 8
    samples: int = 4
    max_events: int = 2000
    provider_candidates: int = 32
    provider_cover_nodes: int = 2000
    protect_evidence: bool = True
    max_wait: float = 2.
    slack_fraction: float = .25
    continuations: tuple = ('evidence', 'provider')
    surrogate_path: str = ''
    surrogate_sha256: str = ''
    device: str = 'cpu'
    shortlist: int = 4

    def __post_init__(self):
        if min(self.candidates, self.samples, self.max_events, self.shortlist,
               self.provider_candidates, self.provider_cover_nodes) < 1:
            raise ValueError('Positive computation budgets required')
        if self.candidates < 2 or self.max_wait < 0 or not 0 <= self.slack_fraction <= 1:
            raise ValueError('Invalid evidence options')
        if not self.continuations or len(set(self.continuations)) != len(self.continuations) or set(self.continuations)-{'evidence','provider','handoff'}:
            raise ValueError('Unknown/duplicate continuation')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('device must be cpu or cuda[:index]')
        if bool(self.surrogate_path) != bool(self.surrogate_sha256):
            raise ValueError('Surrogate requires both path and frozen sha256')


class EvidenceRecoveryPlanner(JointRecoveryPlanner):
    def __init__(self, seed=0, options=None):
        self.evidence_options = options or EvidenceOptions()
        o = self.evidence_options
        super().__init__(seed, JointOptions(candidates=o.candidates, samples=o.samples,
            max_events=o.max_events, provider_candidates=o.provider_candidates,
            provider_cover_nodes=o.provider_cover_nodes, robust_models=False, tail_weight=0.))
        self.name = 'jrs_evidence'
        self.evidence = EvidenceFrontier(self.frontier.options, o.max_wait, o.slack_fraction, o.protect_evidence)
        self.surrogate = None
        if o.surrogate_path:
            from .surrogate import Surrogate
            self.surrogate = Surrogate(o.surrogate_path, o.surrogate_sha256, o.device)

    def diagnostics(self):
        return dict(super().diagnostics(), evidence_options=asdict(self.evidence_options),
                    surrogate_training=False, gpu_scope='optional candidate-policy screening only')

    def candidates(self, obs, allowed):
        base = super().candidates(obs, allowed)
        evidence = tuple(self.evidence.select(obs, allowed))
        # Reserve capacity for both baseline and evidence-aware proposals.
        result, seen = [], {}
        proposals = [base[0], (evidence, {'sources':['evidence_aware']})] + base[1:]
        for acts, meta in proposals:
            key = tuple(sorted(_akey(a) for a in acts))
            if key in seen:
                result[seen[key]][1]['sources'].extend(s for s in meta['sources'] if s not in result[seen[key]][1]['sources'])
            elif len(result) < self.options.candidates:
                seen[key] = len(result)
                result.append((tuple(acts), {'sources':list(meta['sources'])}))
        return result

    def continuation_action(self, obs, allowed, policy):
        if policy == 'evidence':
            return self.evidence.select(obs, allowed)
        if policy == 'handoff':
            return self.frontier.select(obs, allowed)
        # Hydrate from actual public packets, not speculative proposal state.
        p = FixedBudgetPlanner('jrs', self.seed, self.options.provider_candidates, self.options.provider_cover_nodes)
        versions = {c.id:c.version for c in obs.cells}
        for packet in obs.pending_inspections:
            p._pending_inspections[packet.view_id] = (packet.delivery_time, dict(packet.captured_versions))
        for a in obs.active:
            if a.kind == 'inspect':
                p._pending_inspections[a.target_id] = (a.finish+obs.measurement_delay, {c:versions[c] for c in a.cells})
        return p.select(obs, allowed)

    def forecast_policy(self, obs, actions, seed, policy):
        env = public_belief_environment(obs, seed, 'cascade')
        env.step_joint(actions)
        events = 1
        while not env.done and events < self.options.max_events:
            current = env.observe()
            env.step_joint(self.continuation_action(current, env.allowed_actions(), policy))
            events += 1
        if not env.done:
            raise RuntimeError('Evidence forecast event budget exceeded')
        self.stats['rollout_worlds'] += 1
        self.stats['rollout_events'] += events
        return {k:float(env.result()[k]) for k in
                ('quality_success','J','qualified_tasks','successor_revocations','inspections','time')}

    def select(self, obs, allowed):
        if not isinstance(obs, Observation):
            raise TypeError('Public Observation required')
        from .surrogate import candidate_features
        start = time.perf_counter()
        self.stats['calls'] += 1
        candidates = self.candidates(obs, allowed)
        self.stats['master_candidates'] += len(candidates)
        policies = self.evidence_options.continuations
        pairs = [(i,p) for i in range(len(candidates)) for p in policies]
        features = [candidate_features(obs,candidates[i][0],p) for i,p in pairs]
        evaluated = list(range(len(pairs)))
        screening = None
        if self.surrogate is not None:
            if self.surrogate.target_protocol != self.target_protocol():
                raise ValueError('Surrogate label protocol differs from current planner')
            predicted = self.surrogate.predict(features)
            mandatory = {k for k,(i,p) in enumerate(pairs)
                         if ('provider' in candidates[i][1]['sources'] and p == 'provider')
                         or ('evidence_aware' in candidates[i][1]['sources'] and p == 'evidence')}
            ranked = sorted(range(len(pairs)),key=lambda k:(-predicted[k][0],predicted[k][1],k))
            evaluated = sorted(mandatory | set(ranked[:self.evidence_options.shortlist]))
            screening = {'predictions':predicted,'evaluated_pair_indices':evaluated,
                         'mandatory_pair_indices':sorted(mandatory),'device':self.evidence_options.device}
        state = hashlib.sha256(json.dumps(asdict(obs),sort_keys=True).encode()).hexdigest()
        seeds = [int(hashlib.sha256(f'joint:{self.seed}:{state}:{i}'.encode()).hexdigest()[:15],16)
                 for i in range(self.options.samples)]
        predictions = []
        for k in evaluated:
            i, policy = pairs[k]
            outcomes = [self.forecast_policy(obs,candidates[i][0],s,policy) for s in seeds]
            predictions.append({'pair_index':k,'candidate_index':i,'continuation':policy,
                'features':features[k],'by_model':{'central':outcomes},
                'score':self.score({'central':outcomes})})
        best = min(predictions,key=lambda r:(r['score'],r['pair_index']))
        index = best['candidate_index']
        actions, meta = candidates[index]
        self.stats['provider_selected'] += int('provider' in meta['sources'])
        self.stats['total_wall_s'] += time.perf_counter()-start
        self.last_diagnostics = {'method':self.name,'selected_index':index,'selected':meta,
            'selected_continuation':best['continuation'],'candidate_count':len(candidates),
            'candidates':[{'actions':[asdict(a) for a in acts],**m} for acts,m in candidates],
            'predictions':predictions,'screening':screening,'dependencies':dependency_graph(obs),
            'useful_arrivals':useful_arrivals(obs),'target_protocol':self.target_protocol(),
            'all_candidates_evaluated':len(evaluated)==len(pairs),
            'selection_objective':'max empirical completion; min mean J; central public model',
            'continuation_is_approximation_to_replanning':True}
        return actions

    def target_protocol(self):
        o = self.evidence_options
        return {'version':'evidence-label-1','samples':o.samples,'max_events':o.max_events,
                'provider_candidates':o.provider_candidates,'provider_cover_nodes':o.provider_cover_nodes,
                'protect_evidence':o.protect_evidence,'max_wait':o.max_wait,
                'slack_fraction':o.slack_fraction,'continuations':list(o.continuations)}
