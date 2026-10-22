"""Portable NumPy / PyTorch CUDA MLP for candidate-policy screening.

Weights are plain .npz numeric arrays plus JSON metadata (never pickle).
The scalar outputs approximate synthetic continuation returns, not calibrated
physical success probabilities. Only exact CPU forecasts select the action.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FEATURE_NAMES = (
    'time_fraction','remaining_fraction','log_cells','log_tasks','log_robots',
    'certified_fraction','pending_task_fraction','done_task_fraction','pending_job_fraction',
    'active_fraction','packet_fraction','min_slack','mean_slack','freshness_ratio',
    'delay_ratio','collateral_assumption','failure_assumption','sensor_assumption',
    'travel_per_cell','repairs_per_cell','action_fraction','repair_fraction',
    'expanded_fraction','inspect_fraction','successor_fraction','impact_fraction',
    'fix_fraction','recertify_fraction','done_task_impact_fraction','inspection_coverage',
    'action_duration_ratio','incoming_impact_fraction','evidence_policy','provider_policy','handoff_policy')
FEATURE_SCHEMA = 'public-aggregate-35-v1'


def candidate_features(obs, actions, policy):
    from .planners import _impact, _fixes, _dist
    from .evidence_recovery import useful_arrivals
    cells = {c.id:c for c in obs.cells}; jobs = {j.id:j for j in obs.jobs}
    views = {v.id:v for v in obs.views}; tasks = {t.id:t for t in obs.tasks}
    robots = {r.id:r for r in obs.robots}
    n, nt, nr, h = max(1,len(cells)),max(1,len(tasks)),max(1,len(robots)),max(1.,obs.horizon)
    pending = [t for t in obs.tasks if t.status == 'pending']
    slacks = [(t.deadline-obs.time-t.duration)/h for t in pending] or [0.]
    impact, fixes, coverage = set(),set(),set()
    duration = 0.
    for a in actions:
        if a.kind == 'repair':
            target = jobs[a.target_id]
            impact |= set(cells) if obs.evidence_policy == 'whole_reset' else _impact(target,a.mode)
            fixes |= _fixes(target,a.mode)
            duration += target.expanded_duration if a.mode == 'expanded' else target.duration
        elif a.kind == 'inspect':
            target = views[a.target_id]; coverage |= set(target.covers); duration += target.duration
        else:
            target = tasks[a.target_id]; duration += target.duration
        duration += _dist(robots[a.robot_id].position,target.position)/obs.travel_speed
    x = [obs.time/h,(obs.horizon-obs.time)/h,np.log1p(n),np.log1p(nt),np.log1p(nr),
         sum(c.certified for c in obs.cells)/n,len(pending)/nt,sum(t.status=='done' for t in obs.tasks)/nt,
         sum(j.status=='pending' for j in obs.jobs)/n,len(obs.active)/nr,len(obs.pending_inspections)/n,
         min(slacks),sum(slacks)/len(slacks),obs.freshness/h,obs.measurement_delay/h,
         obs.assumed_collateral_probability,obs.assumed_repair_failure_prob,obs.assumed_sensor_error,
         obs.accrued_travel/(30*n),obs.accrued_repairs/n,len(actions)/nr,
         sum(a.kind=='repair' for a in actions)/nr,sum(a.kind=='repair' and a.mode=='expanded' for a in actions)/nr,
         sum(a.kind=='inspect' for a in actions)/nr,sum(a.kind=='successor' for a in actions)/nr,
         len(impact)/n,len(fixes)/n,sum(cells[q].certified for q in impact)/n,
         sum(t.status=='done' and bool(set(t.requires)&impact) for t in obs.tasks)/nt,
         len(coverage)/n,duration/h,len(impact & useful_arrivals(obs).keys())/n,
         float(policy=='evidence'),float(policy=='provider'),float(policy=='handoff')]
    if len(x) != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid public feature vector')
    return [float(v) for v in x]


class Surrogate:
    def __init__(self, path, expected_sha256, device='cpu'):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Surrogate SHA256 mismatch')
        with np.load(path,allow_pickle=False) as z:
            self.meta = json.loads(str(z['metadata']))
            self.weights = {k:z[k].astype(np.float32) for k in ('mean','scale','w1','b1','w2','b2','w3','b3')}
        if self.meta['feature_schema'] != FEATURE_SCHEMA:
            raise ValueError('Surrogate feature schema mismatch')
        self.target_protocol = self.meta['target_protocol']
        self.device = device
        self.torch = None
        if device != 'cpu':
            if not device.startswith('cuda'):
                raise ValueError('Only cpu or cuda device supported')
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
            self.torch = torch
            self.tensors = {k:torch.as_tensor(v,device=device) for k,v in self.weights.items()}

    def predict(self, features):
        x = np.asarray(features,dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError('Invalid surrogate input')
        if self.torch is None:
            w = self.weights
            x = (x-w['mean'])/w['scale']
            x = np.maximum(x@w['w1']+w['b1'],0)
            x = np.maximum(x@w['w2']+w['b2'],0)
            out = x@w['w3']+w['b3']
            out[:,0] = 1/(1+np.exp(-np.clip(out[:,0],-40,40)))
            out[:,1] = np.maximum(0,out[:,1]*5.)
        else:
            t,w = self.torch,self.tensors
            with t.inference_mode():
                x = (t.as_tensor(x,device=self.device)-w['mean'])/w['scale']
                x = t.relu(x@w['w1']+w['b1'])
                x = t.relu(x@w['w2']+w['b2'])
                out = x@w['w3']+w['b3']
                out = t.stack((t.sigmoid(out[:,0]),t.clamp(out[:,1]*5.,min=0)),dim=1).cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite surrogate output')
        return out.tolist()
