from dataclasses import replace
from .recovery_graph import GraphOptions,RecoveryGraphPlanner
EXTRA={'jrs_rg_no_batch':dict(batch=False),'jrs_rg_no_reuse':{},'jrs_rg_serial':dict(parallel=False),
       'jrs_rg_fixed_phase':dict(continuation='fixed'),'jrs_rg_no_temporal':dict(temporal=False),'jrs_rg_random':dict(random_candidates=True)}
FACTORIAL=[f'jrs_rg_f{j}{u}{e}' for j in (0,1) for u in (0,1) for e in (0,1)]
RG_METHODS=['jrs_rg']+[m for m in FACTORIAL if m!='jrs_rg_f111']+list(EXTRA)
BASELINES=['fixed_jrs','qa_sequential','robust_local','whole_reset','overlap_batch','independent','fixed_random','jrs_release']
ALL_METHODS=['jrs_rg']+BASELINES+RG_METHODS[1:]
def make_rg(method,seed,budget):
    opt=GraphOptions(device=budget.get('rg_device','cpu'),candidates=budget.get('rg_candidates',16),samples=budget.get('rg_samples',64),phase_events=budget.get('rg_phase_events',2))
    if method in FACTORIAL:
        j,u,e=map(int,method[-3:]);opt=replace(opt,joint=bool(j),stochastic=bool(u),continuation='adaptive' if e else 'none')
    elif method in EXTRA:opt=replace(opt,**EXTRA[method])
    elif method!='jrs_rg':raise ValueError('Unknown graph variant')
    return RecoveryGraphPlanner(seed,opt)
