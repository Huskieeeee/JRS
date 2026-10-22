"""Candidate-budget curves on paired worlds; report actual wall cost, not nominal equality."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from .campaigns import prepare,ROOT
from .runner import run
from .analysis import analyze,csv_write

def sweep(device,workers,caps):
    summary=[]
    for cap in caps:
        cfg=prepare('main',device)
        cfg.update(split='v16_budget',world_start=16050000,worlds_per_layout=20,
                   methods=['jrs_rg','fixed_jrs','fixed_random','jrs_rg_random'])
        cfg['conditions']=[c for c in cfg['conditions'] if c['id'] in ('base','failure_high','deadline_tight')]
        cfg['planning_budget'].update(rg_candidates=cap,max_candidates=cap)
        path=ROOT/'configs/generated'/f'v16_budget_{device.replace(":","_")}_cap{cap}.json'
        path.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(cfg,indent=2)+'\n'
        if path.exists() and path.read_text()!=payload:raise ValueError('Existing budget protocol differs')
        path.write_text(payload)
        folder=ROOT/'results'/path.stem;out=ROOT/'analysis'/path.stem
        status=run(path,folder,False,None,workers)
        if status:raise RuntimeError(f'Budget cap {cap} did not finish: {status}')
        result=analyze(folder,out)
        latency={}
        for p in (folder/'jobs').glob('*/result.json'):
            r=json.loads(p.read_text());key=r['job']['condition'],r['job']['method']
            v=latency.setdefault(key,dict(wall=[],decisions=[]));v['wall'].append(r['planner_wall_seconds'])
            v['decisions'].extend(d['wall_seconds'] for d in json.loads((p.parent/'decisions.json').read_text()))
        for row in result['summary']:
            v=latency[row['condition'],row['method']]
            summary.append(dict(cap=cap,condition=row['condition'],method=row['method'],completion=row['qualified_completion'],
              J=row['J'],planning_wall=float(np.mean(v['wall'])),decision_p95=float(np.quantile(v['decisions'],.95))))
    target=ROOT/'analysis'/f'v16_budget_curve_{device.replace(":","_")}_{"_".join(map(str,caps))}.csv'
    csv_write(target,summary);print(target)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=1)
    p.add_argument('--caps',nargs='+',type=int,default=[4,8,16,32]);a=p.parse_args(argv)
    if any(c<2 for c in a.caps):p.error('Every cap must be >=2')
    sweep(a.device,a.workers,a.caps)


if __name__ == "__main__":
    main()
