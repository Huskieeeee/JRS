"""Matched CPU/CUDA graph-score parity, evaluator latency and online decision timing."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from .paths import ROOT
from floor_tase.model import ScenarioConfig,make_scenario
from floor_tase.environment import RecoveryEnv
from floor_tase.recovery_graph import GraphOptions,GraphCompiler,BatchedCompletion,RecoveryGraphPlanner,templates
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--out',type=Path,default=Path('analysis/v16_gpu_benchmark.json'))
    a=p.parse_args(argv);rows=[];parity=[];online=[]
    for seed in (16090001,16090002,16090003):
        scenario=make_scenario(seed,ScenarioConfig(assumed_repair_failure_prob=.15,repair_failure_prob=.15));obs=RecoveryEnv(scenario).observe()
        for cap in (4,8,16,32):
            opt=GraphOptions(candidates=cap);plans=[GraphCompiler(obs,opt).compile(t) for t in templates(obs,opt,seed)]
            cpu=BatchedCompletion('cpu');device=BatchedCompletion(a.device)
            c=cpu.evaluate(plans,64,seed);g=device.evaluate(plans,64,seed)
            matrix=lambda v:np.array([[x[k] for k in ('completion','qualified_fraction','cost','finish_p90')] for x in v])
            np.testing.assert_allclose(matrix(c),matrix(g),rtol=1e-4,atol=2e-5)
            rank=lambda v:min(range(len(v)),key=lambda i:(-v[i]['completion'],-v[i]['qualified_fraction'],v[i]['cost'],v[i]['finish_p90'],i))
            parity.append(dict(seed=seed,cap=cap,max_difference=float(abs(matrix(c)-matrix(g)).max()),same_selected=rank(c)==rank(g)))
            for label,ev in [('cpu',cpu),(a.device,device)]:
                for _ in range(5):ev.evaluate(plans,64,seed)
                times=[]
                for _ in range(20):
                    start=time.perf_counter();ev.evaluate(plans,64,seed);times.append(time.perf_counter()-start)
                rows.append(dict(seed=seed,candidates=len(plans),device=label,mean_seconds=float(np.mean(times)),p95_seconds=float(np.quantile(times,.95))))
        for device in ('cpu',a.device):
            warm=RecoveryGraphPlanner(seed,GraphOptions(device=device));warm.select(obs,RecoveryEnv(scenario).allowed_actions())
            env=RecoveryEnv(scenario);planner=RecoveryGraphPlanner(seed,GraphOptions(device=device));times=[]
            while not env.done:
                start=time.perf_counter();acts=planner.select(env.observe(),env.allowed_actions());times.append(time.perf_counter()-start)
                env.step_joint(acts) if acts else env.advance()
            online.append(dict(seed=seed,device=device,planning_wall=sum(times),decision_p95=float(np.quantile(times,.95)),
                               completion=env.result()['quality_success'],J=env.result()['J']))
    if not all(x['same_selected'] for x in parity):raise RuntimeError('CPU/CUDA selected candidate differs')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(parity=parity,evaluator=rows,online=online,includes_transfers=True),indent=2)+'\n');print(a.out)


if __name__ == "__main__":
    main()
