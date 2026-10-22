import argparse,hashlib,json,sys
from pathlib import Path
from .paths import ROOT
from floor_tase.rg_variants import ALL_METHODS

def prepare(stage,device,shard=0,shards=1):
    c=json.loads((ROOT/'configs/v15_main_template.json').read_text());width={'smoke':1,'probe':4,'main':200}[stage]
    if shards>width or not 0<=shard<shards:raise ValueError('Invalid world shard')
    c.update(split='v16_'+stage,world_start={'smoke':16000000,'probe':16001000,'main':16010000}[stage],
        worlds_per_layout=width,world_shard={'index':shard,'count':shards},methods=ALL_METHODS,primary_method='jrs_rg',formal=False,
        claims='Recovery-graph scheduling, fixed baselines and J/U/E mechanism experiments.',
        stop_rule='Run the complete frozen manifest. Execution errors remain distinct from mission failures.')
    if stage!='main':c['conditions']=[x for x in c['conditions'] if x['id'] in ('base','failure_high','deadline_tight')]
    b=c['planning_budget'];b.update(rg_device=device,rg_candidates=16,rg_samples=64,rg_phase_events=2,
                                 release_device=device,release_model_path='models/v15_ensemble_cpu.npz')
    b['release_model_sha256']=hashlib.sha256((ROOT/b['release_model_path']).read_bytes()).hexdigest()
    c['factorial']={'J':'joint recovery-chain objective','U':'stochastic duration propagation','E':'adaptive chain execution','full_alias':'jrs_rg_f111 = jrs_rg'}
    c['acceptance']={'target_gain_pp':8,'base_noninferiority_pp':3,'p95_seconds':.020,'planning_ratio_to_v15':1.2,
                     'sample_size':'600 worlds planned; use development discordance to confirm power before freezing main'}
    return c

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','probe','main']);p.add_argument('--device',default='cuda')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--out',type=Path)
    a=p.parse_args(argv);c=prepare(a.stage,a.device,a.shard,a.shards)
    out=a.out or ROOT/'configs/generated'/f'v16_{a.stage}_{a.device.replace(":","_")}_{a.shard}of{a.shards}.json'
    out.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(c,indent=2)+'\n'
    if out.exists() and out.read_text()!=content:raise ValueError('Config changed; use a new --out path')
    out.write_text(content);print(out)


if __name__ == "__main__":
    main()
