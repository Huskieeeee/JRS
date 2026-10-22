import argparse
from pathlib import Path
from .runner import load_json,receipt_for,write_json,jobs_for

def merge(runs,out):
    protocols=[load_json(r/'protocol.json') for r in runs];canonical=[]
    for c in protocols:d=dict(c);d.pop('world_shard',None);canonical.append(d)
    if any(c!=canonical[0] for c in canonical):raise ValueError('Shard protocols differ')
    expected=receipt_for(canonical[0]);sources={}
    for r,c in zip(runs,protocols):
        if not load_json(r/'COMPLETION.json')['data_complete']:raise ValueError('Incomplete shard: '+str(r))
        if load_json(r/'FREEZE.json')['source']['sha256']!=expected['source']['sha256']:raise ValueError('Source differs')
        for j in jobs_for(c):
            if j['job_id'] in sources:raise ValueError('Duplicate job')
            sources[j['job_id']]=(r/'jobs'/j['job_id']).resolve()
    if {j['job_id'] for j in jobs_for(canonical[0])}!=set(sources):raise ValueError('Incomplete coverage')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use empty merge output')
    (out/'jobs').mkdir()
    for name,source in sources.items():(out/'jobs'/name).symlink_to(source,target_is_directory=True)
    write_json(out/'protocol.json',canonical[0]);write_json(out/'FREEZE.json',expected)
    write_json(out/'COMPLETION.json',dict(data_complete=True,planned=len(sources),complete=len(sources),errors=0))
    write_json(out/'SHARDS.json',dict(source_runs=[str(r.resolve()) for r in runs],retain_source_directories=True))
    print(out)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);merge(a.runs,a.out)


if __name__ == "__main__":
    main()
