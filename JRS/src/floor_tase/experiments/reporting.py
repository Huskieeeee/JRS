"""Paired main contrasts, complete factorial effects, latency and all baselines."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from .analysis import analyze,csv_write,holm,signflip_p
LABELS={'jrs_rg':'JRS-RG完整方法','fixed_jrs':'原始JRS','jrs_release':'冻结v1.5','qa_sequential':'QA顺序',
'robust_local':'保守局部','whole_reset':'全部重置','overlap_batch':'重叠批次','independent':'检查优先',
'fixed_random':'随机候选/确定性预报','jrs_rg_random':'随机候选/同一评估器','jrs_rg_no_batch':'去批次',
'jrs_rg_no_reuse':'去复用','jrs_rg_serial':'去并行','jrs_rg_fixed_phase':'固定续接','jrs_rg_no_temporal':'去时序耦合'}

def interval(values,strata,level=.95):
    values=np.asarray(values,float);strata=np.asarray(strata);rng=np.random.default_rng(1616);samples=np.zeros(4000)
    for label in sorted(set(strata)):
        ids=np.flatnonzero(strata==label)
        for start in range(0,4000,250):
            picks=rng.choice(ids,size=(min(250,4000-start),len(ids)),replace=True)
            samples[start:start+len(picks)]+=values[picks].sum(1)/len(values)
    a=(1-level)/2;return list(map(float,np.quantile(samples,[a,1-a])))

def report(run,out):
    result=analyze(run,out);cfg=json.loads((run/'protocol.json').read_text())
    rows=[];lookup={};mechanisms=[];decision_times={}
    for path in sorted((run/'jobs').glob('*/result.json')):
        r=json.loads(path.read_text());j=r['job'];m=r['metrics'];lookup[j['world_id'],j['condition'],j['method']]=r
        d=json.loads((path.parent/'decisions.json').read_text());times=[x['wall_seconds'] for x in d]
        decision_times.setdefault((j['condition'],j['method']),[]).extend(times)
        rows.append(dict(world=j['world_id'],layout=j['layout'],condition=j['condition'],method=j['method'],completion=int(m['quality_success']),
          J=m['J'],planning_wall=r['planner_wall_seconds'],initialization_wall=r['planner_initialization_wall_seconds'],episode_wall=r['wall_seconds'],
          decision_p95=float(np.quantile(times,.95)) if times else 0.,false_releases=m['false_releases'],waiting=m['waiting_time'],
          inspections=m['inspections'],revocations=m['successor_revocations']))
        stats=r['planner'];mechanisms.append(dict(world=j['world_id'],condition=j['condition'],method=j['method'],
          searches=stats.get('searches',0),phase_steps=stats.get('phase_steps',0),interruptions=stats.get('phase_interruptions',0),
          candidates=stats.get('candidates',0),gpu_batches=stats.get('model_batches',0),baseline_calls=stats.get('baseline_policy_calls','')))
    csv_write(out/'episode_readout.csv',rows);csv_write(out/'mechanisms.csv',mechanisms)
    conditions=[x['id'] for x in cfg['conditions']];summary=[]
    lines=['# JRS-RG v1.6 实验结果','',f"完成{len(rows)}回合；{result['unique_worlds']}个独立基础世界。",'',
      '|条件|方法|按期合格完成|J↓|规划墙钟秒/回合|初始化秒/回合|回合内决策P95均值（秒）|','|---|---|---:|---:|---:|---:|---:|']
    for c in conditions:
        for method in cfg['methods']:
            group=[x for x in rows if x['condition']==c and x['method']==method]
            v={k:float(np.mean([x[k] for x in group])) for k in ('completion','J','planning_wall','initialization_wall','decision_p95')}
            summary.append(dict(condition=c,method=method,**v))
            lines.append(f"|{c}|{LABELS.get(method,method)}|{sum(x['completion'] for x in group)}/{len(group)}|{v['J']:.4f}|{v['planning_wall']:.4f}|{v['initialization_wall']:.4f}|{v['decision_p95']:.4f}|")
    lines+=['','J沿用环境原定义，任务失败回合保留。延迟来自本次回合运行，独立预热计算测试另见benchmark_v16.py。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines));csv_write(out/'summary_v16.csv',summary)
    speed=[]
    for c in conditions:
        full=[x for x in rows if x['condition']==c and x['method']=='jrs_rg']
        old=[x for x in rows if x['condition']==c and x['method']=='jrs_release']
        denom=sum(x['planning_wall'] for x in old)
        ratio=sum(x['planning_wall'] for x in full)/denom if denom else float('inf')
        p95=float(np.quantile(decision_times[c,'jrs_rg'],.95))
        speed.append(dict(condition=c,wall_ratio_to_v15=ratio,pooled_decision_p95=p95,
                          ratio_at_most_1p2=ratio<=1.2,p95_at_most_20ms=p95<=.020))
    csv_write(out/'speed_acceptance.csv',speed)
    primary=[]
    for c in ('base','failure_high','deadline_tight'):
        worlds=sorted(w for w,cond,m in lookup if cond==c and m=='jrs_rg')
        if not worlds:continue
        a=[lookup[w,c,'jrs_rg'] for w in worlds];b=[lookup[w,c,'fixed_jrs'] for w in worlds]
        diff=[int(x['metrics']['quality_success'])-int(y['metrics']['quality_success']) for x,y in zip(a,b)]
        lo,hi=interval(diff,[x['job']['layout'] for x in a],1-.05/3)
        primary.append(dict(condition=c,difference_pp=100*float(np.mean(diff)),simultaneous_low_pp=100*lo,simultaneous_high_pp=100*hi,
          base_noninferior=lo>-.03 if c=='base' else '',positive_interval=lo>0 if c!='base' else '',
          target_8pp=float(np.mean(diff))>=.08 if c!='base' else ''))
    csv_write(out/'primary_acceptance.csv',primary)
    effects=[]
    for c in conditions:
        worlds=sorted(w for w,cond,m in lookup if cond==c and m=='jrs_rg')
        for axes in ((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)):
            for metric in ('quality_success','J'):
                values=[];strata=[]
                for w in worlds:
                    value=0.
                    for bits in itertools.product((0,1),repeat=3):
                        method='jrs_rg' if bits==(1,1,1) else 'jrs_rg_f'+''.join(map(str,bits))
                        value+=np.prod([2*bits[i]-1 for i in axes])*float(lookup[w,c,method]['metrics'][metric])/(2**(3-len(axes)))
                    values.append(value);strata.append(lookup[w,c,'jrs_rg']['job']['layout'])
                lo,hi=interval(values,strata)
                effects.append(dict(condition=c,factor=''.join('JUE'[i] for i in axes),metric=metric,effect=float(np.mean(values)),ci_low=lo,ci_high=hi,raw_p=signflip_p(values)))
    for metric in ('quality_success','J'):
        group=[r for r in effects if r['metric']==metric]
        for row,pv in zip(group,holm([r['raw_p'] for r in group])):row['holm_p']=pv
    csv_write(out/'factorial_effects.csv',effects)
    lines=['# 全部配对对照','', '|条件|对照|完成差pp|完成95%区间pp|J差|J95%区间|完成Holm p|J Holm p|','|---|---|---:|---|---:|---|---:|---:|']
    for r in result['comparisons']:
        lines.append(f"|{r['condition']}|{r['control']}|{100*r['completion_difference']:.2f}|{[round(100*x,2) for x in r['completion_ci95']]}|{r['J_difference']:.4f}|{[round(x,4) for x in r['J_ci95']]}|{r['completion_holm_p_exploratory']:.4f}|{r['J_holm_p_exploratory']:.4f}|")
    lines+=['','本表完成率与J各自按全部对照×条件校正。主验收使用primary_acceptance.csv，因子效应独立成族。']
    (out/'ALL_BASELINES_ZH.md').write_text('\n'.join(lines))
    lines=['# v1.6 主验收读数','', '|条件|完成差pp|同时区间下界pp|同时区间上界pp|','|---|---:|---:|---:|']
    for r in primary:lines.append(f"|{r['condition']}|{r['difference_pp']:.2f}|{r['simultaneous_low_pp']:.2f}|{r['simultaneous_high_pp']:.2f}|")
    lines+=['','区间按基础世界、布局分层配对重抽样，三项主比较使用Bonferroni水平。3世界smoke的区间可退化，只用于执行检查。',
      '主结果RESULTS_ZH.md；全部对照ALL_BASELINES_ZH.md；机制mechanisms.csv；交互factorial_effects.csv；速度验收speed_acceptance.csv。']
    (out/'READOUT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(episodes=len(rows),worlds=result['unique_worlds'],out=str(out))))
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True);report(a.run,a.out)


if __name__ == "__main__":
    main()
