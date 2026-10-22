"""Export paired completion, cost, latency and factorial figures as PDF/PNG."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#777777','#332288','#999933']
MAIN=['jrs_rg','fixed_jrs','qa_sequential','robust_local','whole_reset','overlap_batch','independent','fixed_random','jrs_release']
LABELS=['JRS-RG','Original JRS','QA sequential','Local conservative','Full reset','Overlap batching','Inspection first','Random search','v1.5']
def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=250,bbox_inches='tight');plt.close(fig)
def draw(folder):
    out=folder/'figures';out.mkdir(exist_ok=True)
    d=pd.read_csv(folder/'summary_v16.csv');conditions=list(d.condition.unique())
    fig,axes=plt.subplots(1,len(conditions),figsize=(3.8*len(conditions),3.7),squeeze=False)
    for ax,c in zip(axes[0],conditions):
        g=d[d.condition==c].set_index('method').reindex(MAIN)
        ax.bar(range(len(MAIN)),100*g.completion,color=COLORS)
        ax.set_xticks(range(len(MAIN)),LABELS,rotation=60,ha='right',fontsize=8);ax.set_title(c);ax.set_ylim(0,105)
        ax.set_ylabel('On-time qualified completion (%)');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save(fig,out,'completion')
    fig,axes=plt.subplots(1,len(conditions),figsize=(3.8*len(conditions),3.3),squeeze=False)
    for ax,c in zip(axes[0],conditions):
        for method,label,color in zip(MAIN,LABELS,COLORS):
            row=d[(d.condition==c)&(d.method==method)].iloc[0]
            ax.scatter(row.planning_wall,row.J,color=color,label=label,s=45)
        ax.set_xscale('log');ax.set_xlabel('Planning wall time / episode (s)');ax.set_ylabel('Original J (lower is better)');ax.set_title(c)
        ax.spines[['top','right']].set_visible(False)
    axes[0,-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);fig.tight_layout();save(fig,out,'cost_compute')
    p=pd.read_csv(folder/'primary_acceptance.csv');fig,ax=plt.subplots(figsize=(5.8,3.5))
    ax.errorbar(p.difference_pp,range(len(p)),xerr=[p.difference_pp-p.simultaneous_low_pp,p.simultaneous_high_pp-p.difference_pp],fmt='o',color=COLORS[0],capsize=4)
    ax.axvline(0,color='#888888',lw=1);ax.set_yticks(range(len(p)),p.condition);ax.set_xlabel('Completion difference vs. original JRS (pp)')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();save(fig,out,'primary_paired_contrasts')
    f=pd.read_csv(folder/'factorial_effects.csv');g=f[f.metric=='quality_success'].pivot(index='factor',columns='condition',values='effect')*100
    fig,ax=plt.subplots(figsize=(6,4));extent=max(abs(g.to_numpy()).max(),1);im=ax.imshow(g,vmin=-extent,vmax=extent,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(len(g.columns)),g.columns,rotation=30,ha='right');ax.set_yticks(range(len(g)),g.index)
    for y in range(len(g)):
        for x in range(len(g.columns)):ax.text(x,y,f'{g.iloc[y,x]:.1f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Factorial contrast (pp)');fig.tight_layout();save(fig,out,'factorial_effects')
    print(out)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args(argv);draw(a.analysis)


if __name__ == "__main__":
    main()
