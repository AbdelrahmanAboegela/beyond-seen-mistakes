"""Aggregate protocol-v2 runs with held-out composition as the statistical unit."""
import argparse,glob,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def bootstrap_mean(x,n=100000,seed=20260820):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1)
    return [float(v) for v in np.quantile(means,[.025,.975])]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',default='results/runs/*.json')
    ap.add_argument('--outdir',default='results/context');a=ap.parse_args();rows=[];skipped=[]
    for p in glob.glob(a.runs):
        d=json.loads(Path(p).read_text());model=d.get('model','fact' if d.get('method')=='FACT' else 'unknown')
        for split in ('val','test'):
            # a --no-test run stores test=None; record the omission rather than crashing
            if not isinstance(d.get(split),dict):skipped.append((Path(p).name,split));continue
            for metric,val in d[split].items():
                if isinstance(val,(int,float)):rows.append(dict(model=model,exercise=d['exercise'],target=str(d['target']),seed=d['seed'],split=split,metric=metric,value=val))
    if skipped:print(f'note: {len(skipped)} run/split records had no metrics and were skipped, e.g. {skipped[:3]}')
    raw=pd.DataFrame(rows);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);raw.to_csv(out/'metrics_long.csv',index=False)
    tm=raw.groupby(['model','exercise','target','split','metric'],as_index=False).value.mean()
    tm.to_csv(out/'target_mean_metrics.csv',index=False)
    summary=[]
    for (model,split,metric),g in tm.groupby(['model','split','metric']):
        summary.append(dict(model=model,split=split,metric=metric,n_targets=len(g),mean=float(g.value.mean()),bootstrap95=bootstrap_mean(g.value.to_numpy())))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    paired=[]
    for (split,metric),g in tm.groupby(['split','metric']):
        f=g[g.model=='fact']
        for other in sorted(set(g.model)-{'fact'}):
            t=g[g.model==other];z=f.merge(t,on=['exercise','target'],suffixes=('_fact','_other'))
            if not len(z):continue
            d=(z.value_fact-z.value_other).to_numpy();lo,hi=bootstrap_mean(d)
            paired.append(dict(other=other,split=split,metric=metric,n_targets=len(d),fact_mean=float(z.value_fact.mean()),
                other_mean=float(z.value_other.mean()),difference=float(d.mean()),difference_bootstrap95=[lo,hi],
                wilcoxon_two_sided=float(wilcoxon(d,zero_method='zsplit').pvalue),wins=int((d>0).sum()),ties=int((d==0).sum()),losses=int((d<0).sum())))
    (out/'paired_fact_models.json').write_text(json.dumps(paired,indent=2))
    print(json.dumps({'runs':int(raw[['model','exercise','target','seed']].drop_duplicates().shape[0]),'summary':summary,'paired':paired},indent=2))

if __name__=='__main__':main()
