"""Training-label-manifold risk for selective criterion assessment.

For each predicted diagnosis, the pressure feature counts training repetitions
with the same predicted context but the opposite predicted criterion bit.  A
logistic risk calibrator is fitted only on the fold validation predictions and
then evaluated on the unseen-composition test fold.
"""
import argparse,glob,json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .alexgym_data import load_exercise
    from .analyze_context_evidence import load_model, probabilities, split_rule
    from .loco_split import make_loco_split
except ImportError:
    from alexgym_data import load_exercise
    from analyze_context_evidence import load_model, probabilities, split_rule
    from loco_split import make_loco_split

CANONICAL_OUT = 'results/supplementary/run_metrics.csv'


def bootstrap_mean(x,n=100000,seed=20260820):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);v=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
    return [float(q) for q in np.quantile(v,[.025,.975])]

def flip(s,c):
    q=list(s);q[c]='1' if q[c]=='0' else '0';return ''.join(q)

def features(P,train_compositions):
    pred=(P>=.5).astype(int);strings=[''.join(map(str,row)) for row in pred];counts=Counter(train_compositions)
    rows=[]
    for i,s in enumerate(strings):
        for c in range(P.shape[1]):
            confidence=P[i,c] if pred[i,c] else 1-P[i,c]
            rows.append((1-confidence,np.log1p(counts.get(flip(s,c),0)),np.log1p(counts.get(s,0))))
    return np.asarray(rows,float),pred.reshape(-1)

def aurc(error,risk):
    order=np.argsort(risk);e=np.asarray(error)[order];curve=np.cumsum(e)/np.arange(1,len(e)+1)
    return float(curve.mean())

def selective_accuracy(error,risk,coverage):
    n=max(1,int(np.floor(len(error)*coverage)));idx=np.argsort(risk)[:n]
    return float(1-np.asarray(error)[idx].mean())

def safe_metric(error,risk,kind):
    if len(np.unique(error))<2:return float('nan')
    return float(average_precision_score(error,risk) if kind=='auprc' else roc_auc_score(error,risk))

def analyze(checkpoint,data):
    meta,m=load_model(checkpoint);ex=meta['exercise'];target=str(meta['target']);seed=int(meta['seed'])
    min_train_state,min_val_state=split_rule(meta)
    X,Y,co,g,_=load_exercise(data,ex,T=16)
    tr,va,te,_=make_loco_split(Y,co,g,target,seed,min_train_state=min_train_state,min_val_state=min_val_state)
    pv=probabilities(m,X[va]);pt=probabilities(m,X[te]);fv,qv=features(pv,co[tr]);ft,qt=features(pt,co[tr])
    ev=(qv!=Y[va].astype(int).reshape(-1)).astype(int);et=(qt!=Y[te].astype(int).reshape(-1)).astype(int)
    risks={'confidence':ft[:,0],'pressure':ft[:,1]}
    if len(np.unique(ev))>=2:
        cal=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000,random_state=seed))
        cal.fit(fv,ev);risks['combined']=cal.predict_proba(ft)[:,1]
    else:risks['combined']=ft[:,0]
    rows=[]
    for name,risk in risks.items():
        rows.append(dict(model=meta['model_kind'],exercise=ex,target=target,seed=seed,risk=name,n=len(et),error_rate=float(et.mean()),
                         error_auprc=safe_metric(et,risk,'auprc'),error_auroc=safe_metric(et,risk,'auroc'),aurc=aurc(et,risk),
                         accuracy_at_80=selective_accuracy(et,risk,.8),accuracy_at_60=selective_accuracy(et,risk,.6)))
    return rows

def safe_wilcoxon(d, zero_method='zsplit', alternative='two-sided'):
    d = np.asarray(d, float)
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method=zero_method, alternative=alternative).pvalue)
    except ValueError:
        # scipy raises ValueError for the genuinely degenerate cases (too few
        # samples, all-zero differences). Anything else is a real fault and
        # must surface rather than be reported as a null result.
        return 1.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoints',nargs='*');ap.add_argument('--data',default='data');ap.add_argument('--jobs',type=int,default=3)
    # results/supplementary is the single canonical home for this experiment. An
    # earlier default wrote a parallel results/selective_risk/v2 tree whose
    # run_metrics.csv was left empty.
    ap.add_argument('--out',default=CANONICAL_OUT);ap.add_argument('--from-csv')
    a=ap.parse_args();paths=[]
    for p in a.checkpoints:paths.extend(glob.glob(p))
    paths=[p for p in sorted(set(paths)) if Path(p).name.startswith(('fact_','tcn_','gru_','transformer_','ssm_'))]
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if not paths and not a.from_csv:
        if Path(CANONICAL_OUT).exists():
            a.from_csv = CANONICAL_OUT
        else:
            paths = [p for p in sorted(glob.glob('results/checkpoints/*.pt')) if Path(p).name.startswith(('fact_','tcn_','gru_','transformer_','ssm_'))]
    if a.from_csv:df=pd.read_csv(a.from_csv,dtype={'target':str})
    else:
        if not paths:raise FileNotFoundError('No checkpoints or pre-existing CSV found.')
        rows=[]
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for part in pool.map(lambda p:analyze(p,a.data),paths):rows.extend(part)
        df=pd.DataFrame(rows);df.to_csv(out,index=False)
    tm=df.groupby(['model','exercise','target','risk'],as_index=False).mean(numeric_only=True)
    tm.to_csv(out.with_name('target_mean_metrics.csv'),index=False)
    summary=tm.groupby(['model','risk'],as_index=False).mean(numeric_only=True)
    summary.to_csv(out.with_name('summary.csv'),index=False)
    paired=[]
    for model,g in tm.groupby('model'):
        c=g[g.risk=='combined'];b=g[g.risk=='confidence'];z=c.merge(b,on=['exercise','target'],suffixes=('_combined','_confidence'))
        for metric,direction in [('error_auprc',1),('error_auroc',1),('aurc',-1),('accuracy_at_80',1),('accuracy_at_60',1)]:
            d=(z[f'{metric}_combined']-z[f'{metric}_confidence'])*direction
            paired.append(dict(model=model,metric=metric,n=len(d),improvement=float(d.mean()),bootstrap95=bootstrap_mean(d),
                               wilcoxon_two_sided=safe_wilcoxon(d,alternative='two-sided',zero_method='zsplit')))
    overall=[]
    c=tm[tm.risk=='combined'];b=tm[tm.risk=='confidence'];z=c.merge(b,on=['model','exercise','target'],suffixes=('_combined','_confidence'))
    for metric,direction in [('error_auprc',1),('error_auroc',1),('aurc',-1),('accuracy_at_80',1),('accuracy_at_60',1)]:
        z['gain']=(z[f'{metric}_combined']-z[f'{metric}_confidence'])*direction
        target_gain=z.groupby(['exercise','target']).gain.mean().to_numpy()
        overall.append(dict(metric=metric,n_targets=len(target_gain),mean_improvement=float(target_gain.mean()),bootstrap95=bootstrap_mean(target_gain),
                            wilcoxon_two_sided=safe_wilcoxon(target_gain,alternative='two-sided',zero_method='zsplit')))
    payload={'checkpoints':len(paths) if paths else int(df[['model','exercise','target','seed']].drop_duplicates().shape[0]),'summary':summary.to_dict(orient='records'),'paired':paired,'overall_target_blocked':overall}
    out.with_name('stats.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
