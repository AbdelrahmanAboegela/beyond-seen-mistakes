"""Joint label-context and anatomy-conditioned perturbation diagnostic.

This module intentionally reports *prediction sensitivity*, not causal feature
importance.  For every criterion it compares the change in correct-class
confidence after perturbing the pre-specified criterion region with changes
under size-matched random joint sets from the same view.  Context pressure is
computed only from the fold's training labels.
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata,spearmanr,wilcoxon

try:
    from .alexgym_data import CRITERIA, load_exercise
    from .loco_split import make_loco_split, DEFAULT_MIN_TRAIN_STATE, DEFAULT_MIN_VAL_STATE
    from .modern_models import ANAT_MAP
    from .train_backbones import build
    from .train_fact import FACT
except ImportError:
    from alexgym_data import CRITERIA, load_exercise
    from loco_split import make_loco_split, DEFAULT_MIN_TRAIN_STATE, DEFAULT_MIN_VAL_STATE
    from modern_models import ANAT_MAP
    from train_backbones import build
    from train_fact import FACT


CANONICAL_OUT = 'results/context/criterion_rows.csv'


def load_model(checkpoint):
    d=torch.load(checkpoint,map_location='cpu',weights_only=False)
    kind=d['model_kind'];ex=d['exercise']
    if kind=='fact':
        m=FACT(ex,features=d.get('features','j'),soft_mask=d.get('soft_mask',True),
               use_proto=d.get('use_proto',True),global_mask=d.get('global_mask',False),
               shared_adapter=d.get('shared_adapter',False),map_control=d.get('map_control','anatomy'))
    else:
        m=build(kind,int(d['input_dim']),int(d['n_outputs']))
    m.load_state_dict(d['model']);m.eval()
    return d,m


def probabilities(model,X,batch=128):
    out=[]
    with torch.no_grad():
        for lo in range(0,len(X),batch):
            z=torch.as_tensor(X[lo:lo+batch],dtype=torch.float32)
            out.append(torch.sigmoid(model(z)).cpu().numpy())
    return np.concatenate(out)


def perturb(X,view,joints,mode):
    z=np.asarray(X,dtype=np.float32).copy();off=0 if view=='F' else 99
    jj=np.asarray(joints,dtype=int)
    cols=np.concatenate([np.arange(off+3*j,off+3*j+3) for j in jj])
    if mode=='zero':z[:,:,cols]=0
    elif mode=='temporal_mean':z[:,:,cols]=z[:,:,cols].mean(1,keepdims=True)
    else:raise ValueError(mode)
    return z


def correct_confidence(p,y):
    return np.where(y==1,p,1-p)


def flip_bit(target,c):
    q=list(target);q[c]='1' if q[c]=='0' else '0';return ''.join(q)


def blocked_permutation(df,x,y,n=50000,seed=20260820):
    """Shuffle x within held-out compositions and average within-composition rho."""
    groups=[]
    for _,g in df.groupby(['exercise','target'],sort=True):
        rx=rankdata(g[x].to_numpy());ry=rankdata(g[y].to_numpy())
        rx=rx-rx.mean();ry=ry-ry.mean();den=np.linalg.norm(rx)*np.linalg.norm(ry)
        if den>0:groups.append((rx,ry,den))
    if not groups:
        # Every composition was constant in x or y, so no within-target rank
        # association is defined; returning 0/1 here would fake a null result.
        raise ValueError(f'No held-out composition has variation in both {x!r} and {y!r}.')
    observed=float(np.mean([np.dot(rx,ry)/den for rx,ry,den in groups]))
    rng=np.random.default_rng(seed);null=np.zeros(n)
    for rx,ry,den in groups:
        order=np.argsort(rng.random((n,len(rx))),axis=1)
        null+=(rx[order]@ry)/den
    null/=len(groups)
    if observed>=0:p=(1+int(np.sum(null>=observed)))/(n+1)
    else:p=(1+int(np.sum(null<=observed)))/(n+1)
    return dict(mean_within_target_rho=observed,p_one_sided=float(p),n_permutations=n,
                null_mean=float(np.mean(null)),n_informative_targets=len(groups))


def split_rule(meta):
    """Reuse the support thresholds a checkpoint was trained under, so the
    reconstructed split is byte-identical to the one that produced it."""
    audit=meta.get('split_audit') or {}
    return (int(audit.get('min_train_state',DEFAULT_MIN_TRAIN_STATE)),
            int(audit.get('min_val_state',DEFAULT_MIN_VAL_STATE)))


def analyze_one(checkpoint,data,n_random=50,modes=('zero','temporal_mean')):
    meta,m=load_model(checkpoint);ex=meta['exercise'];target=str(meta['target']);seed=int(meta['seed'])
    X,Y,co,g,_=load_exercise(data,ex,T=16)
    min_train_state,min_val_state=split_rule(meta)
    tr,_,te,audit=make_loco_split(Y,co,g,target,seed,min_train_state=min_train_state,min_val_state=min_val_state)
    base=probabilities(m,X[te]);truth=Y[te].astype(int);base_correct=(base>=.5)==truth
    rng=np.random.default_rng(seed+1701);rows=[]
    for c,(view,joints) in enumerate(ANAT_MAP[ex]):
        # The size-matched control samples from all 33 joints, so it may overlap
        # the annotated set.  That makes relative sensitivity conservative.
        joints=sorted(set(map(int,joints)));available=np.arange(33)
        pressure=int(np.sum(co[tr]==flip_bit(target,c)))
        for mode in modes:
            mapped=probabilities(m,perturb(X[te],view,joints,mode))
            mapped_drop=correct_confidence(base[:,c],truth[:,c])-correct_confidence(mapped[:,c],truth[:,c])
            random_inputs=[]
            for _ in range(n_random):
                rj=rng.choice(available,size=len(joints),replace=False)
                random_inputs.append(perturb(X[te],view,rj,mode))
            rp=probabilities(m,np.concatenate(random_inputs)).reshape(n_random,len(te),Y.shape[1])
            random_drops=correct_confidence(base[None,:,c],truth[None,:,c])-correct_confidence(rp[:,:,c],truth[None,:,c])
            mapped_spill=np.abs(mapped-base)[:,np.arange(Y.shape[1])!=c].mean() if Y.shape[1]>1 else 0.
            rows.append(dict(checkpoint=str(checkpoint),model=meta['model_kind'],exercise=ex,
                target=target,seed=seed,criterion=c,criterion_name=CRITERIA[ex][c],target_bit=int(target[c]),
                perturbation=mode,n_test=len(te),n_mapped_joints=len(joints),context_pressure=pressure,
                baseline_accuracy=float(base_correct[:,c].mean()),baseline_correct_confidence=float(correct_confidence(base[:,c],truth[:,c]).mean()),
                mapped_confidence_drop=float(mapped_drop.mean()),random_confidence_drop=float(random_drops.mean()),
                relative_evidence_sensitivity=float(mapped_drop.mean()-random_drops.mean()),
                random_drop_sd=float(random_drops.mean(1).std(ddof=1)) if n_random>1 else float('nan'),mapped_spillover=float(mapped_spill),
                split_offset=audit['split_offset']))
    return rows


def safe_wilcoxon(d, zero_method='zsplit', alternative='two-sided'):
    d = np.asarray(d, float)
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method=zero_method, alternative=alternative).pvalue)
    except Exception:
        return 1.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoints',nargs='*',help='Checkpoint paths or glob patterns.')
    # results/context is the single canonical home for these outputs. An earlier
    # default wrote a second copy under results/context_evidence, leaving two
    # trees that disagreed in the last float digit.
    ap.add_argument('--data',default='data');ap.add_argument('--out',default=CANONICAL_OUT)
    ap.add_argument('--random-masks',type=int,default=50)
    ap.add_argument('--from-csv',help='Reuse an existing raw criterion-row CSV and only recompute statistics.')
    a=ap.parse_args();paths=[]
    for pattern in a.checkpoints:paths.extend(glob.glob(pattern))
    paths=sorted(set(paths))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if not paths and not a.from_csv:
        default_csv = Path(CANONICAL_OUT)
        if default_csv.exists():
            a.from_csv = str(default_csv)
        else:
            paths = sorted(glob.glob('results/checkpoints/*.pt'))
    if a.from_csv:df=pd.read_csv(a.from_csv)
    else:
        if not paths:raise FileNotFoundError('No checkpoints or pre-existing CSV found.')
        rows=[]
        for p in paths:rows.extend(analyze_one(p,a.data,a.random_masks))
        df=pd.DataFrame(rows);df.to_csv(out,index=False)
    # The dependence-aware statistical unit is a target-criterion pair, averaged
    # across seeds.  Raw seed rows remain in the CSV for reproducibility.
    agg=df.groupby(['model','perturbation','exercise','target','criterion'],as_index=False).mean(numeric_only=True)
    agg.to_csv(out.with_name(out.stem+'_target_criterion.csv'),index=False)
    stats=[]
    for (model,mode),q in agg.groupby(['model','perturbation']):
        r,p=spearmanr(q.context_pressure,q.baseline_accuracy)
        er,ep=spearmanr(q.relative_evidence_sensitivity,q.baseline_accuracy)
        cr,cp=spearmanr(q.context_pressure,q.relative_evidence_sensitivity)
        stats.append(dict(model=model,perturbation=mode,n=len(q),pressure_vs_accuracy_rho=r,
                          pressure_vs_accuracy_p=p,evidence_vs_accuracy_rho=er,evidence_vs_accuracy_p=ep,
                          pressure_vs_evidence_rho=cr,pressure_vs_evidence_p=cp,
                          blocked_pressure_accuracy=blocked_permutation(q,'context_pressure','baseline_accuracy'),
                          blocked_evidence_accuracy=blocked_permutation(q,'relative_evidence_sensitivity','baseline_accuracy')))
    paired=[]
    for mode,q in agg.groupby('perturbation'):
        f=q[q.model=='fact'];t=q[q.model=='tcn'];z=f.merge(t,on=['exercise','target','criterion'],suffixes=('_fact','_tcn'))
        for metric in ('baseline_accuracy','relative_evidence_sensitivity'):
            d=z[f'{metric}_fact']-z[f'{metric}_tcn']
            paired.append(dict(perturbation=mode,metric=metric,n=len(d),mean_difference=float(d.mean()),
                               wilcoxon_two_sided=safe_wilcoxon(d,zero_method='zsplit')))
    payload={'criterion_level':stats,'paired_fact_minus_tcn':paired}
    stat_path=out.with_name(out.stem+'_stats.json');stat_path.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'checkpoints':len(paths),'rows':len(df),'aggregated_rows':len(agg),'out':str(out),'stats':payload},indent=2))


if __name__=='__main__':main()
