"""Single source of truth for the three predeclared v2 research questions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from analyze_context_evidence import blocked_permutation

def boot(x,n=100000,seed=20260820):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);v=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
    return [float(q) for q in np.quantile(v,[.025,.975])]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--classification',default='results/context/target_mean_metrics.csv')
    ap.add_argument('--context',default='results/context/criterion_rows_target_criterion.csv')
    ap.add_argument('--selective',default='results/supplementary/target_mean_metrics.csv')
    ap.add_argument('--out',default='results/rq_stats.json');a=ap.parse_args()

    cls=pd.read_csv(a.classification,dtype={'target':str});rq1={}
    for metric in ('exact_match','bit_accuracy'):
        q=cls[cls.metric==metric].pivot_table(index=['model','exercise','target'],columns='split',values='value').reset_index()
        q['gap']=q.val-q.test;tg=q.groupby(['exercise','target']).gap.mean().to_numpy()
        rq1[metric]={'n_targets':len(tg),'mean_validation_minus_loco':float(tg.mean()),'bootstrap95':boot(tg),
                     'wilcoxon_two_sided':float(wilcoxon(tg,alternative='two-sided',zero_method='zsplit').pvalue)}

    ctx=pd.read_csv(a.context,dtype={'target':str});ctx=ctx[ctx.perturbation=='temporal_mean']
    ctx=ctx.groupby(['exercise','target','criterion'],as_index=False).agg(context_pressure=('context_pressure','mean'),baseline_accuracy=('baseline_accuracy','mean'))
    rq2=blocked_permutation(ctx,'context_pressure','baseline_accuracy')
    rq2['p_two_sided']=min(1.0,2*rq2['p_one_sided'])

    sel=pd.read_csv(a.selective,dtype={'target':str});c=sel[sel.risk=='combined'];b=sel[sel.risk=='confidence']
    z=c.merge(b,on=['model','exercise','target'],suffixes=('_lmr','_confidence'))
    z['gain']=z.error_auprc_lmr-z.error_auprc_confidence;tg=z.groupby(['exercise','target']).gain.mean().to_numpy()
    rq3={'n_targets':len(tg),'mean_error_auprc_gain':float(tg.mean()),'bootstrap95':boot(tg),
         'wilcoxon_two_sided':float(wilcoxon(tg,alternative='two-sided',zero_method='zsplit').pvalue)}

    raw=[rq1['exact_match']['wilcoxon_two_sided'],rq2['p_two_sided'],rq3['wilcoxon_two_sided']]
    adjusted=multipletests(raw,alpha=.05,method='holm')[1].tolist()
    payload={'primary_tests':{
        'H1_exact_match_gap':{**rq1['exact_match'],'holm_p':adjusted[0]},
        'H2_pressure_accuracy':{**rq2,'holm_p':adjusted[1]},
        'H3_lmr_error_auprc':{**rq3,'holm_p':adjusted[2]}},
        'secondary':{'H1_bit_accuracy_gap':rq1['bit_accuracy']},
        'statistical_unit':'held-out natural composition (12 targets)'}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
