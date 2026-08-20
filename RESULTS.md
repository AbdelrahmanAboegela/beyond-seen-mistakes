# Audited results

The statistical unit is a held-out natural diagnosis composition. Results cover 12 eligible targets (7 squat, 5 deadlift), five models, and seeds 7, 42, and 123.

| Research question | Estimate | 95% bootstrap CI | Two-sided test | Holm-adjusted |
|---|---:|---:|---:|---:|
| RQ1: ordinary grouped validation minus targeted LOCO exact match | 11.34 pp | [8.09, 14.54] pp | .000488 | .000977 |
| RQ2: within-target Spearman association of LOP with target-bit accuracy | -0.651 | blocked permutation | <.00004 | <.00012 |
| Supplemental RQ3: LMR error-AUPRC gain over confidence | +4.99 pp | [-0.15, 10.26] pp | .0923 | .0923 |

RQ1 and RQ2 support the paper's bounded contribution: ordinary grouped validation can hide failures on deliberately held-out, naturally observed error sets, and a training-label-only diagnostic anticipates vulnerable criteria. RQ3 is not supported overall and is retained only to preserve the original multiple-testing family and avoid selective reporting.

Exact diagnosis match is the primary RQ1 outcome. Bit accuracy and micro-F1 describe criterion-level behavior. Fold-level macro-F1 is included in raw records for transparency but is not used as a primary LOCO discrimination metric because every target fold has a constant state for each criterion.

Machine-readable values are in [`results/rq_stats.json`](results/rq_stats.json); the 180 per-run records are in [`results/runs`](results/runs).

