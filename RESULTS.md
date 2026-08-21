# Audited results

The statistical unit is a held-out natural diagnosis composition. Results cover 12 eligible targets (7 squat, 5 deadlift), seeds 7, 42, and 123, four matched architectures, and a six-model full-target stress test.

| Research question | Estimate | 95% bootstrap CI | Two-sided test |
|---|---:|---:|---:|
| RQ1: optimized-exchange target-present minus target-absent exact match | 21.13 pp | [9.86, 34.73] pp | .000488 |
| Secondary: optimized-exchange bit-accuracy gap | 12.80 pp | [7.38, 18.79] pp | .000488 |
| RQ2 descriptive: LOP versus target-bit accuracy | ρ=-.632 | blocked permutation | <.00004 |
| RQ2 control: global opposing support versus accuracy | ρ=-.742 | blocked permutation | <.00004 |
| RQ2 incremental LOP beyond support and target bit | partial ρ=-.182 | blocked permutation | .200 |

RQ1 supports the paper's central contribution using identical evaluation recordings and matched training budget. A deterministic minimax-then-L1 exchange reduces the mean worst criterion-count mismatch by 50.6% and total mismatch by 39.3%; the earlier random-exchange exact-match gap is a compatible 25.44 pp. Marginals are improved rather than identical, so the result is not presented as causal. RQ2's raw association generalizes across models, but its incremental form is unsupported: simple marginal criterion support explains more than local opposition.

Exact diagnosis match is the primary RQ1 outcome. Bit accuracy and micro-F1 describe criterion-level behavior. Fold-level macro-F1 is included in raw records for transparency but is not used as a primary LOCO discrimination metric because every target fold has a constant state for each criterion.

Machine-readable values are in [`results/rq_stats.json`](results/rq_stats.json). The original grid, random matched-v2 experiment, optimized matched-v3 experiment, and graph baseline are retained separately under `results/`.
