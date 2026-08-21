# Audited results

The statistical unit is a held-out natural diagnosis composition. Results cover 12 eligible targets (7 squat, 5 deadlift), seeds 7, 42, and 123, four matched architectures, and a six-model full-target stress test.

| Research question | Estimate | 95% bootstrap CI | Two-sided test |
|---|---:|---:|---:|
| RQ1: matched target-present minus target-absent exact match | 25.44 pp | [17.31, 33.32] pp | .000488 |
| Secondary: matched bit-accuracy gap | 19.08 pp | [13.62, 26.10] pp | .000488 |
| RQ2 descriptive: LOP versus target-bit accuracy | ρ=-.632 | blocked permutation | <.00004 |
| RQ2 control: global opposing support versus accuracy | ρ=-.742 | blocked permutation | <.00004 |
| RQ2 incremental LOP beyond support and target bit | partial ρ=-.182 | blocked permutation | .200 |

RQ1 supports the paper's central contribution using identical evaluation recordings and matched training budget. RQ2's raw association generalizes across models, but its incremental form is unsupported: simple marginal criterion support explains more than local opposition.

Exact diagnosis match is the primary RQ1 outcome. Bit accuracy and micro-F1 describe criterion-level behavior. Fold-level macro-F1 is included in raw records for transparency but is not used as a primary LOCO discrimination metric because every target fold has a constant state for each criterion.

Machine-readable values are in [`results/rq_stats.json`](results/rq_stats.json). The original grid, matched v2 experiment, and graph baseline are retained separately under `results/`.
