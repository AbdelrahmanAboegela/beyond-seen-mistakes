# Audited results

The statistical unit is a held-out natural diagnosis composition. Results cover 12 eligible targets (7 squat, 5 deadlift), seeds 7, 42, and 123, four matched architectures, and a six-model full-target stress test.

| Research question | Estimate | 95% bootstrap CI | Two-sided test |
|---|---:|---:|---:|
| RQ1: optimized-exchange target-present minus target-absent exact match | 21.13 pp | [9.86, 34.73] pp | .000488 |
| Secondary: optimized-exchange bit-accuracy gap | 12.80 pp | [7.38, 18.79] pp | .000488 |
| RQ2 descriptive: LOP versus target-bit accuracy | ρ=-.632 | blocked permutation | <.00004 |
| RQ2 control: global opposing support versus accuracy | ρ=-.742 | blocked permutation | <.00004 |
| RQ2 incremental LOP beyond support and target bit | partial ρ=-.182 | blocked permutation | .200 |

RQ1 supports the paper's central contribution using identical evaluation recordings and matched training budget. A deterministic minimax-then-L1 exchange reduces the mean worst criterion-count mismatch by 50.6% and total mismatch by 39.3%; the earlier random-exchange exact-match gap is a compatible 25.44 pp. Marginals are improved rather than identical, so the result is not presented as causal. RQ2's raw association generalizes across models, but its incremental form is unsupported: simple marginal criterion support (`global_opposing_support`, $G_c$) explains more than local opposition.

$G_c$ is ordinary per-criterion class imbalance, not a new statistic — its link to degraded minority-state generalization is established in the imbalanced-learning literature (Buda et al. 2018, He & Garcia 2009, Cui et al. 2019). The paper's contribution here is confirming that it subsumes the more localized LOP account under composition shift, and that the effect is practically large enough to matter: among the 67 target-criterion units, the bottom $G_c$ quartile averages 82.70% held-out accuracy versus 40.79% for the top quartile. Because $G_c$ is a count over training labels only, it can be computed for a proposed diagnosis composition before any model is trained, serving as a training-free pre-training check for which criteria are likely to fail under composition shift. Reproduce the quartile split with `results/lop_controls/target_criterion_rows.csv` (grouped by `exercise`, `target`, `criterion`, averaged over models — the same 67 rows RQ2 uses) split on the `global_opposing_support` quartiles.

Exact diagnosis match is the primary RQ1 outcome. Bit accuracy and micro-F1 describe criterion-level behavior. Fold-level macro-F1 is included in raw records for transparency but is not used as a primary LOCO discrimination metric because every target fold has a constant state for each criterion.

Machine-readable values are in [`results/rq_stats.json`](results/rq_stats.json). The original grid, random matched-v2 experiment, optimized matched-v3 experiment, and graph baseline are retained separately under `results/`.

## External replication on CPR-Coach

| Quantity | ALEX-GYM-1 | CPR-Coach |
|---|---:|---:|
| Eligible compositions | 12 | **72** |
| Exact match, target present | 22.65% | 15.72% |
| Exact match, target absent | 1.52% | 3.92% |
| Gap | 21.13 pp | **11.80 pp** |
| 95% bootstrap CI | [9.86, 34.73] | **[9.03, 14.77]** |
| Compositions falling | 12/12 | 62/72 |
| RQ2 partial LOP (controlled) | −.182 (*p*=.200) | **−.005 (*p*=.87)** |

Frozen in [`results/cpr_rq_stats.json`](results/cpr_rq_stats.json) and [`results/cpr_lop_controls/stats.json`](results/cpr_lop_controls/stats.json); both are checked by `scripts/verify_release.py`.

Two architectures added on CPR-Coach do **not** support the effect and are reported as such: ST-GCN reaches 2.22% vs 0.35% exact match with 64/72 compositions tied at zero (it fails the task outright), and FACT shows no gap under its anatomical prior (−3.40 pp) while a random joint map of equal size gives +4.86 pp — so its anatomical prior does not transfer.

One published claim does **not** replicate and is no longer stated unqualified: on ALEX-GYM-1 marginal support was the stronger raw predictor (ρ=−.742 vs LOP's −.632), but on CPR-Coach the ordering reverses (−.255 vs −.316). What holds on both datasets is the *null* — local opposition adds nothing once ordinary support is controlled — not the ranking of the two raw predictors.

## Temporal representation

| Criterion type (CPR-Coach) | Lift over own trivial baseline | With rate features |
|---|---:|---:|
| Posture (6) | +7.71 pp | +8.38 pp |
| Hand (3) | ~0 | ~0 |
| **Rate/timing (4)** | **−14.50 pp** | **+3.23 pp** |

Reproduce with `python src/compose_cpr_results.py` and `python src/analyze_cpr_lop.py`.
