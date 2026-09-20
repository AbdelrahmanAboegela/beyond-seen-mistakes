"""Freeze every CPR-Coach number the paper quotes into one verifiable file.

Mirrors ``compose_final_results`` for the second dataset.  Everything here is
recomputed from the run records, so a reader can regenerate it without a GPU
and CI can check it has not drifted.

Model groupings matter and are kept separate on purpose:

* ``rq1_backbones`` is the four generic temporal backbones, the same set the
  published ALEX-GYM-1 matched test uses.  This is the headline comparison.
* ``stgcn`` and ``fact_*`` are additions on this dataset only and are reported
  apart from the headline rather than averaged into it.
* ``rate_features`` is the same folds and manifest with the duration/speed/
  cadence channels restored, so the contrast is an input ablation, not a
  different experiment.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BACKBONES = ("tcn", "gru", "transformer", "ssm")


def target_equal(run_glob, metric, models=None):
    """Average seeds within (model, target), then models within target."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for name in glob.glob(run_glob):
        d = json.loads(Path(name).read_text())
        if models and d["model"] not in models:
            continue
        for cond in ("seen", "unseen"):
            acc[(d["target"], cond)][d["model"]].append(d[cond]["test"][metric])
    per = collections.defaultdict(dict)
    for (target, cond), by_model in acc.items():
        per[target][cond] = float(np.mean([np.mean(v) for v in by_model.values()]))
    return per


def contrast(run_glob, metric, models=None, bootstrap=100000, seed=20260921):
    per = target_equal(run_glob, metric, models)
    targets = sorted(t for t in per if {"seen", "unseen"} <= set(per[t]))
    seen = np.array([per[t]["seen"] for t in targets])
    unseen = np.array([per[t]["unseen"] for t in targets])
    # Round before classifying direction. A target whose two conditions are
    # genuinely identical can come out at -2.8e-17 from the nested averaging,
    # which would be counted as a reversal rather than a tie.
    gap = np.round(seen - unseen, 9)
    rng = np.random.default_rng(seed)
    draws = np.array([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(bootstrap)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # The exact method underflows to 0.0 well before this many units; the normal
    # approximation is reported so the magnitude is at least representable.
    p = float(wilcoxon(seen, unseen, alternative="two-sided", method="approx").pvalue)
    return {
        "metric": metric, "n_targets": int(len(targets)),
        "seen_mean": float(seen.mean()), "unseen_mean": float(unseen.mean()),
        "seen_minus_unseen": float(gap.mean()),
        "bootstrap95": [float(lo), float(hi)],
        "wilcoxon_two_sided_approx": p,
        "targets_lower_when_absent": int((gap > 0).sum()),
        "targets_tied": int((gap == 0).sum()),
        "targets_higher_when_absent": int((gap < 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="results/cpr_matched_v3_optimized/runs/*.json")
    ap.add_argument("--rate", default="results/cpr_matched_v3_rate/runs/*.json")
    ap.add_argument("--fact-anatomy", default="results/cpr_fact_anatomy/runs/*.json")
    ap.add_argument("--fact-random", default="results/cpr_fact_random/runs/*.json")
    ap.add_argument("--lop", default="results/cpr_lop_controls/stats.json")
    ap.add_argument("--out", default="results/cpr_rq_stats.json")
    a = ap.parse_args()

    payload = {
        "dataset": "CPR-Coach",
        "protocol": "matched target-present/absent, optimized minimax-then-L1 exchange",
        "rq1_backbones": {m: contrast(a.base, m, BACKBONES)
                          for m in ("exact_match", "bit_accuracy", "micro_f1")},
        "rq1_per_backbone": {b: contrast(a.base, "exact_match", (b,)) for b in BACKBONES},
        "stgcn": contrast(a.base, "exact_match", ("stgcn",)),
        "rate_features": {m: contrast(a.rate, m, BACKBONES)
                          for m in ("exact_match", "bit_accuracy")},
        "fact_anatomy_map": contrast(a.fact_anatomy, "exact_match"),
        "fact_random_map": contrast(a.fact_random, "exact_match"),
    }
    lop = json.loads(Path(a.lop).read_text())
    payload["rq2"] = {
        "all_criteria": lop["all_criteria"],
        "excluding_hand_criteria": lop["excluding_hand_criteria"],
    }
    Path(a.out).write_text(json.dumps(payload, indent=2))

    r = payload["rq1_backbones"]["exact_match"]
    print(f"RQ1 (4 backbones): {r['seen_mean']*100:.2f}% -> {r['unseen_mean']*100:.2f}%  "
          f"gap {r['seen_minus_unseen']*100:+.2f} pp  n={r['n_targets']}")
    q = payload["rate_features"]["exact_match"]
    print(f"with rate features: {q['seen_mean']*100:.2f}% -> {q['unseen_mean']*100:.2f}%  "
          f"gap {q['seen_minus_unseen']*100:+.2f} pp")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
