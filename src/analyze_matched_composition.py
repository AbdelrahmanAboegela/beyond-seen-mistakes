"""Analyze paired target-present versus target-absent training conditions."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap(x, n=100000, seed=20260821):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    values = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(v) for v in np.quantile(values, [.025, .975])]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="results/matched_composition/runs/*.json")
    ap.add_argument("--outdir", default="results/matched_composition"); args = ap.parse_args()
    rows = []
    for path in glob.glob(args.runs):
        d = json.loads(Path(path).read_text())
        for condition in ("seen", "unseen"):
            for metric, value in d[condition]["test"].items():
                if isinstance(value, (int, float)):
                    rows.append({"model": d["model"], "exercise": d["exercise"], "target": str(d["target"]),
                                 "seed": d["seed"], "condition": condition, "metric": metric, "value": value})
    raw = pd.DataFrame(rows); out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "metrics_long.csv", index=False)
    target = raw.groupby(["model", "exercise", "target", "condition", "metric"], as_index=False).value.mean()
    target.to_csv(out / "target_mean_metrics.csv", index=False)
    stats = []
    for (model, metric), group in target.groupby(["model", "metric"]):
        z = group.pivot(index=["exercise", "target"], columns="condition", values="value").dropna()
        delta = (z.seen - z.unseen).to_numpy()
        stats.append({"model": model, "metric": metric, "n_targets": len(delta),
                      "seen_mean": float(z.seen.mean()), "unseen_mean": float(z.unseen.mean()),
                      "seen_minus_unseen": float(delta.mean()), "bootstrap95": bootstrap(delta),
                      "wilcoxon_two_sided": float(wilcoxon(delta, zero_method="zsplit").pvalue)})
    for metric, group in target.groupby("metric"):
        pooled = group.groupby(["exercise", "target", "condition"], as_index=False).value.mean()
        z = pooled.pivot(index=["exercise", "target"], columns="condition", values="value").dropna()
        delta = (z.seen - z.unseen).to_numpy()
        stats.append({"model": "model_mean", "metric": metric, "n_targets": len(delta),
                      "seen_mean": float(z.seen.mean()), "unseen_mean": float(z.unseen.mean()),
                      "seen_minus_unseen": float(delta.mean()), "bootstrap95": bootstrap(delta),
                      "wilcoxon_two_sided": float(wilcoxon(delta, zero_method="zsplit").pvalue)})
    (out / "stats.json").write_text(json.dumps(stats, indent=2)); print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
