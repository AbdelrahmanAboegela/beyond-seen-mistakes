"""RQ2 on CPR-Coach: does local label opposition beat ordinary marginal support?

This mirrors ``analyze_lop_controls`` exactly -- same blocked permutation, same
residualised partial rank test, same diagnosis-level dependence block -- so the
two datasets answer RQ2 by the same method.  Two things differ by necessity:

* ALEX-GYM-1 reads Label-Opposition Pressure from a precomputed context table
  produced by the FACT sweep.  CPR-Coach has no such table, so LOP is computed
  here directly from its definition,
  ``LOP_c = |{i in train : y_{i,-c} = y*_{-c} and y_{i,c} != y*_c}|``,
  over the fold's own training indices in the frozen manifest.
* The statistical unit is taken from the **target-absent** condition of the
  matched runs, which is the analogue of the LOCO runs ALEX-GYM-1 uses: in both
  the model has never seen the held-out composition.

Criteria 1-3 of CPR-Coach are hand-placement errors that the body skeleton
cannot observe (hand keypoint confidence 0.058), so their accuracy is driven by
observability rather than by support.  The sensitivity excluding them is
reported alongside, and was specified before these numbers were computed.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_lop_controls import blocked_result, residualized_rank_test
from cpr_data import load_cpr

HAND_CRITERIA = (0, 1, 2)


def label_space_statistics(Y: np.ndarray, train_idx: np.ndarray, target_bits: np.ndarray):
    """Return per-criterion ``(LOP, opposing support, matching support)``."""
    train = Y[train_idx].astype(int)
    criteria = len(target_bits)
    lop = np.zeros(criteria)
    opposing = np.zeros(criteria)
    matching = np.zeros(criteria)
    for c in range(criteria):
        others = [j for j in range(criteria) if j != c]
        context_match = np.all(train[:, others] == target_bits[others], axis=1)
        differs = train[:, c] != target_bits[c]
        lop[c] = float(np.sum(context_match & differs))
        opposing[c] = float(np.sum(differs))
        matching[c] = float(np.sum(~differs))
    return lop, opposing, matching


def build_rows(run_glob: str, manifest_path: str, data: str) -> pd.DataFrame:
    _, Y, _, _, _ = load_cpr(data)
    Y = np.asarray(Y, dtype=int)
    manifest = json.loads(Path(manifest_path).read_text())["splits"]["cpr"]
    cache: dict[tuple[str, int], tuple] = {}
    rows = []
    for name in sorted(glob.glob(run_glob)):
        d = json.loads(Path(name).read_text())
        target, seed = str(d["target"]), int(d["seed"])
        bits = np.asarray([int(v) for v in target])
        key = (target, seed)
        if key not in cache:
            train_idx = np.asarray(manifest[target][str(seed)]["unseen_train"], dtype=int)
            cache[key] = label_space_statistics(Y, train_idx, bits) + (len(train_idx),)
        lop, opposing, matching, n_train = cache[key]
        for c, accuracy in enumerate(d["unseen"]["test"]["per_criterion_accuracy"]):
            rows.append({
                "model": d["model"], "exercise": "cpr", "target": target, "seed": seed,
                "criterion": c, "target_bit": int(bits[c]),
                "accuracy": float(accuracy), "lop": float(lop[c]),
                "global_opposing_support": float(opposing[c]),
                "global_matching_support": float(matching[c]),
                "global_opposing_rate": float(opposing[c]) / float(n_train),
                "target_cardinality": float(bits.sum()),
                "total_hamming1_support": float(lop.sum()),
                "is_hand_criterion": c in HAND_CRITERIA,
            })
    return pd.DataFrame(rows)


def analyse(agg: pd.DataFrame, permutations: int) -> dict:
    mean = agg.groupby(["exercise", "target", "criterion"], as_index=False).mean(numeric_only=True)
    out = {
        "n_target_criterion_units": int(len(mean)),
        "lop": blocked_result(mean, "lop", permutations),
        "global_opposing_support": blocked_result(mean, "global_opposing_support", permutations),
        "partial_beyond_simple_controls": residualized_rank_test(mean, permutations),
    }
    if mean.global_opposing_support.nunique() > 3:
        quartile = mean.global_opposing_support.quantile([.25, .75])
        low = mean[mean.global_opposing_support <= quartile.loc[.25]].accuracy.mean()
        high = mean[mean.global_opposing_support >= quartile.loc[.75]].accuracy.mean()
        out["quartile_contrast"] = {
            "bottom_quartile_accuracy": float(low), "top_quartile_accuracy": float(high),
            "note": "Descriptive; reuses the RQ2 sample rather than validating on held-out units.",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/cpr_matched_v3_optimized/runs/*.json")
    ap.add_argument("--manifest", default="configs/cpr_matched_manifest_v3_optimized.json")
    ap.add_argument("--data", default="data/cpr_coach")
    ap.add_argument("--outdir", default="results/cpr_lop_controls")
    ap.add_argument("--permutations", type=int, default=50000)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.runs, args.manifest, args.data)
    agg = rows.groupby(["model", "exercise", "target", "criterion"], as_index=False).mean(numeric_only=True)
    rows.to_csv(out / "seed_rows.csv", index=False)
    agg.to_csv(out / "target_criterion_rows.csv", index=False)

    payload = {
        "protocol": "matched target-absent condition; LOP computed from the frozen manifest",
        "all_criteria": analyse(agg, args.permutations),
        "excluding_hand_criteria": analyse(
            agg[~agg.criterion.isin(HAND_CRITERIA)], args.permutations),
        "per_model": {m: blocked_result(q, "lop", args.permutations)
                      for m, q in agg.groupby("model", sort=True)},
    }
    (out / "stats.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["all_criteria"], indent=2))
    print(f"\nwrote {out/'stats.json'}")


if __name__ == "__main__":
    main()
