"""Model-general robustness and simple-control analysis for Label-Opposition Pressure.

Uses the checked-in run JSONs only; no checkpoints or GPU are required. The
held-out diagnosis is the dependence block throughout.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from .analyze_context_evidence import blocked_permutation
except ImportError:
    from analyze_context_evidence import blocked_permutation


def build_rows(run_glob: str, context_csv: str) -> pd.DataFrame:
    context = pd.read_csv(context_csv, dtype={"target": str})
    context = context[(context.model == "fact") & (context.perturbation == "temporal_mean")]
    pressure = {
        (r.exercise, str(r.target), int(r.seed), int(r.criterion)): float(r.context_pressure)
        for r in context.itertuples()
    }
    rows = []
    for name in sorted(glob.glob(run_glob)):
        d = json.loads(Path(name).read_text())
        ex, target, seed = d["exercise"], str(d["target"]), int(d["seed"])
        model = d.get("model", "fact" if d.get("method") == "FACT" else "unknown")
        bits = np.asarray([int(v) for v in target])
        audit = d["split_audit"]
        lop = np.asarray([pressure[(ex, target, seed, c)] for c in range(len(bits))])
        for c, accuracy in enumerate(d["test"]["per_criterion_accuracy"]):
            target_bit = int(bits[c])
            state = audit["train_state_counts"][c]
            opposing = float(state[1 - target_bit])
            matching = float(state[target_bit])
            rows.append({
                "model": model, "exercise": ex, "target": target, "seed": seed,
                "criterion": c, "target_bit": target_bit,
                "accuracy": float(accuracy), "lop": float(lop[c]),
                "global_opposing_support": opposing,
                "global_matching_support": matching,
                "global_opposing_rate": opposing / float(audit["n_train"]),
                "target_frequency": float(audit["n_test"]),
                "target_cardinality": float(bits.sum()),
                "total_hamming1_support": float(lop.sum()),
            })
    return pd.DataFrame(rows)


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "exercise", "target", "criterion"]
    return rows.groupby(keys, as_index=False).mean(numeric_only=True)


def blocked_result(df: pd.DataFrame, x: str, permutations: int) -> dict:
    result = blocked_permutation(df, x, "accuracy", n=permutations)
    result["p_two_sided"] = min(1.0, 2.0 * result["p_one_sided"])
    return result


def residualized_rank_test(df: pd.DataFrame, permutations: int, seed: int = 20260821) -> dict:
    """Partial rank association of LOP after simple label-imbalance controls.

    All variables are ranked and centered inside diagnosis. Controls are target
    bit and global opposing-state rate. Target frequency, cardinality, and total
    Hamming-1 support are diagnosis constants and therefore cancel by design.
    """
    q = df.copy().reset_index(drop=True)
    group_indices = [g.index.to_numpy() for _, g in q.groupby(["exercise", "target"], sort=True)]

    def within_rank(column: str) -> np.ndarray:
        out = np.zeros(len(q), dtype=float)
        values = q[column].to_numpy(float)
        for idx in group_indices:
            r = rankdata(values[idx])
            out[idx] = r - r.mean()
        return out

    y = within_rank("accuracy")
    x = within_rank("lop")
    controls = np.column_stack([within_rank("global_opposing_rate"), within_rank("target_bit")])
    keep = np.linalg.norm(controls, axis=0) > 0
    controls = controls[:, keep]
    projection = np.eye(len(q)) - controls @ np.linalg.pinv(controls) if controls.size else np.eye(len(q))
    yr = projection @ y
    xr = projection @ x

    def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        numerator = a @ b
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    observed = float(corr_rows(xr[None, :], yr)[0])
    rng = np.random.default_rng(seed)
    permuted = np.tile(x, (permutations, 1))
    for idx in group_indices:
        order = np.argsort(rng.random((permutations, len(idx))), axis=1)
        permuted[:, idx] = x[idx][order]
    permuted_residual = permuted @ projection.T
    null = corr_rows(permuted_residual, yr)
    p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (permutations + 1)
    return {
        "partial_rank_correlation": observed,
        "p_two_sided": float(p),
        "n_target_criterion_units": int(len(q)),
        "n_targets": int(len(group_indices)),
        "controls": ["global_opposing_rate", "target_bit"],
        "note": "Target frequency, target cardinality, and total Hamming-1 support are constant within diagnosis and cancel under within-diagnosis ranking.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/*.json")
    ap.add_argument("--context", default="results/context/criterion_rows.csv")
    ap.add_argument("--outdir", default="results/lop_controls")
    ap.add_argument("--permutations", type=int, default=50000)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.runs, args.context)
    agg = aggregate(rows)
    rows.to_csv(out / "seed_rows.csv", index=False)
    agg.to_csv(out / "target_criterion_rows.csv", index=False)

    payload = {"per_model": {}, "all_model_mean": {}, "leave_one_model_out": {}, "controls": {}}
    for model, q in agg.groupby("model", sort=True):
        payload["per_model"][model] = blocked_result(q, "lop", args.permutations)

    mean = agg.groupby(["exercise", "target", "criterion"], as_index=False).mean(numeric_only=True)
    payload["all_model_mean"] = blocked_result(mean, "lop", args.permutations)
    payload["all_model_mean"]["partial_beyond_simple_controls"] = residualized_rank_test(mean, args.permutations)

    for omitted in sorted(agg.model.unique()):
        q = agg[agg.model != omitted].groupby(["exercise", "target", "criterion"], as_index=False).mean(numeric_only=True)
        payload["leave_one_model_out"][omitted] = blocked_result(q, "lop", args.permutations)

    for predictor in ["global_opposing_support", "global_opposing_rate"]:
        payload["controls"][predictor] = blocked_result(mean, predictor, args.permutations)

    (out / "stats.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
