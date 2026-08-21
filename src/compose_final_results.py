"""Compose the paper's final machine-readable RQ summary from audited analyses."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--matched", default="results/matched_composition_v3_optimized/stats.json")
    ap.add_argument("--lop", default="results/lop_controls/stats.json")
    ap.add_argument("--out", default="results/rq_stats.json"); args = ap.parse_args()
    matched = json.loads(Path(args.matched).read_text())
    lookup = {(row["model"], row["metric"]): row for row in matched}
    lop = json.loads(Path(args.lop).read_text())
    payload = {
        "protocol_version": "matched-composition-v3-optimized-marginals",
        "statistical_unit": "held-out natural diagnosis (12 targets)",
        "primary": {
            "RQ1_matched_exact_match": lookup[("model_mean", "exact_match")],
        },
        "secondary": {
            "RQ1_matched_bit_accuracy": lookup[("model_mean", "bit_accuracy")],
            "RQ1_matched_micro_f1": lookup[("model_mean", "micro_f1")],
        },
        "diagnostic": {
            "RQ2_raw_lop": lop["all_model_mean"],
            "RQ2_global_opposing_support": lop["controls"]["global_opposing_support"],
            "RQ2_lop_partial_beyond_support_and_target_bit": lop["all_model_mean"]["partial_beyond_simple_controls"],
        },
        "inference_note": "Exact match was fixed before the v2 and optimized-v3 reruns. The v3 exchange optimization was specified before inspecting v3 model outcomes; all probabilities are two-sided.",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2)); print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
