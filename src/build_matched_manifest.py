"""Freeze exact indices for random-v2 or optimized-v3 matched protocols."""
import argparse
import json
from pathlib import Path

from matched_composition import (load_dataset, make_matched_split,
                                 make_optimized_manifest_split)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("--protocol", default="configs/final_protocol.json")
    ap.add_argument("--source-manifest", help="Freeze an optimized replacement while retaining these folds")
    ap.add_argument("--out", default="configs/matched_manifest_v2.json"); args = ap.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    version = "matched-composition-v3-optimized-marginals" if args.source_manifest else "matched-composition-v2-shared-class-weights"
    payload = {"version": version, "splits": {}}
    for exercise, targets in protocol["default_targets"].items():
        if not targets:
            continue
        _, Y, compositions, groups = load_dataset(args.data, exercise, T=16)
        payload["splits"][exercise] = {}
        for target in targets:
            payload["splits"][exercise][target] = {}
            for seed in protocol["seeds"]:
                if args.source_manifest:
                    seen, unseen, val, test, audit = make_optimized_manifest_split(
                        Y, compositions, groups, exercise, target, seed, args.source_manifest)
                else:
                    seen, unseen, val, test, audit = make_matched_split(Y, compositions, groups, target, seed)
                payload["splits"][exercise][target][str(seed)] = {
                    "seen_train": seen.tolist(), "unseen_train": unseen.tolist(),
                    "validation": val.tolist(), "test": test.tolist(), "audit": audit,
                }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2)); print(out)


if __name__ == "__main__":
    main()
