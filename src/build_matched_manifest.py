"""Freeze exact indices for the matched composition-presence v2 protocol."""
import argparse
import json
from pathlib import Path

from alexgym_data import load_exercise
from matched_composition import make_matched_split


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("--protocol", default="configs/final_protocol.json")
    ap.add_argument("--out", default="configs/matched_manifest_v2.json"); args = ap.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    payload = {"version": "matched-composition-v2-shared-class-weights", "splits": {}}
    for exercise, targets in protocol["default_targets"].items():
        if not targets:
            continue
        _, Y, compositions, groups, _ = load_exercise(args.data, exercise, T=16)
        payload["splits"][exercise] = {}
        for target in targets:
            payload["splits"][exercise][target] = {}
            for seed in protocol["seeds"]:
                seen, unseen, val, test, audit = make_matched_split(Y, compositions, groups, target, seed)
                payload["splits"][exercise][target][str(seed)] = {
                    "seen_train": seen.tolist(), "unseen_train": unseen.tolist(),
                    "validation": val.tolist(), "test": test.tolist(), "audit": audit,
                }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2)); print(out)


if __name__ == "__main__":
    main()

