#!/usr/bin/env python3
"""Audit raw ALEX-GYM pose files before any preprocessing/training.

This script is intentionally independent of ``alexgym_data.load_exercise``:
it records quality and provenance of the raw public files so a paper release can
make preprocessing decisions explicit rather than silently consuming a cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXERCISES = ("squat", "deadlift", "lunges")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sequence_quality(seq: object) -> dict[str, int | bool]:
    x = np.asarray(seq, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (33, 3):
        return {"malformed": True, "frames": int(x.shape[0]) if x.ndim else 0,
                "nonfinite_frames": 0, "zero_frames": 0, "zero_joint_frames": 0,
                "degenerate_scale_frames": 0}
    finite = np.isfinite(x).all(axis=(1, 2))
    zero_frame = np.all(np.abs(x) <= 1e-8, axis=(1, 2))
    zero_joint = np.all(np.abs(x) <= 1e-8, axis=2)
    # The primary normalization uses these two body widths.  A nonpositive width
    # makes the frame unsuitable as a scale source, even if other joints exist.
    shoulder = np.linalg.norm(x[:, 11] - x[:, 12], axis=1)
    hip = np.linalg.norm(x[:, 23] - x[:, 24], axis=1)
    bad_scale = (~np.isfinite(shoulder)) | (~np.isfinite(hip)) | ((shoulder <= 1e-5) & (hip <= 1e-5))
    return {
        "malformed": False,
        "frames": int(len(x)),
        "nonfinite_frames": int((~finite).sum()),
        "zero_frames": int(zero_frame.sum()),
        "zero_joint_frames": int(zero_joint.sum()),
        "degenerate_scale_frames": int(bad_scale.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--outdir", default="results/data_audit")
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, rows = {"schema": 1, "data_root": str(root.resolve()), "files": {}}, []

    for ex in EXERCISES:
        xlsx = root / f"{ex}.xlsx"
        front_path, lat_path = root / f"front_pose_{ex}.json", root / f"lat_pose_{ex}.json"
        for p in (xlsx, front_path, lat_path):
            if not p.exists():
                raise FileNotFoundError(p)
            manifest["files"][str(p.name)] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        df = pd.read_excel(xlsx)
        with front_path.open(encoding="utf-8") as f:
            front = json.load(f)
        with lat_path.open(encoding="utf-8") as f:
            lat = json.load(f)
        if not (len(df) == len(front) == len(lat)):
            raise ValueError(f"{ex}: workbook/front/lateral lengths differ: {len(df)}, {len(front)}, {len(lat)}")
        for i, (fs, ls) in enumerate(zip(front, lat)):
            rec = {"exercise": ex, "index": i}
            for view, seq in (("front", fs), ("lateral", ls)):
                for key, value in sequence_quality(seq).items():
                    rec[f"{view}_{key}"] = value
            rec["any_quality_flag"] = bool(
                rec["front_malformed"] or rec["lateral_malformed"] or
                rec["front_nonfinite_frames"] or rec["lateral_nonfinite_frames"] or
                rec["front_zero_frames"] or rec["lateral_zero_frames"] or
                rec["front_degenerate_scale_frames"] or rec["lateral_degenerate_scale_frames"]
            )
            rows.append(rec)

    detail = pd.DataFrame(rows)
    detail.to_csv(out / "raw_pose_sequence_quality.csv", index=False)
    summary = detail.groupby("exercise", as_index=False).agg(
        repetitions=("index", "size"), flagged_repetitions=("any_quality_flag", "sum"),
        front_zero_frames=("front_zero_frames", "sum"), lateral_zero_frames=("lateral_zero_frames", "sum"),
        front_nonfinite_frames=("front_nonfinite_frames", "sum"), lateral_nonfinite_frames=("lateral_nonfinite_frames", "sum"),
        front_degenerate_scale_frames=("front_degenerate_scale_frames", "sum"),
        lateral_degenerate_scale_frames=("lateral_degenerate_scale_frames", "sum"),
    )
    summary.to_csv(out / "raw_pose_quality_summary.csv", index=False)
    manifest["quality_summary"] = summary.to_dict(orient="records")
    (out / "raw_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
