# Data setup

The ALEX-GYM-1 data are not redistributed in this repository. Download them from the official dataset source described by Hassan et al. (ICINCO 2025), then place the exercise workbooks and pose arrays in this directory using the original filenames.

Expected exercises are `squat`, `deadlift`, and `lunges`. Run the audit before training:

```bash
python src/audit_data_quality.py --data data --outdir results/data_audit
python src/build_split_manifest.py --data data --out configs/split_manifest_v2.json
```

The checked-in [raw data manifest](../results/data_audit/raw_data_manifest.json) records expected SHA-256 provenance. The processed release contains 294 squat, 269 deadlift, and 106 lunge repetitions after excluding one squat recording with an entirely missing frontal view. Missing whole frames are linearly interpolated; isolated zero-valued joints are preserved because their semantics cannot be verified.

The split unit is the paired frontal/lateral recording identifier available in the release—not a verified participant identity. Do not describe these experiments as participant-disjoint.

