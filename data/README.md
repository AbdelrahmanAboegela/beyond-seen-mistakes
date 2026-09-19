# Data setup

The ALEX-GYM-1 data are not redistributed in this repository. Download them from the official dataset source described by Hassan et al. (ICINCO 2025), then place the exercise workbooks and pose arrays in this directory using the original filenames.

Expected exercises are `squat`, `deadlift`, and `lunges`. Run the audit before training:

```bash
python src/audit_data_quality.py --data data --outdir results/data_audit
python src/build_split_manifest.py --data data --out configs/split_manifest_v2.json
```

The checked-in [raw data manifest](../results/data_audit/raw_data_manifest.json) records expected SHA-256 provenance. The processed release contains 294 squat, 269 deadlift, and 106 lunge repetitions after excluding one squat recording with an entirely missing frontal view. Missing whole frames are linearly interpolated; isolated zero-valued joints are preserved because their semantics cannot be verified.

The split unit is the paired frontal/lateral recording identifier available in the release—not a verified participant identity. Do not describe these experiments as participant-disjoint.

## CPR-Coach (second dataset, external composition check)

[CPR-Coach](https://github.com/Shunli-Wang/CPR-Coach) (Wang et al., CVPR 2024) annotates 13 error types and releases recordings in which two, three or four co-occur, which is why it is the external check on the matched result. Only the keypoints and annotations are needed — **2.6 GB**, not the ~408 GB full release:

```bash
mkdir -p data/cpr_coach && cd data/cpr_coach
curl -LO https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/main/ann.tar.gz
curl -LO https://huggingface.co/datasets/ShunliWang/CPR-Coach/resolve/main/Keypoints.tar.gz
tar -xzf ann.tar.gz && tar -xzf Keypoints.tar.gz
```

Then build the cached arrays:

```bash
python src/cpr_data.py --data data/cpr_coach
```

Notes that differ from ALEX-GYM-1 and matter for interpretation:

- Poses are **2D COCO-17** from AlphaPose, not 3D MediaPipe-33, and each take has **four synchronised channels** rather than two views. The loader concatenates the channels into one repetition; treating them as separate rows would place the same performance in train and test.
- The recording group is the **class folder** (e.g. `CPR_Double_Dataset_S0/DC00_S0`). All 234 folders are composition-pure, and every eligible composition has exactly three folders, which matches the protocol's three-seed distinct-test-group rule.
- Applying the frozen support rule (≥10 repetitions, ≥3 source groups, ≥8 training examples per criterion state) yields **72 eligible targets** — 13 single-error and 59 double-error compositions — from 1,416 repetitions. The 15 triple- and quadruple-error compositions have 8 repetitions in a single folder and are excluded, as lunge is in ALEX-GYM-1.
- Folder names carry `S0`/`S1` session tags, but these are not verified participant identities either. The same recording-disjoint, **not** participant-disjoint caveat applies.

