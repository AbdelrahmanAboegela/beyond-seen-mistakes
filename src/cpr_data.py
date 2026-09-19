"""Loader for the CPR-Coach composite-error dataset.

CPR-Coach is the closest published analogue to this study's composition
protocol: it annotates 13 error types and releases recordings in which two,
three or four of them co-occur.  That makes it the natural external check on
the matched present/absent result, which ALEX-GYM-1 alone cannot supply.

This module mirrors ``alexgym_data.load_exercise`` closely enough that
``loco_split.make_loco_split`` and the matched-composition machinery run
against it unchanged: both return a float32 ``X``, a binary criterion matrix
``Y``, per-repetition composition strings, and recording-group identifiers.

Three facts about the release drive the implementation:

* Each take is recorded on four synchronised channels (``ch0``-``ch3``).  They
  are one repetition, not four, so the channels are concatenated into a single
  feature vector exactly as ALEX-GYM-1 concatenates its frontal and lateral
  views.  Treating them as separate rows would put the same performance in
  train and test.
* Keypoints are 2D COCO-17 from AlphaPose, not 3D MediaPipe-33.  The
  normalisation below is the same pelvis-centre/body-scale recipe as
  ALEX-GYM-1 with COCO joint indices substituted.
* Class folders are composition-pure, so the recording group can be the folder
  (``CPR_Double_Dataset_S0/DC00_S0``).  Every eligible composition has exactly
  three such folders, which is what the protocol's three-seed, distinct
  test-group rule expects.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

PREPROCESS_VERSION = "cpr-coach-v1-coco17-4ch"

# Label 0 is the correct action; 1..13 are the error criteria, in ActionList order.
CPR_ERRORS = [
    "Overlap Hands", "Clenching Hands", "Single Hand", "Bending Arms",
    "Tilting Arms", "Jump Pressing", "Squatting", "Standing", "Wrong Position",
    "Insufficient Pressing", "Slow Frequency", "Excessive Pressing",
    "Random Position Pressing",
]
NCRIT = len(CPR_ERRORS)

# COCO-17 indices used for normalisation.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# Some frame_dir values carry a _BIG suffix that the annotation lists omit.
_SUFFIX = re.compile(r"_BIG$")


def _parse_frame_dir(frame_dir: str):
    """``dataset/class/r##/ch#`` -> (group, repetition_key, channel).

    The dataset segment is normalised so keypoint entries join against the
    annotation lists, which spell the same folder without ``_BIG``.
    """
    parts = frame_dir.strip("/").split("/")
    if len(parts) != 4:
        raise ValueError(f"unexpected frame_dir layout: {frame_dir!r}")
    dataset, cls, rep, channel = parts
    dataset = _SUFFIX.sub("", dataset)
    return f"{dataset}/{cls}", f"{dataset}/{cls}/{rep}", channel


def resample(x, T=16):
    """Linear resampling to a fixed length, matching the ALEX-GYM-1 pipeline."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == T:
        return x
    old, new = np.linspace(0, 1, len(x)), np.linspace(0, 1, T)
    out = np.empty((T, x.shape[1], x.shape[2]), np.float32)
    for j in range(x.shape[1]):
        for c in range(x.shape[2]):
            out[:, j, c] = np.interp(new, old, x[:, j, c])
    return out


def fill_missing_frames(seq):
    """Interpolate frames whose entire pose is absent.

    AlphaPose writes all-zero coordinates for a failed detection, the same
    convention ALEX-GYM-1 uses, so the same whole-frame interpolation applies.
    Partially missing joints are left alone: the release carries no confidence
    channel that would distinguish a true zero from a dropped joint.
    """
    x = np.asarray(seq, dtype=np.float32).copy()
    valid = np.abs(x).sum(axis=(1, 2)) > 1e-8
    if not valid.any():
        return x, valid
    if not valid.all():
        t = np.arange(len(x))
        tv = t[valid]
        for j in range(x.shape[1]):
            for c in range(x.shape[2]):
                x[:, j, c] = np.interp(t, tv, x[valid, j, c])
    return x, valid


def normalize_pose(x):
    """Pelvis-centre and divide by a robust body scale (COCO-17 joints)."""
    pelvis = (x[:, L_HIP, :] + x[:, R_HIP, :]) / 2
    x = x - pelvis[:, None, :]
    shoulder = np.linalg.norm(x[:, L_SHOULDER] - x[:, R_SHOULDER], axis=-1)
    hip = np.linalg.norm(x[:, L_HIP] - x[:, R_HIP], axis=-1)
    sizes = np.r_[shoulder[shoulder > 1e-5], hip[hip > 1e-5]]
    scale = float(np.median(sizes)) if len(sizes) else 1.0
    return x / max(scale, 1e-3)


def load_cpr(root, T=16, keypoints_file="Keypoints/all_errors_keypoints.pkl"):
    """Return ``(X, Y, compositions, groups, meta)`` for every repetition.

    ``X`` is ``(N, T, n_channels * 17 * 2)``, ``Y`` is ``(N, 13)`` binary error
    bits, ``compositions`` are the bit strings the split logic keys on, and
    ``groups`` are composition-pure folder identifiers.
    """
    root = Path(root)
    cache = root / f"cpr_T{T}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["preprocess_version"].item()) == PREPROCESS_VERSION:
            return (z["X"], z["Y"], z["co"], z["g"], z["meta"].item())

    with open(root / keypoints_file, "rb") as fh:
        records = pickle.load(fh)

    # Gather the channels of each take, and the labels they agree on.
    takes: dict[str, dict] = {}
    for rec in records:
        group, key, channel = _parse_frame_dir(rec["frame_dir"])
        labels = frozenset(int(v) for v in rec["label"] if int(v) > 0)
        take = takes.setdefault(key, {"group": group, "labels": labels, "channels": {}})
        if take["labels"] != labels:
            raise ValueError(f"{key}: channels disagree on labels "
                             f"({sorted(take['labels'])} vs {sorted(labels)})")
        take["channels"][channel] = np.asarray(rec["keypoint"], dtype=np.float32)

    channel_names = sorted({c for t in takes.values() for c in t["channels"]})
    X, Y, compositions, groups, keys, dropped = [], [], [], [], [], []
    for key in sorted(takes):
        take = takes[key]
        if set(take["channels"]) != set(channel_names):
            dropped.append({"take": key, "channels": sorted(take["channels"])})
            continue
        views = []
        incomplete = False
        for channel in channel_names:
            seq = take["channels"][channel]
            # AlphaPose arrays are (persons, frames, joints, coords); this
            # release is single-person, so the first track is the performer.
            if seq.ndim == 4:
                seq = seq[0]
            seq, valid = fill_missing_frames(seq)
            if not valid.any():
                incomplete = True
                break
            views.append(normalize_pose(resample(seq, T)).reshape(T, -1))
        if incomplete:
            dropped.append({"take": key, "reason": "channel with no valid frame"})
            continue
        X.append(np.concatenate(views, axis=-1))
        Y.append([1 if (c + 1) in take["labels"] else 0 for c in range(NCRIT)])
        groups.append(take["group"])
        keys.append(key)

    X = np.stack(X).astype(np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    compositions = np.array(["".join(map(str, row.astype(int))) for row in Y])
    groups = np.asarray(groups)
    meta = {"preprocess_version": PREPROCESS_VERSION, "criteria": CPR_ERRORS,
            "channels": channel_names, "n_takes": len(keys), "dropped": dropped,
            "keys": keys}

    np.savez_compressed(cache, X=X, Y=Y, co=compositions, g=groups,
                        preprocess_version=PREPROCESS_VERSION,
                        meta=np.array(meta, dtype=object))
    return X, Y, compositions, groups, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/cpr_coach")
    ap.add_argument("--frames", type=int, default=16)
    a = ap.parse_args()
    X, Y, co, g, meta = load_cpr(a.data, T=a.frames)
    print(f"X={X.shape}  Y={Y.shape}  compositions={len(set(co))}  groups={len(set(g))}")
    print(f"channels={meta['channels']}  dropped={len(meta['dropped'])}")
