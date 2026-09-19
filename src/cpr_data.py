"""Loader for the CPR-Coach composite-error dataset.

CPR-Coach is the closest published analogue to this study's composition
protocol: it annotates 13 error types and releases recordings in which two,
three or four of them co-occur.  That makes it the natural external check on
the matched present/absent result, which ALEX-GYM-1 alone cannot supply.

This module mirrors ``alexgym_data.load_exercise`` closely enough that
``loco_split.make_loco_split`` and the matched-composition machinery run
against it unchanged: both return a float32 ``X``, a binary criterion matrix
``Y``, per-repetition composition strings, and recording-group identifiers.

Four facts about the release drive the implementation:

* Each take is recorded on four synchronised channels (``ch0``-``ch3``).  They
  are one repetition, not four, so the channels are concatenated into a single
  feature vector exactly as ALEX-GYM-1 concatenates its frontal and lateral
  views.  Treating them as separate rows would put the same performance in
  train and test.
* Poses are AlphaPose **Halpe-136 truncated to 68 joints**: 26 body joints
  followed by 42 hand joints.  The hand joints are not usable -- measured mean
  confidence is 0.057 against 0.606 for the body -- so only the 26 body joints
  are kept.  They would otherwise be 62% of the input width and carry noise.
  The first 17 body joints follow the COCO ordering, so the normalisation is
  the same pelvis-centre/body-scale recipe ALEX-GYM-1 uses.
* ``label`` is a bare int in the single-error files and a list in the
  composite files.  Both spellings mean the same thing and are normalised.
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

from temporal_features import (N_RATE_FEATURES, append_constant_channels,
                               rate_features, standardize)

PREPROCESS_VERSION = "cpr-coach-v1-halpe26-4ch"

# Label 0 is the correct action; 1..13 are the error criteria, in ActionList order.
CPR_ERRORS = [
    "Overlap Hands", "Clenching Hands", "Single Hand", "Bending Arms",
    "Tilting Arms", "Jump Pressing", "Squatting", "Standing", "Wrong Position",
    "Insufficient Pressing", "Slow Frequency", "Excessive Pressing",
    "Random Position Pressing",
]
NCRIT = len(CPR_ERRORS)

# Halpe body joints; the trailing 42 hand joints are discarded (see module docstring).
N_BODY_JOINTS = 26
# Indices 0..16 follow COCO, so these mean the same thing as in alexgym_data.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# The composite pickles carry the multi-error takes; the train/test pickles carry
# the single-error and correct takes.  double_errors_keypoints.pkl is a subset of
# all_errors_keypoints.pkl and is deliberately not read.
DEFAULT_KEYPOINT_FILES = (
    "Keypoints/all_errors_keypoints.pkl",
    "Keypoints/train_keypoints.pkl",
    "Keypoints/test_keypoints.pkl",
)

# Some frame_dir values carry a _BIG suffix that the annotation lists omit.
_SUFFIX = re.compile(r"_BIG$")


def _parse_frame_dir(frame_dir: str):
    """``[dataset/]class/r##/ch#`` -> (group, repetition_key, channel).

    The release uses two layouts: the main recordings spell
    ``dataset/class/r##/ch#``, while the supplementary ``Sup*`` folders omit
    the dataset segment.  Both identify a composition-pure class folder, which
    is the recording group.  The dataset segment is also stripped of the
    ``_BIG`` suffix some paths carry, so one session does not fork into two
    groups and break train/test group-disjointness.
    """
    parts = frame_dir.strip("/").split("/")
    if len(parts) == 4:
        dataset, cls, rep, channel = parts
        group = f"{_SUFFIX.sub('', dataset)}/{cls}"
    elif len(parts) == 3:
        cls, rep, channel = parts
        group = _SUFFIX.sub("", cls)
    else:
        raise ValueError(f"unexpected frame_dir layout: {frame_dir!r}")
    return group, f"{group}/{rep}", channel


def labels_of(record) -> frozenset:
    """Error labels of a record, as a set, dropping the 'correct' label 0.

    The single-error files store ``label`` as an int and the composite files
    store a list.  Iterating the int spelling would raise; indexing the list
    spelling would silently keep only the first error.
    """
    raw = record["label"]
    values = [raw] if np.isscalar(raw) else list(raw)
    return frozenset(int(v) for v in values if int(v) > 0)


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
    Partially missing joints are left alone: the release carries a confidence
    channel, but ALEX-GYM-1 does not, and imputing here would make the two
    datasets incomparable on the one axis the paper is testing.
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
    """Pelvis-centre and divide by a robust body scale (COCO-compatible joints)."""
    pelvis = (x[:, L_HIP, :] + x[:, R_HIP, :]) / 2
    x = x - pelvis[:, None, :]
    shoulder = np.linalg.norm(x[:, L_SHOULDER] - x[:, R_SHOULDER], axis=-1)
    hip = np.linalg.norm(x[:, L_HIP] - x[:, R_HIP], axis=-1)
    sizes = np.r_[shoulder[shoulder > 1e-5], hip[hip > 1e-5]]
    scale = float(np.median(sizes)) if len(sizes) else 1.0
    return x / max(scale, 1e-3)


def load_cpr(root, T=16, keypoint_files=DEFAULT_KEYPOINT_FILES, with_rate=False):
    """Return ``(X, Y, compositions, groups, meta)`` for every repetition.

    ``X`` is ``(N, T, n_channels * 26 * 2)``, ``Y`` is ``(N, 13)`` binary error
    bits, ``compositions`` are the bit strings the split logic keys on, and
    ``groups`` are composition-pure folder identifiers.
    """
    root = Path(root)
    version = PREPROCESS_VERSION + ("-rate" if with_rate else "")
    cache = root / (f"cpr_T{T}_rate.npz" if with_rate else f"cpr_T{T}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["preprocess_version"].item()) == version:
            return (z["X"], z["Y"], z["co"], z["g"], z["meta"].item())

    takes: dict[str, dict] = {}
    confidence = {"body": [], "hand": []}
    empty_channels: list[dict] = []
    for name in keypoint_files:
        with open(root / name, "rb") as fh:
            records = pickle.load(fh)
        for rec in records:
            group, key, channel = _parse_frame_dir(rec["frame_dir"])
            labels = labels_of(rec)
            take = takes.setdefault(key, {"group": group, "labels": labels, "channels": {}})
            if take["labels"] != labels:
                raise ValueError(f"{key}: channels disagree on labels "
                                 f"({sorted(take['labels'])} vs {sorted(labels)})")
            seq = np.asarray(rec["keypoint"], dtype=np.float32)
            # AlphaPose arrays are (persons, frames, joints, coords); this
            # release is single-person, so the first track is the performer.
            if seq.ndim == 4:
                seq = seq[0]
            # Two takes in the release have a channel the detector failed on
            # entirely, stored as zero frames rather than zero coordinates.
            # Record it and leave the channel absent, which drops the take
            # below -- the same treatment ALEX-GYM-1 gives its one repetition
            # with no valid frontal view.
            if seq.ndim != 3 or seq.shape[0] == 0:
                empty_channels.append({"take": key, "channel": channel,
                                       "shape": list(np.shape(rec["keypoint"]))})
                continue
            if (score := rec.get("keypoint_score")) is not None:
                score = np.asarray(score)
                score = score[0] if score.ndim == 3 else score
                if score.size:
                    confidence["body"].append(float(score[:, :N_BODY_JOINTS].mean()))
                    if score.shape[1] > N_BODY_JOINTS:
                        confidence["hand"].append(float(score[:, N_BODY_JOINTS:].mean()))
            take["channels"][channel] = seq[:, :N_BODY_JOINTS, :]

    channel_names = sorted({c for t in takes.values() for c in t["channels"]})
    X, Y, groups, keys, dropped, rates = [], [], [], [], [], []
    for key in sorted(takes):
        take = takes[key]
        if set(take["channels"]) != set(channel_names):
            dropped.append({"take": key, "reason": "missing channel",
                            "channels": sorted(take["channels"])})
            continue
        views, incomplete, take_rates = [], False, []
        for channel in channel_names:
            seq, valid = fill_missing_frames(take["channels"][channel])
            if not valid.any():
                incomplete = True
                break
            # measured before resampling, which is what destroys these
            take_rates.append(rate_features(seq))
            views.append(normalize_pose(resample(seq, T)).reshape(T, -1))
        if incomplete:
            dropped.append({"take": key, "reason": "channel with no valid frame"})
            continue
        X.append(np.concatenate(views, axis=-1))
        rates.append(np.concatenate(take_rates))
        Y.append([1 if (c + 1) in take["labels"] else 0 for c in range(NCRIT)])
        groups.append(take["group"])
        keys.append(key)

    X = np.stack(X).astype(np.float32)
    if with_rate:
        X = append_constant_channels(X, standardize(np.stack(rates)))
    Y = np.asarray(Y, dtype=np.float32)
    compositions = np.array(["".join(map(str, row.astype(int))) for row in Y])
    groups = np.asarray(groups)
    meta = {"preprocess_version": version, "criteria": CPR_ERRORS,
            "rate_features_per_channel": N_RATE_FEATURES if with_rate else 0,
            "channels": channel_names, "n_takes": len(keys), "dropped": dropped,
            "empty_channels": empty_channels,
            "body_joints": N_BODY_JOINTS, "keys": keys,
            "mean_confidence": {k: (float(np.mean(v)) if v else None)
                                for k, v in confidence.items()}}

    np.savez_compressed(cache, X=X, Y=Y, co=compositions, g=groups,
                        preprocess_version=version,
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
    print(f"channels={meta['channels']}  dropped={len(meta['dropped'])}  "
          f"confidence={meta['mean_confidence']}")
