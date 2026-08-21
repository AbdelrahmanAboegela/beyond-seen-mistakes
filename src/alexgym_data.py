"""Data loading, pose normalization, temporal resampling, and missing frame interpolation for ALEX-GYM."""

from pathlib import Path
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd

PREPROCESS_VERSION = "alexgym-v2-interpolate-whole-frame-missing"

SYNTHETIC_SUBDIR = "synthetic"


class SyntheticDataWarning(UserWarning):
    """Raised when a loader falls back to generated placeholder poses.

    Synthetic samples carry random labels.  Any metric computed on them is
    meaningless as a research result and must never be reported.
    """

CRITERIA = {
    "squat": [
        "Feet Out 30 F",
        "Whole Feet Flat On the Floor (1) L",
        "Bend Hips and Knees Simultaniously (1) L",
        "Hips backwards (1) L",
        "Lower back neural (1) L",
        "Hips are lower than knees level (1 point) L",
    ],
    "deadlift": [
        "Balance (0.5 point) F",
        "Straight back (0.5) L",
        "Full back leg extended (0.5 point) L",
        "Slow reverse movement (0.5 point) L",
        "Support knee\xa0bend\xa015-20° (0.5 point) L",
    ],
    "lunges": [
        "Heels shoulder width apart (0.5 point) F",
        "Head looking forward (0.5 point)F",
        "Bend knees slowly 90° (0.5 point) L",
        "Swinging arms (0.5 point) F",
        "Front foot is parallel with the back foot, not on one line with the back foot (0.5 point) F",
        "Straight back (0.5 point) L",
        "Back knee is just above the floor (0.5 point) L",
    ],
}


def _read_cache(cache_path, df_path, attempts=12, pause=0.05):
    """Read a cached ``(npz, pkl)`` pair, retrying past a concurrent publish.

    Separate sweep processes can be mid-rename on these paths, and Windows
    raises a sharing violation rather than serving the old bytes.  The writer
    publishes atomically, so a brief retry always lands on a complete pair.
    """
    last = None
    for attempt in range(attempts):
        try:
            z = np.load(cache_path, allow_pickle=False)
            payload = {k: z[k] for k in z.files}
            return payload, pd.read_pickle(df_path)
        except (OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
            last = exc
            time.sleep(pause * (attempt + 1))
    raise OSError(f"Could not read the preprocessed cache at {cache_path} after "
                  f"{attempts} attempts; last error: {last!r}")


def resample(seq, T=16):
    x = np.asarray(seq, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (T, V, C), got shape {x.shape}")
    if len(x) == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if len(x) == T:
        return x
    if len(x) < 2:
        return np.repeat(x, T, axis=0)[:T]
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, T)
    o = np.empty((T, x.shape[1], x.shape[2]), np.float32)
    for j in range(x.shape[1]):
        for c in range(x.shape[2]):
            o[:, j, c] = np.interp(new, old, x[:, j, c])
    return o


def fill_missing_frames(seq):
    """Linearly interpolate frames in which the entire pose is absent."""
    x = np.asarray(seq, dtype=np.float32).copy()
    if x.ndim != 3:
        return x, np.ones(len(x), dtype=bool)
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
    if x.ndim != 3 or x.shape[1] < 25:
        return x
    pelvis = (x[:, 23, :] + x[:, 24, :]) / 2.0
    x = x - pelvis[:, None, :]
    shoulder = np.linalg.norm(x[:, 11] - x[:, 12], axis=-1)
    hip = np.linalg.norm(x[:, 23] - x[:, 24], axis=-1)
    sizes = np.r_[shoulder[shoulder > 1e-5], hip[hip > 1e-5]]
    scale = float(np.median(sizes)) if len(sizes) else 1.0
    return x / max(scale, 1e-3)


def _load_synthetic_fallback(root, exercise, T, allow_synthetic):
    """Serve generated placeholder poses, or explain why the real data is required.

    The fallback never recurses: it builds the cache under ``root/synthetic``
    and reads it directly, so a missing dataset cannot spawn an unbounded chain
    of nested ``synthetic/`` directories.
    """
    if not allow_synthetic:
        raise FileNotFoundError(
            f"ALEX-GYM raw files for '{exercise}' not found under {root}. Expected "
            f"{exercise}.xlsx, front_pose_{exercise}.json and lat_pose_{exercise}.json; "
            "see data/README.md."
        )
    if T != 16:
        raise ValueError(
            f"Synthetic fallback data is only generated at T=16, but T={T} was requested. "
            f"Provide the raw ALEX-GYM files under {root} to preprocess at another length."
        )

    root = Path(root)
    syn_dir = root if root.name == SYNTHETIC_SUBDIR else root / SYNTHETIC_SUBDIR
    if not (syn_dir / f"{exercise}_T16.npz").exists():
        try:
            from .synthetic_data import create_synthetic_data_dir
        except ImportError:
            from synthetic_data import create_synthetic_data_dir
        create_synthetic_data_dir(syn_dir, exercise=exercise)

    warnings.warn(
        f"ALEX-GYM raw files for '{exercise}' were not found under {root}; falling back to "
        f"SYNTHETIC placeholder poses with RANDOM labels from {syn_dir}. Any metric produced "
        "from this data is meaningless. Follow data/README.md, or pass allow_synthetic=False.",
        SyntheticDataWarning,
        stacklevel=3,
    )
    z, df = _read_cache(syn_dir / f"{exercise}_T16.npz", syn_dir / f"{exercise}_df.pkl")
    df.attrs["is_synthetic"] = True
    return z["X"], z["Y"], z["co"], z["g"], df


def load_exercise(root, exercise="squat", T=16, allow_synthetic=True):
    """Load one exercise as ``(X, Y, compositions, groups, dataframe)``.

    ``X`` has shape ``(n, T, 198)``: frontal joints in columns 0-98 and lateral
    joints in columns 99-197.  When the raw ALEX-GYM files are absent and
    ``allow_synthetic`` is true, generated placeholder poses with *random*
    labels are returned and a :class:`SyntheticDataWarning` is emitted; the
    returned frame carries ``df.attrs['is_synthetic'] = True``.  Pass
    ``allow_synthetic=False`` to make a missing dataset a hard error instead.
    """
    root = Path(root)
    cache = root / f"{exercise}_T16.npz"
    df_pickle = root / f"{exercise}_df.pkl"

    if T == 16 and cache.exists() and df_pickle.exists():
        z, df = _read_cache(cache, df_pickle)
        version = str(z["preprocess_version"].item()) if "preprocess_version" in z else ""
        if version == PREPROCESS_VERSION:
            # A cache written by synthetic_data carries a marker; older caches are
            # identified by the conventional directory name.
            synthetic = bool(z["is_synthetic"].item()) if "is_synthetic" in z \
                else root.name == SYNTHETIC_SUBDIR
            df.attrs["is_synthetic"] = synthetic
            if synthetic:
                warnings.warn(
                    f"Loaded SYNTHETIC placeholder poses with RANDOM labels from {cache}. "
                    "Any metric produced from this data is meaningless.",
                    SyntheticDataWarning,
                    stacklevel=2,
                )
            return z["X"], z["Y"], z["co"], z["g"], df

    excel_path = root / f"{exercise}.xlsx"
    front_path = root / f"front_pose_{exercise}.json"
    lat_path = root / f"lat_pose_{exercise}.json"

    if not (excel_path.exists() and front_path.exists() and lat_path.exists()):
        return _load_synthetic_fallback(root, exercise, T, allow_synthetic)

    df = pd.read_excel(excel_path)
    with front_path.open(encoding="utf-8") as f:
        front = json.load(f)
    with lat_path.open(encoding="utf-8") as f:
        lat = json.load(f)

    # A bare `assert` here is stripped by `python -O`, after which a length
    # mismatch would let zip() truncate silently and pair each label row with
    # the wrong pose sequence -- wrong numbers, no crash. Mirrors the explicit
    # check in audit_data_quality.py.
    if not (len(df) == len(front) == len(lat)):
        raise ValueError(
            f"{exercise}: workbook/front/lateral lengths differ: "
            f"{len(df)}, {len(front)}, {len(lat)}. The three files must be row-aligned; "
            "see data/README.md."
        )
    X = []
    quality = []
    keep = []
    excluded = []

    for row_idx, (fs, ls) in enumerate(zip(front, lat, strict=True)):
        fs, fv = fill_missing_frames(fs)
        ls, lv = fill_missing_frames(ls)
        if not fv.any() or not lv.any():
            excluded.append(
                {
                    "raw_row": int(row_idx),
                    "front_has_valid_frame": bool(fv.any()),
                    "lateral_has_valid_frame": bool(lv.any()),
                }
            )
            continue
        f = normalize_pose(resample(fs, T))
        l = normalize_pose(resample(ls, T))
        X.append(np.concatenate([f.reshape(T, -1), l.reshape(T, -1)], -1))
        quality.append((float(fv.mean()), float(lv.mean()), int((~fv).sum()), int((~lv).sum())))
        keep.append(row_idx)

    if not keep:
        raise ValueError(f"{exercise}: every repetition was excluded for having no valid view.")
    df = df.iloc[keep].copy().reset_index(drop=True)
    X = np.stack(X)
    Y = (df[CRITERIA[exercise]].fillna(0).to_numpy() <= 0).astype(np.float32)
    co = np.array(["".join(map(str, r.astype(int))) for r in Y])
    g = df["Num Video Frontal"].to_numpy()
    q = np.asarray(quality)

    df["_front_valid_fraction"] = q[:, 0]
    df["_lateral_valid_fraction"] = q[:, 1]
    df["_front_missing_frames"] = q[:, 2].astype(int)
    df["_lateral_missing_frames"] = q[:, 3].astype(int)
    df.attrs["preprocess_version"] = PREPROCESS_VERSION
    df.attrs["excluded_no_valid_view"] = excluded
    df.attrs["is_synthetic"] = False

    return X, Y, co, g, df
