"""Synthetic dataset generator for Beyond Seen Mistakes.

Provides utilities to generate realistic synthetic 3D pose data, criteria matrices,
and cached/raw file structures matching ALEX-GYM-1 schemas for testing and verification
without requiring external dataset downloads.
"""

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import threading

import numpy as np
import pandas as pd
try:
    from .alexgym_data import CRITERIA, PREPROCESS_VERSION
except ImportError:
    from alexgym_data import CRITERIA, PREPROCESS_VERSION


def generate_synthetic_pose(T_raw=24, num_joints=33, missing_frames=0, seed=None):
    """Generate a single raw 3D pose sequence (T_raw, 33, 3)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, T_raw)[:, None, None]
    base_joints = rng.normal(size=(1, num_joints, 3)) * 0.1
    # Simple harmonic motion for joints over time
    pose = base_joints + np.sin(t) * 0.2

    # Set realistic pelvis and shoulder joints to ensure valid normalization
    # Joints 11, 12 (shoulders), 23, 24 (pelvis/hips)
    pose[:, 11, :] = np.array([-0.2, 0.5, 0.0])
    pose[:, 12, :] = np.array([0.2, 0.5, 0.0])
    pose[:, 23, :] = np.array([-0.15, 0.0, 0.0])
    pose[:, 24, :] = np.array([0.15, 0.0, 0.0])

    if missing_frames > 0:
        missing_indices = rng.choice(T_raw, size=min(missing_frames, T_raw - 1), replace=False)
        pose[missing_indices] = 0.0

    return pose.tolist()


def generate_synthetic_exercise_dataset(exercise="squat", n_samples=30, T=16, seed=42):
    """Generate synthetic array features and metadata matching load_exercise return signature."""
    rng = np.random.default_rng(seed)
    n_criteria = len(CRITERIA[exercise])

    # Generate 198-D preprocessed features (2 views * 16 frames * 33 joints * 3 coords)
    X = rng.normal(size=(n_samples, T, 198)).astype(np.float32)
    # Generate binary label matrix
    Y = rng.integers(0, 2, size=(n_samples, n_criteria)).astype(np.float32)
    # Composition string identifiers
    co = np.array(["".join(map(str, row.astype(int))) for row in Y])
    # Paired recording groups (e.g. 5 groups with 6 reps each)
    groups = np.repeat([f"rec_group_{i}" for i in range(5)], n_samples // 5)
    if len(groups) < n_samples:
        groups = np.pad(groups, (0, n_samples - len(groups)), mode="edge")

    # Mock DataFrame
    df = pd.DataFrame(Y, columns=CRITERIA[exercise])
    df["Num Video Frontal"] = groups
    df["_front_valid_fraction"] = 1.0
    df["_lateral_valid_fraction"] = 1.0
    df["_front_missing_frames"] = 0
    df["_lateral_missing_frames"] = 0
    df.attrs["preprocess_version"] = PREPROCESS_VERSION
    df.attrs["excluded_no_valid_view"] = []
    df.attrs["is_synthetic"] = True

    return X, Y, co, groups, df


_CREATE_LOCK = threading.Lock()


def _publish(tmp_path, final_path):
    """Move a staged file into place, tolerating a concurrent winner.

    On Windows ``os.replace`` raises ``PermissionError`` when the destination is
    already open elsewhere, which happens whenever a parallel sweep worker is
    mid-read.  Generation is seeded and deterministic, so if the destination is
    already there another worker produced byte-identical content and this copy
    can simply be dropped.
    """
    try:
        os.replace(tmp_path, final_path)
    except OSError:
        if not Path(final_path).exists():
            raise


def create_synthetic_data_dir(target_dir, exercise="squat", n_samples=30, seed=42):
    """Write synthetic cached .npz and .pkl files into target_dir for load_exercise.

    Both files are staged under a process-unique name and then atomically moved
    into place, and the ``.npz`` lands last.  ``load_exercise`` gates on the
    ``.npz`` existing, so a concurrent reader either sees no cache at all or a
    complete pair.  Parallel sweep workers otherwise race on first use and one
    of them reads a half-written archive.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / f"{exercise}_T16.npz"
    df_path = target_dir / f"{exercise}_df.pkl"

    # Threads in one sweep process contend on the very first load; serializing
    # them means only the first does any work and the rest see finished files.
    with _CREATE_LOCK:
        if cache_path.exists() and df_path.exists():
            return target_dir

        X, Y, co, g, df = generate_synthetic_exercise_dataset(
            exercise=exercise, n_samples=n_samples, seed=seed)

        # The staged archive must itself end in .npz: np.savez silently appends the
        # extension otherwise, and the rename would then target a file that does not exist.
        stamp = f".{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_cache = target_dir / f"{exercise}_T16{stamp}.npz"
        tmp_df = target_dir / f"{exercise}_df{stamp}.pkl"
        try:
            np.savez(
                tmp_cache,
                X=X,
                Y=Y,
                co=co,
                g=g,
                preprocess_version=PREPROCESS_VERSION,
                is_synthetic=True,
            )
            df.to_pickle(tmp_df)
            _publish(tmp_df, df_path)
            _publish(tmp_cache, cache_path)
        finally:
            for leftover in (tmp_cache, tmp_df):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:  # another worker still has it mapped; harmless
                    pass

    return target_dir
