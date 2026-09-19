"""Motion statistics that fixed-length resampling destroys.

Both loaders resample every repetition to a fixed frame count.  That maps a
two-second and a six-second repetition onto the same tensor, which normalises
away exactly the quantities several annotation criteria are *defined by*:

* CPR-Coach: "Slow Frequency", "Insufficient Pressing", "Excessive Pressing",
  "Jump Pressing".
* ALEX-GYM-1: "Slow reverse movement", "Bend knees slowly", "Bend Hips and
  Knees Simultaniously".

Measured on the CPR-Coach matched sweep, the four rate criteria score 14.27
points *below* their own trivial baselines while posture criteria score 7.50
points above -- the models do worse than guessing on precisely the criteria
whose evidence resampling removes.

These three statistics are computed on the original sequence, before
resampling, and are passed alongside the resampled pose so the information is
available again.  They use only input coordinates, never labels.
"""
from __future__ import annotations

import numpy as np

N_RATE_FEATURES = 3


def rate_features(seq) -> np.ndarray:
    """Return ``[log duration, mean speed, cadence]`` for one original sequence.

    ``seq`` is ``(frames, joints, coords)`` *before* resampling.

    * duration -- how long the repetition took, which a fixed frame count erases.
    * speed -- mean absolute displacement per original frame.  Resampling
      rescales the time axis, so post-resampling velocity is duration-invariant
      and cannot express "slow".
    * cadence -- dominant non-DC frequency of the highest-variance coordinate,
      in cycles per frame.  This is what "Slow Frequency" and the pressing-rate
      criteria are about.
    """
    x = np.asarray(seq, dtype=np.float64)
    if x.ndim != 3 or len(x) == 0:
        return np.zeros(N_RATE_FEATURES)
    n = len(x)
    duration = float(np.log1p(n))
    speed = float(np.abs(np.diff(x, axis=0)).mean()) if n > 1 else 0.0

    cadence = 0.0
    if n >= 4:
        flat = x.reshape(n, -1)
        variance = flat.var(axis=0)
        if variance.max() > 1e-12:
            signal = flat[:, int(variance.argmax())]
            signal = signal - signal.mean()
            spectrum = np.abs(np.fft.rfft(signal))
            if len(spectrum) > 1:
                # bin 0 is the (already removed) mean; the dominant remaining
                # bin over the original frame axis is the repetition rate.
                peak = int(np.argmax(spectrum[1:])) + 1
                cadence = float(peak / n)
    return np.array([duration, speed, cadence], dtype=np.float64)


def standardize(features: np.ndarray) -> np.ndarray:
    """Z-score each column across the dataset.

    The three statistics live on wildly different scales (log-frames, pixels
    per frame, cycles per frame). Left raw, the largest would dominate the
    first layer purely through magnitude.
    """
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def append_constant_channels(X: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Broadcast per-repetition statistics across time and concatenate.

    ``X`` is ``(N, T, D)`` and ``features`` is ``(N, F)``; the result is
    ``(N, T, D + F)``. The models pool over time, so a constant channel reaches
    them intact rather than being averaged away.
    """
    X = np.asarray(X, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if len(X) != len(features):
        raise ValueError(f"X has {len(X)} rows but features has {len(features)}")
    tiled = np.repeat(features[:, None, :], X.shape[1], axis=1)
    return np.concatenate([X, tiled], axis=-1).astype(np.float32)
