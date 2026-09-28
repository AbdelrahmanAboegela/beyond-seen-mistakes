"""Guards for the rate statistics that repair fixed-length resampling.

The whole point of these features is that they distinguish sequences which
resampling makes identical. The first test is therefore the load-bearing one:
a slow repetition and a fast one that resample to the same tensor must not
produce the same features.
"""
import numpy as np
import pytest

from temporal_features import (N_RATE_FEATURES, append_constant_channels,
                               rate_features, standardize)


def _oscillation(n_frames, cycles, joints=4):
    """A sequence with a known number of compression cycles."""
    t = np.linspace(0, 1, n_frames)
    seq = np.zeros((n_frames, joints, 2))
    seq[:, 0, 1] = np.sin(2 * np.pi * cycles * t)
    seq[:, 1, 0] = np.linspace(0, 1, n_frames)
    return seq


def test_resampling_collapses_what_these_features_preserve():
    """The motivating case: same shape after resampling, different rates.

    A 60-frame and a 300-frame repetition of the same movement resample to an
    identical tensor. If the features did not separate them, there would be no
    point computing them.
    """
    from cpr_data import resample
    slow, fast = _oscillation(300, 5), _oscillation(60, 5)
    # Interpolating at different sampling densities leaves a small residue, so
    # these are not bit-identical -- but on a unit-amplitude signal they agree
    # to within a few percent, i.e. the model receives substantively the same
    # input for a repetition that took five times as long.
    resampled_gap = np.abs(resample(slow, 16) - resample(fast, 16)).max()
    assert resampled_gap < 0.05

    f_slow, f_fast = rate_features(slow), rate_features(fast)
    assert f_slow[0] != pytest.approx(f_fast[0])      # duration differs
    assert not np.allclose(f_slow, f_fast)
    # the separation the features restore dwarfs the residue resampling leaves
    assert abs(f_slow[0] - f_fast[0]) > 20 * resampled_gap


def test_cadence_tracks_the_number_of_cycles():
    """Twice the compressions in the same window is twice the cadence."""
    few = rate_features(_oscillation(200, 4))[2]
    many = rate_features(_oscillation(200, 8))[2]
    assert many > few
    assert many == pytest.approx(2 * few, rel=0.25)


def test_cadence_is_measured_per_original_frame():
    """Same cycles over a longer recording is a slower cadence."""
    quick = rate_features(_oscillation(100, 5))[2]
    drawn_out = rate_features(_oscillation(400, 5))[2]
    assert drawn_out < quick


def test_speed_separates_a_slow_movement_from_a_fast_one():
    fast = rate_features(_oscillation(60, 5))[1]
    slow = rate_features(_oscillation(300, 5))[1]
    assert fast > slow


def test_degenerate_sequences_do_not_raise():
    assert rate_features(np.zeros((0, 4, 2))).shape == (N_RATE_FEATURES,)
    assert rate_features(np.zeros((1, 4, 2))).shape == (N_RATE_FEATURES,)
    assert np.isfinite(rate_features(np.zeros((50, 4, 2)))).all()   # no motion


def test_standardize_centres_each_column_independently():
    raw = np.array([[5.0, 1000.0, 0.01], [6.0, 2000.0, 0.02], [7.0, 3000.0, 0.03]])
    out = standardize(raw)
    assert out.mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-9)
    assert out.std(axis=0) == pytest.approx(np.ones(3), abs=1e-6)


def test_standardize_survives_a_constant_column():
    out = standardize(np.array([[1.0, 2.0], [1.0, 5.0]]))
    assert np.isfinite(out).all()


def test_channels_are_appended_without_disturbing_the_pose():
    X = np.random.default_rng(0).normal(size=(6, 16, 20)).astype(np.float32)
    feats = np.random.default_rng(1).normal(size=(6, N_RATE_FEATURES))
    out = append_constant_channels(X, feats)
    assert out.shape == (6, 16, 20 + N_RATE_FEATURES)
    assert out[:, :, :20] == pytest.approx(X)
    # the appended value is constant along time, so pooling cannot erase it
    assert out[:, :, 20:].std(axis=1) == pytest.approx(np.zeros((6, N_RATE_FEATURES)), abs=1e-6)


def test_row_count_mismatch_is_rejected():
    with pytest.raises(ValueError):
        append_constant_channels(np.zeros((3, 16, 5)), np.zeros((2, N_RATE_FEATURES)))


def test_rate_scaler_is_fitted_on_training_rows_only():
    """The scaler must not see held-out rows.

    Fitting over the whole dataset moved a standardised channel by up to 0.30
    of its own standard deviation on CPR-Coach, so this is a real leak rather
    than a formality.
    """
    import numpy as np
    from matched_composition import standardize_rate_block
    rng = np.random.default_rng(0)
    X = np.zeros((10, 4, 6), dtype=np.float32)
    raw = rng.normal(size=(10, 3))
    raw[8:] += 50.0                                   # held-out rows are extreme
    X[:, :, 3:] = raw[:, None, :]
    fit_rows = np.arange(8)
    out = standardize_rate_block(X, 3, fit_rows)
    # the fitted rows are centred; the held-out rows are not, and must not be
    assert out[fit_rows][:, 0, 3:].mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-5)
    assert abs(out[8:][:, 0, 3:].mean()) > 5
    assert out[:, :, :3] == pytest.approx(X[:, :, :3])   # pose block untouched


def test_rate_scaler_is_identical_across_the_two_conditions():
    """Both conditions must receive the same transform, or the match breaks."""
    import numpy as np
    from matched_composition import standardize_rate_block
    X = np.zeros((6, 2, 5), dtype=np.float32)
    X[:, :, 2:] = np.arange(18, dtype=np.float32).reshape(6, 1, 3)
    fit = np.array([0, 1, 2, 3])
    a = standardize_rate_block(X, 3, fit)
    b = standardize_rate_block(X, 3, fit)
    assert a == pytest.approx(b)
