"""The production loader: it must consume real files and fail without them."""
import numpy as np
import pytest

from alexgym_data import load_exercise, resample, fill_missing_frames, normalize_pose


def test_missing_dataset_is_a_hard_error(tmp_path):
    """No fallback: a research command must stop when ALEX-GYM is absent."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_exercise(tmp_path, "squat", T=16)


def test_loader_reads_a_real_trio(fake_dataset):
    root = fake_dataset(n_rows=12)
    X, Y, co, g, df = load_exercise(root, "squat", T=16)
    assert X.shape == (12, 16, 198)          # 2 views x 33 joints x 3 coords
    assert Y.shape[0] == 12 and len(co) == 12 and len(g) == 12
    assert all(len(s) == Y.shape[1] for s in co)


def test_misaligned_workbook_and_poses_raise(fake_dataset):
    """A bare `assert` here is stripped by `python -O`, after which zip() would
    truncate and pair each label row with the wrong pose sequence."""
    root = fake_dataset(n_rows=12, n_poses=9)
    with pytest.raises(ValueError, match="lengths differ"):
        load_exercise(root, "squat", T=16)


def test_frontal_and_lateral_are_packed_in_order(fake_dataset):
    X, *_ = load_exercise(fake_dataset(n_rows=6), "squat", T=16)
    assert X.shape[-1] == 198 and np.isfinite(X).all()


def test_resample_and_interpolation_helpers():
    assert resample(np.random.randn(30, 33, 3).astype(np.float32), T=16).shape == (16, 33, 3)
    seq = np.ones((10, 33, 3), dtype=np.float32); seq[4] = 0.0
    filled, valid = fill_missing_frames(seq)
    assert valid.sum() == 9 and not valid[4]
    np.testing.assert_allclose(filled[4], 1.0)
    norm = normalize_pose(np.random.randn(16, 33, 3).astype(np.float32))
    np.testing.assert_allclose((norm[:, 23] + norm[:, 24]) / 2.0, 0.0, atol=1e-5)
