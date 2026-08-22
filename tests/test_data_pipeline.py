"""Unit and end-to-end integration tests for data loading and preprocessing pipeline."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alexgym_data import resample, fill_missing_frames, normalize_pose, load_exercise
from synthetic_data import create_synthetic_data_dir
from loco_split import make_loco_split
from train_backbones import TCN
from train_fact import FACT


def test_resample_sequence_length():
    seq = np.random.randn(30, 33, 3).astype(np.float32)
    resampled = resample(seq, T=16)
    assert resampled.shape == (16, 33, 3)


def test_fill_missing_frames_interpolation():
    seq = np.ones((10, 33, 3), dtype=np.float32)
    seq[4] = 0.0

    filled, valid = fill_missing_frames(seq)
    assert valid.sum() == 9
    assert not valid[4]
    np.testing.assert_allclose(filled[4], 1.0)


def test_normalize_pose_pelvis_centering():
    seq = np.random.randn(16, 33, 3).astype(np.float32)
    seq[:, 23, :] = np.array([2.0, 4.0, 6.0])
    seq[:, 24, :] = np.array([4.0, 6.0, 8.0])

    norm = normalize_pose(seq)
    pelvis_mid = (norm[:, 23, :] + norm[:, 24, :]) / 2.0
    np.testing.assert_allclose(pelvis_mid, 0.0, atol=1e-5)


def test_load_exercise_with_synthetic_dir(tmp_path):
    create_synthetic_data_dir(tmp_path, exercise="squat", n_samples=20)
    X, Y, co, g, df = load_exercise(tmp_path, exercise="squat", T=16)

    assert X.shape == (20, 16, 198)
    assert Y.shape[0] == 20
    assert len(co) == 20
    assert len(g) == 20
    assert len(df) == 20


def test_end_to_end_mini_training(tmp_path):
    create_synthetic_data_dir(tmp_path, exercise="squat", n_samples=30)
    X, Y, co, g, df = load_exercise(tmp_path, exercise="squat", T=16)

    target = co[0]
    train_idx, val_idx, test_idx, _ = make_loco_split(
        Y, co, g, target=target, seed=42, min_train_state=1, min_val_state=1
    )

    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(Y[train_idx]))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    tcn_model = TCN(din=198, nout=Y.shape[1])
    optimizer = torch.optim.Adam(tcn_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    tcn_model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = tcn_model(bx)
        loss = loss_fn(out, by)
        loss.backward()
        optimizer.step()

    assert loss.item() > 0.0

    fact_model = FACT(exercise="squat")
    optimizer_fact = torch.optim.Adam(fact_model.parameters(), lr=1e-3)

    fact_model.train()
    for bx, by in train_loader:
        optimizer_fact.zero_grad()
        out, _ = fact_model(bx, return_tokens=True)
        loss = loss_fn(out, by)
        loss.backward()
        optimizer_fact.step()

    assert loss.item() > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
