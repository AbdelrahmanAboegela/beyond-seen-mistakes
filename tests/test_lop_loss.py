"""Unit tests for LOP-aware loss functions and loss factory utilities."""

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

from lop_loss import (
    compute_lop_counts,
    compute_lop_weights,
    LOPWeightedBCELoss,
    FocalBCELoss,
    get_loss_fn,
)
from synthetic_data import generate_synthetic_exercise_dataset
from train_fact import FACT


def test_compute_lop_counts():
    Y_train = np.array([
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 0],
    ])
    counts = compute_lop_counts(Y_train, target_str="111")
    np.testing.assert_array_equal(counts, [1.0, 1.0, 0.0])


def test_compute_lop_weights():
    Y_train = np.array([
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 0],
    ])
    weights = compute_lop_weights(Y_train, target_str="111", alpha=0.5)
    assert weights.shape == (3,)
    np.testing.assert_allclose(weights.numpy(), [1.5, 1.5, 1.0])


def test_lop_weighted_bce_loss_gradients():
    batch_size, n_criteria = 4, 3
    logits = torch.randn(batch_size, n_criteria, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]] * batch_size)
    lop_weights = torch.tensor([1.5, 1.0, 0.5])

    loss_fn = LOPWeightedBCELoss(lop_weights=lop_weights)
    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
    assert logits.grad.shape == (batch_size, n_criteria)


def test_focal_bce_loss():
    batch_size, n_criteria = 4, 3
    logits = torch.randn(batch_size, n_criteria, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]] * batch_size)

    loss_fn = FocalBCELoss(gamma=2.0)
    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None


@pytest.mark.parametrize("loss_type", ["bce", "lop_weighted", "focal"])
def test_loss_factory(loss_type):
    logits = torch.randn(2, 3, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    lop_weights = torch.tensor([1.2, 1.0, 1.1])

    loss_fn = get_loss_fn(loss_type=loss_type, lop_weights=lop_weights)
    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.item() > 0.0


def test_end_to_end_lop_training():
    X, Y, co, g, df = generate_synthetic_exercise_dataset(exercise="squat", n_samples=20)
    target = co[0]

    lop_weights = compute_lop_weights(Y, target_str=target, alpha=0.5)
    loss_fn = LOPWeightedBCELoss(lop_weights=lop_weights)

    model = FACT(exercise="squat")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        out, _ = model(bx, return_tokens=True)
        loss = loss_fn(out, by)
        loss.backward()
        optimizer.step()

    assert loss.item() > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
