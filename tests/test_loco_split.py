"""Unit tests for LOCO (Leave-One-Composition-Out) group-disjoint splitting."""

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
from loco_split import make_loco_split


def test_make_loco_split_disjointness():
    np.random.seed(42)
    Y = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]]), (6, 1))
    compositions = np.array(["00"] * 15 + ["11"] * 15)
    groups = np.array([f"g{i//5 + 1}" for i in range(30)])

    train, val, test, audit = make_loco_split(
        Y, compositions, groups, target="11", seed=42, min_train_state=1, min_val_state=1
    )

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

    assert audit["group_overlap"]["train_val"] == 0
    assert audit["group_overlap"]["train_test"] == 0
    assert audit["group_overlap"]["val_test"] == 0

    assert (compositions[test] == "11").all()
    assert audit["target"] == "11"
    assert audit["n_test"] == len(test)


def test_make_loco_split_insufficient_groups():
    Y = np.zeros((10, 2))
    compositions = np.array(["11"] * 8 + ["00"] * 2)
    groups = np.array(["g1"] * 8 + ["g2"] * 2)

    with pytest.raises(ValueError, match="Fewer than two non-target recording groups remain"):
        make_loco_split(Y, compositions, groups, target="11", seed=42)


def test_make_loco_split_unsupported_state():
    Y = np.zeros((15, 2))
    compositions = np.array(["00"] * 10 + ["11"] * 5)
    groups = np.array(["g1"] * 5 + ["g2"] * 5 + ["g3"] * 5)

    with pytest.raises(ValueError, match="No recording-disjoint split satisfies"):
        make_loco_split(Y, compositions, groups, target="11", seed=42, min_train_state=1)


if __name__ == "__main__":
    pytest.main([__file__])
