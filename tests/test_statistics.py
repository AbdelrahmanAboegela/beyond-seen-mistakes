"""Unit tests for statistical diagnostic and bootstrap functions."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

# `boot` used to live in analyze_research_questions; that module is now a thin
# shim for the matched-composition pipeline. The identical percentile bootstrap
# still backs the aggregation path, so the test targets it there.
from aggregate_runs import bootstrap_mean as boot
from analyze_context_evidence import flip_bit, correct_confidence, blocked_permutation


def test_boot_confidence_interval():
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=1.0, size=100)
    ci = boot(data, n=1000, seed=42)
    assert len(ci) == 2
    assert ci[0] < 10.0 < ci[1]
    assert ci[0] < ci[1]


def test_flip_bit():
    assert flip_bit("001", 0) == "101"
    assert flip_bit("001", 2) == "000"
    assert flip_bit("1111", 1) == "1011"


def test_correct_confidence():
    p = np.array([0.9, 0.2, 0.8])
    y = np.array([1, 0, 0])
    conf = correct_confidence(p, y)
    np.testing.assert_allclose(conf, [0.9, 0.8, 0.2])


def test_blocked_permutation():
    df = pd.DataFrame({
        'exercise': ['squat'] * 8,
        'target': ['00'] * 4 + ['11'] * 4,
        'context_pressure': [1, 2, 3, 4, 10, 20, 30, 40],
        'baseline_accuracy': [0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.75, 0.65]
    })
    res = blocked_permutation(df, 'context_pressure', 'baseline_accuracy', n=1000, seed=42)
    assert 'mean_within_target_rho' in res
    assert 'p_one_sided' in res
    assert res['mean_within_target_rho'] < -0.9
    assert res['n_informative_targets'] == 2


if __name__ == "__main__":
    pytest.main([__file__])
