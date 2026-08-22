"""Test-only fixtures.

Synthetic arrays live here and nowhere else. The production loader must fail
when ALEX-GYM is absent, so nothing in ``src`` may import this module.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alexgym_data import CRITERIA  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path):
    """Write a minimal, row-aligned ALEX-GYM trio that load_exercise accepts."""
    def build(exercise="squat", n_rows=12, n_frames=8, n_poses=None, groups=None):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({c: rng.integers(0, 2, n_rows) for c in CRITERIA[exercise]})
        df["Num Video Frontal"] = groups if groups is not None else [f"g{i // 3}" for i in range(n_rows)]
        df.to_excel(tmp_path / f"{exercise}.xlsx", index=False)
        pose = (rng.normal(size=(n_frames, 33, 3)) * 0.1).tolist()
        count = n_rows if n_poses is None else n_poses
        import json
        for name in (f"front_pose_{exercise}.json", f"lat_pose_{exercise}.json"):
            (tmp_path / name).write_text(json.dumps([pose] * count))
        return tmp_path
    return build
