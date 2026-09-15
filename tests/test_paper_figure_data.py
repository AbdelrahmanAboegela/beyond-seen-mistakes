"""The paper's Figure 2 is drawn natively in LaTeX, so its numbers live in
``paper/main.tex`` rather than being read from ``results/`` at build time.

That buys matching fonts and a vector figure with no external asset, at the
cost of a silent-drift hazard: nothing would notice if the run records were
regenerated and the hardcoded coordinates were left behind. These tests close
that hole by recomputing every plotted point from the result CSVs.
"""
import re
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MATCHED = ROOT / "results/matched_composition_v3_optimized"
TEX = (ROOT / "paper/main.tex").read_text(encoding="utf-8")

# The coordinates are written with four decimals, so a point may sit half a unit
# in the last place from the recomputed value and still be the same number.
TOL = 5e-5


def _per_target():
    """Per-target present/absent exact match and residual mismatch, as plotted."""
    df = pd.read_csv(MATCHED / "target_mean_metrics.csv", dtype={"target": str})
    pairs = (df[df.metric == "exact_match"]
             .groupby(["exercise", "target", "condition"]).value.mean()
             .unstack().mul(100).reset_index())
    balance = pd.read_csv(MATCHED / "balance_audit.csv", dtype={"target": str})
    pairs = pairs.merge(
        balance.groupby(["exercise", "target"], as_index=False).optimized_max.mean(),
        on=["exercise", "target"])
    pairs["gap"] = pairs.seen - pairs.unseen
    return pairs


def _close(produced, published):
    """Match two unordered point sets within TOL, pairing each point greedily."""
    remaining = list(published)
    for point in produced:
        for i, other in enumerate(remaining):
            if all(abs(a - b) <= TOL for a, b in zip(point, other)):
                remaining.pop(i)
                break
        else:
            return False, point
    return not remaining, remaining


def test_panel_a_slopes_match_the_run_records():
    """Every (present, absent) pair drawn in panel (a) is in the CSVs."""
    drawn = [(float(seen), float(absent)) for _, seen, absent in re.findall(
        r"\\addplot\[(sq|dl)\] coordinates \{\(0,([\d.]+)\) \(1,([\d.]+)\)\};", TEX)]
    assert len(drawn) == 12, f"expected 12 slope lines in main.tex, found {len(drawn)}"

    pairs = _per_target()
    expected = list(zip(pairs.seen, pairs.unseen))
    ok, leftover = _close(drawn, expected)
    assert ok, f"panel (a) coordinates drifted from the result records: {leftover}"


def test_panel_b_points_match_the_run_records():
    """Every (residual mismatch, gap) point drawn in panel (b) is in the CSVs."""
    blocks = re.findall(
        r"\\addplot\[only marks[^\]]*color=(sqc|dlc)\] coordinates\s*\{([^}]*)\}", TEX)
    assert len(blocks) == 2, "expected one scatter series per exercise"

    drawn = {"sqc": "squat", "dlc": "deadlift"}
    pairs = _per_target()
    for color, body in blocks:
        points = [(float(x), float(y))
                  for x, y in re.findall(r"\(([\d.]+),([\d.]+)\)", body)]
        group = pairs[pairs.exercise == drawn[color]]
        ok, leftover = _close(points, list(zip(group.optimized_max, group.gap)))
        assert ok, f"panel (b) {drawn[color]} points drifted: {leftover}"


def test_reported_rho_matches_and_preserves_the_tie():
    """The rho printed in the panel title is the tie-correct value.

    Two squat targets have an exactly equal gap (25/3 points). Summing those
    means in a different order leaves a ~1e-15 discrepancy that unties them and
    moves rho from -.26 to -.27, so the rounding here is load-bearing, not
    cosmetic.
    """
    printed = re.search(r"title=\{\(b\) \$\\rho=(-?\.\d+)\$\}", TEX)
    assert printed, "could not find the rho reported in the panel (b) title"

    pairs = _per_target()
    rho = spearmanr(pairs.optimized_max.round(9), pairs.gap.round(9)).statistic
    assert f"{rho:.2f}".replace("0.", ".") == printed.group(1)

    # The tie is real, and the guard against float noise breaking it must work.
    gaps = sorted(round(g, 9) for g in pairs.gap)
    assert any(a == b for a, b in zip(gaps, gaps[1:])), "expected a tied gap"
    untied = spearmanr(pairs.optimized_max, pairs.gap).statistic
    assert untied != pytest.approx(rho), "rounding no longer changes the ranking"


def test_results_text_quotes_the_same_rho():
    """The prose and the figure must not report different values."""
    body = re.search(r"rank association is negative \(\$\\rho=(-?\.\d+)\$\)", TEX)
    title = re.search(r"title=\{\(b\) \$\\rho=(-?\.\d+)\$\}", TEX)
    assert body and title and body.group(1) == title.group(1)
