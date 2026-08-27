"""Aggregation robustness and reproduction of the published statistics."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import aggregate_runs

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run(path, **overrides):
    record = {"model": "tcn", "exercise": "squat", "target": "000000", "seed": 42,
              "val": {"exact_match": 0.5}, "test": {"exact_match": 0.25}}
    record.update(overrides)
    path.write_text(json.dumps(record))


def test_aggregation_survives_a_no_test_record(tmp_path, monkeypatch, capsys):
    """A run produced with --no-test stores test=None, which used to crash."""
    _run(tmp_path / "a.json", test=None)
    _run(tmp_path / "b.json")
    monkeypatch.setattr(sys, "argv", ["aggregate_runs", "--runs", str(tmp_path / "*.json"),
                                      "--outdir", str(tmp_path / "out")])
    aggregate_runs.main()
    assert "were skipped" in capsys.readouterr().out
    assert (tmp_path / "out" / "metrics_long.csv").is_file()


# A resampled mean summed in a different SIMD width, or a percentile taken by a
# different numpy, lands a unit or two in the last place away from the value the
# release was cut with -- roughly 1e-16 relative.  That is float arithmetic, not
# a failure to reproduce.  This bound sits four orders of magnitude above that
# noise and nine below the precision anything is reported at, so a real change
# in the numbers cannot hide underneath it.
_REL, _ABS = 1e-12, 1e-12


def _mismatches(produced, published, path):
    """Every value that differs, as ``(path, produced, published)`` triples.

    Structure, keys, list lengths and non-numeric values must match exactly;
    only floats get the tolerance.  Reporting all of them at once beats a
    whole-dict ``==`` that dumps both trees and leaves you to spot the digit.
    """
    if isinstance(published, dict):
        if not isinstance(produced, dict) or produced.keys() != published.keys():
            return [(path, produced, published)]
        return [m for k in published for m in _mismatches(produced[k], published[k], f"{path}.{k}")]
    if isinstance(published, list):
        if not isinstance(produced, list) or len(produced) != len(published):
            return [(path, produced, published)]
        return [m for i, (a, b) in enumerate(zip(produced, published))
                for m in _mismatches(a, b, f"{path}[{i}]")]
    if isinstance(published, float) and isinstance(produced, float):
        close = abs(produced - published) <= max(_ABS, _REL * abs(published))
        return [] if close else [(path, produced, published)]
    return [] if produced == published and type(produced) is type(published) \
        else [(path, produced, published)]


def test_the_comparison_tolerates_last_place_noise_but_not_real_drift():
    """Guards the guard: a tolerance nothing exercises drifts into always-true."""
    ref = {"seen_mean": 0.6883179012345679, "bootstrap95": [0.07379629629629633, 0.1879],
           "n_targets": 12, "model": "model_mean"}
    ulp = {**ref, "bootstrap95": [0.0737962962962963, 0.1879]}  # the exact CI discrepancy
    assert not _mismatches(ulp, ref, "s")

    # a change in the fourth decimal -- the precision the paper reports at
    assert _mismatches({**ref, "seen_mean": 0.6884}, ref, "s") == [
        ("s.seen_mean", 0.6884, 0.6883179012345679)]
    assert _mismatches({**ref, "n_targets": 11}, ref, "s")          # count changed
    assert _mismatches({**ref, "model": "tcn"}, ref, "s")           # different model
    assert _mismatches({k: v for k, v in ref.items() if k != "model"}, ref, "s")  # key dropped
    assert _mismatches({**ref, "bootstrap95": [0.0737962962962963]}, ref, "s")   # bound dropped


def test_published_statistics_reproduce_from_the_run_records(tmp_path):
    """rq_stats.json must be reconstructible from results/, not merely present."""
    lop, matched = tmp_path / "lop", tmp_path / "matched"
    subprocess.run([sys.executable, str(SRC / "analyze_lop_controls.py"),
                    "--outdir", str(lop)], check=True, cwd=ROOT, capture_output=True)
    subprocess.run([sys.executable, str(SRC / "analyze_matched_composition.py"),
                    "--runs", "results/matched_composition_v3_optimized/runs/*.json",
                    "--outdir", str(matched)], check=True, cwd=ROOT, capture_output=True)
    out = tmp_path / "rq_stats.json"
    subprocess.run([sys.executable, str(SRC / "compose_final_results.py"),
                    "--matched", str(matched / "stats.json"), "--lop", str(lop / "stats.json"),
                    "--out", str(out)], check=True, cwd=ROOT, capture_output=True)

    produced = json.loads(out.read_text())
    published = json.loads((ROOT / "results/rq_stats.json").read_text())
    for block in ("primary", "secondary", "diagnostic"):
        bad = _mismatches(produced[block], published[block], block)
        assert not bad, "rq_stats.json did not reproduce:\n" + "\n".join(
            f"  {path}: produced {a!r}, published {b!r}" for path, a, b in bad)


def test_release_verification_passes():
    proc = subprocess.run([sys.executable, "scripts/verify_release.py"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
