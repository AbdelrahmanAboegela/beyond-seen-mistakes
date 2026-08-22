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
    assert produced["primary"] == published["primary"], "the headline result did not reproduce"
    assert produced["secondary"] == published["secondary"]

    # The permutation null means carry float noise across library versions; the
    # reported associations themselves must still match exactly.
    for key, node in produced["diagnostic"].items():
        ref = published["diagnostic"][key]
        for field, value in node.items():
            if field == "null_mean":
                assert abs(value - ref[field]) < 1e-12, key
            else:
                assert value == ref[field], f"{key}.{field}"


def test_release_verification_passes():
    proc = subprocess.run([sys.executable, "scripts/verify_release.py"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
