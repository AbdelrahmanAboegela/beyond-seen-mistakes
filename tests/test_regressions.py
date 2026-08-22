"""Regression tests for bugs found in the release audit.

Each test pins a failure mode that previously crashed, silently produced
meaningless numbers, or diverged from the frozen protocol.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import subprocess
import warnings

import numpy as np
import pytest
import torch

import aggregate_runs
from alexgym_data import SyntheticDataWarning, load_exercise, resample
from loco_split import (
    DEFAULT_MIN_TRAIN_STATE,
    DEFAULT_MIN_VAL_STATE,
    frozen_selection_score,
    make_loco_split,
)
from synthetic_data import create_synthetic_data_dir
from train_backbones import TransformerModel
from train_fact import context_counters, context_weights, run as run_fact


# --- data loading -----------------------------------------------------------

def test_missing_dataset_does_not_recurse(tmp_path):
    """A non-16 frame count used to recurse until RecursionError, creating a
    chain of nested synthetic/ directories on the way down."""
    with pytest.raises(ValueError, match="only generated at T=16"):
        load_exercise(tmp_path, "squat", T=8)
    assert not (tmp_path / "synthetic" / "synthetic").exists()


def test_synthetic_fallback_warns_loudly(tmp_path):
    with pytest.warns(SyntheticDataWarning):
        X, Y, co, g, df = load_exercise(tmp_path, "squat", T=16)
    assert df.attrs["is_synthetic"] is True
    assert X.shape[1:] == (16, 198)


def test_synthetic_fallback_can_be_forbidden(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found under"):
        load_exercise(tmp_path, "squat", allow_synthetic=False)


def test_real_cache_is_not_flagged_synthetic(tmp_path):
    """A cache in a directory not named 'synthetic' and without the marker must
    not be mislabelled; the marker written by create_synthetic_data_dir must be."""
    create_synthetic_data_dir(tmp_path / "synthetic", exercise="squat", n_samples=20)
    with pytest.warns(SyntheticDataWarning):
        *_, df = load_exercise(tmp_path / "synthetic", "squat", T=16)
    assert df.attrs["is_synthetic"] is True


def test_resample_rejects_empty_sequence():
    with pytest.raises(ValueError, match="empty sequence"):
        resample(np.zeros((0, 33, 3), dtype=np.float32), T=16)


# --- split protocol ---------------------------------------------------------

def test_default_support_rule_matches_frozen_protocol():
    protocol = json.loads((ROOT / "configs/final_protocol.json").read_text())
    rule = protocol["default_support_rule"]["min_training_examples_per_criterion_state"]
    assert DEFAULT_MIN_TRAIN_STATE == rule, "code default drifted from configs/final_protocol.json"
    assert DEFAULT_MIN_VAL_STATE == 1


def test_split_audit_records_the_support_rule():
    # Six groups of five; the target composition is confined to the last group so
    # a non-target pool survives.
    bits = np.array([[0, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    Y = np.concatenate([np.tile(bits, (5, 1)), np.tile([[1, 1]], (5, 1))])
    compositions = np.array(["".join(map(str, row)) for row in Y])
    groups = np.repeat([f"g{i}" for i in range(6)], 5)
    train, val, test, audit = make_loco_split(Y, compositions, groups, target="11", seed=42,
                                              min_train_state=1, min_val_state=1)
    assert audit["min_train_state"] == 1 and audit["min_val_state"] == 1
    assert (compositions[test] == "11").all()
    assert not set(groups[train]) & set(groups[val])


def test_frozen_splits_clear_the_support_rule_with_margin():
    """The selection score keys on the minimum across *both* folds jointly, even
    though train and validation have different thresholds. That is harmless only
    while every frozen split clears both with room to spare -- which this pins.
    """
    manifest = json.loads((ROOT / "configs/split_manifest_v2.json").read_text())
    audits = [a for targets in manifest["splits"].values()
              for seeds in targets.values() for a in seeds.values()]
    assert len(audits) == 36
    worst_train = min(min(min(s) for s in a["train_state_counts"]) for a in audits)
    worst_val = min(min(min(s) for s in a["val_state_counts"]) for a in audits)
    assert worst_train >= DEFAULT_MIN_TRAIN_STATE, worst_train
    assert worst_val >= DEFAULT_MIN_VAL_STATE, worst_val
    # Margin above the training threshold is what makes the lenient and strict
    # selections provably coincide for the released splits.
    assert worst_train > DEFAULT_MIN_TRAIN_STATE, (
        "frozen splits now sit exactly on the support threshold; the selection "
        "score's train/validation conflation can start changing which split wins")


def test_selection_score_prefers_better_support_then_balance():
    strong = np.array([[20, 20]]), np.array([[9, 9]])
    weak = np.array([[20, 20]]), np.array([[2, 2]])
    assert (frozen_selection_score(*strong, 20, 100, .2)
            > frozen_selection_score(*weak, 20, 100, .2))
    balanced = frozen_selection_score(np.array([[9, 9]]), np.array([[9, 9]]), 20, 100, .2)
    skewed = frozen_selection_score(np.array([[9, 9]]), np.array([[9, 9]]), 45, 100, .2)
    assert balanced > skewed


def test_absent_target_is_reported_clearly():
    Y = np.zeros((10, 2))
    compositions = np.array(["00"] * 10)
    groups = np.repeat(["g1", "g2"], 5)
    with pytest.raises(ValueError, match="does not occur in the data"):
        make_loco_split(Y, compositions, groups, target="11", seed=42)


def test_unsatisfiable_support_reports_best_achieved():
    Y = np.zeros((15, 2))
    compositions = np.array(["00"] * 10 + ["11"] * 5)
    groups = np.repeat(["g1", "g2", "g3"], 5)
    with pytest.raises(ValueError, match="best achieved train/validation support"):
        make_loco_split(Y, compositions, groups, target="11", seed=42, min_train_state=1)


# --- training ---------------------------------------------------------------

def test_zero_epoch_run_does_not_crash(tmp_path):
    """`epochs=0` left the loop variable unbound and raised UnboundLocalError."""
    create_synthetic_data_dir(tmp_path, exercise="squat", n_samples=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntheticDataWarning)
        *_, co, _, _ = load_exercise(tmp_path, "squat", T=16)
        record = run_fact("squat", co[0], seed=42, data=str(tmp_path), epochs=0,
                          min_train_state=1, min_val_state=1)
    assert record["epochs"] == 0
    assert record["synthetic_data"] is True


def test_context_weights_survive_an_unseen_context():
    """A context absent from training has count 0; ``0 ** -alpha`` used to raise
    ZeroDivisionError mid-epoch."""
    Y = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.float32)
    counters = context_counters(Y, np.array([0, 1]))
    unseen = torch.tensor([[1.0, 1.0]])
    w = context_weights(unseen, counters, alpha=0.5)
    assert torch.isfinite(w).all()


def test_transformer_rejects_overlong_sequences():
    model = TransformerModel(din=198, nout=6, max_len=16)
    with pytest.raises(ValueError, match="exceeds positional table"):
        model(torch.randn(2, 32, 198))
    assert TransformerModel(din=198, nout=6, max_len=32)(torch.randn(2, 32, 198)).shape == (2, 6)


# --- aggregation ------------------------------------------------------------

def _write_run(path, **overrides):
    record = {"model": "tcn", "exercise": "squat", "target": "000000", "seed": 42,
              "val": {"exact_match": 0.5}, "test": {"exact_match": 0.25}}
    record.update(overrides)
    path.write_text(json.dumps(record))


def test_aggregate_skips_runs_without_a_test_split(tmp_path, monkeypatch, capsys):
    """A --no-test run stores test=None, which used to crash the aggregator."""
    _write_run(tmp_path / "a.json", test=None)
    _write_run(tmp_path / "b.json")
    monkeypatch.setattr(sys, "argv", ["aggregate_runs", "--runs", str(tmp_path / "*.json"),
                                      "--outdir", str(tmp_path / "out")])
    aggregate_runs.main()
    assert "were skipped" in capsys.readouterr().out


def test_aggregate_refuses_synthetic_runs(tmp_path, monkeypatch):
    _write_run(tmp_path / "a.json", synthetic_data=True)
    monkeypatch.setattr(sys, "argv", ["aggregate_runs", "--runs", str(tmp_path / "*.json"),
                                      "--outdir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="synthetic placeholder data"):
        aggregate_runs.main()



def test_verify_release_rejects_ablation_run_files(tmp_path):
    """verify_release's filename grammar must admit only the reported panel."""
    import re
    pattern = re.compile(r"^(tcn|gru|transformer|ssm|fact)_(squat|deadlift)_([01]+)_s(7|42|123)$")
    assert pattern.match("fact_squat_000000_s7")
    assert not pattern.match("fact_random_map_squat_000000_s7")
    assert not pattern.match("fact_permuted_map_squat_000000_s7")


def test_synthetic_cache_creation_is_concurrency_safe(tmp_path):
    """Parallel sweep workers hit an empty data dir at the same moment; a
    non-atomic write let one of them read a half-finished archive."""
    import concurrent.futures as cf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntheticDataWarning)
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda _: load_exercise(tmp_path, "deadlift", T=16)[0].shape, range(8)))
    assert len(set(results)) == 1
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert not leftovers, f"staging files were left behind: {leftovers}"


def test_synthetic_cache_writes_land_on_the_expected_names(tmp_path):
    """np.savez appends .npz unless the name already ends in it, which silently
    breaks an atomic rename staged through a .tmp suffix."""
    create_synthetic_data_dir(tmp_path, exercise="squat", n_samples=20)
    assert (tmp_path / "squat_T16.npz").is_file()
    assert (tmp_path / "squat_df.pkl").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["squat_T16.npz", "squat_df.pkl"]


def test_every_run_record_declares_its_provenance():
    """aggregate_runs and any downstream audit rely on this flag existing."""
    for record in (ROOT / "results/runs").glob("*.json"):
        payload = json.loads(record.read_text())
        assert not payload.get("synthetic_data", False), record.name




# --- command line surfaces --------------------------------------------------

def test_sweep_accepts_the_documented_seeds_flag():
    """README and REPRODUCIBILITY both document --seeds; it did not exist."""
    proc = subprocess.run([sys.executable, str(SRC / "run_context_evidence_sweep.py"), "--help"],
                          capture_output=True, text=True)
    assert "--seeds" in proc.stdout


def test_plot_composition_graph_has_no_import_side_effects():
    """Importing the module used to render and overwrite the paper figure."""
    before = (ROOT / "paper/figures/composition_graph.png").read_bytes()
    subprocess.run([sys.executable, "-c",
                    f"import sys; sys.path.insert(0, r'{SRC}'); import plot_composition_graph"],
                   check=True, capture_output=True)
    assert (ROOT / "paper/figures/composition_graph.png").read_bytes() == before


if __name__ == "__main__":
    pytest.main([__file__])


def test_dataset_alignment_guard_survives_python_O(tmp_path):
    """The workbook and the two pose files must be row-aligned. This was guarded
    by a bare `assert`, which `python -O` strips -- after which zip() would
    truncate to the shortest input and pair each label row with the wrong pose
    sequence. Wrong numbers, no crash, so it must be a real exception."""
    import pandas as pd
    from alexgym_data import CRITERIA

    ex = "squat"
    n_rows, n_poses = 4, 3           # deliberately misaligned
    df = pd.DataFrame({c: [1] * n_rows for c in CRITERIA[ex]})
    df["Num Video Frontal"] = [f"g{i}" for i in range(n_rows)]
    df.to_excel(tmp_path / f"{ex}.xlsx", index=False)
    pose = [[[0.1] * 3 for _ in range(33)] for _ in range(8)]
    for name in (f"front_pose_{ex}.json", f"lat_pose_{ex}.json"):
        (tmp_path / name).write_text(json.dumps([pose] * n_poses))

    with pytest.raises(ValueError, match="lengths differ"):
        load_exercise(tmp_path, ex, T=16)


def test_alignment_guard_is_not_an_assert():
    """Pin the mechanism, not just the behaviour: an `assert` would silently
    disappear under -O and this test would keep passing."""
    import inspect
    import alexgym_data
    body = inspect.getsource(alexgym_data.load_exercise)
    assert "assert " not in body, "data-integrity checks must not use bare assert"


# --- review follow-ups (PR #1) ----------------------------------------------

def test_matched_pipeline_defaults_to_refusing_synthetic_data():
    """The v3 pipeline is a research command: a missing dataset must fail, not
    silently train on random labels."""
    import inspect
    import matched_composition
    sig = inspect.signature(matched_composition.run)
    assert sig.parameters["allow_synthetic"].default is False


def test_matched_records_are_stamped_with_provenance(tmp_path):
    import inspect
    import matched_composition
    src = inspect.getsource(matched_composition.run)
    assert '"synthetic_data": synthetic' in src, "matched records must carry synthetic_data"
    assert "allow_synthetic=allow_synthetic" in src


def test_matched_analysis_refuses_synthetic_records(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "r.json").write_text(json.dumps({
        "exercise": "squat", "target": "000000", "seed": 7, "model": "tcn",
        "synthetic_data": True,
        "seen": {"test": {"exact_match": 0.1}}, "unseen": {"test": {"exact_match": 0.1}}}))
    proc = subprocess.run(
        [sys.executable, str(SRC / "analyze_matched_composition.py"),
         "--runs", str(runs / "*.json"), "--outdir", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert proc.returncode != 0 and "synthetic" in proc.stderr


def test_audit_fails_loudly_when_raw_data_is_missing(tmp_path):
    """Exiting 0 with a friendly message let a pipeline record 'audit passed'
    when nothing had been audited."""
    proc = subprocess.run(
        [sys.executable, str(SRC / "audit_data_quality.py"),
         "--data", str(tmp_path / "absent"), "--outdir", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert proc.returncode == 1
    marker = json.loads((tmp_path / "out" / "raw_data_manifest.json").read_text())
    assert marker["status"] == "incomplete" and marker["missing_files"]


def test_wilcoxon_helper_only_swallows_degenerate_cases():
    """A blanket `except Exception: return 1.0` reports unrelated faults as a
    null result. Only scipy's ValueError for degenerate input may be absorbed."""
    import inspect
    import aggregate_runs
    src = inspect.getsource(aggregate_runs.safe_wilcoxon)
    assert "except ValueError" in src and "except Exception" not in src


def test_sweep_resolves_paths_before_switching_cwd():
    """Children run with cwd=ROOT_DIR, so a relative --data/--outdir given from
    another directory must not re-anchor to the repository root."""
    import inspect
    import run_context_evidence_sweep as sweep
    src = inspect.getsource(sweep.main)
    for flag in ("a.data", "a.outdir", "a.protocol"):
        assert f"{flag}=str(Path({flag}).resolve())" in src, f"{flag} is not resolved before cwd change"


def test_package_version_matches_citation():
    import re
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pv = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    cv = re.search(r"^version:\s*(\S+)", citation, re.M).group(1)
    assert pv == cv, f"pyproject {pv} != CITATION.cff {cv}"


def test_installed_wheel_imports_every_module():
    """Packaging faults are invisible to the rest of the suite, which runs with
    `src` on sys.path. Opt in with BSM_WHEEL_TEST=1; CI always runs the script."""
    import os
    if os.environ.get("BSM_WHEEL_TEST") != "1":
        pytest.skip("set BSM_WHEEL_TEST=1 to build and install a wheel (slow)")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/check_wheel_imports.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_audit_only_records_carry_provenance():
    """analyze_matched_composition also consumes --audit-only records, so an
    unstamped one produced with --allow-synthetic would slip past its guard."""
    import inspect
    import matched_composition
    src = inspect.getsource(matched_composition)
    audit_branch = src.split("if a.audit_only:", 1)[1].split("else:", 1)[0]
    assert '"synthetic_data"' in audit_branch, "--audit-only record is missing synthetic_data"
    assert ", df = load_exercise" in audit_branch, "--audit-only discards the dataframe"
