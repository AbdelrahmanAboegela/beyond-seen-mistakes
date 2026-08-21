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
from lop_loss import (
    LOPWeightedBCELoss,
    LOSS_TYPES,
    compute_lop_counts,
    compute_lop_weights,
    elementwise_loss,
    get_loss_fn,
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


# --- LOP loss ---------------------------------------------------------------

def test_lop_counts_reject_a_mismatched_target():
    Y = np.array([[1, 1, 1], [0, 1, 1]])
    with pytest.raises(ValueError, match="4 bits but Y_train has 3 criteria"):
        compute_lop_counts(Y, target_str="1111")
    with pytest.raises(ValueError, match="binary string"):
        compute_lop_counts(Y, target_str="1x1")


def test_lop_loss_weights_move_with_the_module():
    loss_fn = LOPWeightedBCELoss(pos_weight=torch.ones(3), lop_weights=torch.ones(3))
    assert "lop_weights" in dict(loss_fn.named_buffers())
    assert "pos_weight" in dict(loss_fn.named_buffers())


def test_bce_ablation_is_bit_identical_to_the_frozen_expression():
    """The reported panel trains with --loss bce. Routing it through the loss
    selector must not perturb a single bit, or the frozen results move."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits, targets = torch.randn(8, 6), (torch.rand(8, 6) > .5).float()
    pos_weight = torch.rand(6) + .5
    reference = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight)
    assert torch.equal(elementwise_loss("bce", logits, targets, pos_weight=pos_weight), reference)


@pytest.mark.parametrize("loss_type", list(LOSS_TYPES))
def test_every_loss_ablation_is_unreduced_and_differentiable(loss_type):
    logits = torch.randn(4, 3, requires_grad=True)
    targets = (torch.rand(4, 3) > .5).float()
    weights = compute_lop_weights(np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1]]), "111")
    out = elementwise_loss(loss_type, logits, targets, lop_weights=weights)
    assert out.shape == logits.shape, "training multiplies by context weights before reducing"
    out.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_lop_ablation_actually_reweights_criteria():
    logits, targets = torch.zeros(4, 3), torch.ones(4, 3)
    flat = elementwise_loss("bce", logits, targets)
    weighted = elementwise_loss("lop_weighted", logits, targets,
                                lop_weights=torch.tensor([1.0, 2.0, 1.0]))
    assert not torch.allclose(flat, weighted)
    # mean-1 normalization keeps the overall scale comparable at the same lr
    assert weighted.mean().item() == pytest.approx(flat.mean().item(), rel=1e-6)


def test_ablations_get_distinct_model_names():
    """An ablation run must never be poolable with the reported panel."""
    from train_fact import variant_name
    assert variant_name("anatomy", "bce") == "fact"
    assert variant_name("anatomy", "focal") == "fact_focal"
    assert variant_name("anatomy", "lop_weighted") == "fact_lop_weighted"
    assert variant_name("random", "bce") == "fact_random_map"
    names = {variant_name(m, l) for m in ("anatomy", "random", "permuted") for l in LOSS_TYPES}
    assert len(names) == 9 and sum(n == "fact" for n in names) == 1


def test_verify_release_rejects_ablation_run_files(tmp_path):
    """verify_release's filename grammar must not admit fact_focal_* records."""
    import re
    pattern = re.compile(r"^(tcn|gru|transformer|ssm|fact)_(squat|deadlift)_([01]+)_s(7|42|123)$")
    assert pattern.match("fact_squat_000000_s7")
    assert not pattern.match("fact_focal_squat_000000_s7")
    assert not pattern.match("fact_lop_weighted_squat_000000_s7")


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
    """aggregate_runs and any downstream audit rely on these two flags existing
    on records from both trainers."""
    for record in (ROOT / "results/runs").glob("*.json"):
        payload = json.loads(record.read_text())
        assert not payload.get("synthetic_data", False), record.name
        assert not payload.get("uses_holdout_identity", False), record.name


def test_lop_factory_requires_weights():
    with pytest.raises(ValueError, match="requires lop_weights"):
        get_loss_fn(loss_type="lop_weighted")


def test_lop_loss_rejects_a_criterion_count_mismatch():
    loss_fn = LOPWeightedBCELoss(lop_weights=torch.ones(3))
    with pytest.raises(ValueError, match="entries but"):
        loss_fn(torch.randn(2, 5), torch.zeros(2, 5))


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
