"""Edge cases in the training entry points, run against the real loader."""
import numpy as np
import pytest

from alexgym_data import load_exercise
import train_fact


def _dataset(fake_dataset):
    """A fixture trio whose most common composition is a usable LOCO target."""
    root = fake_dataset(exercise="squat", n_rows=24)
    X, Y, co, g, df = load_exercise(root, "squat", T=16)
    target = max(set(co.tolist()), key=co.tolist().count)
    return root, target


def test_zero_epoch_run_does_not_crash(fake_dataset):
    """`epochs=0` never entered the loop, leaving the epoch counter unbound and
    raising UnboundLocalError while building the result record."""
    root, target = _dataset(fake_dataset)
    record = train_fact.run("squat", target, seed=42, data=str(root), epochs=0,
                            min_train_state=1, min_val_state=1)
    assert record["epochs"] == 0
    assert record["val"] and record["split_audit"]["min_train_state"] == 1


def test_context_weighting_survives_an_unseen_context(fake_dataset):
    """A context absent from the training pool has count 0, and `0 ** -alpha`
    raised ZeroDivisionError part-way through an epoch."""
    root, target = _dataset(fake_dataset)
    record = train_fact.run("squat", target, seed=42, data=str(root), epochs=1,
                            context_alpha=0.5, min_train_state=1, min_val_state=1)
    assert record["epochs"] == 1
    assert np.isfinite(record["val"]["macro_f1"])


def test_split_audit_records_the_support_rule(fake_dataset):
    root, target = _dataset(fake_dataset)
    record = train_fact.run("squat", target, seed=42, data=str(root), epochs=0,
                            min_train_state=1, min_val_state=1)
    audit = record["split_audit"]
    assert audit["min_train_state"] == 1 and audit["min_val_state"] == 1
    assert audit["group_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
