"""Frozen split protocol: the support rule, group disjointness, manifests."""
import json
from pathlib import Path

import numpy as np
import pytest

from loco_split import DEFAULT_MIN_TRAIN_STATE, DEFAULT_MIN_VAL_STATE, make_loco_split

ROOT = Path(__file__).resolve().parents[1]


def test_default_support_rule_matches_frozen_protocol():
    """The rule was documented and enforced in the manifest builder, but the
    trainers defaulted to 1, so a run could be scored on an unsanctioned split."""
    protocol = json.loads((ROOT / "configs/final_protocol.json").read_text())
    assert DEFAULT_MIN_TRAIN_STATE == protocol["default_support_rule"]["min_training_examples_per_criterion_state"]
    assert DEFAULT_MIN_VAL_STATE == 1


@pytest.mark.parametrize("manifest_name", ["split_manifest_v2.json"])
def test_frozen_splits_are_unchanged_under_the_centralized_rule(manifest_name):
    """Every published split must still satisfy the rule now that it is enforced
    at training time, and must clear it with margin -- the selection score keys
    on the minimum across both folds jointly, which is only harmless while no
    split sits on the threshold."""
    manifest = json.loads((ROOT / "configs" / manifest_name).read_text())
    audits = [a for targets in manifest["splits"].values()
              for seeds in targets.values() for a in seeds.values()]
    assert len(audits) == 36, f"expected 36 target x seed splits, found {len(audits)}"
    worst_train = min(min(min(s) for s in a["train_state_counts"]) for a in audits)
    worst_val = min(min(min(s) for s in a["val_state_counts"]) for a in audits)
    assert worst_train >= DEFAULT_MIN_TRAIN_STATE, worst_train
    assert worst_val >= DEFAULT_MIN_VAL_STATE, worst_val
    assert worst_train > DEFAULT_MIN_TRAIN_STATE, "a frozen split now sits on the threshold"


def test_split_is_group_disjoint_and_target_only_in_test():
    bits = np.array([[0, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    Y = np.concatenate([np.tile(bits, (5, 1)), np.tile([[1, 1]], (5, 1))])
    compositions = np.array(["".join(map(str, r)) for r in Y])
    groups = np.repeat([f"g{i}" for i in range(6)], 5)
    train, val, test, audit = make_loco_split(Y, compositions, groups, target="11", seed=42,
                                              min_train_state=1, min_val_state=1)
    assert (compositions[test] == "11").all()
    assert not (set(groups[train]) & set(groups[val]))
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[val]) & set(groups[test]))
    assert audit["group_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert audit["min_train_state"] == 1 and audit["min_val_state"] == 1


def test_unsatisfiable_support_is_an_explicit_failure():
    Y = np.zeros((15, 2))
    compositions = np.array(["00"] * 10 + ["11"] * 5)
    groups = np.repeat(["g1", "g2", "g3"], 5)
    with pytest.raises(ValueError, match="No recording-disjoint split satisfies"):
        make_loco_split(Y, compositions, groups, target="11", seed=42, min_train_state=1)


def test_matched_v3_manifest_constraints_hold():
    """Matched conditions must differ only in the target composition."""
    manifest = json.loads((ROOT / "configs/matched_manifest_v3_optimized.json").read_text())
    entries = [e for ex in manifest.get("splits", manifest).values()
               for tgt in (ex.values() if isinstance(ex, dict) else [])
               for e in (tgt.values() if isinstance(tgt, dict) else [])]
    assert entries, "no matched manifest entries found"
    for e in entries:
        if not isinstance(e, dict):
            continue
        for key in ("seen", "unseen"):
            if key in e and isinstance(e[key], list):
                assert len(e[key]) == len(set(e[key])), "duplicate indices in a matched condition"
