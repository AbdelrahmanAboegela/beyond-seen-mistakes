"""Deterministic, recording-disjoint splits for composition holdout experiments.

The split selector rejects candidate group splits that leave a criterion state
absent from either training or validation.  This prevents early stopping on a
validation macro-F1 that is undefined for a label state.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Frozen support rule, mirroring
# configs/final_protocol.json -> default_support_rule.  Training, manifest
# construction and analysis must all use the same thresholds, otherwise a run
# can silently be scored on a split the published manifest never sanctioned.
DEFAULT_MIN_TRAIN_STATE = 8
DEFAULT_MIN_VAL_STATE = 1


def frozen_selection_score(train_counts, val_counts, n_val, n_pool, val_fraction):
    """Rank a candidate split. Higher is better; ties broken by fold balance.

    The published splits were selected with this exact rule, so it is frozen and
    must not be tuned. It is deliberately documented rather than left inline
    because it has a real quirk: the primary key is the minimum over *both*
    folds jointly, even though the two folds have different thresholds (8 for
    training, 1 for validation). A candidate with abundant training support but
    a thin validation state can therefore be ranked below one that is merely
    even. That does not affect the released results -- every frozen split clears
    both thresholds with margin, the smallest observed state count being 9 --
    but it means the score is a balance heuristic, not a direct encoding of the
    support rule. The support rule itself is enforced separately, as a hard
    admissibility filter, which is what actually guarantees the protocol.
    """
    return (int(min(train_counts.min(), val_counts.min())),
            -abs(n_val / n_pool - val_fraction))


def make_loco_split(Y, compositions, groups, target, seed, *, val_fraction=.2,
                    min_train_state=DEFAULT_MIN_TRAIN_STATE,
                    min_val_state=DEFAULT_MIN_VAL_STATE, candidates=512):
    """Return recording-disjoint train/validation/test indices and an audit.

    All recordings carrying ``target`` are test-only.  Among deterministic
    candidate group splits of the remaining recordings, select the split that
    maximizes the minimum per-criterion/state support and then minimizes the
    validation-size deviation.  A failure is explicit rather than silently
    admitting an unsupported validation fold.
    """
    Y = np.asarray(Y, dtype=int)
    compositions = np.asarray(compositions).astype(str)
    groups = np.asarray(groups).astype(str)
    target = str(target)
    target_groups = np.unique(groups[compositions == target])
    if len(test := np.flatnonzero(compositions == target)) == 0:
        raise ValueError(f"Target composition {target!r} does not occur in the data.")
    # Only target repetitions are scored.  Non-target repetitions from their
    # recording pairs are deliberately excluded from every partition.
    pool = np.flatnonzero(~np.isin(groups, target_groups))
    if len(np.unique(groups[pool])) < 2:
        raise ValueError("Fewer than two non-target recording groups remain.")

    best = None
    best_support = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction,
                                     random_state=int(seed) + offset)
        tr_rel, va_rel = next(splitter.split(pool, groups=groups[pool]))
        train, val = pool[tr_rel], pool[va_rel]
        train_counts = np.stack([(Y[train, c] == state).sum()
                                 for c in range(Y.shape[1]) for state in (0, 1)]).reshape(-1, 2)
        val_counts = np.stack([(Y[val, c] == state).sum()
                               for c in range(Y.shape[1]) for state in (0, 1)]).reshape(-1, 2)
        valid = train_counts.min() >= min_train_state and val_counts.min() >= min_val_state
        support = (int(train_counts.min()), int(val_counts.min()))
        if best_support is None or support > best_support:
            best_support = support
        score = frozen_selection_score(train_counts, val_counts, len(val), len(pool), val_fraction)
        if valid and (best is None or score > best[0]):
            best = (score, train, val, train_counts, val_counts, offset)
    if best is None:
        raise ValueError(f"No recording-disjoint split satisfies train>={min_train_state} and "
                         f"validation>={min_val_state} examples for every criterion state "
                         f"(target {target!r}, seed {seed}, {candidates} candidates; best "
                         f"achieved train/validation support was {best_support}).")
    _, train, val, train_counts, val_counts, offset = best
    audit = dict(target=target, seed=int(seed), split_offset=int(offset),
                 min_train_state=int(min_train_state), min_val_state=int(min_val_state),
                 target_groups=target_groups.tolist(), n_train=int(len(train)),
                 n_val=int(len(val)), n_test=int(len(test)),
                 train_state_counts=train_counts.tolist(), val_state_counts=val_counts.tolist(),
                 group_overlap=dict(train_val=int(len(set(groups[train]) & set(groups[val]))),
                                    train_test=int(len(set(groups[train]) & set(groups[test]))),
                                    val_test=int(len(set(groups[val]) & set(groups[test])))))
    return train, val, test, audit
