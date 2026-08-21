"""Matched target-present versus target-absent composition experiment.

The same recording-disjoint validation and test indices are used in both
conditions. Training-set size is matched by exchanging target-diagnosis rows for
an equal number of non-target rows from the same training-group pool.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from alexgym_data import CRITERIA, load_exercise
from train_backbones import DS, build, metrics, pred


def state_counts(Y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray([[(Y[idx, c] == s).sum() for s in (0, 1)] for c in range(Y.shape[1])])


def make_matched_split(Y, compositions, groups, target, seed, *, val_fraction=.2,
                       min_train_state=8, min_val_state=1, candidates=512):
    Y = np.asarray(Y, int); compositions = np.asarray(compositions).astype(str)
    groups = np.asarray(groups).astype(str); target = str(target)
    target_groups = np.unique(groups[compositions == target])
    if len(target_groups) < 3:
        raise ValueError("Matched experiment requires at least three target groups.")
    # The three reported seeds select distinct target groups whenever possible;
    # this prevents repeated evaluation of the same recording group.
    canonical_rank = {7: 0, 42: 1, 123: 2}.get(int(seed), int(seed))
    test_group = str(target_groups[(int(target, 2) + canonical_rank) % len(target_groups)])
    test = np.flatnonzero((groups == test_group) & (compositions == target))
    clean_groups = np.asarray([g for g in np.unique(groups) if g != test_group and not np.any((groups == g) & (compositions == target))])
    clean_rows = np.flatnonzero(np.isin(groups, clean_groups))
    remaining_target_groups = set(target_groups) - {test_group}

    best = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=int(seed) + offset)
        _, val_rel = next(splitter.split(clean_rows, groups=groups[clean_rows]))
        val = clean_rows[val_rel]
        val_groups = set(groups[val])
        train_pool = np.flatnonzero(~np.isin(groups, list(val_groups | {test_group})))
        target_train = train_pool[compositions[train_pool] == target]
        non_target = train_pool[compositions[train_pool] != target]
        if not len(target_train) or len(non_target) <= len(target_train):
            continue
        reserve_rng = np.random.default_rng(int(seed) * 1000 + offset + 97)
        reserve = reserve_rng.choice(non_target, size=len(target_train), replace=False)
        seen = np.concatenate([target_train, np.setdiff1d(non_target, reserve, assume_unique=False)])
        unseen = non_target.copy()
        sc_seen, sc_unseen, sc_val = state_counts(Y, seen), state_counts(Y, unseen), state_counts(Y, val)
        valid = sc_seen.min() >= min_train_state and sc_unseen.min() >= min_train_state and sc_val.min() >= min_val_state
        score = (int(min(sc_seen.min(), sc_unseen.min(), sc_val.min())), -abs(len(val) / len(clean_rows) - val_fraction))
        if valid and (best is None or score > best[0]):
            best = (score, seen, unseen, val, test, target_train, reserve, sc_seen, sc_unseen, sc_val, offset)
    if best is None:
        raise ValueError(f"No feasible matched split for {target}, seed {seed}.")
    _, seen, unseen, val, test, target_train, reserve, sc_seen, sc_unseen, sc_val, offset = best
    audit = {
        "target": target, "seed": int(seed), "test_group": test_group,
        "remaining_target_groups": sorted(remaining_target_groups), "split_offset": int(offset),
        "n_train_each": int(len(seen)), "n_target_rows_added": int(len(target_train)),
        "n_non_target_rows_exchanged": int(len(reserve)), "n_val": int(len(val)), "n_test": int(len(test)),
        "seen_train_state_counts": sc_seen.tolist(), "unseen_train_state_counts": sc_unseen.tolist(),
        "val_state_counts": sc_val.tolist(),
        "group_overlap": {
            "train_val": int(len(set(groups[seen]) & set(groups[val]))),
            "train_test": int(len(set(groups[seen]) & set(groups[test]))),
            "val_test": int(len(set(groups[val]) & set(groups[test]))),
        },
    }
    return seen, unseen, val, test, audit


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_condition(X, Y, train, val, test, kind, seed, shared_pos_weight, epochs=55, patience=15):
    seed_all(seed)
    train_loader = DataLoader(DS(X, Y, train), 32, shuffle=True)
    val_loader = DataLoader(DS(X, Y, val), 64)
    test_loader = DataLoader(DS(X, Y, test), 64)
    model = build(kind, X.shape[-1], Y.shape[1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shared_pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    best, state, stale, started = -1.0, None, 0, time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        yv, pv = pred(model, val_loader); score = metrics(yv, pv)["macro_f1"]
        if score > best + 1e-4:
            best, state, stale = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    yv, pv = pred(model, val_loader); yt, pt = pred(model, test_loader)
    return {"val": metrics(yv, pv), "test": metrics(yt, pt), "epochs": epoch + 1,
            "train_seconds": time.time() - started, "n_params": sum(p.numel() for p in model.parameters())}


def run(exercise, target, seed, model, data, epochs=55):
    X, Y, compositions, groups, _ = load_exercise(data, exercise, T=16)
    seen, unseen, val, test, audit = make_matched_split(Y, compositions, groups, target, seed)
    # Identical initialization seed isolates the training-set composition change.
    result = {"exercise": exercise, "target": target, "seed": seed, "model": model,
              "criteria": CRITERIA[exercise], "split_audit": audit}
    common_union = np.union1d(seen, unseen)
    pos = Y[common_union].sum(0); neg = len(common_union) - pos
    shared_pos_weight = np.clip(neg / np.maximum(pos, 1), .25, 8)
    result["shared_pos_weight"] = shared_pos_weight.tolist()
    result["seen"] = train_condition(X, Y, seen, val, test, model, seed, shared_pos_weight, epochs)
    result["unseen"] = train_condition(X, Y, unseen, val, test, model, seed, shared_pos_weight, epochs)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("--exercise", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--model", default="tcn", choices=["tcn", "gru", "transformer", "ssm", "stgcn"])
    ap.add_argument("--epochs", type=int, default=55); ap.add_argument("--out", required=True)
    a = ap.parse_args(); result = run(a.exercise, a.target, a.seed, a.model, a.data, a.epochs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
