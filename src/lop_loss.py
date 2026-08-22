"""Label-Opposition Pressure (LOP) aware loss functions and criterion calibration utilities.

These losses are selectable training *ablations*, reached through
``train_fact.py --loss``.  The reported five-model panel is trained with plain
``bce``; nothing here is used to produce the numbers in ``RESULTS.md``.

Scope caveat, stated once here and repeated in the CLI help: LOP weights are
built from the training labels **and the identity of the held-out composition**.
No test label is read, but the weighting does condition training on which
diagnosis was withheld.  That makes ``lop_weighted`` an oracle-flavoured upper
bound, not a deployable method, and it must never be compared against ``bce``
as if the two had the same information.  ``focal`` carries no such caveat.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_TYPES = ("bce", "lop_weighted", "focal")


def compute_lop_counts(Y_train, target_str):
    """Compute 1-bit opposing neighbor counts for each criterion from training fold labels."""
    Y_train = np.asarray(Y_train, dtype=int)
    if Y_train.ndim != 2:
        raise ValueError(f"Y_train must be 2-D (n_samples, n_criteria), got shape {Y_train.shape}")
    n_samples, n_criteria = Y_train.shape
    target_str = str(target_str)
    if len(target_str) != n_criteria:
        raise ValueError(f"target {target_str!r} has {len(target_str)} bits but Y_train has {n_criteria} criteria")
    if set(target_str) - {"0", "1"}:
        raise ValueError(f"target {target_str!r} must be a binary string")
    target_vec = np.array([int(b) for b in target_str], dtype=int)

    lop_counts = np.zeros(n_criteria, dtype=np.float32)
    for c in range(n_criteria):
        # 1-bit opposing neighbor pattern: all criteria except c match target_vec
        mask_other = np.ones(n_criteria, dtype=bool)
        mask_other[c] = False
        match_other = (Y_train[:, mask_other] == target_vec[mask_other]).all(axis=1)
        opposing = match_other & (Y_train[:, c] != target_vec[c])
        lop_counts[c] = float(opposing.sum())

    return lop_counts


def compute_lop_weights(Y_train, target_str, alpha=0.5):
    """Compute per-criterion loss weights based on normalized Label Opposition Pressure (LOP).

    Weights lie in ``[1, 1 + alpha]``: criteria under the most opposition
    pressure are upweighted.  See the module docstring for why this needs the
    held-out target and what that costs in interpretation.
    """
    lop_counts = compute_lop_counts(Y_train, target_str)
    if lop_counts.max() > 0:
        norm_lop = lop_counts / lop_counts.max()
    else:
        norm_lop = lop_counts
    weights = 1.0 + alpha * norm_lop
    return torch.tensor(weights, dtype=torch.float32)


class LOPWeightedBCELoss(nn.Module):
    """Binary Cross Entropy loss weighted per-criterion by Label Opposition Pressure."""

    def __init__(self, pos_weight=None, lop_weights=None):
        super().__init__()
        # Both tensors are buffers so ``.to(device)`` moves them with the module.
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("lop_weights", lop_weights)

    def forward(self, logits, targets):
        raw_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if self.lop_weights is not None:
            if self.lop_weights.shape[-1] != logits.shape[-1]:
                raise ValueError(f"lop_weights has {self.lop_weights.shape[-1]} entries but "
                                 f"logits have {logits.shape[-1]} criteria")
            raw_bce = raw_bce * self.lop_weights[None, :]
        return raw_bce.mean()


class FocalBCELoss(nn.Module):
    """Focal Binary Cross Entropy loss to emphasize hard/opposing criterion predictions."""

    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma

        raw_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        focal_bce = focal_weight * raw_bce
        return focal_bce.mean()


def elementwise_loss(loss_type, logits, targets, pos_weight=None, lop_weights=None, gamma=2.0):
    """Per-element criterion loss, shaped like ``logits``.

    Training multiplies this by per-sample context weights before reducing, so
    it must stay unreduced.  ``bce`` is bit-identical to a direct
    ``binary_cross_entropy_with_logits(..., reduction='none')`` call, which is
    what keeps the frozen results reproducible when the flag is left at default.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unknown loss type: {loss_type}; choose from {list(LOSS_TYPES)}")
    raw = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if loss_type == "bce":
        return raw
    if loss_type == "lop_weighted":
        if lop_weights is None:
            raise ValueError("loss_type='lop_weighted' requires lop_weights; "
                             "build them with compute_lop_weights(Y_train, target).")
        if lop_weights.shape[-1] != logits.shape[-1]:
            raise ValueError(f"lop_weights has {lop_weights.shape[-1]} entries but "
                             f"logits have {logits.shape[-1]} criteria")
        # Normalized to mean 1 so the ablation stays comparable to bce at the
        # same learning rate; only the relative emphasis across criteria matters.
        w = lop_weights.to(raw.device, raw.dtype)
        return raw * (w / w.mean().clamp_min(1e-8))[None, :]
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1.0, probs, 1.0 - probs)
    return ((1.0 - pt) ** gamma) * raw


def get_loss_fn(loss_type="bce", pos_weight=None, lop_weights=None, gamma=2.0):
    """Factory function for criteria loss functions."""
    if loss_type == "bce":
        return lambda logits, targets: F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
    elif loss_type == "lop_weighted":
        if lop_weights is None:
            raise ValueError("loss_type='lop_weighted' requires lop_weights; "
                             "build them with compute_lop_weights(Y_train, target).")
        return LOPWeightedBCELoss(pos_weight=pos_weight, lop_weights=lop_weights)
    elif loss_type == "focal":
        return FocalBCELoss(pos_weight=pos_weight, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}; choose from {list(LOSS_TYPES)}")
