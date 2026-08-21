"""Coverage for the exploratory model zoo in ``src/modern_models.py``.

None of these models produce a reported result, but they were entirely
untested, which is how a phantom import and four no-op ablations survived.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from modern_models import (
    ANAT_MAP,
    BACKBONE_BUILDERS,
    CAPER_VARIANTS,
    UNIMPLEMENTED_CAPER_VARIANTS,
    build_modern,
    pose_features,
    split_pose,
)


def _batch(n=2, T=16):
    return torch.randn(n, T, 198)


# --- shared feature plumbing (this part IS used by the reported models) ------

def test_split_pose_puts_frontal_first():
    """X packs frontal joints in columns 0-98 and lateral in 99-197; several
    modules index views by that convention independently."""
    x = torch.zeros(1, 16, 198)
    x[..., :99] = 1.0
    p = split_pose(x)
    assert p.shape == (1, 2, 16, 33, 3)
    assert (p[:, 0] == 1.0).all() and (p[:, 1] == 0.0).all()


def test_split_pose_rejects_wrong_width():
    """A bare assert here would vanish under `python -O`, so this must be a
    real exception carrying the offending width."""
    with pytest.raises(ValueError, match="expected 198 features"):
        split_pose(torch.randn(1, 16, 200))


@pytest.mark.parametrize("mode,channels", [("j", 3), ("jb", 6), ("jbm", 9), ("m", 3)])
def test_pose_features_channel_count(mode, channels):
    assert pose_features(_batch(), mode).shape == (2, 2, 16, 33, channels)


def test_motion_feature_is_a_first_difference():
    x = _batch(n=1)
    p = split_pose(x)
    m = pose_features(x, "m")
    assert torch.allclose(m[:, :, 0], torch.zeros_like(m[:, :, 0]))
    assert torch.allclose(m[:, :, 1:], p[:, :, 1:] - p[:, :, :-1], atol=1e-6)


# --- the zoo ----------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(BACKBONE_BUILDERS))
def test_style_backbones_forward(kind):
    model = build_modern(kind, nout=6)
    model.eval()
    with torch.no_grad():
        logits, aux = model(_batch(), return_aux=True)
    assert logits.shape == (2, 6)
    assert isinstance(aux, dict)


@pytest.mark.parametrize("kind", sorted(CAPER_VARIANTS))
@pytest.mark.parametrize("exercise", ["squat", "deadlift"])
def test_caper_variants_forward(kind, exercise):
    model = build_modern(kind, nout=len(ANAT_MAP[exercise]), exercise=exercise)
    model.eval()
    with torch.no_grad():
        logits, aux = model(_batch(), return_aux=True)
    assert logits.shape == (2, len(ANAT_MAP[exercise]))
    assert {"local_logits", "residual_logits", "gate", "evidence", "context"} <= set(aux)


def test_every_caper_ablation_actually_changes_the_model():
    """Four names used to fall through the elif chain and rebuild the baseline,
    which would have read as 'this component does nothing'."""
    def config(kind):
        m = build_modern(kind, nout=6, exercise="squat")
        return (m.use_masks, m.use_proto, m.use_context,
                m.slots.soft_masks, m.slots.hard_views,
                # graph=False swaps the encoder block type rather than a flag
                type(m.enc.blocks[0]).__name__,
                hasattr(m, "proto"), hasattr(m, "ctx_proj"))

    baseline = config("caper")
    for kind in CAPER_VARIANTS:
        if kind in ("caper", "caper_hard_view"):  # hard views are already the default
            continue
        assert config(kind) != baseline, f"{kind} is indistinguishable from the caper baseline"


@pytest.mark.parametrize("kind", sorted(UNIMPLEMENTED_CAPER_VARIANTS))
def test_unimplemented_ablations_fail_loudly(kind):
    with pytest.raises(NotImplementedError, match="does not have"):
        build_modern(kind, nout=6, exercise="squat")


def test_phantom_blockgcn_core_is_gone():
    """build_modern imported a module that does not exist in this repository."""
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_modern("blockgcn_core", nout=6)


def test_unknown_kind_lists_the_alternatives():
    with pytest.raises(ValueError, match="choose from"):
        build_modern("not_a_model", nout=6, exercise="squat")


def test_criterion_conditioned_model_requires_an_exercise():
    with pytest.raises(ValueError, match="requires an exercise"):
        build_modern("caper", nout=6)


def test_anat_map_joint_indices_are_in_range():
    for exercise, entries in ANAT_MAP.items():
        assert len(entries) > 0
        for view, joints in entries:
            assert view in ("F", "L"), f"{exercise}: unknown view {view!r}"
            assert joints, f"{exercise}: empty joint set"
            assert min(joints) >= 0 and max(joints) < 33, f"{exercise}: joint index out of range"


if __name__ == "__main__":
    pytest.main([__file__])
