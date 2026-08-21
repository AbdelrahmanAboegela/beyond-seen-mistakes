"""Unit tests for deep learning model architectures (TCN, GRU, Transformer, SSM, STGCN, and FACT)."""

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

from train_backbones import TCN, GRUModel, TransformerModel, SSMModel, STGCNModel
from train_fact import FACT


@pytest.mark.parametrize("nout", [5, 6])
@pytest.mark.parametrize("model_cls", [TCN, GRUModel, TransformerModel, SSMModel])
def test_backbone_models_forward_shape(model_cls, nout):
    batch_size = 4
    seq_len = 16
    in_dim = 198  # 2 views * 33 joints * 3 coords
    x = torch.randn(batch_size, seq_len, in_dim)

    model = model_cls(din=in_dim, nout=nout)
    model.eval()
    with torch.no_grad():
        out = model(x)
        assert out.shape == (batch_size, nout), f"{model.__class__.__name__} output shape mismatch"


@pytest.mark.parametrize("nout", [5, 6])
def test_stgcn_model_forward_shape(nout):
    batch_size = 4
    seq_len = 16
    in_dim = 198
    x = torch.randn(batch_size, seq_len, in_dim)

    model = STGCNModel(nout=nout)
    model.eval()
    with torch.no_grad():
        out = model(x)
        assert out.shape == (batch_size, nout), "STGCNModel output shape mismatch"


@pytest.mark.parametrize("exercise", ["squat", "deadlift"])
def test_fact_model_forward_shape(exercise):
    batch_size = 4
    seq_len = 16
    in_dim = 198
    x = torch.randn(batch_size, seq_len, in_dim)

    model = FACT(exercise=exercise)
    model.eval()

    with torch.no_grad():
        out, tokens = model(x, return_tokens=True)
        num_criteria = len(model.adapter)
        assert out.shape == (batch_size, num_criteria)
        assert tokens.shape == (batch_size, num_criteria, 32)


if __name__ == "__main__":
    pytest.main([__file__])
