"""Unit tests wrapping fast release verification checks."""

import sys
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_final_protocol_targets():
    protocol_path = ROOT / "configs" / "final_protocol.json"
    assert protocol_path.is_file(), "configs/final_protocol.json missing"
    protocol = json.loads(protocol_path.read_text())
    targets = [(ex, target) for ex, values in protocol["default_targets"].items() for target in values]
    assert len(targets) == 12, f"Expected 12 eligible targets, got {len(targets)}"
    assert "squat" in protocol["default_targets"]
    assert "deadlift" in protocol["default_targets"]


def test_run_records_count():
    runs_dir = ROOT / "results" / "runs"
    assert runs_dir.is_dir(), "results/runs directory missing"
    json_files = list(runs_dir.glob("*.json"))
    assert len(json_files) == 180, f"Expected 180 run record JSON files, got {len(json_files)}"


def test_paper_artifacts_exist():
    paper_pdf = ROOT / "paper" / "Beyond_Seen_Mistakes.pdf"
    paper_tex = ROOT / "paper" / "main.tex"
    assert paper_pdf.is_file(), "Paper PDF snapshot missing"
    assert paper_tex.is_file(), "Paper LaTeX source missing"


if __name__ == "__main__":
    pytest.main([__file__])
