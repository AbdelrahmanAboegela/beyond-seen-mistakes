"""Fast integrity checks for the public research artifact."""
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = {"tcn", "gru", "transformer", "ssm", "fact"}
SEEDS = {7, 42, 123}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


protocol = json.loads((ROOT / "configs/final_protocol.json").read_text())
targets = {(ex, target) for ex, values in protocol["default_targets"].items() for target in values}
require(len(targets) == 12, f"expected 12 eligible targets, found {len(targets)}")

runs = []
pattern = re.compile(r"^(tcn|gru|transformer|ssm|fact)_(squat|deadlift)_([01]+)_s(7|42|123)$")
for path in (ROOT / "results/runs").glob("*.json"):
    match = pattern.match(path.stem)
    require(match is not None, f"unexpected run filename: {path.name}")
    payload = json.loads(path.read_text())
    model, exercise, target, seed = match.groups()
    require((exercise, target) in targets, f"unsupported target in {path.name}")
    require(int(seed) == int(payload["seed"]), f"seed mismatch in {path.name}")
    runs.append((model, exercise, target, int(seed)))

require(len(runs) == 180, f"expected 180 run records, found {len(runs)}")
require(len(set(runs)) == 180, "duplicate run records")
require({r[0] for r in runs} == MODELS, "model grid is incomplete")
require({r[3] for r in runs} == SEEDS, "seed grid is incomplete")

stats = json.loads((ROOT / "results/rq_stats.json").read_text())["primary_tests"]
require(abs(stats["H1_exact_match_gap"]["mean_validation_minus_loco"] - 0.11336171781452319) < 1e-12, "RQ1 drift")
require(abs(stats["H2_pressure_accuracy"]["mean_within_target_rho"] + 0.6507475231012901) < 1e-12, "RQ2 drift")

metrics = pd.read_csv(ROOT / "results/context/target_mean_metrics.csv", dtype={"target": str})
require(set(metrics.model) == MODELS, "aggregated metrics model set is incomplete")
require(set(zip(metrics.exercise, metrics.target)) == targets, "aggregated metrics target set is incomplete")

for relative in ["paper/main.tex", "paper/Beyond_Seen_Mistakes.pdf", "paper/figures/composition_graph.png"]:
    require((ROOT / relative).is_file(), f"missing release artifact: {relative}")

print("Release verified: 5 models × 12 targets × 3 seeds, frozen statistics, paper, and figures.")

