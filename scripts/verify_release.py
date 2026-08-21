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
matched_manifest = json.loads((ROOT / "configs/matched_manifest_v2.json").read_text())
manifest_folds = [fold for exercise in matched_manifest["splits"].values() for target in exercise.values() for fold in target.values()]
require(len(manifest_folds) == 36, f"expected 36 matched manifest folds, found {len(manifest_folds)}")
for fold in manifest_folds:
    require(len(fold["seen_train"]) == len(fold["unseen_train"]), "manifest training sizes differ")
    require(not (set(fold["seen_train"]) & set(fold["validation"])), "manifest train/validation row overlap")
    require(not (set(fold["seen_train"]) & set(fold["test"])), "manifest train/test row overlap")

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

stats = json.loads((ROOT / "results/rq_stats.json").read_text())
rq1 = stats["primary"]["RQ1_matched_exact_match"]
require(abs(rq1["seen_minus_unseen"] - 0.25439814814814815) < 1e-12, "matched RQ1 drift")
require(abs(stats["diagnostic"]["RQ2_raw_lop"]["mean_within_target_rho"] + 0.6324804937729948) < 1e-12, "LOP drift")
require(abs(stats["diagnostic"]["RQ2_global_opposing_support"]["mean_within_target_rho"] + 0.7424785893093744) < 1e-12, "support-control drift")

matched = list((ROOT / "results/matched_composition_v2/runs").glob("*.json"))
require(len(matched) == 144, f"expected 144 matched-v2 records, found {len(matched)}")
for path in matched:
    payload = json.loads(path.read_text())
    audit = payload["split_audit"]
    require("shared_pos_weight" in payload, f"missing shared class weights: {path.name}")
    require(audit["n_train_each"] > 0 and audit["n_target_rows_added"] == audit["n_non_target_rows_exchanged"], f"unmatched training size: {path.name}")
    require(all(value == 0 for value in audit["group_overlap"].values()), f"recording overlap: {path.name}")

stgcn = list((ROOT / "results/stgcn_loco/runs").glob("*.json"))
require(len(stgcn) == 36, f"expected 36 ST-GCN LOCO records, found {len(stgcn)}")

metrics = pd.read_csv(ROOT / "results/context/target_mean_metrics.csv", dtype={"target": str})
require(set(metrics.model) == MODELS, "aggregated metrics model set is incomplete")
require(set(zip(metrics.exercise, metrics.target)) == targets, "aggregated metrics target set is incomplete")

for relative in ["paper/main.tex", "paper/Beyond_Seen_Mistakes.pdf", "paper/figures/composition_graph.png"]:
    require((ROOT / relative).is_file(), f"missing release artifact: {relative}")

print("Release verified: original 180-run grid, 144 matched-v2 pairs, 36 ST-GCN runs, frozen statistics, paper, and figures.")
