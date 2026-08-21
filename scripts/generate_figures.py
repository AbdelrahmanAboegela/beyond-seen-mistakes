"""Regenerate the result figures used by the paper and README."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
MODELS = ["tcn", "gru", "transformer", "ssm", "fact"]
LABELS = ["TCN", "GRU", "Transformer", "SSM", "FACT"]


def rq1() -> None:
    df = pd.read_csv(ROOT / "results/context/target_mean_metrics.csv", dtype={"target": str})
    q = df[df.metric == "exact_match"].pivot_table(
        index=["model", "exercise", "target"], columns="split", values="value"
    ).reset_index()
    means = q.groupby("model")[["val", "test"]].mean().reindex(MODELS) * 100
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = range(len(MODELS))
    ax.bar([i - .18 for i in x], means.val, width=.36, label="Grouped validation", color="#7aa6c2")
    ax.bar([i + .18 for i in x], means.test, width=.36, label="Targeted LOCO", color="#d97963")
    ax.set_xticks(list(x), LABELS)
    ax.set_ylabel("Exact diagnosis match (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rq1_exact_gap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rq1_exact_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rq2() -> None:
    df = pd.read_csv(ROOT / "results/context/criterion_rows_target_criterion.csv", dtype={"target": str})
    q = df[df.perturbation == "temporal_mean"].groupby(
        ["exercise", "target", "criterion"], as_index=False
    ).agg(label_opposition_pressure=("context_pressure", "mean"), accuracy=("baseline_accuracy", "mean"))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for exercise, g in q.groupby("exercise"):
        ax.scatter(g.label_opposition_pressure, 100 * g.accuracy, s=23, alpha=.78, label=exercise.capitalize())
    ax.set_xlabel("Label-opposition pressure (training labels only)")
    ax.set_ylabel("Held-out target-bit accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_pressure_error.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rq2_pressure_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def matched() -> None:
    df = pd.read_csv(ROOT / "results/matched_composition_v3_optimized/target_mean_metrics.csv", dtype={"target": str})
    q = df[df.metric == "exact_match"]
    means = q.groupby(["model", "condition"]).value.mean().unstack().reindex(MODELS[:4]) * 100
    fig, ax = plt.subplots(figsize=(5.1, 2.8))
    x = range(4)
    ax.bar([i - .18 for i in x], means.seen, width=.36, label="Target seen in training", color="#3f7f93")
    ax.bar([i + .18 for i in x], means.unseen, width=.36, label="Target absent from training", color="#d66b55")
    ax.set_xticks(list(x), LABELS[:4])
    ax.set_ylabel("Exact diagnosis match (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "matched_exact.pdf", bbox_inches="tight")
    fig.savefig(OUT / "matched_exact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rq1()
    rq2()
    matched()
    print(f"Wrote figures to {OUT}")
