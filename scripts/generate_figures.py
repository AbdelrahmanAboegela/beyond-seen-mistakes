"""Regenerate the result figures used by the paper and README."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


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
    """Per-target matched pairs, plus the gap against residual marginal imbalance.

    The paired view matters more than architecture-average bars because the
    statistical unit is the held-out diagnosis and there are only twelve of
    them. Panel (b) addresses the obvious alternative explanation directly: if
    the residual post-exchange marginal mismatch drove the effect, the largest
    gaps would sit at the largest mismatch.
    """
    df = pd.read_csv(ROOT / "results/matched_composition_v3_optimized/target_mean_metrics.csv", dtype={"target": str})
    pairs = (df[df.metric == "exact_match"]
             .groupby(["exercise", "target", "condition"]).value.mean()
             .unstack().mul(100).reset_index())
    balance = pd.read_csv(ROOT / "results/matched_composition_v3_optimized/balance_audit.csv", dtype={"target": str})
    pairs = pairs.merge(
        balance.groupby(["exercise", "target"], as_index=False).optimized_max.mean(),
        on=["exercise", "target"],
    )
    pairs["gap"] = pairs.seen - pairs.unseen
    colors = {"squat": "#3f7f93", "deadlift": "#d66b55"}

    # Authored at final IEEE column width so LaTeX does not downscale the text.
    # Panel labels sit inside the axes rather than as titles: a title row costs
    # vertical space the paper's four-page budget does not have.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.45))
    for _, row in pairs.iterrows():
        ax1.plot([0, 1], [row.seen, row.unseen], marker="o", ms=2.4, lw=.8,
                 color=colors[row.exercise], alpha=.85)
    ax1.set_xlim(-.32, 1.32)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Present", "Absent"], fontsize=7)
    ax1.set_ylabel("Exact match (%)", fontsize=7)
    ax1.text(.97, .93, f"(a) {len(pairs)}/{len(pairs)} fall", transform=ax1.transAxes,
             ha="right", va="top", fontsize=7)

    for exercise, group in pairs.groupby("exercise"):
        ax2.scatter(group.optimized_max, group.gap, s=11, alpha=.85,
                    color=colors[exercise], label=exercise.capitalize())
    # Round before ranking: two squat targets have an exactly tied gap (25/3 pp),
    # and float noise of ~1e-15 from a different summation order would untie them
    # and move rho from -.27 to -.26. Rounding keeps the tie, and the value, stable.
    rho = spearmanr(pairs.optimized_max.round(9), pairs.gap.round(9)).statistic
    ax2.set_xlabel("Residual mismatch", fontsize=7)
    ax2.set_ylabel("Gap (pp)", fontsize=7)
    ax2.text(.97, .93, rf"(b) $\rho={rho:.2f}$", transform=ax2.transAxes,
             ha="right", va="top", fontsize=7)
    ax2.legend(frameon=False, fontsize=6, handletextpad=.3, borderpad=.2,
               loc="center right")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6, length=2.5, pad=1.5)
    fig.tight_layout(pad=.3, w_pad=.8)
    # PNG only: the paper draws this figure natively in LaTeX (see main.tex), so
    # the vector copy had no consumer. This one is the README's.
    fig.savefig(OUT / "matched_exact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def cpr_composition() -> None:
    """Second-dataset composition space; skipped when CPR-Coach is absent."""
    import os, subprocess, sys
    if not (ROOT / "data" / "cpr_coach").exists():
        print("skipping CPR composition figure: data/cpr_coach not present")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")   # the plotter imports cpr_data
    subprocess.run([sys.executable, str(ROOT / "src" / "plot_cpr_composition_graph.py")],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rq1()
    rq2()
    matched()
    cpr_composition()
    print(f"Wrote figures to {OUT}")
