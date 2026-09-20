"""Composition spaces of both datasets, in one figure.

This replaces the two separate composition figures.  They answered the same
question -- what does the observed diagnosis space look like, and what does the
support rule keep -- so splitting them across a full-width float and a
single-column float cost space and made the two datasets look like separate
stories rather than the same protocol applied twice.

The layouts still differ, because the data differ.  ALEX-GYM-1's exercises have
16-30 compositions over 5-7 criteria and project legibly under PCA.  CPR-Coach
has 88 over 13, where the same projection is an unreadable tangle but the
cardinality structure is strictly layered, so it is drawn by layer instead.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from alexgym_data import CRITERIA, load_exercise
from cpr_data import load_cpr

OUT = Path("paper/figures/composition_spaces.png")


def alex_panel(ax, exercise, eligible):
    _, _, compositions, _, _ = load_exercise("data", exercise, T=16)
    labels, counts = np.unique(compositions, return_counts=True)
    bits = np.array([[int(c) for c in s] for s in labels])
    xy = (PCA(n_components=2, random_state=0).fit_transform(bits) if bits.shape[1] >= 2
          else np.c_[bits[:, 0], np.zeros(len(bits))])
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if int(np.sum(bits[i] != bits[j])) == 1:
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.85", lw=.5, zorder=1)
    for i, s in enumerate(labels):
        keep = s in eligible
        ax.scatter(xy[i, 0], xy[i, 1], s=8 + 2.0 * counts[i],
                   facecolors="0.15" if keep else "white",
                   edgecolors="0.25", linewidths=.5, zorder=3)
    ax.set_title(f"{exercise.capitalize()} ({len(labels)} comps., "
                 f"{sum(1 for s in labels if s in eligible)} kept)", fontsize=7)


def cpr_panel(ax):
    _, _, compositions, _, _ = load_cpr("data/cpr_coach")
    eligible = set(json.loads(Path("configs/cpr_protocol.json").read_text())["default_targets"]["cpr"])
    labels, counts = np.unique(compositions, return_counts=True)
    bits = np.array([[int(c) for c in s] for s in labels])
    card = bits.sum(1)
    freq = dict(zip(labels, counts))
    by_layer = collections.defaultdict(list)
    for i, s in enumerate(labels):
        by_layer[int(card[i])].append(i)
    pos = {}
    for layer, idx in by_layer.items():
        for rank, i in enumerate(sorted(idx, key=lambda k: labels[k])):
            pos[i] = ((rank + .5) / len(idx), float(layer))
    for i in range(len(labels)):
        for j in range(len(labels)):
            if card[j] == card[i] + 1 and int(np.sum(bits[i] != bits[j])) == 1:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color="0.88", lw=.3, zorder=1)
    for i, s in enumerate(labels):
        keep = s in eligible
        ax.scatter(pos[i][0], pos[i][1], s=3 + .45 * freq[s],
                   facecolors="0.15" if keep else "white",
                   edgecolors="0.25", linewidths=.4, zorder=3)
    for layer in sorted(by_layer):
        kept = sum(1 for i in by_layer[layer] if labels[i] in eligible)
        ax.text(1.03, layer, f"{len(by_layer[layer])} ({kept})", fontsize=5.5, va="center")
    ax.set_yticks(sorted(by_layer))
    ax.set_yticklabels([f"{k}" for k in sorted(by_layer)], fontsize=6)
    ax.set_ylabel("errors per diagnosis", fontsize=6)
    ax.set_xlim(-.05, 1.30)
    ax.invert_yaxis()
    ax.set_title(f"CPR-Coach ({len(labels)} comps., {len(eligible)} kept)", fontsize=7)


def main() -> None:
    eligible = json.loads(Path("configs/final_protocol.json").read_text())["default_targets"]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75))
    for ax, exercise in zip(axs, CRITERIA):
        alex_panel(ax, exercise, set(eligible[exercise]))
        ax.set_xticks([]); ax.set_yticks([])
    cpr_panel(axs[3])
    axs[3].set_xticks([])
    for ax in axs:
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.text(.5, .005, "Node area = diagnosis frequency; filled nodes are retained by the frozen "
                       "support rule; edges join diagnoses differing in one criterion.",
             ha="center", fontsize=6)
    fig.tight_layout(rect=[0, .06, 1, 1], w_pad=.9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
