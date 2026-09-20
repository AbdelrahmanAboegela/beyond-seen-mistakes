"""Composition space of CPR-Coach, the second dataset.

ALEX-GYM-1's figure lays its diagnoses out by PCA, which works because each
exercise has at most 30 compositions over 5-7 criteria.  CPR-Coach has 88
compositions over 13 criteria, and the same projection collapses into an
unreadable tangle.

Its structure is, however, strictly layered -- one correct action, 13 single
errors, 59 doubles, 10 triples and 5 quadruples -- so the honest layout is by
cardinality, with Hamming-distance-one edges running between adjacent layers.
That shows directly what the support rule keeps: every single and double
qualifies, while the triples and quadruples have too few repetitions in too few
folders and drop out, exactly as lunge does on ALEX-GYM-1.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cpr_data import load_cpr

OUT = Path("paper/figures/cpr_composition_graph.png")


def main() -> None:
    _, _, compositions, _, _ = load_cpr("data/cpr_coach")
    eligible = set(json.loads(Path("configs/cpr_protocol.json").read_text())["default_targets"]["cpr"])

    labels, counts = np.unique(compositions, return_counts=True)
    bits = np.array([[int(ch) for ch in s] for s in labels])
    cardinality = bits.sum(1)
    frequency = dict(zip(labels, counts))

    # x position: evenly spread within the layer, ordered for a tidy layout
    by_layer = collections.defaultdict(list)
    for i, s in enumerate(labels):
        by_layer[int(cardinality[i])].append(i)
    pos = {}
    for layer, idx in by_layer.items():
        idx = sorted(idx, key=lambda i: labels[i])
        for rank, i in enumerate(idx):
            pos[i] = ((rank + .5) / len(idx), float(layer))

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cardinality[j] == cardinality[i] + 1 and int(np.sum(bits[i] != bits[j])) == 1:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color="0.86", lw=.35, zorder=1)
    for i, s in enumerate(labels):
        keep = s in eligible
        ax.scatter(pos[i][0], pos[i][1], s=4 + .55 * frequency[s],
                   facecolors="0.15" if keep else "white",
                   edgecolors="0.25", linewidths=.45, zorder=3)

    for layer in sorted(by_layer):
        n = len(by_layer[layer])
        kept = sum(1 for i in by_layer[layer] if labels[i] in eligible)
        ax.text(1.02, layer, f"{n} ({kept} kept)", fontsize=6, va="center", ha="left")
    ax.set_yticks(sorted(by_layer))
    ax.set_yticklabels([f"{k} errors" if k != 1 else "1 error" for k in sorted(by_layer)], fontsize=6.5)
    ax.set_xticks([])
    ax.set_xlim(-.04, 1.28)
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"{OUT}  ({len(labels)} compositions, {len(eligible)} eligible)")


if __name__ == "__main__":
    main()
