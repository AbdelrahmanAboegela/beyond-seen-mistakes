"""Render the observed-diagnosis composition graph used in the paper and README."""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse, json
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    from .alexgym_data import load_exercise, CRITERIA
except ImportError:
    from alexgym_data import load_exercise, CRITERIA


def build_figure(data, protocol):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
    synthetic = False
    # strict=: a new exercise in CRITERIA must not be silently dropped from the figure
    for ax, ex in zip(axs, CRITERIA, strict=True):
        X, Y, co, g, df = load_exercise(data, ex, T=16)
        synthetic |= bool(df.attrs.get("is_synthetic", False))
        u, c = np.unique(co, return_counts=True)
        bits = np.array([[int(ch) for ch in s] for s in u])
        if bits.shape[1] >= 2:
            xy = PCA(n_components=2, random_state=0).fit_transform(bits)
        else:
            xy = np.c_[bits[:, 0], np.zeros(len(bits))]
        # Hamming-1 edges
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                if np.sum(bits[i] != bits[j]) == 1:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.82", lw=.6, zorder=1)
        eligible = set(protocol["default_targets"].get(ex, []))
        for i, s in enumerate(u):
            isel = s in eligible
            ax.scatter(xy[i, 0], xy[i, 1], s=10 + 2.4 * c[i],
                       facecolors="0.15" if isel else "white", edgecolors="0.2",
                       linewidths=.6, zorder=3)
        ax.set_title(f"{ex.capitalize()} ({len(u)} comps.)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    fig.text(.5, .01, "Node area = diagnosis frequency; filled nodes are the 12 final eligible "
                      "LOCO targets; edges connect Hamming-distance-one diagnoses.",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=[0, .08, 1, 1], w_pad=1.0)
    return fig, synthetic


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--protocol", default=str(ROOT_DIR / "configs/final_protocol.json"))
    ap.add_argument("--out", default=str(ROOT_DIR / "paper/figures/composition_graph.png"))
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="Write the figure even when the real dataset is unavailable.")
    a = ap.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    fig, synthetic = build_figure(a.data, protocol)
    if synthetic and not a.allow_synthetic:
        plt.close(fig)
        raise SystemExit(
            f"Refusing to overwrite {a.out}: no real ALEX-GYM data was found under "
            f"'{a.data}', so this figure would show synthetic placeholder compositions. "
            "See data/README.md, or pass --allow-synthetic with a scratch --out path."
        )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
