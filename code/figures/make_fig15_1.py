"""Figure 15.1: SEIR non-identifiability from incidence data (Chapter 15, section 15.1).
Data: results/chapter15/seir_trajectories.npz.

Usage:
    python code/figures/make_fig15_1.py
Output: figures/chapter15/F15.1_seir_nonidentifiability.png and .pdf (reads stored results only; runtime a few seconds).
Colors: Okabe-Ito order (colorblind-safe), with line style / marker as secondary encoding.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter15"
FIG_DIR = REPO_ROOT / "figures" / "chapter15"

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


z = np.load(RESULTS / "seir_trajectories.npz")
fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.3))
ax[0].plot(z["t"], z["y"], color="k", label="both parameter sets (identical)")
ax[0].set_title("Incidence $y(t)$"); ax[0].set_xlabel("$t$")
ax[1].plot(z["t"], z["E_true"], color=BLUE, label="data-generating parameters"); ax[1].plot(z["t"], z["E_alt"], color=VERM, ls="--", label="alternative parameters")
ax[1].set_title("Exposed $E(t)$"); ax[1].set_xlabel("$t$")
ax[2].plot(z["t"], z["I_true"], color=BLUE); ax[2].plot(z["t"], z["I_alt"], color=VERM, ls="--")
ax[2].set_title("Infectious $I(t)$"); ax[2].set_xlabel("$t$")
h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
save(fig, "F15.1_seir_nonidentifiability.png")
