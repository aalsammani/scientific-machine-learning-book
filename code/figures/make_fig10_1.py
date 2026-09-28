"""Figure 10.1: SIR non-identifiability (top; Chapter 10, section 10.3) and the offline-online gap for a
learned closure (bottom; section 10.4).
Data: results/chapter10/sir_trajectories.npz and results/chapter10/closure_instability.json. The
script that produced closure_instability.json was not preserved (see code/chapter10/README.md); the
bottom row is plotted from the stored file.

Usage:
    python code/figures/make_fig10_1.py
Output: figures/chapter10/F10.1_nonidentifiability_and_instability.png and .pdf (reads stored results only; runtime a few seconds).
Colors: Okabe-Ito order (colorblind-safe), with line style / marker as secondary encoding.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter10"
FIG_DIR = REPO_ROOT / "figures" / "chapter10"

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


s = np.load(RESULTS / "sir_trajectories.npz")
c = json.load(open(RESULTS / "closure_instability.json"))
fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
ax[0, 0].plot(s["t"], s["y"], color="k", label="both parameter sets")
ax[0, 0].set_title("SIR incidence $y(t)$ (identical)"); ax[0, 0].set_xlabel("$t$"); ax[0, 0].legend(fontsize=9.5)
ax[0, 1].plot(s["t"], s["I_true"], color=BLUE, label=r"$\gamma=1/7$, $I_0=0.01$ (data-generating)")
ax[0, 1].plot(s["t"], s["I_alt"], color=VERM, ls="--", label=r"$\gamma'=1/10$, $I_0'=0.005$ (alternative)")
ax[0, 1].set_title("Infectious $I(t)$"); ax[0, 1].set_xlabel("$t$")
# direct labels (a legend box collides with the curves in this narrow panel)
ax[0, 1].text(27, 0.40, r"$\gamma'=1/10$, $I_0'=0.005$ (alternative)", color=VERM, fontsize=9.5)
ax[0, 1].text(31, 0.30, r"$\gamma=1/7$, $I_0=0.01$ (data-generating)", color=BLUE, fontsize=9.5)
rs = sorted(c["l2_by_r"], key=float)
ax[1, 0].loglog([float(r) for r in rs], [c["l2_by_r"][r] for r in rs], marker="o", color=BLUE)
ax[1, 0].set_xlabel("bump radius $r$"); ax[1, 0].set_ylabel(r"$\|g-g^\dagger\|_{L^2(\mu)}$")
ax[1, 0].set_title("Offline error\n(perturbed eigenvalues unchanged)")
xt, xp = np.array(c["x_true"]), np.array(c["x_pert"])
ax[1, 1].semilogy(c["t_true"], np.linalg.norm(xt, axis=-1 if xt.shape[-1] == 2 else 0), color=BLUE, label="true closure")
ax[1, 1].semilogy(c["t_pert"], np.linalg.norm(xp, axis=-1 if xp.shape[-1] == 2 else 0), color=VERM, ls="--",
                  label="perturbed closure")
ax[1, 1].set_xlabel("$t$"); ax[1, 1].set_ylabel(r"$\|x(t)\|$"); ax[1, 1].set_title("Online behavior"); ax[1, 1].legend(fontsize=9.5, loc="lower left")
ax[1, 0].yaxis.label.set_size(13)
fig.tight_layout()
save(fig, "F10.1_nonidentifiability_and_instability.png")
