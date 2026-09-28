"""Figure 2.1: Lorenz-63 separation of two nearby trajectories (Chapter 2, section 2.3).
Plots the stored result results/chapter02/lorenz_sensitivity.json.

Usage:
    python code/figures/make_fig2_1.py
Output: figures/chapter02/F2.1_lorenz_sensitivity.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter02"
FIG_DIR = REPO_ROOT / "figures" / "chapter02"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

with open(RESULTS / "lorenz_sensitivity.json") as f:
    R = json.load(f)
t = np.asarray(R["t"], dtype=float)
d = np.asarray(R["diff"], dtype=float)
eps = float(R["eps"])
stored_rate = float(R["fitted_rate_14_to_29"])

# Least-squares fit of log(separation) over the growth window t in [14, 29] (as stated in the text).
w = (t >= 14.0 - 1e-9) & (t <= 29.0 + 1e-9)
rate, intercept = np.polyfit(t[w], np.log(d[w]), 1)
print(f"refit rate over [14,29] = {rate:.6f}; stored rate = {stored_rate:.6f}")
if abs(rate - stored_rate) > 1e-3:
    raise SystemExit("refitted rate disagrees with stored rate; not plotting")

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.semilogy(t, d, color=BLUE, lw=1.4, label="separation of the two trajectories")
tw = t[w]
ax.semilogy(tw, np.exp(intercept + rate * tw), color=VERM, ls="--", lw=1.8,
            label=rf"least-squares fit on $t\in[14,29]$ (rate {stored_rate:.2f})")
ax.axhline(eps, color=GRAY, ls=":", lw=1.3, label=r"initial separation $10^{-8}$")
ax.set_xlabel(r"time $t$")
ax.set_ylabel("distance between trajectories")
ax.set_title("Lorenz-63: separation of two nearby trajectories", fontsize=12)
ax.set_xlim(0, t[-1])
ax.set_ylim(3e-9, 1e1)
ax.legend(loc="upper left", fontsize=9.5)

name = "F2.1_lorenz_sensitivity.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
