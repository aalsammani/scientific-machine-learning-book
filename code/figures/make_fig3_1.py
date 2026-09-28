"""Figure 3.1: RK4 order verification (left; Chapter 3, section 3.6) and Monte Carlo variance-rate
check (right; section 3.9). Plots the stored results results/chapter03/rk4_order_verification.json
and results/chapter03/monte_carlo_rate_verification.json.

Usage:
    python code/figures/make_fig3_1.py
Output: figures/chapter03/F3.1_rk4_and_mc_verification.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter03"
FIG_DIR = REPO_ROOT / "figures" / "chapter03"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

with open(RESULTS / "rk4_order_verification.json") as f:
    RK = json.load(f)
with open(RESULTS / "monte_carlo_rate_verification.json") as f:
    MC = json.load(f)

h = np.array([r["h"] for r in RK["rows"]], dtype=float)
err = np.array([r["err"] for r in RK["rows"]], dtype=float)
orders = RK["observed_order"]
n = np.array([r["n"] for r in MC["rows"]], dtype=float)
var_n = np.array([r["empirical_var"] for r in MC["rows"]], dtype=float) * n
print("observed orders:", [f"{o:.2f}" for o in orders])
print("Var*n:", [f"{v:.3f}" for v in var_n], "mean", f"{var_n.mean():.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

# Left: RK4 error vs step size, slope-4 reference anchored at the finest step.
ref = err[-1] * (h / h[-1]) ** 4
a1.loglog(h, err, "o-", color=BLUE, ms=7, lw=1.2, label="RK4 error", zorder=2)
a1.loglog(h, ref, color=VERM, ls="--", lw=1.5, label=r"slope-4 reference ($\propto h^4$)", zorder=3)
a1.set_xlabel(r"step size $h$")
a1.set_ylabel(r"error at $t=1$")
a1.set_title(f"RK4 convergence (observed order {orders[-1]:.2f})", fontsize=12)
a1.set_xticks(h)
a1.set_xticklabels([f"{v:g}" for v in h], fontsize=9.5)
a1.minorticks_off()
a1.legend(loc="upper left", fontsize=9.5)

# Right: empirical Var(estimator) x n vs n.
a2.semilogx(n, var_n, "o-", color=GREEN, ms=6, label=r"empirical Var(estimator)$\,\times\, n$")
a2.axhline(var_n.mean(), color=GRAY, ls="--", lw=1.4, label=f"mean = {var_n.mean():.3f}")
a2.set_xlabel(r"sample count $n$")
a2.set_ylabel(r"Var(estimator)$\,\times\, n$")
a2.set_title(r"Monte Carlo: Var $\times\, n$ stays flat", fontsize=12)
a2.set_ylim(0.0, 0.16)
a2.legend(loc="lower center", fontsize=9.5)

fig.tight_layout(w_pad=2.0)
name = "F3.1_rk4_and_mc_verification.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
