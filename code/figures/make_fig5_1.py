"""Figure 5.1: Tikhonov regularization trade-off (left; Chapter 5, section 5.3) and two-compartment
non-identifiability (right; section 5.4). Plots the stored results
results/chapter05/tikhonov_discrepancy_verification.json and
results/chapter05/two_compartment_nonidentifiability.json.

Usage:
    python code/figures/make_fig5_1.py
Output: figures/chapter05/F5.1_regularization_and_identifiability.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter05"
FIG_DIR = REPO_ROOT / "figures" / "chapter05"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

with open(RESULTS / "tikhonov_discrepancy_verification.json") as f:
    TK = json.load(f)
with open(RESULTS / "two_compartment_nonidentifiability.json") as f:
    TC = json.load(f)

alphas = np.asarray(TK["alphas"], dtype=float)
errs = np.asarray(TK["errs"], dtype=float)
a_or, e_or = float(TK["oracle_best_alpha"]), float(TK["oracle_best_err"])
a_dp, e_dp = float(TK["discrepancy_alpha"]), float(TK["discrepancy_err"])
print(f"alpha grid {alphas[0]:.1e}..{alphas[-1]:.1e} ({alphas.size} pts); err range {errs.min():.4f}..{errs.max():.3f}")
print(f"oracle alpha={a_or:.4g} err={e_or:.4f}; discrepancy alpha={a_dp:.4g} err={e_dp:.4f}")

t = np.asarray(TC["t"], dtype=float)
yA = np.asarray(TC["yA"], dtype=float)
yB = np.asarray(TC["yB"], dtype=float)
maxdiff = float(TC["max_abs_diff_y"])
print(f"recomputed max|yA-yB| = {np.max(np.abs(yA - yB)):.3e}; stored = {maxdiff:.3e}")
A, B = TC["setA"], TC["setB"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

a1.loglog(alphas, errs, color=BLUE, label="reconstruction error")
YLO, YHI = 0.35, 25.0
a1.set_ylim(YLO, YHI)
# Vertical markers stop below the legend so they do not cross it.
a1.vlines(a_or, YLO, 3.0, color=GREEN, ls="--", lw=1.5)
a1.vlines(a_dp, YLO, 3.0, color=VERM, ls=":", lw=1.8)
a1.plot([a_or], [e_or], "o", color=GREEN, ms=6, zorder=4,
        label=f"oracle best (error {e_or:.3f})")
a1.plot([a_dp], [e_dp], "s", color=VERM, ms=6, zorder=4,
        label=f"discrepancy principle (error {e_dp:.3f})")
a1.set_xlabel(r"regularization parameter $\alpha$")
a1.set_ylabel("reconstruction error")
a1.set_title("Backward heat equation: Tikhonov trade-off", fontsize=12)
a1.legend(loc="upper center", bbox_to_anchor=(0.52, 1.0), fontsize=9.5)

fmt = lambda s: ", ".join(f"{s[key]:.3g}" for key in ("k10", "k12", "k21", "k20"))
a2.plot(t, yA, color=BLUE, lw=3.0, label=f"set A: ({fmt(A)})")
a2.plot(t, yB, color=VERM, ls="--", lw=1.8, label=f"set B: ({fmt(B)})")
a2.set_xlabel(r"time $t$")
a2.set_ylabel(r"observed output $x_1(t)$")
mant, expo = f"{maxdiff:.1e}".split("e")
a2.set_title("Two-compartment model: identical output\n"
             + rf"(max difference ${float(mant):.0f}\times10^{{{int(expo)}}}$)", fontsize=12)
a2.set_xlim(t[0], t[-1])
leg = a2.legend(loc="upper right", fontsize=9.5,
                title=r"$(k_{10}, k_{12}, k_{21}, k_{20})$", title_fontsize=9.5)

fig.tight_layout(w_pad=2.0)
name = "F5.1_regularization_and_identifiability.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
