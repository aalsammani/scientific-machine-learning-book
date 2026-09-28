"""Figure 4.1: gradient-descent bounds (Theorems 4.4 and 4.5) versus observed behavior (Chapter 4,
section 4.5). Plots the stored result results/chapter04/gd_convergence_verification.json.
Notation follows section 4.5: objective J, iterates theta_k, minimizer theta^*, smoothness L_nabla.

Usage:
    python code/figures/make_fig4_1.py
Output: figures/chapter04/F4.1_gd_convergence_verification.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter04"
FIG_DIR = REPO_ROOT / "figures" / "chapter04"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

with open(RESULTS / "gd_convergence_verification.json") as f:
    R = json.load(f)
L, mu = float(R["L"]), float(R["mu"])

K = np.array([r["K"] for r in R["convex_check"]], dtype=float)
gmin = np.array([r["min_grad_sq"] for r in R["convex_check"]], dtype=float)
gbound = np.array([r["bound"] for r in R["convex_check"]], dtype=float)
k = np.array([r["k"] for r in R["strongly_convex_check"]], dtype=float)
dist = np.array([r["actual"] for r in R["strongly_convex_check"]], dtype=float)
dbound = np.array([r["predicted"] for r in R["strongly_convex_check"]], dtype=float)

# Consistency checks of the stored bounds against the theorem formulas.
assert np.allclose(gbound * K, gbound[0] * K[0]), "Theorem 4.4 bound is not C/K"
assert np.allclose(dbound, dbound[0] * (1 - mu / L) ** k), "Theorem 4.5 bound is not (1-mu/L)^k d0"
print("ratio bound/observed (left):", [f"{v:.3g}" for v in gbound / gmin])
print("ratio bound/observed (right):", [f"{v:.3g}" for v in dbound / dist])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

a1.loglog(K, gbound, color=VERM, ls="--", marker="s", ms=4.5, label="Theorem 4.4 bound")
a1.loglog(K, gmin, "o-", color=BLUE, ms=6, label="observed")
a1.set_xlabel(r"iteration count $K$")
a1.set_ylabel(r"$\mathrm{min}_{k<K}\,\|\nabla\mathcal{J}(\theta_k)\|^2$")
a1.set_title("Gradient-norm rate (Theorem 4.4)", fontsize=12)
a1.legend(loc="lower left", fontsize=9.5)

a2.semilogy(k, dbound, color=VERM, ls="--", marker="s", ms=4.5, label="Theorem 4.5 bound")
a2.semilogy(k, dist, "o-", color=BLUE, ms=6, label="observed")
a2.set_xlabel(r"iteration $k$")
a2.set_ylabel(r"$\|\theta_k-\theta^*\|^2$")
a2.set_title("Strongly convex rate (Theorem 4.5)", fontsize=12)
a2.legend(loc="lower left", fontsize=9.5)

fig.tight_layout(w_pad=2.0)
name = "F4.1_gd_convergence_verification.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
