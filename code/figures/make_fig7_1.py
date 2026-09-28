"""Figure 7.1: residual NTK spectrum at initialization (left; Chapter 7, section 7.4) and loss
balancing (right; section 7.5).

Data (stored results, not recomputed here):
  results/chapter07/ntk_spectrum.json        (eigenvalues of the residual NTK, seed 0)
  results/chapter07/lossbalance_{fixed,gradnorm}_seed{0,1,2}.json
Only the twelve largest eigenvalues are plotted: the stored tail of the spectrum lies at the
rounding level of double precision (many entries are ~1e-13 or negative) and carries no information.

Usage:
    python code/figures/make_fig7_1.py
Output: figures/chapter07/F7.1_ntk_and_lossbalance.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter07"
FIG_DIR = REPO_ROOT / "figures" / "chapter07"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

RES = str(RESULTS)
with open(f"{RES}/ntk_spectrum.json") as f:
    eig = np.array(json.load(f)["eigvals"], dtype=float)
eig = np.sort(eig)[::-1][:12]
assert np.all(eig > 0), "twelve largest eigenvalues must be positive"

seeds = [0, 1, 2]
err = {}
for mode in ("fixed", "gradnorm"):
    err[mode] = []
    for s in seeds:
        with open(f"{RES}/lossbalance_{mode}_seed{s}.json") as f:
            err[mode].append(100.0 * json.load(f)["rel_l2"])

fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)

# Left: spectrum
rank = np.arange(1, 13)
axl.semilogy(rank, eig, "o-", color=BLUE, markersize=5.5)
axl.set_xticks(rank[::1])
axl.set_xlabel("eigenvalue rank")
axl.set_ylabel("eigenvalue")
axl.set_title("Residual NTK spectrum at initialization", fontsize=11.5)
axl.set_ylim(3e-10, 3e4)

# Right: grouped bars
x = np.arange(len(seeds))
w = 0.36
b1 = axr.bar(x - w / 2, err["fixed"], w, color=BLUE, label="fixed (underweighted)", zorder=3)
b2 = axr.bar(x + w / 2, err["gradnorm"], w, color=VERM, label="gradient-magnitude adaptive", zorder=3)
for bars in (b1, b2):
    for r in bars:
        v = r.get_height()
        lab = f"{v:.2f}%" if v < 1 else f"{v:.1f}%"
        axr.text(r.get_x() + r.get_width() / 2, v + 0.25, lab, ha="center", va="bottom",
                 fontsize=9.5, color="0.2")
axr.set_xticks(x)
axr.set_xticklabels([f"seed {s}" for s in seeds])
axr.set_ylabel(r"relative $L^2$ error (%)")
axr.set_ylim(0, 19.5)
axr.grid(axis="x", visible=False)
axr.set_title("Loss balancing, three seeds", fontsize=11.5)
axr.legend(loc="upper left", fontsize=9.5, ncol=1)

name = "F7.1_ntk_and_lossbalance.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
print("eigvals top12:", ", ".join(f"{v:.4g}" for v in eig))
print("orders spanned (top12): %.2f" % np.log10(eig[0] / eig[-1]))
print("fixed %:", [round(v, 2) for v in err["fixed"]], "gradnorm %:", [round(v, 2) for v in err["gradnorm"]])
