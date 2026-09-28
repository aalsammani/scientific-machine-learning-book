"""Figure 12.2: Fisher-information spectrum for 15 and 441 observation nodes (left; Chapter 12,
section 12.4) and recovery of c_1 with the finite-difference solver and two DeepONet surrogates over
200 noise realizations (right; section 12.6).
Data: results/chapter12/darcy_inversion_dense.json (surrogates identified by the substrings
"normalized" and "firstedition" in their stored paths).

Usage:
    python code/figures/make_fig12_2.py
Output: figures/chapter12/F12.2_identifiability_and_contamination.png and .pdf (reads stored results only; runtime a few seconds).
Colors: Okabe-Ito order (colorblind-safe), with line style / marker as secondary encoding.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter12"
FIG_DIR = REPO_ROOT / "figures" / "chapter12"

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


inv = json.load(open(RESULTS / "darcy_inversion_dense.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.0))
ev = inv["fim_sparse15"]["eigenvalues"]; evd = inv["fim_dense441"]["eigenvalues"]
a.semilogy(range(1, 6), ev, marker="o", color=BLUE, label=f"15 nodes ($\\kappa$ = {ev[-1]/ev[0]:.0f})")
a.semilogy(range(1, 6), evd, marker="s", ls="--", color=VERM, label=f"441 nodes ($\\kappa$ = {evd[-1]/evd[0]:.0f})")
a.set_xticks(range(1, 6)); a.set_xlabel("eigenvalue index (ascending)"); a.set_ylabel("Fisher information eigenvalue")
a.set_title("Fisher information spectrum, five coefficients"); a.legend(fontsize=9.5)
cl = inv["classical"]; su = inv["surrogates"]
keys = [k for k in su]
norm_k = [k for k in keys if "normalized" in k][0]; fe_k = [k for k in keys if "firstedition" in k][0]
labels = ["finite-difference\nsolver",
          f"DeepONet\n(field error\n{100*su[norm_k]['field_relerr_at_c_true']:.2f}%)",
          f"DeepONet\n(field error\n{100*su[fe_k]['field_relerr_at_c_true']:.2f}%)"]
rm = [cl["rmse"], su[norm_k]["rmse"], su[fe_k]["rmse"]]
bias = [abs(cl["bias"]), abs(su[norm_k]["bias"]), abs(su[fe_k]["bias"])]
x = np.arange(3)
b.bar(x - 0.18, rm, width=0.34, color=BLUE, label="root-mean-square error")
b.bar(x + 0.18, bias, width=0.34, color=ORANGE, label="|bias|")
b.axhline(inv["cramer_rao_std_c1_dense"], color=GRAY, ls="--", lw=1.2, label="Cramér–Rao bound (std)")
b.set_xticks(x); b.set_xticklabels(labels, fontsize=9.5); b.set_ylabel("error in recovered $c_1$")
b.set_title("Recovery of $c_1$ from 441 noisy observations\n(200 noise realizations)")
b.legend(fontsize=9.5, loc="upper left")
save(fig, "F12.2_identifiability_and_contamination.png")
