"""Figure 1.5: frequency-mode learning curves for the physics-informed network of Running Example A
(Chapter 1, section 1.9).

Data: the per-seed training histories results/chapter01/ch01-e03_pinn_seed{0..4}_history.json
(JAX implementation, Adam for 5,000 iterations followed by L-BFGS, logged every 100 iterations).
Each record holds the fitted sine coefficients b_k(t) of the network output for k = 1..8 at several
times; the "t0.0" entry is used here. Exact coefficients at t = 0: b_1 = 1.0, b_5 = 0.5.
Relative error of mode k = |b_k - b_k^exact| / |b_k^exact|; median over the 5 seeds, min-max shading.
The medians are cross-checked against results/chapter01/ch01-f01_spectral_modes_summary.json.

Usage:
    python code/figures/make_fig1_5.py
Output: figures/chapter01/F1.5_spectral_bias.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json, os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter01"
FIG_DIR = REPO_ROOT / "figures" / "chapter01"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

R = str(RESULTS)
NAME = "F1.5_spectral_bias.png"
EXACT = {1: 1.0, 5: 0.5}
ADAM_ITERS = 5000

# ---- load per-seed histories -------------------------------------------------------------
per_seed = []
for s in range(5):
    h = json.load(open(f"{R}/ch01-e03_pinn_seed{s}_history.json"))
    recs = h if isinstance(h, list) else next(v for v in h.values() if isinstance(v, list))
    d = {}
    for r in recs:
        if r.get("phase") == "final":
            continue
        c = r["modes"]["t0.0"]
        d[int(r["iter"])] = {k: abs(c[k - 1] - a) / abs(a) for k, a in EXACT.items()}
    per_seed.append(d)
common = sorted(set.intersection(*[set(d) for d in per_seed]))
print("iterations per seed:", [len(d) for d in per_seed], "common:", len(common), common[0], common[-1])
it = np.array(common)
E = {k: np.array([[d[i][k] for i in common] for d in per_seed]) for k in EXACT}  # (seed, iter)
med = {k: np.median(E[k], axis=0) for k in EXACT}
lo = {k: E[k].min(axis=0) for k in EXACT}
hi = {k: E[k].max(axis=0) for k in EXACT}

# ---- cross-check against stored summary, and against the caption / text claims -------------
summ = json.load(open(f"{R}/ch01-f01_spectral_modes_summary.json"))["ch01-e03_pinn"]
for i, m1, m5 in zip(summ["iters"], summ["median_rel_err_mode1_t0"], summ["median_rel_err_mode5_t0"]):
    if i in common:
        j = common.index(i)
        print(f"iter {i:5d}: summary m1={m1:.4g} m5={m5:.4g} | recomputed m1={med[1][j]:.4g} m5={med[5][j]:.4g}")
adam = (it > 0) & (it < ADAM_ITERS)
print("median mode-5 error over Adam phase (iter 100..4900): min %.3f max %.3f"
      % (med[5][adam].min(), med[5][adam].max()))
print("per-seed mode-5 error over Adam phase: min %.3f max %.3f" % (E[5][:, adam].min(), E[5][:, adam].max()))
print("first iteration with median mode-1 error < 0.1:", it[np.argmax(med[1] < 0.1)])
print("first iteration with every seed's mode-1 error < 0.1:", it[np.argmax(E[1].max(axis=0) < 0.1)])
print("first iteration with median mode-5 error < 0.1:", it[np.argmax(med[5] < 0.1)])
print("ratio median(m5/m1) over Adam phase:", np.median((E[5] / E[1])[:, adam]))

# ---- figure ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.1))
for k, col, ls in ((1, BLUE, "-"), (5, VERM, "-")):
    ax.fill_between(it, lo[k], hi[k], color=col, alpha=0.18, linewidth=0)
    ax.plot(it, med[k], color=col, ls=ls, label=f"mode $k={k}$, median")
ax.set_yscale("log")
ax.set_xlim(0, it[-1])
ymin = 10 ** np.floor(np.log10(min(lo[1].min(), lo[5].min())))
ax.set_ylim(ymin, 25)
ax.axvline(ADAM_ITERS, color=GRAY, ls=":", lw=1.3, zorder=1)
# phase labels sit above all data, on either side of the transition line (never crossed by it)
ytxt = 7
ax.text(ADAM_ITERS - 250, ytxt, "Adam", ha="right", va="center", color=GRAY, fontsize=10.5)
ax.text(ADAM_ITERS + 250, ytxt, "L-BFGS", ha="left", va="center", color=GRAY, fontsize=10.5)
ax.set_yticks([10.0 ** e for e in range(int(np.log10(ymin)), 1)])
ax.set_xlabel("training iteration")
ax.set_ylabel("relative error of Fourier\ncoefficient at $t=0$")
ax.set_yticks([], minor=True)
ax.legend(fontsize=10, loc="lower left", borderaxespad=0.3, frameon=True, facecolor="white",
          edgecolor="none", framealpha=1.0)

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / NAME, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{NAME[:-4]}.pdf", bbox_inches="tight")
print("wrote", NAME)
