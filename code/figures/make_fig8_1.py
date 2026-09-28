"""Figure 8.1: Deep Ritz with fixed versus resampled collocation (Chapter 8, section 8.3).

Data (stored results, not recomputed here):
  results/chapter08/deepritz_{fixed,resample}_seed{0,1,2}.json
Left: relative L2 error (log scale). Right: final empirical energy against the true minimum
-pi^2/4 (dashed); an inset zooms on the resampled runs, which lie just above the minimum.

Usage:
    python code/figures/make_fig8_1.py
Output: figures/chapter08/F8.1_deepritz_collocation.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter08"
FIG_DIR = REPO_ROOT / "figures" / "chapter08"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

RES = str(RESULTS)
seeds = [0, 1, 2]
data = {}
for mode in ("fixed", "resample"):
    data[mode] = []
    for s in seeds:
        with open(f"{RES}/deepritz_{mode}_seed{s}.json") as f:
            data[mode].append(json.load(f))
emin = data["fixed"][0]["true_min_energy"]
assert abs(emin + np.pi ** 2 / 4) < 1e-9

err = {m: [100.0 * d["rel_l2"] for d in data[m]] for m in data}
en = {m: [d["final_energy"] for d in data[m]] for m in data}
COL = {"fixed": BLUE, "resample": VERM}
LAB = {"fixed": "fixed collocation", "resample": "resampled collocation"}

fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)

# Left: error bars, log scale
x = np.arange(len(seeds))
w = 0.36
for k, mode in enumerate(("fixed", "resample")):
    bars = axl.bar(x + (k - 0.5) * w, err[mode], w, color=COL[mode], label=LAB[mode], zorder=3)
    for r in bars:
        v = r.get_height()
        lab = f"{v:.0f}%" if v >= 10 else (f"{v:.1f}%" if v >= 1 else f"{v:.2f}%")
        axl.text(r.get_x() + r.get_width() / 2, v * 1.12, lab, ha="center", va="bottom",
                 fontsize=9.5, color="0.2")
axl.set_yscale("log")
axl.set_ylim(0.2, 2e4)
axl.set_xticks(x)
axl.set_xticklabels([f"seed {s}" for s in seeds])
axl.set_ylabel(r"relative $L^2$ error (%)")
axl.grid(axis="x", visible=False)
axl.set_title("Deep Ritz: fixed vs. resampled", fontsize=11.5)
axl.legend(loc="upper left", fontsize=9.5, ncol=2, columnspacing=1.0, handlelength=1.4)

# Right: final energy vs true minimum
off = np.array([-0.08, 0.0, 0.08])
pos = {"fixed": 0.0, "resample": 1.0}
for mode in ("fixed", "resample"):
    axr.scatter(pos[mode] + off, en[mode], s=48, color=COL[mode], zorder=4,
                edgecolor="white", linewidth=0.8)
axr.axhline(emin, color="0.15", ls="--", lw=1.3, zorder=2)
axr.text(0.5, emin + 0.7, r"true minimum $-\pi^2/4 \approx %.3f$" % emin, ha="center",
         va="bottom", fontsize=9.5, color="0.15")
for s, e in zip(seeds, en["fixed"]):
    axr.text(pos["fixed"] + off[s] + 0.1, e, f"seed {s}", va="center", fontsize=9.5, color="0.3")
axr.set_xticks([0, 1])
axr.set_xticklabels(["fixed", "resampled"])
axr.set_xlim(-0.35, 1.35)
axr.set_ylim(-26.5, 1.0)
axr.grid(axis="x", visible=False)
axr.set_ylabel("final empirical energy")
axr.set_title("Empirical energy vs. true minimum", fontsize=11.5)

# Inset: zoom on the resampled runs
ins = axr.inset_axes([0.52, 0.10, 0.44, 0.44])
ins.scatter(off, en["resample"], s=40, color=VERM, zorder=4, edgecolor="white", linewidth=0.8)
ins.axhline(emin, color="0.15", ls="--", lw=1.2)
for s, xo, e in zip(seeds, off, en["resample"]):
    ins.text(xo, e + 0.018, f"seed {s}", ha="center", va="bottom", fontsize=9.5, color="0.3")
ins.set_xlim(-0.2, 0.2)
ins.set_ylim(-2.52, -2.13)
ins.set_xticks([])
ins.set_yticks([-2.5, -2.4, -2.3, -2.2])
ins.tick_params(labelsize=9.5)
ins.set_title("resampled (zoom)", fontsize=9.5, pad=3)
for sp in ("top", "right"):
    ins.spines[sp].set_visible(True)
ins.set_facecolor("white")

name = "F8.1_deepritz_collocation.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
print("fixed err %:", [round(v, 2) for v in err["fixed"]], "resample err %:", [round(v, 3) for v in err["resample"]])
print("fixed energy:", [round(v, 3) for v in en["fixed"]], "resample energy:", [round(v, 4) for v in en["resample"]])
print("resample gap above min:", [round(v - emin, 4) for v in en["resample"]])
print("error ratio fixed/resample:", [round(a / b, 1) for a, b in zip(err["fixed"], err["resample"])])
