"""Figure 9.1: FNO resolution transfer (left; Chapter 9, section 9.4), aliasing sweep (center;
section 9.3) and failure modes (right; section 9.5).

Data (stored results, not recomputed here):
  results/chapter09/fno/resolution_transfer.json  (five seeds, resolutions 32-256)
  results/chapter09/fno/failure_ood.json          (aliasing sweep 8-64; shifted-input and low-mode errors)
All plotted errors are median relative L2 errors over the test set.

Usage:
    python code/figures/make_fig9_1.py
Output: figures/chapter09/F9.1_fno_resolution_and_failure.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter09"
FIG_DIR = REPO_ROOT / "figures" / "chapter09"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"
SEED_COLORS = [BLUE, VERM, GREEN, ORANGE, PURPLE]

RES = str(RESULTS / "fno")
with open(f"{RES}/resolution_transfer.json") as f:
    rt = json.load(f)
with open(f"{RES}/failure_ood.json") as f:
    fo = json.load(f)

fig, (axl, axc, axr) = plt.subplots(1, 3, figsize=(9.4, 3.6), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.0, 1.0, 1.12]})

# Left: resolution transfer, five seeds
resol = rt["resolutions"]
for i, r in enumerate(rt["results"]):
    y = [100.0 * r["per_resolution"][str(n)]["median_rel_l2"] for n in resol]
    axl.plot(resol, y, "o-", color=SEED_COLORS[i], markersize=4.5, label=f"seed {r['seed']}")
axl.set_xscale("log", base=2)
axl.set_xticks(resol)
axl.set_xticklabels([str(n) for n in resol])
axl.minorticks_off()
axl.set_ylim(0.20, 0.40)
axl.set_xlabel("evaluation grid points")
axl.set_ylabel(r"relative $L^2$ error (%)")
axl.set_title("Resolution transfer", fontsize=11.5)
axl.legend(loc="upper center", fontsize=9.5, ncol=3, columnspacing=0.8, handlelength=1.3,
           handletextpad=0.4, borderaxespad=0.2)

# Center: aliasing sweep
sw = fo["aliasing_sweep"]
ns = sorted(int(k) for k in sw)
ya = [100.0 * sw[str(n)]["median_rel_l2"] for n in ns]
axc.semilogy(ns, ya, "o-", color=BLUE, markersize=5)
nyq = 16  # 8-mode input: Nyquist limit n/2 = 8 reached at n = 16
axc.axvline(nyq, color=GRAY, ls="--", lw=1.3)
axc.text(nyq + 1.5, 8.0, "Nyquist limit\nfor 8 modes\n($n = 16$)", fontsize=9.5, color="0.25",
         va="center", ha="left")
axc.set_xticks([8, 16, 32, 48, 64])
axc.set_xlabel("evaluation grid points")
axc.set_ylabel(r"relative $L^2$ error (%)")
axc.set_title("Aliasing below Nyquist", fontsize=11.5)
axc.set_ylim(0.15, 60)

# Right: three failure mechanisms, log scale with value labels
vals = [100.0 * fo["indist_relerr_res64"], 100.0 * fo["ood_relerr_modes9to16_res64"],
        100.0 * fo["lowmodes_indist_relerr_res64"]]
labels = ["in-distr.\nmodes 1–8", "shifted\nmodes 9–16",
          f"{fo['lowmodes_model_modes']}-mode FNO\nmodes 1–8"]
cols = [BLUE, VERM, ORANGE]
xb = np.arange(3)
bars = axr.bar(xb, vals, 0.62, color=cols, zorder=3)
axr.set_yscale("log")
axr.set_ylim(0.1, 3e4)
for r, v in zip(bars, vals):
    lab = f"{v:,.0f}%" if v >= 100 else (f"{v:.1f}%" if v >= 1 else f"{v:.2f}%")
    axr.text(r.get_x() + r.get_width() / 2, v * 1.35, lab, ha="center", va="bottom",
             fontsize=9.5, color="0.2")
axr.set_xticks(xb)
axr.set_xticklabels(labels, fontsize=9.5)
axr.grid(axis="x", visible=False)
axr.set_ylabel(r"relative $L^2$ error (%)")
axr.set_title("Shift vs. truncation", fontsize=11.5)

name = "F9.1_fno_resolution_and_failure.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
for r in rt["results"]:
    print("seed", r["seed"], [round(100 * r["per_resolution"][str(n)]["median_rel_l2"], 4) for n in resol])
print("aliasing sweep %:", dict(zip(ns, [round(v, 3) for v in ya])))
print("bars %:", [round(v, 3) for v in vals])
