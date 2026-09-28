"""Figure 9.2: matched-seed mode-truncation ablation, four versus twelve retained Fourier modes
(Chapter 9, section 9.5).

Data (stored results, not recomputed here):
  results/chapter09/fno/matched_seed_ablation.json  (seeds 1-4, fresh training runs)
Plotted: median relative L2 error over the test set.

Usage:
    python code/figures/make_fig9_2.py
Output: figures/chapter09/F9.2_matched_seed_ablation.png and .pdf (reads stored results only; runtime a few seconds).
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

with open(RESULTS / "fno" / "matched_seed_ablation.json") as f:
    d = json.load(f)
seeds = [r["seed"] for r in d["results"]]
e4 = np.array([100.0 * r["modes4"]["median_rel_l2"] for r in d["results"]])
e12 = np.array([100.0 * r["modes12"]["median_rel_l2"] for r in d["results"]])

fig, ax = plt.subplots(figsize=(5.8, 3.3), constrained_layout=True)
x = np.arange(len(seeds))
w = 0.36
b4 = ax.bar(x - w / 2, e4, w, color=ORANGE, label="4 retained modes", zorder=3)
b12 = ax.bar(x + w / 2, e12, w, color=BLUE, label="12 retained modes", zorder=3)
for bars in (b4, b12):
    for r in bars:
        v = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, v + 0.07, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=9.5, color="0.2")
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel(r"relative $L^2$ error (%)")
ax.set_ylim(0, 5.6)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=9.5, ncol=2, columnspacing=1.2, handlelength=1.4)

name = "F9.2_matched_seed_ablation.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
diff = e4 - e12
print("wrote", name)
print("4-mode %:", np.round(e4, 3), "12-mode %:", np.round(e12, 3))
print("paired diff pp:", np.round(diff, 3), "mean %.3f sd %.3f" % (diff.mean(), diff.std(ddof=1)))
print("mean 4-mode %.3f  mean 12-mode %.3f" % (e4.mean(), e12.mean()))
