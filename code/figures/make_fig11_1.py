"""Figure 11.1: SINDy on Lorenz-63 under increasing noise (Chapter 11, section 11.5), from
results/chapter11/sindy_noise_sweep.json.

Usage:
    python code/figures/make_fig11_1.py
Output: figures/chapter11/F11.1_sindy_noise_sweep.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json, os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter11"
FIG_DIR = REPO_ROOT / "figures" / "chapter11"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#555555"
res = json.load(open(RESULTS / "sindy_noise_sweep.json"))
rows = res["levels"]
lev = np.array([100 * r["noise_rel"] for r in rows])
pos = np.arange(len(lev))
med = np.array([r["spurious_median"] for r in rows])
lo = np.array([r["spurious_min"] for r in rows])
hi = np.array([r["spurious_max"] for r in rows])
miss = np.array([100 * r["frac_any_missing"] for r in rows])

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.7))
for p, r in zip(pos, rows):
    s = np.array(r["spurious"])
    jit = (np.random.default_rng(0).random(len(s)) - 0.5) * 0.25
    a.plot(p + jit, s, ".", color=BLUE, alpha=0.25, ms=4)
a.fill_between(pos, lo, hi, color=BLUE, alpha=0.12, lw=0)
a.plot(pos, med, "o-", color=BLUE, label="median over realizations")
a.axhline(23, color=GRAY, lw=0.8, ls="--")
a.text(0.05, 23.4, "23 coefficients that should be zero", fontsize=9.5, color=GRAY)
a.set_xticks(pos, [f"{v:g}%" for v in lev])
a.set_ylim(-0.5, 25.5)
a.set_xlabel("relative measurement noise"); a.set_ylabel("spurious terms (all three equations)")
a.set_title("False positives")
a.legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))

b.set_axisbelow(True)
b.bar(pos, miss, width=0.6, color=VERM)
for p, m in zip(pos, miss):
    b.text(p, m + 0.8, f"{m:.0f}%", ha="center", fontsize=9.5)
b.set_xticks(pos, [f"{v:g}%" for v in lev])
b.set_ylim(0, 32)
b.set_xlabel("relative measurement noise"); b.set_ylabel("realizations missing a true term (%)")
b.set_title("False negatives")
fig.tight_layout()
name = "F11.1_sindy_noise_sweep.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
