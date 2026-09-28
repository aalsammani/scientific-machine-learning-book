"""Figure 12.1: the omega_r sweep for joint state-parameter estimation (Chapter 12, section 12.5), from
results/chapter12/omega_r_sweep.json. The omega_r = 0 run is drawn at a separate position labeled
"0", with a break in the log axis.

Usage:
    python code/figures/make_fig12_1.py
Output: figures/chapter12/F12.1_omega_r_sweep.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json, os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter12"
FIG_DIR = REPO_ROOT / "figures" / "chapter12"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#555555"
d = json.load(open(RESULTS / "omega_r_sweep.json"))
rows = d["results"]
zero = [r for r in rows if r["omega_r"] == 0.0][0]
pos = [r for r in rows if r["omega_r"] > 0]
X0 = 1e-5  # plotting position of omega_r = 0
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot([r["omega_r"] for r in pos], [r["nu_hat"] for r in pos], marker="o", color=VERM, label=r"recovered $\hat\nu$")
ax.plot([X0], [zero["nu_hat"]], marker="s", color=VERM, markersize=7, linestyle="none")
ax.axhline(d["nu_true"], color=GRAY, ls="--", lw=1.2, label=r"true $\nu = 0.5$")
ax.axhline(0.2, color=BLUE, ls=":", lw=1.2, label=r"initial value $\nu = 0.2$")
ax.set_xscale("log")
ax.set_xticks([X0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
ax.set_xticklabels(["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
# axis break marks between the omega_r = 0 position and 1e-4
for xb in (2.4e-5, 2.9e-5):
    ax.plot([xb, xb * 1.08], [-0.02, 0.02], color="k", lw=1.0, clip_on=False, transform=ax.get_xaxis_transform())
ax.set_ylim(0.1, 0.55)
ax.set_xlabel(r"residual weight $\omega_r$")
ax.set_ylabel(r"recovered $\hat\nu$")
ax.set_title(r"Joint estimation of $\nu$ against the residual weight")
ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0.2, 0.3))
fig.tight_layout()
name = "F12.1_omega_r_sweep.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
