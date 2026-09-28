"""Figure 14.1: reliability diagram of the ten-member ensemble (Chapter 14, section 14.3), from
results/chapter14/calibration_check_rerun.json.

Usage:
    python code/figures/make_fig14_1.py
Output: figures/chapter14/F14.1_reliability_diagram.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json, os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter14"
FIG_DIR = REPO_ROOT / "figures" / "chapter14"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, GRAY = "#0072B2", "#555555"
d = json.load(open(RESULTS / "calibration_check_rerun.json"))
lev = d["levels"]
fig, a = plt.subplots(figsize=(4.6, 4.2))
a.plot([0.4, 1.0], [0.4, 1.0], color=GRAY, lw=1.0, ls="--", label="perfect calibration")
for key, lab, st in (("eval_seed_999", "200 points (set 1)", dict(color=BLUE, alpha=0.45, marker="^", ls=":")),
                     ("eval_seed_1999", "200 points (set 2)", dict(color=BLUE, alpha=0.45, marker="v", ls=":")),
                     ("eval_seed_2999", "2000 points", dict(color=BLUE, marker="o", ls="-"))):
    a.plot(lev, [d[key]["coverage"][str(q)] for q in lev], label=lab, markersize=6, **st)
a.set_xlim(0.45, 1.0); a.set_ylim(0.45, 1.0); a.set_aspect("equal")
a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage")
a.set_title("Ten-member PINN ensemble, heat equation")
a.legend(fontsize=8, loc="upper left")
name = "F14.1_reliability_diagram.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
