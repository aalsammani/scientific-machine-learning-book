"""Figure 14.2: in-family and out-of-family test error of the Darcy DeepONet (five seeds), with the
exact in-family operator and the training-mean field on the shifted set (Chapter 14, section 14.5).
Data: results/chapter14/deeponet_distribution_shift.json.

Usage:
    python code/figures/make_fig14_2.py
Output: figures/chapter14/F14.2_distribution_shift.png and .pdf (reads stored results only; runtime a few seconds).
Colors: Okabe-Ito order (colorblind-safe), with line style / marker as secondary encoding.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter14"
FIG_DIR = REPO_ROOT / "figures" / "chapter14"

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


sh = json.load(open(RESULTS / "deeponet_distribution_shift.json"))
fig, a = plt.subplots(figsize=(6.6, 4.0))
seeds = sorted(sh["deeponet_normalized_n200"], key=int)
inm = [100*sh["deeponet_normalized_n200"][s]["in_family"]["median"] for s in seeds]
outm = [100*sh["deeponet_normalized_n200"][s]["out_of_family"]["median"] for s in seeds]
x = np.arange(len(seeds))
a.bar(x - 0.2, inm, width=0.38, color=BLUE, label="in family (modes 1–5)")
a.bar(x + 0.2, outm, width=0.38, color=VERM, label="out of family (modes 6–10)")
a.axhline(100*sh["exact_in_family_surrogate_on_out_of_family"]["median"], color=GRAY, ls="--", lw=1.2,
          label="exact in-family operator on the shifted set")
a.axhline(100*sh["mean_predictor"]["out_of_family"]["median"], color=GREEN, ls=":", lw=1.6,
          label="training-mean field on the shifted set")
a.set_xticks(x); a.set_xticklabels([f"seed {s}" for s in seeds]); a.set_ylabel("median relative $L^2$ error (%)")
a.set_title("DeepONet ($n=200$): in-family and out-of-family test error")
a.legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(0, 1.0), frameon=True, facecolor="white",
         edgecolor="none", framealpha=1.0); a.set_axisbelow(True); a.set_ylim(0, 2.6)
save(fig, "F14.2_distribution_shift.png")
