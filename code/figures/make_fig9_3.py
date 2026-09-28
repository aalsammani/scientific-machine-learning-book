"""Figure 9.3: Darcy DeepONet test error against training-set size under both output protocols, with
the non-learned baselines (left), and one training solution (right) (Chapter 9, section 9.6).
Data: results/chapter09/deeponet_sweep_summary.json, regeneration_prepare.json, darcy_dataset.npz.
The legend's "unscaled"/"scaled outputs" are the raw and standardized output protocols of the text.

Usage:
    python code/figures/make_fig9_3.py
Output: figures/chapter09/F9.3_darcy_dataset_and_cost.png and .pdf (reads stored results only; runtime a few seconds).
Colors: Okabe-Ito order (colorblind-safe), with line style / marker as secondary encoding.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter09"
FIG_DIR = REPO_ROOT / "figures" / "chapter09"

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)


sw = json.load(open(f"{RESULTS}/deeponet_sweep_summary.json"))
prep = json.load(open(f"{RESULTS}/regeneration_prepare.json"))
ns = [25, 50, 100, 200, 400]
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
for key, col, mk, lab in (("firstedition_protocol", VERM, "o", "DeepONet, unscaled outputs"),
                          ("normalized", BLUE, "s", "DeepONet, scaled outputs")):
    med = np.array([100*np.median(sw[key][str(n)]["medians_by_seed"]) for n in ns])
    lo = np.array([100*np.min(sw[key][str(n)]["medians_by_seed"]) for n in ns])
    hi = np.array([100*np.max(sw[key][str(n)]["medians_by_seed"]) for n in ns])
    a.fill_between(ns, lo, hi, color=col, alpha=0.18, linewidth=0)
    a.plot(ns, med, marker=mk, color=col, label=lab, markersize=6)
bl = prep["baselines_by_n_train"]
a.plot(ns, [100*bl[str(n)]["mean_predictor_median"] for n in ns], ls="--", color=GRAY, label="training-mean field")
a.plot(ns, [100*bl[str(n)]["affine_ls_median"] for n in ns], ls="-.", marker="^", color=GREEN, markersize=6, label="affine least squares in $c$")
a.plot(ns, [100*bl[str(n)]["quadratic_ls_median"] for n in ns], ls=":", marker="D", color=PURPLE, markersize=5, label="quadratic least squares in $c$")
a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel("training pairs $n$"); a.set_ylabel("median relative $L^2$ test error (%)")
a.set_title("Test error against training-set size")
a.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
a.set_ylim(1e-3, 3e3)
d = np.load(f"{RESULTS}/darcy_dataset.npz")
im = b.imshow(d["U"][0].T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
fig.colorbar(im, ax=b, fraction=0.046); b.set_xlabel("$x$"); b.set_ylabel("$y$"); b.grid(False)
b.set_title("One training solution $u(x,y)$")
save(fig, "F9.3_darcy_dataset_and_cost.png")
