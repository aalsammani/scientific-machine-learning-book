"""Figure 1.3: the heat equation (Running Example A) three ways: classical solver, regression network,
physics-informed network (Chapter 1, section 1.8.3).

Left:   exact solution u(x,t) = e^{-nu pi^2 t} sin(pi x) + 1/2 e^{-25 nu pi^2 t} sin(5 pi x), nu = 0.5,
        on the 201 x 101 evaluation grid (formula as in code/reference_problems/exA_heat.py).
Center: relative L2 error by method; FD (one run) and five seeds per learned method, JAX and PyTorch
        shown side by side within each method; short bar = median.
Right:  training cost (Adam + L-BFGS wall time; FD solve time) against relative L2 error, one point
        per seed.
Data:   results/chapter01/ch01-e01_fd.json,
        ch01-e02_regression_seed{0..4}.json (JAX), ch01-e02_regression_seed{0..4}_pytorch.json,
        ch01-e03_pinn_seed{0..4}.json (JAX),       ch01-e03_pinn_seed{0..4}_pytorch.json.
Medians are cross-checked against results/chapter01/ch01_final_summary_both_frameworks.json.

Usage:
    python code/figures/make_fig1_3_comparison.py
Output: figures/chapter01/F1.3_three_way_comparison.png and .pdf (reads stored results only; runtime a few seconds).
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
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

R = str(RESULTS)
NAME = "F1.3_three_way_comparison.png"

# ---- data ------------------------------------------------------------------------------------
fd = json.load(open(f"{R}/ch01-e01_fd.json"))
def load(stem, suffix):
    rows = [json.load(open(f"{R}/{stem}_seed{s}{suffix}.json")) for s in range(5)]
    err = np.array([r["rel_l2"] for r in rows])
    cost = np.array([r["adam_s"] + r["lbfgs_s"] for r in rows])
    return err, cost
runs = {("Regression", "JAX"): load("ch01-e02_regression", ""),
        ("Regression", "PyTorch"): load("ch01-e02_regression", "_pytorch"),
        ("PINN", "JAX"): load("ch01-e03_pinn", ""),
        ("PINN", "PyTorch"): load("ch01-e03_pinn", "_pytorch")}
summ = json.load(open(f"{R}/ch01_final_summary_both_frameworks.json"))
skey = {("Regression", "JAX"): "regression NN (JAX)", ("Regression", "PyTorch"): "regression NN (PyTorch)",
        ("PINN", "JAX"): "PINN (JAX)", ("PINN", "PyTorch"): "PINN (PyTorch)"}
print(f"FD: rel L2 {fd['rel_l2']:.3e}, wall {fd['wall_s']:.3f} s")
for k, (e, c) in runs.items():
    s = summ.get(skey[k], {})
    print(f"{k}: rel L2 median {100*np.median(e):.3f}% [{100*e.min():.3f}, {100*e.max():.3f}]"
          f" (summary {100*s.get('rel_l2', {}).get('median', np.nan):.3f}%);"
          f" cost median {np.median(c):.1f} s [{c.min():.1f}, {c.max():.1f}]"
          f" (summary {s.get('train_s', {}).get('median', np.nan):.1f} s)")

# ---- encoding: color = method, marker = framework ----------------------------------------------
MCOL = {"FD": "black", "Regression": VERM, "PINN": BLUE}
FMK = {"JAX": "o", "PyTorch": "^"}
FOFF = {"JAX": -0.14, "PyTorch": 0.14}
XPOS = {"FD": 0, "Regression": 1, "PINN": 2}

fig = plt.figure(figsize=(9.4, 3.55), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.0])
a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2], sharey=a1)

# left: exact solution
nu = 0.5
x = np.linspace(0, 1, 201); t = np.linspace(0, 1, 101)
Xg, Tg = np.meshgrid(x, t)
U = np.exp(-nu * np.pi ** 2 * Tg) * np.sin(np.pi * Xg) + 0.5 * np.exp(-25 * nu * np.pi ** 2 * Tg) * np.sin(5 * np.pi * Xg)
print("exact solution range on grid: [%.3f, %.3f]" % (U.min(), U.max()))
im = a0.pcolormesh(Xg, Tg, U, shading="nearest", cmap="viridis", rasterized=True)
cb = fig.colorbar(im, ax=a0, pad=0.02, aspect=25, ticks=[0, 0.5, 1.0, 1.5])
a0.set_xlabel("$x$"); a0.set_ylabel("$t$"); a0.grid(False)
a0.set_xticks([0, 0.5, 1]); a0.set_yticks([0, 0.5, 1])
a0.set_title("Exact solution $u(x,t)$")

# center: accuracy by method
a1.axhline(fd["rel_l2"], color="black", ls="--", lw=1.0, zorder=1)
a1.plot([XPOS["FD"]], [fd["rel_l2"]], marker="s", ms=7, color=MCOL["FD"], ls="none", zorder=3)
for (m, f), (e, c) in runs.items():
    xp = XPOS[m] + FOFF[f]
    a1.plot(np.full(e.size, xp), e, marker=FMK[f], ms=6.5, color=MCOL[m], ls="none", alpha=0.9, zorder=3)
    a1.plot([xp - 0.1, xp + 0.1], [np.median(e)] * 2, color=MCOL[m], lw=2.2, zorder=2)
a1.set_yscale("log")
a1.set_xticks([0, 1, 2]); a1.set_xticklabels(["FD", "Regression", "PINN"])
a1.set_xlim(-0.45, 2.45)
a1.grid(axis="x", visible=False)
a1.set_ylabel("relative $L^2$ error")
a1.set_title("Accuracy by method")

# right: cost vs accuracy
a2.plot([fd["wall_s"]], [fd["rel_l2"]], marker="s", ms=7, color=MCOL["FD"], ls="none", zorder=3)
for (m, f), (e, c) in runs.items():
    a2.plot(c, e, marker=FMK[f], ms=6.5, color=MCOL[m], ls="none", alpha=0.9, zorder=3)
a2.set_xscale("log")
a2.set_xlim(0.02, 600)
a2.set_xlabel("training cost (s)")
a2.tick_params(labelleft=False)
a2.set_title("Cost vs. accuracy")
lo = min(fd["rel_l2"], min(e.min() for e, _ in runs.values()))
hi = max(e.max() for e, _ in runs.values())
a1.set_ylim(lo / 2.5, hi * 2.5)

# column-major fill with ncol=3: rows read (FD | Regression JAX | PINN JAX), (median | ... PyTorch)
handles = [Line2D([], [], marker="s", ms=7, color="black", ls="--", lw=1.0, label="FD, 81 points"),
           Line2D([], [], color=GRAY, lw=2.2, label="median of 5 seeds"),
           Line2D([], [], marker="o", ms=6.5, color=VERM, ls="none", label="Regression, JAX"),
           Line2D([], [], marker="^", ms=6.5, color=VERM, ls="none", label="Regression, PyTorch"),
           Line2D([], [], marker="o", ms=6.5, color=BLUE, ls="none", label="PINN, JAX"),
           Line2D([], [], marker="^", ms=6.5, color=BLUE, ls="none", label="PINN, PyTorch")]
fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10, columnspacing=1.6,
           handletextpad=0.5)

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / NAME, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{NAME[:-4]}.pdf", bbox_inches="tight")
print("wrote", NAME)
