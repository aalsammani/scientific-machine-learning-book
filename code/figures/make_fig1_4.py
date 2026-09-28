"""Figure 1.4: pointwise squared residual versus pointwise squared solution error, PyTorch seed-0
physics-informed network of Running Example A (Chapter 1, section 1.8.5).

Data: results/chapter01/fig15_fields.npz (written by code/chapter01/residual_error_fields.py; arrays
X, T, residual = N[u_theta] = u_t - nu u_xx, error = u_theta - u, on a 101 x 101 grid over
[0,1] x [0,1]). The color scale is logarithmic (ten decades below each field's maximum; smaller
values are shown at the floor color) so that the structure of each field away from its peak is
visible; the location of each field's maximum is marked.

Usage:
    python code/figures/make_fig1_4.py
Output: figures/chapter01/F1.4_residual_vs_error.png and .pdf (reads stored results only; runtime a few seconds).
"""
import os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter01"
FIG_DIR = REPO_ROOT / "figures" / "chapter01"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#555555"

NAME = "F1.4_residual_vs_error.png"
d = np.load(RESULTS / "fig15_fields.npz")
X, T = d["X"], d["T"]
R2, E2 = d["residual"] ** 2, d["error"] ** 2

# diagnostics: location of each maximum, and the relative L2 error on this grid
for lab, F in (("squared residual", R2), ("squared error", E2)):
    i, j = np.unravel_index(np.argmax(F), F.shape)
    print(f"{lab}: max {F.max():.3e} at x={X[i, j]:.3f}, t={T[i, j]:.3f}; min {F.min():.3e}")
# consistency check against the stored seed-0 result (rel. L2 = 0.22% on the 201 x 101 grid);
# exact solution as in code/examples/exA_heat.py (nu = 0.5, modes (1, 1.0), (5, 0.5))
nu = 0.5
U = sum(a * np.exp(-nu * (k * np.pi) ** 2 * T) * np.sin(k * np.pi * X) for k, a in ((1, 1.0), (5, 0.5)))
print("relative L2 error of u_theta on this 101 x 101 grid: %.4f%%" % (100 * np.sqrt(E2.sum() / (U ** 2).sum())))

DECADES = 10
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained")
panels = ((axes[0], R2, r"Squared residual $(\mathcal{N}[u_\theta])^2$"),
          (axes[1], E2, r"Squared error $(u_\theta-u)^2$"))
outline = [pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()]
for ax, F, title in panels:
    vmax = F.max(); vmin = vmax * 10.0 ** (-DECADES)
    print(f"{title}: fraction of grid points below the color floor: {(F < vmin).mean():.3f}")
    im = ax.pcolormesh(X, T, np.maximum(F, vmin), shading="nearest", cmap="viridis",
                       norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
    i, j = np.unravel_index(np.argmax(F), F.shape)
    h, = ax.plot(X[i, j], T[i, j], marker="o", ms=10, mfc="none", mec="white", mew=1.8, ls="none",
                 clip_on=False, zorder=5, path_effects=outline)
    fig.colorbar(im, ax=ax, pad=0.02)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(False)
fig.legend([h], ["location of each field's maximum"], loc="outside lower center",
           fontsize=10)

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / NAME, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{NAME[:-4]}.pdf", bbox_inches="tight")
print("wrote", NAME)
