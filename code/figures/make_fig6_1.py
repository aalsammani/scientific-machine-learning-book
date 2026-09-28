"""Figure 6.1: the physics-informed residual as two automatic-differentiation passes (schematic;
Chapter 6, section 6.4). No data.

Usage:
    python code/figures/make_fig6_1.py
Output: figures/chapter06/F6.1_ad_computational_graph.png and .pdf (reads stored results only; runtime a few seconds).
"""
import matplotlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter06"
FIG_DIR = REPO_ROOT / "figures" / "chapter06"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10.0, 3.6))
ax.set_xlim(0, 11.6); ax.set_ylim(0, 4); ax.axis("off")
BLUE, GREEN, VERM, PURPLE = "#0072B2", "#009E73", "#D55E00", "#CC79A7"


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="none", alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=10.5)
    return (x, y, w, h)


def arrow(p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3,
                                                    shrinkA=0, shrinkB=0))


inp = box(0.1, 2.5, 1.3, 0.9, r"$(x,t)$", BLUE)
net = box(2.0, 2.5, 2.1, 0.9, r"network $u_\theta(x,t)$", GREEN)
p1 = box(4.7, 2.5, 2.5, 0.9, r"$u_{\theta,t},\ \nabla u_\theta$" + "\n(first reverse pass)", VERM)
p2 = box(8.7, 2.5, 2.8, 0.9, r"$\Delta u_\theta$" + "\n(second, nested pass)", VERM)
res = box(4.7, 0.3, 6.8, 0.9, r"interior residual $\mathcal{N}[u_\theta] = u_{\theta,t} - \nu\,\Delta u_\theta$", PURPLE)
arrow((1.4, 2.95), (2.0, 2.95))
arrow((4.1, 2.95), (4.7, 2.95))
arrow((7.2, 2.95), (8.7, 2.95))
ax.text(7.95, 3.05, "on the graph\n" + r"of $\nabla u_\theta$", ha="center", va="bottom", fontsize=9.5, color="#444444")
arrow((5.95, 2.5), (5.95, 1.2))
arrow((10.1, 2.5), (10.1, 1.2))
name = "F6.1_ad_computational_graph.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
