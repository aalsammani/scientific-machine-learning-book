"""Figure 1.1: the partial-knowledge spectrum (schematic; Chapter 1, section 1.2). No data.
The script checks that no two text labels overlap and prints the result.

Usage:
    python code/figures/make_fig1_1.py
Output: figures/chapter01/F1.1_partial_knowledge_spectrum.png and .pdf (reads stored results only; runtime a few seconds).
"""
import matplotlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter01"
FIG_DIR = REPO_ROOT / "figures" / "chapter01"
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GRAY = "#0072B2", "#555555"
# Line breaks keep every label within its column at a printable font size (the figure is set at
# \textwidth, so a 9.4 in canvas is reduced by about 0.69).
marks = [
    ("Fully\nspecified\nmodel", "classical\nsolvers\n(Ch. 3)", "nothing\nto learn"),
    ("Uncertain\nparameters", "parameter\nand field\nestimation\n(Ch. 5, 12)", "parameter-\nization,\nloss, prior"),
    ("Unknown\nconstitutive\nrelation", "closures,\nuniversal DEs\n(Ch. 10)", "parameter-\nization"),
    ("Partially\nknown\ndynamics", "hybrid\nmodels\n(Ch. 10)", "parameter-\nization,\narchitecture"),
    ("Unknown\ngoverning\nequations", "sparse equation\ndiscovery\n(Ch. 11)", "candidate\nlibrary,\nsparse loss"),
    ("Largely\ndata-driven\nprediction", "operator\nlearning\n(Ch. 9)", "architecture,\nprior"),
]
FS_LAND, FS_ROW = 11.5, 11
fig, ax = plt.subplots(figsize=(9.4, 3.4))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis("off")
n = len(marks)
ax.plot([-0.55, n - 0.45], [0, 0], color="k", lw=1.5)
ax.annotate("", xy=(n - 0.4, 0), xytext=(n - 0.6, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
texts = []
for x, (land, fam, mech) in enumerate(marks):
    ax.plot([x, x], [-0.07, 0.07], color="k", lw=1.5)
    texts.append(ax.text(x, 0.17, mech, ha="center", va="bottom", fontsize=FS_ROW, color=BLUE,
                         linespacing=1.15))
    texts.append(ax.text(x, -0.17, land, ha="center", va="top", fontsize=FS_LAND, linespacing=1.15))
    texts.append(ax.text(x, -0.93, fam, ha="center", va="top", fontsize=FS_ROW, style="italic",
                         color=GRAY, linespacing=1.15))
ax.set_xlim(-0.6, n - 0.35); ax.set_ylim(-2.2, 1.1)
# direction labels at the two ends of the axis
texts.append(ax.text(-0.55, 0.98, "← more known structure", fontsize=FS_ROW, ha="left", va="center"))
texts.append(ax.text(n - 0.45, 0.98, "more reliance on data →", fontsize=FS_ROW, ha="right", va="center"))
# key to the three rows (replaces a label column, which would squeeze the six columns together);
# segments are placed one after another from their rendered widths
yk = -2.05
key = [("Above the axis: where structure enters (§1.1).", BLUE, "normal"),
       ("Below: landmark and", "black", "normal"),
       ("method family.", GRAY, "italic")]
fig.canvas.draw()
xk = -0.55
inv = ax.transData.inverted()
for s, c, st in key:
    t = ax.text(xk, yk, s, fontsize=FS_ROW, color=c, style=st, ha="left", va="center")
    texts.append(t)
    ext = t.get_window_extent(fig.canvas.get_renderer())
    xk = inv.transform((ext.x1, ext.y0))[0] + 0.08

# check that no two text boxes overlap (with a small margin), and report the tightest gap
fig.canvas.draw()
r = fig.canvas.get_renderer()
bb = [t.get_window_extent(r).expanded(1.0, 1.0) for t in texts]
pad = 6  # pixels at the figure dpi
bad = [(texts[i].get_text(), texts[j].get_text()) for i in range(len(bb)) for j in range(i + 1, len(bb))
       if bb[i].x0 - pad < bb[j].x1 and bb[j].x0 - pad < bb[i].x1 and bb[i].y0 - pad < bb[j].y1 and bb[j].y0 - pad < bb[i].y1]
print("overlapping text pairs:", bad if bad else "none")

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "F1.1_partial_knowledge_spectrum.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "F1.1_partial_knowledge_spectrum.pdf", bbox_inches="tight")
