"""Figure 1.6: chapter dependency graph (Chapter 1, section 1.10), drawn as its transitive reduction
(an arrow i -> j is omitted when j already depends on i through another chapter). The hard
dependencies and the division into parts are those of the book's 16-chapter structure and are
listed below. The node layout is chosen by a seeded random search that maximizes the clearance
between edges and nodes.

Usage:
    python code/figures/make_fig1_6.py
Output: figures/chapter01/F1.6_dependency_graph.png and .pdf (reads stored results only; runtime a few seconds).
"""
import matplotlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter01"
FIG_DIR = REPO_ROOT / "figures" / "chapter01"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Hard chapter dependencies of the book (chapter -> chapters it depends on); Chapter 16 depends on all.
deps = {1: [], 2: [1], 3: [1, 2], 4: [1, 2, 3], 5: [2, 3, 4], 6: [1, 2, 3, 4], 7: [2, 3, 4, 6],
        8: [2, 3, 4, 6, 7], 9: [3, 4, 6, 7], 10: [2, 3, 4], 11: [3, 4], 12: [5, 6, 7, 9, 10],
        13: [2, 4, 5, 10], 14: [5, 6, 9, 10, 12, 13], 15: [6, 7, 9, 10, 11, 12, 13],
        16: list(range(1, 16))}
PARTS = {"I": [1, 2, 3, 4, 5], "II": [6, 7, 8, 9], "III": [10, 11, 12], "IV": [13, 14, 15, 16]}
def ancestors(j, seen=None):
    seen = set() if seen is None else seen
    for i in deps[j]:
        if i not in seen:
            seen.add(i); ancestors(i, seen)
    return seen
edges = []
for j, ds in deps.items():
    for i in ds:
        if not any(i in ancestors(k) for k in ds if k != i):
            edges.append((i, j))
part = {c: p for p, cs in PARTS.items() for c in cs}
colors = {"I": "#0072B2", "II": "#009E73", "III": "#D55E00", "IV": "#CC79A7"}
depth = {}
def d(j):
    if j not in depth:
        depth[j] = 0 if not deps[j] else 1 + max(d(i) for i in deps[j])
    return depth[j]
for j in deps: d(j)
layers = {}
for j in sorted(deps): layers.setdefault(depth[j], []).append(j)
import numpy as np
def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b)); ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))
def score(pos):
    # minimum clearance between any edge and any non-incident node, then number of crossings
    clear = min(seg_dist(pos[n], pos[i], pos[j]) for i, j in edges for n in pos if n not in (i, j))
    return clear
rng = np.random.default_rng(0)
slots = np.arange(-2.0, 2.01, 0.5)
best, best_pos = -1, None
for trial in range(20000):
    pos = {}
    for L, nodes in layers.items():
        xs = sorted(rng.choice(slots, size=len(nodes), replace=False)) if len(nodes) > 1 else [0.0]
        for j, x in zip(nodes, rng.permutation(xs)):
            pos[j] = (float(x), -float(L))
    s = score(pos)
    if s > best:
        best, best_pos = s, pos
pos = best_pos
print("min clearance", best)
fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.axis("off")
for i, j in edges:
    ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>", mutation_scale=11, color="0.45", lw=0.9,
                                 shrinkA=13, shrinkB=13))
for j, (x, y) in pos.items():
    ax.scatter([x], [y], s=520, color=colors[part[j]], zorder=3)
    ax.text(x, y, str(j), ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=4)
for p, c in colors.items():
    ax.scatter([], [], color=c, s=80, label={"I": "Part I: Foundations", "II": "Part II: Learning Solutions and Operators",
               "III": "Part III: Learning Dynamics and Hidden Physics", "IV": "Part IV: Reliability, Applications, and Frontiers"}[p])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11, frameon=False)
ax.set_xlim(-2.6, 2.6)
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "F1.6_dependency_graph.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "F1.6_dependency_graph.pdf", bbox_inches="tight")
print(sorted(edges, key=lambda e: (e[1], e[0])))
print(layers)
