"""SINDy (sequentially thresholded least squares) on Lorenz-63 with finite-difference derivatives,
noise sweep over 50 independent noise realizations per level (Chapter 11, sections 11.4-11.5;
Figure 11.1).

Protocol: sigma=10, rho=28, beta=8/3; x(0)=(1,1,1); t = 0, 0.01, ..., 9.99 (1000 samples);
reference trajectory by DOP853 (rtol 1e-12, atol 1e-14); additive Gaussian noise with standard
deviation equal to (noise level) x (standard deviation of the clean component); derivatives by
second-order central differences (numpy.gradient); library of the ten monomials of degree <= 2;
STLSQ with threshold 0.1 and 10 refit iterations.

Noise levels 0, 1, 5, 10, 20 and 50%; realizations use numpy default_rng(0..49) (one noise-free
run). For each run the numbers of spurious and missing terms are recorded; the mutual coherence of
the library columns on the clean trajectory is also reported (raw and centered). The second
paragraph of this docstring is stored as the protocol string in the output.

Usage:
    python code/chapter11/sindy_noise_sweep.py
Output: results/chapter11/sindy_noise_sweep.json
Runtime: a few seconds. Deterministic (fixed seeds).
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

R = Path(__file__).resolve().parents[2] / "results" / "chapter11"
OLD = R / "sindy_noise_sweep.json"

NAMES = ["1", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]
TRUE = [{"x", "y"}, {"x", "y", "xz"}, {"z", "xy"}]
LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
N_SEEDS = 50


def lorenz(t, u, s=10.0, r=28.0, b=8.0 / 3.0):
    x, y, z = u
    return [s * (y - x), x * (r - z) - y, x * y - b * z]


def library(X):
    x, y, z = X.T
    return np.stack([np.ones_like(x), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z], axis=1)


def stlsq(Th, dX, lam=0.1, iters=10):
    Xi = np.linalg.lstsq(Th, dX, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for j in range(dX.shape[1]):
            act = ~small[:, j]
            if act.any():
                Xi[act, j] = np.linalg.lstsq(Th[:, act], dX[:, j], rcond=None)[0]
    return Xi


def support_counts(Xi):
    spurious, missing, per_eq = 0, 0, []
    for j, eq in enumerate("xyz"):
        rec = {NAMES[i] for i in range(10) if Xi[i, j] != 0}
        spurious += len(rec - TRUE[j])
        missing += len(TRUE[j] - rec)
        per_eq.append(dict(eq=eq, recovered=sorted(rec, key=NAMES.index)))
    return spurious, missing, per_eq


def coherence_pairs(Th, center=False):
    A = Th - Th.mean(axis=0) if center else Th
    keep = np.linalg.norm(A, axis=0) > 0
    idx = np.flatnonzero(keep)
    G = A[:, keep] / np.linalg.norm(A[:, keep], axis=0)
    C = np.abs(G.T @ G)
    pairs = [(float(C[a, b]), NAMES[idx[a]], NAMES[idx[b]]) for a in range(len(idx)) for b in range(a + 1, len(idx))]
    return sorted(pairs, reverse=True)[:5]


t = np.arange(1000) * 0.01
sol = solve_ivp(lorenz, (t[0], t[-1]), [1.0, 1.0, 1.0], t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14)
X0 = sol.y.T
scale = X0.std(axis=0)

out = dict(protocol=__doc__.strip().split("\n\n")[1], n_seeds=N_SEEDS,
           coherence_top5=coherence_pairs(library(X0)),
           coherence_top5_centered=coherence_pairs(library(X0), center=True), levels=[])
for lev in LEVELS:
    sp, mi, eqs = [], [], []
    for seed in range(N_SEEDS if lev > 0 else 1):
        rng = np.random.default_rng(seed)
        X = X0 + lev * scale * rng.standard_normal(X0.shape)
        dX = np.gradient(X, t, axis=0)
        s, m, pe = support_counts(stlsq(library(X), dX))
        sp.append(s); mi.append(m); eqs.append(pe)
    sp, mi = np.array(sp), np.array(mi)
    row = dict(noise_rel=lev, n_realizations=len(sp), spurious=sp.tolist(), missing=mi.tolist(),
               spurious_median=float(np.median(sp)), spurious_min=int(sp.min()), spurious_max=int(sp.max()),
               frac_any_missing=float((mi > 0).mean()), missing_max=int(mi.max()), seed0_support=eqs[0])
    out["levels"].append(row)
    print(f"noise {lev:4.2f}: spurious median {row['spurious_median']:.1f} range [{row['spurious_min']}, {row['spurious_max']}], "
          f"runs missing a true term {100 * row['frac_any_missing']:.0f}% (max {row['missing_max']})")
print("coherence top5:", out["coherence_top5"])
print("centered coherence top5:", out["coherence_top5_centered"])
R.mkdir(parents=True, exist_ok=True)
with open(OLD, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OLD)
