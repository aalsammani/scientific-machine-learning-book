"""Kansa's method (Kansa 1990): radial-basis-function collocation for Running Example A, solved by a
single linear least-squares problem in a multiquadric basis. This is the classical mesh-free
comparator of Chapter 6, section 6.10 (Table 6.1).

Space-time collocation: centers and collocation points coincide on a uniform n x n grid in (x, t),
split into interior (residual), boundary and initial rows, as the physics-informed loss is split;
the trial function is linear in its coefficients and derivatives are analytic. Shape parameter
c = 0.3; n in {11, 15, 21, 27, 33}; errors on the 201 x 101 evaluation grid of Chapter 1.

Usage:
    python code/chapter06/kansa_rbf.py
Output: results/chapter06/kansa_convergence.json
Runtime: a few seconds. Deterministic (wall-clock entries are machine specific).
"""
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import exact, NU, T  # noqa: E402


def mq(r2, c):
    return np.sqrt(r2 + c ** 2)


def mq_dt(dt, r2, c):
    # d/dt sqrt((x-xc)^2+(t-tc)^2+c^2) = (t-tc)/sqrt(...)
    return dt / mq(r2, c)


def mq_dxx(dx, r2, c):
    # phi = sqrt(dx^2+dt^2+c^2); phi_x = dx/phi; phi_xx = 1/phi - dx^2/phi^3
    phi = mq(r2, c)
    return 1.0 / phi - dx ** 2 / phi ** 3


def solve_kansa(n_centers_per_dim, c_shape=0.3, n_r=None, n_b=None, n_0=None, seed=0):
    """Centers on a uniform (x,t) grid; collocation points equal to centers (standard Kansa
    symmetric collocation) split into interior/boundary/initial rows exactly as the PINN loss is
    split into three residual types, but here forming one linear system solved by least squares."""
    nx = n_centers_per_dim
    xs = np.linspace(0, 1, nx)
    ts = np.linspace(0, T, nx)
    Xc, Tc = np.meshgrid(xs, ts, indexing="ij")
    xc, tc = Xc.ravel(), Tc.ravel()
    N = xc.size

    interior = (xc > 0) & (xc < 1) & (tc > 0)
    boundary = (xc == 0) | (xc == 1)
    initial_ = (tc == 0) & (xc > 0) & (xc < 1)

    x_r, t_r = xc[interior], tc[interior]
    x_b, t_b = xc[boundary], tc[boundary]
    x_0, t_0 = xc[initial_], tc[initial_]

    def design_value(x, t):
        dx = x[:, None] - xc[None, :]
        dt = t[:, None] - tc[None, :]
        r2 = dx ** 2 + dt ** 2
        return mq(r2, c_shape)

    def design_residual(x, t):
        dx = x[:, None] - xc[None, :]
        dt = t[:, None] - tc[None, :]
        r2 = dx ** 2 + dt ** 2
        phi_t = mq_dt(dt, r2, c_shape)
        phi_xx = mq_dxx(dx, r2, c_shape)
        return phi_t - NU * phi_xx

    A_r = design_residual(x_r, t_r)
    A_b = design_value(x_b, t_b)
    b_b = np.zeros(x_b.size)
    A_0 = design_value(x_0, t_0)
    b_0 = exact(x_0, 0.0)

    A = np.vstack([A_r, A_b, A_0])
    b = np.concatenate([np.zeros(x_r.size), b_b, b_0])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return xc, tc, coef, c_shape


def evaluate_kansa(xc, tc, coef, c_shape, X, Tg):
    dx = X.ravel()[:, None] - xc[None, :]
    dt = Tg.ravel()[:, None] - tc[None, :]
    r2 = dx ** 2 + dt ** 2
    Phi = mq(r2, c_shape)
    return (Phi @ coef).reshape(X.shape)


if __name__ == "__main__":
    import json
    import time
    EX = np.linspace(0, 1, 201)
    ET = np.linspace(0, T, 101)
    X, Tg = np.meshgrid(EX, ET, indexing="ij")
    Uex = exact(X, Tg)
    rows = []
    for n_per_dim in (11, 15, 21, 27, 33):
        t0 = time.perf_counter()
        xc, tc, coef, c_shape = solve_kansa(n_per_dim)
        wall = time.perf_counter() - t0
        Upred = evaluate_kansa(xc, tc, coef, c_shape, X, Tg)
        e = Upred - Uex
        rel_l2 = float(np.linalg.norm(e) / np.linalg.norm(Uex))
        rel_linf = float(np.abs(e).max() / np.abs(Uex).max())
        rows.append(dict(n_per_dim=n_per_dim, n_centers=int(n_per_dim ** 2), c_shape=c_shape,
                          rel_l2=rel_l2, rel_linf=rel_linf, wall_s=wall))
        print(rows[-1])
    json.dump(dict(method="Kansa RBF collocation, multiquadric, space-time", rows=rows),
              open(REPO_ROOT / "results" / "chapter06" / "kansa_convergence.json", "w"), indent=1)
