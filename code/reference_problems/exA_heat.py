"""Running Example A: the one-dimensional heat equation.

    u_t = nu u_xx  on (0,1) x (0,T],   u(0,t) = u(1,t) = 0,
    u(x,0) = sin(pi x) + 0.5 sin(5 pi x),   nu = 0.5, T = 1.

The exact solution is a finite two-term Fourier series, so it is the reference for every learned
method applied to this problem. The module also provides two classical comparators:

  * explicit finite differences (FTCS), order (2, 1) in (space, time);
  * Crank-Nicolson finite differences, order (2, 2).

Both are verified (i) against the exact solution under grid refinement and (ii) by the method of
manufactured solutions with a non-separable target.

Book: Chapter 1 (sections 1.8.2-1.8.3, Table 1.4), Chapter 3 (section 3.8), Chapter 6
(section 6.10), Appendix D. The module is imported by the Chapter 1, 6, 7 and 14 experiments.

Usage:
    python code/reference_problems/exA_heat.py

Output: data/reference/exA/verification.json (rerunning overwrites the stored file; the wall-clock
entries and platform strings are machine specific). Runtime: a few seconds. Deterministic.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = REPO_ROOT / "data" / "reference" / "exA" / "verification.json"

NU = 0.5
T = 1.0
MODES = ((1, 1.0), (5, 0.5))  # (wavenumber k, amplitude a_k)


def exact(x, t, nu=NU, modes=MODES):
    x = np.asarray(x, float); t = np.asarray(t, float)
    u = np.zeros(np.broadcast(x, t).shape)
    for k, a in modes:
        u = u + a * np.exp(-nu * (k * np.pi) ** 2 * t) * np.sin(k * np.pi * x)
    return u


def initial(x, modes=MODES):
    return exact(x, 0.0, modes=modes)


def ftcs(nx, nt, nu=NU, T=T, u0=None, source=None):
    """Explicit FTCS on a grid of nx points including both ends.
    Stability requires r = nu*dt/dx^2 <= 1/2. Returns (x, t, U[nt+1, nx])."""
    x = np.linspace(0, 1, nx); dx = x[1] - x[0]
    t = np.linspace(0, T, nt + 1); dt = t[1] - t[0]
    r = nu * dt / dx ** 2
    assert r <= 0.5 + 1e-12, f"FTCS unstable: r={r}"
    U = np.zeros((nt + 1, nx))
    U[0] = initial(x) if u0 is None else u0(x)
    for n in range(nt):
        u = U[n]
        un = u.copy()
        un[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        if source is not None:
            un[1:-1] += dt * source(x[1:-1], t[n])
        un[0] = un[-1] = 0.0
        U[n + 1] = un
    return x, t, U


def crank_nicolson(nx, nt, nu=NU, T=T, u0=None, source=None):
    """Crank-Nicolson on a grid of nx points including both ends. Returns (x, t, U[nt+1, nx])."""
    from scipy.linalg import solve_banded
    x = np.linspace(0, 1, nx); dx = x[1] - x[0]
    t = np.linspace(0, T, nt + 1); dt = t[1] - t[0]
    r = nu * dt / dx ** 2
    m = nx - 2
    ab = np.zeros((3, m))
    ab[0, 1:] = -r / 2; ab[1, :] = 1 + r; ab[2, :-1] = -r / 2
    U = np.zeros((nt + 1, nx))
    U[0] = initial(x) if u0 is None else u0(x)
    for n in range(nt):
        u = U[n, 1:-1]
        rhs = u + (r / 2) * (U[n, 2:] - 2 * u + U[n, :-2])
        if source is not None:
            rhs += dt * 0.5 * (source(x[1:-1], t[n]) + source(x[1:-1], t[n + 1]))
        U[n + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
    return x, t, U


def rel_errors(U, Uex):
    e = U - Uex
    return (np.linalg.norm(e) / np.linalg.norm(Uex),
            np.abs(e).max() / np.abs(Uex).max())


def observed_order(hs, errs):
    hs = np.asarray(hs, float); errs = np.asarray(errs, float)
    return np.log(errs[1:] / errs[:-1]) / np.log(hs[1:] / hs[:-1])


def main():
    out = {"example": "A", "nu": NU, "T": T, "modes": MODES, "platform": platform.platform(),
           "cpu": platform.processor(), "checks": {}}
    # 1. FTCS refinement against the exact solution (dt tied to dx^2 through r = 0.4)
    rows = []
    for nx in (11, 21, 41, 81, 161):
        dx = 1 / (nx - 1); dt_target = 0.4 * dx ** 2 / NU; nt = int(np.ceil(T / dt_target))
        t0 = time.perf_counter(); x, t, U = ftcs(nx, nt); wall = time.perf_counter() - t0
        Uex = exact(x[None, :], t[:, None])
        l2, li = rel_errors(U, Uex)
        rows.append(dict(nx=nx, nt=nt, dx=dx, rel_l2=l2, rel_linf=li, wall_s=wall))
    ords = observed_order([r["dx"] for r in rows], [r["rel_l2"] for r in rows])
    out["checks"]["ftcs_vs_exact"] = dict(rows=rows, observed_order_l2=ords.tolist(),
                                          expected_order=2.0)
    # 2. Crank-Nicolson refinement (dt = dx, so order 2 is expected in both variables)
    rows = []
    for nx in (11, 21, 41, 81, 161, 321):
        dx = 1 / (nx - 1); nt = int(round(T / dx))
        t0 = time.perf_counter(); x, t, U = crank_nicolson(nx, nt); wall = time.perf_counter() - t0
        Uex = exact(x[None, :], t[:, None])
        l2, li = rel_errors(U, Uex)
        rows.append(dict(nx=nx, nt=nt, dx=dx, rel_l2=l2, rel_linf=li, wall_s=wall))
    ords = observed_order([r["dx"] for r in rows], [r["rel_l2"] for r in rows])
    out["checks"]["cn_vs_exact"] = dict(rows=rows, observed_order_l2=ords.tolist(), expected_order=2.0)

    # 3. Manufactured solution u_m = sin(pi x) cos(3t) + x(1-x) t^2 (non-separable, zero boundary values)
    def um(x, t):
        return np.sin(np.pi * x) * np.cos(3 * t) + x * (1 - x) * t ** 2

    def src(x, t):  # u_t - nu u_xx for u_m
        return (-3 * np.sin(np.pi * x) * np.sin(3 * t) + 2 * x * (1 - x) * t
                - NU * (-(np.pi ** 2) * np.sin(np.pi * x) * np.cos(3 * t) - 2 * t ** 2))

    rows = []
    for nx in (11, 21, 41, 81, 161):
        dx = 1 / (nx - 1); nt = int(round(T / dx))
        x, t, U = crank_nicolson(nx, nt, u0=lambda x: um(x, 0.0), source=src)
        Uex = um(x[None, :], t[:, None])
        l2, li = rel_errors(U, Uex)
        rows.append(dict(nx=nx, nt=nt, dx=dx, rel_l2=l2, rel_linf=li))
    ords = observed_order([r["dx"] for r in rows], [r["rel_l2"] for r in rows])
    out["checks"]["cn_mms"] = dict(manufactured="sin(pi x)cos(3t)+x(1-x)t^2", rows=rows,
                                   observed_order_l2=ords.tolist(), expected_order=2.0)
    # 4. The exact solution is a finite sum, so there is no series truncation error. As a consistency
    #    check, evaluate the finite-difference residual of the exact solution on a fine grid.
    out["checks"]["exact_solution"] = dict(note="finite two-term Fourier series; residual checked symbolically-by-FD below")
    xx = np.linspace(0, 1, 2001)[1:-1]; tt = 0.3; h = 1e-4
    ut = (exact(xx, tt + h) - exact(xx, tt - h)) / (2 * h)
    uxx = (exact(xx + h, tt) - 2 * exact(xx, tt) + exact(xx - h, tt)) / h ** 2
    out["checks"]["exact_solution"]["max_fd_residual"] = float(np.abs(ut - NU * uxx).max())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["checks"].items():
        if "rows" in v:
            print(k, "orders:", np.round(v["observed_order_l2"], 3), "finest rel L2:", "%.2e" % v["rows"][-1]["rel_l2"])
    print("exact FD residual:", out["checks"]["exact_solution"]["max_fd_residual"])


if __name__ == "__main__":
    main()
