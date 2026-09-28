"""Continuous adjoint gradient versus centered finite differences (Chapter 5, section 5.7;
Theorem 5.7).

  * Logistic model x' = p x (1 - x), x(0) = 0.1, T = 3; synthetic data y_obs = trajectory at p = 2.
  * J(p) = 0.5 int_0^T (x - y_obs)^2 dt, trapezoidal rule on linspace(0, 3, 300).
  * Forward and backward solves: solve_ivp RK45, rtol = atol = 1e-12, dense output.
  * Adjoint psi' = -p (1 - 2x) psi - (x - y_obs), psi(T) = 0; dJ/dp = int psi x (1 - x) dt on the
    same 300-point grid.
  * Finite differences: centered, h = 1e-5. Parameter values p in {1.5, 1.8, 2.2, 2.5, 3.0}.
Also computed: the exact gradient (closed-form x(t; p) and adaptive quadrature) and the comparison on
a 3001-point grid, which show that the adjoint/FD discrepancy is dominated by the trapezoidal
quadrature. This protocol reproduces the stored original (adjoint_vs_fd_verification.json) to about
1e-10 relative.

Usage:
    python code/chapter05/adjoint_vs_fd_verification.py
Output: results/chapter05/adjoint_vs_fd_verification_rerun.json
Runtime: under a minute. Deterministic.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp, trapezoid

RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter05"
OUT = RESULTS / "adjoint_vs_fd_verification_rerun.json"
OLD = RESULTS / "adjoint_vs_fd_verification.json"
X0, T, PT, H = 0.1, 3.0, 2.0, 1e-5
RTOL, ATOL = 1e-12, 1e-12
PS = [1.5, 1.8, 2.2, 2.5, 3.0]


def solve(p, dense=True, t_eval=None):
    return solve_ivp(lambda t, x: p * x * (1 - x), (0, T), [X0], rtol=RTOL, atol=ATOL,
                     dense_output=dense, t_eval=t_eval)


def grads(p, N):
    tg = np.linspace(0, T, N)
    ob = solve(PT)
    yo = lambda t: ob.sol(t)[0]
    fw = solve(p)
    xs = lambda t: fw.sol(t)[0]
    bw = solve_ivp(lambda t, psi: -p * (1 - 2 * xs(t)) * psi - (xs(t) - yo(t)), (T, 0), [0.0],
                   rtol=RTOL, atol=ATOL, dense_output=True)
    g_adj = trapezoid(bw.sol(tg)[0] * xs(tg) * (1 - xs(tg)), x=tg)

    def J(pp):
        xx = solve(pp, dense=False, t_eval=tg).y[0]
        return trapezoid(0.5 * (xx - yo(tg)) ** 2, x=tg)

    g_fd = (J(p + H) - J(p - H)) / (2 * H)
    return g_adj, g_fd


def exact_grad(p):
    xa = lambda t, q: 1.0 / (1.0 + (1 / X0 - 1) * np.exp(-q * t))
    dx = lambda t: (1 / X0 - 1) * t * np.exp(-p * t) * xa(t, p) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return quad(lambda t: (xa(t, p) - xa(t, PT)) * dx(t), 0, T, epsabs=1e-16, epsrel=1e-13, limit=400)[0]


def main():
    old = json.load(open(OLD))
    rows = []
    for i, p in enumerate(PS):
        ga, gf = grads(p, 300)
        ga2, gf2 = grads(p, 3001)
        ge = exact_grad(p)
        o = old["rows"][i]
        rows.append({
            "p": p, "adjoint_grad": ga, "fd_grad": gf, "rel_diff": abs(ga - gf) / abs(gf),
            "exact_grad": ge, "adjoint_rel_err_vs_exact": ga / ge - 1, "fd_rel_err_vs_exact": gf / ge - 1,
            "N3001_adjoint_grad": ga2, "N3001_fd_grad": gf2, "N3001_rel_diff": abs(ga2 - gf2) / abs(gf2),
            "dev_adjoint_vs_stored": ga / o["adjoint_grad"] - 1, "dev_fd_vs_stored": gf / o["fd_grad"] - 1,
        })
        r = rows[-1]
        print("p=%.1f adj=%.12e fd=%.12e rel=%.3e | stored rel=%.3e | dev adj %.1e fd %.1e | "
              "err vs exact adj %.2e fd %.2e | N=3001 rel=%.3e" % (
                  p, ga, gf, r["rel_diff"], o["rel_diff"], r["dev_adjoint_vs_stored"], r["dev_fd_vs_stored"],
                  r["adjoint_rel_err_vs_exact"], r["fd_rel_err_vs_exact"], r["N3001_rel_diff"]))
    res = {"p_true": PT, "protocol": "RK45 rtol=atol=1e-12, trapezoid on linspace(0,3,300), FD h=1e-5",
           "rows": rows}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
