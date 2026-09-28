"""FTCS stability threshold on Running Example A (Chapter 3, section 3.3; Theorem 3.2).

  * u_t = nu u_xx on (0,1), nu = 0.5, zero Dirichlet data, u(x,0) = sin(pi x) + 0.5 sin(5 pi x).
  * Explicit FTCS with r = nu dt / dx^2 in {0.4, 0.5, 0.6}, dt = r dx^2 / nu, n = int(T/dt) steps.
  * Primary configuration: 41 grid points, T = 1, update u[1:-1] += r (u[2:] - 2 u[1:-1] + u[:-2]).
    At r = 0.6 this gives 1333 steps and max|u| of order 1e176.
  * Maximum amplification factor over the discrete sine modes, |g| = |1 - 4 r sin^2(m pi / (2M))|.

The discrete initial data contain only sine modes 1 and 5, which are exact eigenvectors of the FTCS
operator and are damped at r = 0.6; the growing high modes are seeded only by round-off. The blow-up
magnitude therefore depends on floating-point details, and its leading digits are implementation
dependent. The stored original result (ftcs_stability_demo.json, 3.25e176 after 1333 steps) was
produced by a script that was not preserved and did not record the grid; this script scans grids,
horizons and algebraically equivalent update formulas that give 1333 steps and reports every
configuration (none reproduces the stored leading digits exactly).

Usage:
    python code/chapter03/ftcs_stability_demo.py
Output: results/chapter03/ftcs_stability_demo_rerun.json
Runtime: a few seconds. Deterministic.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "chapter03" / "ftcs_stability_demo_rerun.json"


def exact(x, t, nu, ic):
    a = np.exp(-nu * np.pi ** 2 * t) * np.sin(np.pi * x)
    if ic == "two_mode":
        a = a + 0.5 * np.exp(-25 * nu * np.pi ** 2 * t) * np.sin(5 * np.pi * x)
    return a


def step(u, r, variant, A=None):
    if variant == "diff":
        un = u.copy()
        un[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        return un
    if variant == "weights":
        un = u.copy()
        un[1:-1] = (1 - 2 * r) * u[1:-1] + r * (u[2:] + u[:-2])
        return un
    if variant == "npdiff_inplace":
        u[1:-1] += r * np.diff(u, 2)
        return u
    if variant == "matrix":
        un = u.copy()
        un[1:-1] = A @ u[1:-1]
        return un
    raise ValueError(variant)


def run(N, T, r, variant, nu=0.5, ic="two_mode", r_mode="recomputed"):
    x = np.linspace(0.0, 1.0, N)
    dx = 1.0 / (N - 1)
    dt = r * dx ** 2 / nu
    n = int(T / dt)
    rr = nu * dt / dx ** 2 if r_mode == "recomputed" else r
    u = exact(x, 0.0, nu, ic)
    u[0] = u[-1] = 0.0
    A = None
    if variant == "matrix":
        m = N - 2
        A = (1 - 2 * rr) * np.eye(m) + rr * (np.eye(m, k=1) + np.eye(m, k=-1))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n):
            u = step(u, rr, variant, A)
    t_final = n * dt
    ue = exact(x, t_final, nu, ic)
    mx = float(np.max(np.abs(u)))
    emx = float(np.max(np.abs(ue)))
    return {"N": N, "T": T, "nu": nu, "ic": ic, "r": r, "r_mode": r_mode, "variant": variant, "steps": n,
            "t_final": float(t_final), "max_abs_u": mx, "max_abs_u_str": f"{mx:.2e}", "exact_max_abs_u": emx,
            "rel_err_max": abs(mx - emx) / emx, "max_abs_err": float(np.max(np.abs(u - ue)))}


def main():
    # primary configuration
    primary = [run(41, 1.0, r, "diff") for r in (0.4, 0.5, 0.6)]
    for d in primary:
        print("PRIMARY", d)
    # scan: all configurations giving 1333 steps at r=0.6
    scan = []
    for nu in (0.5, 1.0):
        for N in (21, 41, 51, 81, 101):
            for T in (0.1, 0.25, 0.5, 1.0):
                dx = 1.0 / (N - 1)
                if int(T / (0.6 * dx ** 2 / nu)) != 1333:
                    continue
                for ic in ("two_mode", "one_mode"):
                    for variant in ("diff", "weights", "npdiff_inplace", "matrix"):
                        for r_mode in ("recomputed", "literal"):
                            d = run(N, T, 0.6, variant, nu, ic, r_mode)
                            d["matches_3.25e+176"] = d["max_abs_u_str"] == "3.25e+176"
                            scan.append(d)
                            print(d["N"], d["T"], d["nu"], d["ic"], d["variant"], d["r_mode"], d["max_abs_u_str"])
    # r=0.4 accuracy on the plausible grids
    acc = [run(N, T, 0.4, "diff", nu) for (N, T, nu) in ((41, 1.0, 0.5), (81, 0.25, 0.5), (41, 0.5, 1.0))]
    g = {}
    for N in (41, 81):
        M = N - 1
        m = np.arange(1, M)
        for r in (0.4, 0.5, 0.6):
            gm = 1 - 4 * r * np.sin(m * np.pi / (2 * M)) ** 2
            unstable = np.abs(gm) > 1
            g[f"N={N},r={r}"] = {"max_abs_g": float(np.max(np.abs(gm))),
                                 "first_unstable_mode": int(m[unstable][0]) if np.any(unstable) else None,
                                 "g_mode1": float(gm[0]), "g_mode5": float(gm[4])}
    out = {"primary": primary, "r04_accuracy": acc, "scan_r06_1333_steps": scan,
           "any_exact_match_3.25e+176": [d for d in scan if d["matches_3.25e+176"]],
           "amplification": g}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"matches": out["any_exact_match_3.25e+176"], "amplification": g, "r04": acc}, indent=1))


if __name__ == "__main__":
    main()
