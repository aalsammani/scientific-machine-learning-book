"""Structural non-identifiability of a two-compartment model (Chapter 5, section 5.4; Theorem 5.4;
Figure 5.1, right panel).

  * x1' = -(k10 + k12) x1 + k21 x2, x2' = k12 x1 - (k21 + k20) x2, x(0) = (1, 0), output y = x1.
  * Set A = (k10, k12, k21, k20) = (0.3, 0.2, 0.4, 0.1); the output depends only on
    a = k21 + k20, b = k10 + k12 + k21 + k20 and c = (k10 + k12) a - k12 k21, so every admissible
    k21 in [P/u, a] (u = b - a, P = u a - c) gives the same output.
  * Alternative set B: k21 on linspace(0.01, 0.499, 200), keep grid points with all four rates
    nonnegative, and take valid[len(valid) // 3]. This selection reproduces the stored alternative
    (k21 = 0.27539) and the stored count of valid grid points exactly.
  * Outputs on t = linspace(0, 10, 200), computed by the matrix exponential and by solve_ivp under
    several solvers and tolerances (RK45 at rtol 1e-12, atol 1e-14 reproduces the stored
    max|yA - yB|).

Usage:
    python code/chapter05/two_compartment_nonidentifiability.py
Output: results/chapter05/two_compartment_nonidentifiability_rerun.json
Runtime: a few seconds. Deterministic.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter05"
OUT = RESULTS / "two_compartment_nonidentifiability_rerun.json"
OLD = RESULTS / "two_compartment_nonidentifiability.json"


def mat(k10, k12, k21, k20):
    return np.array([[-(k10 + k12), k21], [k12, -(k21 + k20)]])


def abc(k10, k12, k21, k20):
    a = k21 + k20
    return a, k10 + k12 + k21 + k20, (k10 + k12) * a - k12 * k21


def main():
    old = json.load(open(OLD))
    kA = (0.3, 0.2, 0.4, 0.1)
    a, b, c = abc(*kA)
    u, P = b - a, (b - a) * a - c
    grid = np.linspace(0.01, 0.499, 200)
    valid = [(u - P / k, P / k, k, a - k) for k in grid if u - P / k >= 0 and a - k >= 0]
    kB = valid[len(valid) // 3]
    aB, bB, cB = abc(*kB)
    t = np.linspace(0, 10, 200)
    x0 = np.array([1.0, 0.0])

    def y_expm(k):
        M = mat(*k)
        return np.array([(expm(M * ti) @ x0)[0] for ti in t]), np.array([(expm(M * ti) @ x0)[1] for ti in t])

    yA_e, x2A_e = y_expm(kA)
    yB_e, x2B_e = y_expm(kB)
    variants = {"expm": (float(np.max(np.abs(yA_e - yB_e))), float(np.max(np.abs(yA_e - np.array(old["yA"])))),
                         float(np.max(np.abs(yB_e - np.array(old["yB"])))))}
    for meth in ["RK45", "DOP853", "LSODA", "Radau"]:
        for rtol, atol in [(1e-10, 1e-12), (1e-12, 1e-14), (1e-13, 1e-15)]:
            ys = []
            for k in (kA, kB):
                M = mat(*k)
                sol = solve_ivp(lambda _t, x, M=M: M @ x, (0, 10), x0, t_eval=t, method=meth, rtol=rtol, atol=atol)
                ys.append(sol.y[0])
            key = "%s rtol=%g atol=%g" % (meth, rtol, atol)
            variants[key] = (float(np.max(np.abs(ys[0] - ys[1]))), float(np.max(np.abs(ys[0] - np.array(old["yA"])))),
                             float(np.max(np.abs(ys[1] - np.array(old["yB"])))))
    for k, v in variants.items():
        print("%-32s max|yA-yB|=%.3e  |yA-stored|=%.3e  |yB-stored|=%.3e" % (k, *v))

    res = {
        "setA": dict(zip(["k10", "k12", "k21", "k20"], kA)),
        "setB": dict(zip(["k10", "k12", "k21", "k20"], kB)),
        "abc_A": {"a": a, "b": b, "c": c}, "abc_B": {"a": aB, "b": bB, "c": cB},
        "u": u, "P": P, "k21_interval": [P / u, a],
        "n_valid_alternatives_in_search": len(valid),
        "max_abs_diff_y_expm": variants["expm"][0],
        "max_abs_diff_x2_expm": float(np.max(np.abs(x2A_e - x2B_e))),
        "solver_variants": {k: {"max_abs_diff_y": v[0], "max_dev_yA_vs_stored": v[1], "max_dev_yB_vs_stored": v[2]}
                            for k, v in variants.items()},
        "setB_dev_vs_stored": {kk: kB[i] - old["setB"][kk] for i, kk in enumerate(["k10", "k12", "k21", "k20"])},
    }
    print("set B:", kB, " n_valid:", len(valid), " abc_B:", (aB, bB, cB))
    print("set B deviation vs stored:", res["setB_dev_vs_stored"])
    print("max|x2A-x2B| (expm):", res["max_abs_diff_x2_expm"])
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
