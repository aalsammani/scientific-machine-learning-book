"""Manufactured-solution verification of the Darcy solver in darcy_data.py (Chapter 9, section 9.6;
Chapter 14, section 14.4; Appendix D). The solver's own assemble/solve functions are called (not a
copy), for both boundary treatments.

Checks the operator sign (-div(a grad u) = f), the coefficient placement ([i, j] <-> (x_i, y_j)),
the boundary treatment and the observed order of convergence under grid refinement
(n = 11, 23, 47, 95, 191 interior points, h halved each time). Forcing terms are derived
symbolically with SymPy. Three manufactured cases:
  M1  u = sin(pi x) sin(pi y),          a = 1 + 0.3 sin(pi x) cos(pi y)
  M2  u = x(1-x) y(1-y) exp(x + 2y),    a = 1 + 0.5 x^2 + 0.3 sin(2 pi y)   (not symmetric in x, y)
  M3  u = sin(pi x) sin(2 pi y),        a = 1
For M2, solving with the transposed coefficient must increase the error (placement check).
The observed orders quoted in the book (2.00, 2.00 and 1.95 for M1, M3 and M2) are the max-norm
orders at the finest level (key order_max); the grid-L2 orders are also stored (order_l2).
A production-family study (f = 1, three coefficient draws from default_rng(0)) compares the n = 21
solution with nested-grid solutions up to n = 351, for the corrected and the legacy boundary
treatment.

Usage:
    python code/chapter09/darcy/darcy_manufactured_verification.py
Output: results/chapter09/darcy_verification/manufactured_solution_convergence.json
Runtime: under a minute. Deterministic.
"""
import sys, json, os
from pathlib import Path
import numpy as np, sympy as sp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from darcy_data import grid, solve_field, darcy_solve

x, y = sp.symbols("x y", real=True)
cases = {
    "M1": (sp.sin(sp.pi*x)*sp.sin(sp.pi*y), 1 + sp.Rational(3, 10)*sp.sin(sp.pi*x)*sp.cos(sp.pi*y)),
    "M2": (x*(1-x)*y*(1-y)*sp.exp(x + 2*y), 1 + sp.Rational(1, 2)*x**2 + sp.Rational(3, 10)*sp.sin(2*sp.pi*y)),
    "M3": (sp.sin(sp.pi*x)*sp.sin(2*sp.pi*y), sp.Integer(1)),
}
resolutions = [11, 23, 47, 95, 191]   # h = 1/12, 1/24, 1/48, 1/96, 1/192 (halving)
out = {"description": __doc__, "cases": {}}

for name, (u_s, a_s) in cases.items():
    # boundary values of u must vanish
    assert all(sp.simplify(u_s.subs(v, val)) == 0 for v in (x, y) for val in (0, 1)), name
    f_s = sp.simplify(-(sp.diff(a_s*sp.diff(u_s, x), x) + sp.diff(a_s*sp.diff(u_s, y), y)))
    u_f, a_f, f_f = (sp.lambdify((x, y), e, "numpy") for e in (u_s, a_s, f_s))
    rec = {"u": str(u_s), "a": str(a_s), "f": str(f_s)}
    for boundary in ("corrected", "legacy"):
        rows = []
        for n in resolutions:
            h, X, Y = grid(n)
            a = np.broadcast_to(a_f(X, Y), X.shape).astype(float)
            f = np.broadcast_to(f_f(X, Y), X.shape).astype(float)
            u = solve_field(a, f, h, boundary)
            e = u - u_f(X, Y)
            rows.append(dict(n=n, h=h, err_l2_grid=float(np.sqrt(np.mean(e**2))), err_max=float(np.abs(e).max()),
                             u_max=float(np.abs(u_f(X, Y)).max())))
        for i in range(1, len(rows)):
            rows[i]["order_l2"] = float(np.log(rows[i-1]["err_l2_grid"]/rows[i]["err_l2_grid"]) / np.log(rows[i-1]["h"]/rows[i]["h"]))
            rows[i]["order_max"] = float(np.log(rows[i-1]["err_max"]/rows[i]["err_max"]) / np.log(rows[i-1]["h"]/rows[i]["h"]))
        rec[boundary] = rows
        print(name, boundary, " ".join(f"n={r['n']}:{r['err_max']:.2e}" + (f"(p={r['order_max']:.2f})" if 'order_max' in r else "") for r in rows))
    # placement check: transposing the coefficient must change the answer for M2
    if name == "M2":
        h, X, Y = grid(47)
        a = a_f(X, Y); f = f_f(X, Y)
        e_T = float(np.abs(solve_field(a.T.copy(), f, h) - u_f(X, Y)).max())
        rec["transposed_coefficient_max_error_n47"] = e_T
        print("  M2 with transposed coefficient, n=47, max error:", e_T)
    out["cases"][name] = rec

# Production family (f = 1): nested-grid self-convergence at the dataset's resolution n = 21.
# Grids n = 21, 43, 87, 175, 351 have h = 1/22, 1/44, ..., and share the coarse nodes.
rng = np.random.default_rng(0)
coeff_samples = rng.uniform(-1, 1, (3, 5))
fam = []
for c in coeff_samples:
    sols = {}
    for n in (21, 43, 87, 175, 351):
        X, Y, a, u = darcy_solve(c, n=n)
        s = (n + 1) // 22
        sols[n] = u[s-1::s, s-1::s]        # restrict to the n = 21 nodes
    ref = sols[351]
    fam.append({"coeffs": c.tolist(),
                "rel_err_vs_n351": {str(n): float(np.linalg.norm(sols[n]-ref)/np.linalg.norm(ref)) for n in (21, 43, 87, 175)}})
    X, Y, a, u_leg = darcy_solve(c, n=21, boundary="legacy")
    fam[-1]["legacy_n21_rel_err_vs_n351"] = float(np.linalg.norm(u_leg - ref)/np.linalg.norm(ref))
    print("family", np.round(c, 3), fam[-1]["rel_err_vs_n351"], "legacy n21:", fam[-1]["legacy_n21_rel_err_vs_n351"])
out["production_family_self_convergence"] = fam

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "chapter09" / "darcy_verification"
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(out, open(OUT_DIR / "manufactured_solution_convergence.json", "w"), indent=1)
print("written", OUT_DIR / "manufactured_solution_convergence.json")
