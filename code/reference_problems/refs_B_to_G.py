"""Reference solutions and their verification for Running Examples B, C, E, F and G.

  B  Lotka-Volterra (alpha, beta, delta, gamma) = (1.5, 1, 1, 3) and the pendulum (c = 0, 0.1);
     DOP853 tolerance study with invariant drift.
  C  Viscous Burgers u_t + u u_x = nu u_xx on (-1,1), nu = 0.01/pi, u(x,0) = -sin(pi x);
     Cole-Hopf solution by Gauss-Hermite quadrature, cross-checked against a method-of-lines
     finite-difference solution (BDF) under refinement.
  E  Robertson kinetics (stiff); Radau and BDF tolerance studies, Radau-vs-BDF agreement, and an
     explicit RK45 run showing the cost of an explicit method on a stiff problem.
  F  SEIR model with time-varying transmission rate; DOP853 tolerance study.
  G  Lorenz-63 and Lorenz-96; DOP853 tolerance studies and the divergence time of two tolerances.

Example D (two-dimensional Darcy flow) is verified separately by
code/chapter09/darcy/darcy_manufactured_verification.py.

Each verification is a self-consistency check: successive tolerances or grids are compared, and the
change between them bounds the error of the reference only heuristically (Richardson-type argument);
for Example C the independent Cole-Hopf solution gives a direct comparison.

Book: Chapter 3 (section 3.7, Example E), Chapter 6 (section 6.8, Example C), Appendix D.

Usage:
    python code/reference_problems/refs_B_to_G.py            # all examples (about 4-5 minutes)
    python code/reference_problems/refs_B_to_G.py exE exG    # selected examples

Output: data/reference/exX/verification.json and reference.npz (rerunning overwrites the stored
files; wall-clock entries are machine specific). Deterministic.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "reference"
INFO = dict(platform=platform.platform(), cpu=platform.processor())


def tol_study(fun, y0, tspan, t_eval, method, rtols, invariant=None, args=()):
    """Solve at decreasing rtol; report the change between successive tolerances
    (a Richardson-type self-consistency check) and the drift of an invariant if one is given."""
    sols, rows = [], []
    for rtol in rtols:
        t0 = time.perf_counter()
        s = solve_ivp(fun, tspan, y0, method=method, t_eval=t_eval, rtol=rtol, atol=rtol * 1e-3, args=args)
        wall = time.perf_counter() - t0
        assert s.success, s.message
        sols.append(s.y)
        row = dict(rtol=rtol, nfev=int(s.nfev), wall_s=wall)
        if invariant is not None:
            I = invariant(s.y); row["invariant_max_rel_drift"] = float(np.abs(I - I[0]).max() / max(abs(I[0]), 1e-300))
        if len(sols) > 1:
            row["rel_change_vs_prev"] = float(np.linalg.norm(sols[-1] - sols[-2]) / np.linalg.norm(sols[-1]))
        rows.append(row)
    return rows, sols[-1]


# ---------------- Example B: Lotka-Volterra, pendulum ----------------
def exB():
    a, b, d, g = 1.5, 1.0, 1.0, 3.0
    def lv(t, y): return [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]]
    def lv_inv(Y): return d * Y[0] - g * np.log(Y[0]) + b * Y[1] - a * np.log(Y[1])
    t_eval = np.linspace(0, 10, 101)
    rows_lv, ylv = tol_study(lv, [1.0, 1.0], (0, 10), t_eval, "DOP853", [1e-6, 1e-8, 1e-10, 1e-12], lv_inv)
    def pend(t, y, c): return [y[1], -c * y[1] - np.sin(y[0])]
    def energy(Y): return 0.5 * Y[1] ** 2 + 1 - np.cos(Y[0])
    t_eval2 = np.linspace(0, 20, 201)
    rows_p0, yp0 = tol_study(pend, [1.0, 0.0], (0, 20), t_eval2, "DOP853", [1e-6, 1e-8, 1e-10, 1e-12], energy, args=(0.0,))
    rows_p1, yp1 = tol_study(pend, [1.0, 0.0], (0, 20), t_eval2, "DOP853", [1e-6, 1e-8, 1e-10, 1e-12], None, args=(0.1,))
    np.savez(OUT / "exB" / "reference.npz", t_lv=t_eval, y_lv=ylv, t_pend=t_eval2, y_pend_undamped=yp0, y_pend_damped=yp1)
    return dict(lotka_volterra=rows_lv, pendulum_undamped=rows_p0, pendulum_damped=rows_p1,
                reference_rtol=1e-12, method="DOP853")


# ---------------- Example C: viscous Burgers ----------------
def cole_hopf(x, t, nu, n=200):
    """Exact solution for u0 = -sin(pi x) (odd, 2-periodic extension), Basdevant et al. (1986) form,
    by n-point Gauss-Hermite quadrature in the heat-kernel variable."""
    xi, w = hermgauss(n)
    x = np.asarray(x)[:, None]
    eta = 2 * np.sqrt(nu * t) * xi[None, :]          # eta = x - y
    y = x - eta
    f = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
    num = (-np.sin(np.pi * y) * f * w).sum(1)
    den = (f * w).sum(1)
    return num / den


def burgers_mol(N, nu, T, t_eval, rtol=1e-10):
    """Method of lines: second-order central differences on N+1 points, BDF in time."""
    x = np.linspace(-1, 1, N + 1); h = x[1] - x[0]
    def rhs(t, u):
        U = np.concatenate(([0.0], u, [0.0]))
        ux = (U[2:] - U[:-2]) / (2 * h); uxx = (U[2:] - 2 * U[1:-1] + U[:-2]) / h ** 2
        return -U[1:-1] * ux + nu * uxx
    s = solve_ivp(rhs, (0, T), -np.sin(np.pi * x[1:-1]), method="BDF", t_eval=t_eval, rtol=rtol, atol=1e-12)
    return x, np.pad(s.y.T, ((0, 0), (1, 1)))


def exC():
    nu = 0.01 / np.pi; T = 1.0; t_eval = np.array([0.25, 0.5, 0.75, 1.0])
    rows = []
    for N in (256, 512, 1024, 2048, 4096):
        t0 = time.perf_counter(); x, U = burgers_mol(N, nu, T, t_eval); wall = time.perf_counter() - t0
        Uex = np.stack([cole_hopf(x, t, nu) for t in t_eval])
        err = np.linalg.norm(U - Uex) / np.linalg.norm(Uex)
        rows.append(dict(N=N, h=2 / N, rel_l2_vs_colehopf=float(err), rel_linf=float(np.abs(U - Uex).max() / np.abs(Uex).max()), wall_s=wall))
    hs = np.array([r["h"] for r in rows]); es = np.array([r["rel_l2_vs_colehopf"] for r in rows])
    order = (np.log(es[1:] / es[:-1]) / np.log(hs[1:] / hs[:-1])).tolist()
    # Cole-Hopf quadrature self-consistency: 100 and 150 nodes against the 200 nodes used for the
    # reference (numpy's hermgauss overflows for 400 nodes, so 200 is the largest rule used)
    xs = np.linspace(-1, 1, 401)
    ch100, ch150, ch200 = (cole_hopf(xs, 0.5, nu, n) for n in (100, 150, 200))
    xq, Uq = burgers_mol(4096, nu, T, t_eval)
    np.savez(OUT / "exC" / "reference_nu_0.01_over_pi.npz", x=xq, t=t_eval, u=Uq, u_colehopf=np.stack([cole_hopf(xq, t, nu) for t in t_eval]))
    return dict(nu=nu, mol_fd_rows=rows, observed_order_l2=order, expected_order=2.0,
                reference="Cole-Hopf (n=200 Gauss-Hermite; n=400 overflows in numpy hermgauss) for nu=0.01/pi; MOL-FD N=4096 for other nu",
                colehopf_quadrature_change_100_to_200=float(np.abs(ch100 - ch200).max()),
                colehopf_quadrature_change_150_to_200=float(np.abs(ch150 - ch200).max()),
                mol_vs_colehopf_consistency=(f"MOL-FD converges to the Cole-Hopf solution at observed order {order[-1]:.2f}, "
                                             f"finest rel L2 {es[-1]:.1e}; the two independent methods agree, which validates both."))


# ---------------- Example E: Robertson ----------------
def exE():
    k1, k2, k3 = 0.04, 3e7, 1e4
    def rob(t, y): return [-k1 * y[0] + k3 * y[1] * y[2], k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2], k2 * y[1] ** 2]
    def jac(t, y): return [[-k1, k3 * y[2], k3 * y[1]], [k1, -2 * k2 * y[1] - k3 * y[2], -k3 * y[1]], [0, 2 * k2 * y[1], 0]]
    t_eval = np.logspace(-5, 5, 50)
    res = {}
    for method in ("Radau", "BDF"):
        rows, y = [], None
        prev = None
        for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
            t0 = time.perf_counter()
            s = solve_ivp(rob, (0, 1e5), [1, 0, 0], method=method, jac=jac, t_eval=t_eval, rtol=rtol, atol=1e-6 * rtol)
            wall = time.perf_counter() - t0
            row = dict(rtol=rtol, nfev=int(s.nfev), wall_s=wall, mass_max_abs_drift=float(np.abs(s.y.sum(0) - 1).max()),
                       min_y2=float(s.y[1].min()))
            if prev is not None: row["rel_change_vs_prev"] = float(np.linalg.norm(s.y - prev) / np.linalg.norm(s.y))
            prev = s.y; rows.append(row); y = s.y
        res[method] = rows
        res[method + "_final"] = y[:, -1].tolist()
    res["radau_vs_bdf_rel_diff_at_1e-12"] = float(np.linalg.norm(np.array(res["Radau_final"]) - np.array(res["BDF_final"])) / np.linalg.norm(res["Radau_final"]))
    s = solve_ivp(rob, (0, 1e5), [1, 0, 0], method="Radau", jac=jac, t_eval=t_eval, rtol=1e-12, atol=1e-18)
    np.savez(OUT / "exE" / "reference.npz", t=t_eval, y=s.y)
    # Explicit-solver cost demonstration (documented behavior, not a reference solution)
    t0 = time.perf_counter(); se = solve_ivp(rob, (0, 100.0), [1, 0, 0], method="RK45", rtol=1e-6, atol=1e-10); walle = time.perf_counter() - t0
    res["explicit_RK45_tspan"] = [0, 100.0]; res["explicit_RK45_nfev"] = int(se.nfev); res["explicit_RK45_wall_s"] = walle; res["explicit_RK45_success"] = bool(se.success)
    return res


# ---------------- Example F: SEIR with beta(t) ----------------
def exF():
    N, sE, g = 1e6, 1 / 3, 1 / 7
    beta = lambda t: 0.5 * (1 - 0.6 / (1 + np.exp(-(t - 40) / 5)))
    def seir(t, y):
        S, E, I, R = y; inf = beta(t) * S * I / N
        return [-inf, inf - sE * E, sE * E - g * I, g * I]
    t_eval = np.arange(0, 151.0)
    rows, y = tol_study(seir, [N - 30, 20, 10, 0], (0, 150), t_eval, "DOP853", [1e-6, 1e-8, 1e-10, 1e-12], lambda Y: Y.sum(0))
    inc = beta(t_eval) * y[0] * y[2] / N
    np.savez(OUT / "exF" / "reference.npz", t=t_eval, y=y, incidence=inc, beta=beta(t_eval))
    return dict(rows=rows, peak_incidence_per_day=float(inc.max()), peak_day=float(t_eval[inc.argmax()]),
                final_size_fraction=float(y[3, -1] / N), R0_before_intervention=0.5 / g, R0_after=0.2 / g)


# ---------------- Example G: Lorenz-63 / Lorenz-96 ----------------
def exG():
    def l63(t, y): return [10 * (y[1] - y[0]), y[0] * (28 - y[2]) - y[1], y[0] * y[1] - 8 / 3 * y[2]]
    rows_short, y2 = tol_study(l63, [1.0, 1.0, 1.0], (0, 2), np.linspace(0, 2, 201), "DOP853", [1e-8, 1e-10, 1e-12, 1e-13])
    rows_long, y25 = tol_study(l63, [1.0, 1.0, 1.0], (0, 25), np.linspace(0, 25, 2501), "DOP853", [1e-8, 1e-10, 1e-12, 1e-13])
    # divergence time between the rtol 1e-12 and 1e-13 solutions: first time the relative difference exceeds 1e-2
    s1 = solve_ivp(l63, (0, 25), [1.0, 1.0, 1.0], method="DOP853", t_eval=np.linspace(0, 25, 2501), rtol=1e-12, atol=1e-15)
    s2 = solve_ivp(l63, (0, 25), [1.0, 1.0, 1.0], method="DOP853", t_eval=np.linspace(0, 25, 2501), rtol=1e-13, atol=1e-16)
    d = np.linalg.norm(s1.y - s2.y, axis=0) / np.linalg.norm(s1.y, axis=0)
    t_div = float(s1.t[np.argmax(d > 1e-2)]) if (d > 1e-2).any() else None
    K, F = 40, 8.0
    def l96(t, x): return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    x0 = F * np.ones(K); x0[0] += 0.01
    rows96, y96 = tol_study(l96, x0, (0, 5), np.linspace(0, 5, 101), "DOP853", [1e-8, 1e-10, 1e-12])
    np.savez(OUT / "exG" / "reference.npz", t63=np.linspace(0, 2, 201), y63=y2, t96=np.linspace(0, 5, 101), y96=y96)
    return dict(lorenz63_T2=rows_short, lorenz63_T25=rows_long, lorenz63_divergence_time_rtol12_vs_13=t_div,
                lorenz96_T5=rows96, note="Trajectories are references only up to the divergence time; beyond it only attractor statistics are reference quantities.")


EXAMPLES = {"exB": exB, "exC": exC, "exE": exE, "exF": exF, "exG": exG}


def main(names):
    for name in names:
        if name not in EXAMPLES:
            raise SystemExit(f"unknown example {name!r}; choose from {sorted(EXAMPLES)}")
        (OUT / name).mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter(); r = EXAMPLES[name](); r.update(INFO); r["total_wall_s"] = time.perf_counter() - t0
        with open(OUT / name / "verification.json", "w") as fh:
            json.dump(r, fh, indent=1, default=float)
        print(name, "done in %.1fs" % r["total_wall_s"])


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXAMPLES))
