"""Lorenz-63 separation of two nearby trajectories (Chapter 2, section 2.3; Figure 2.1).

Protocol:
  * Lorenz-63 with sigma = 10, rho = 28, beta = 8/3.
  * Initial conditions (1, 1, 1) and (1 + 1e-8, 1, 1).
  * scipy.integrate.solve_ivp, DOP853, rtol = 1e-13, atol = 1e-15 (primary configuration).
  * Output every 0.1 on [0, 30]; separation = Euclidean norm of the state difference; growth rate =
    least-squares slope of log(separation) over t in [14, 29].
Also reported: eigenvalues of the Jacobian at the equilibrium C+, lobe-switch times of the reference
trajectory, and the first time the separation reaches 1.

Provenance: results/chapter02/lorenz_sensitivity.json is the stored run plotted in Figure 2.1; the
script that produced it was not preserved. This script reconstructs the protocol, reproduces the
stored fitted rate (0.889), and records the separation checkpoints, eigenvalues and lobe-switch times
quoted in section 2.3. Its separation curve differs from the stored one by at most about 2e-4
relative, so the stored run's absolute tolerance is not confirmed; a solver scan is recorded.
results_lorenz_divergence.json is a second stored run used only for the comparison scan.

Usage:
    python code/chapter02/lorenz_sensitivity.py
Output: results/chapter02/lorenz_sensitivity_rerun.json (the stored files are only read).
Runtime: under a minute. Deterministic.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

SIG, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
EPS = 1e-8
RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter02"
STORED_MAIN = RESULTS / "lorenz_sensitivity.json"
STORED_DIV = RESULTS / "results_lorenz_divergence.json"
OUT = RESULTS / "lorenz_sensitivity_rerun.json"


def lorenz(t, x):
    return [SIG * (x[1] - x[0]), x[0] * (RHO - x[2]) - x[1], x[0] * x[1] - BETA * x[2]]


def run(t_end, n_out, method, rtol, atol):
    t = np.linspace(0.0, t_end, n_out)
    a = solve_ivp(lorenz, (0.0, t_end), [1.0, 1.0, 1.0], method=method, rtol=rtol, atol=atol, t_eval=t)
    b = solve_ivp(lorenz, (0.0, t_end), [1.0 + EPS, 1.0, 1.0], method=method, rtol=rtol, atol=atol, t_eval=t)
    return t, a.y, b.y, np.linalg.norm(a.y - b.y, axis=0)


def fit_rate(t, d, lo, hi):
    m = (t >= lo - 1e-12) & (t <= hi + 1e-12)
    return float(np.polyfit(t[m], np.log(d[m]), 1)[0])


def rel_mismatch(d, ref):
    return float(np.max(np.abs(d - ref) / np.abs(ref)))


def main():
    with open(STORED_MAIN) as f:
        s_main = json.load(f)
    with open(STORED_DIV) as f:
        s_div = json.load(f)
    ref_main = np.array(s_main["diff"])
    ref_div = np.array(s_div["diff"])

    configs = [("DOP853", 1e-13, 1e-15), ("DOP853", 1e-13, 1e-13), ("DOP853", 1e-13, 1e-14),
               ("DOP853", 1e-12, 1e-12), ("DOP853", 1e-12, 1e-14), ("DOP853", 1e-12, 1e-15),
               ("DOP853", 1e-10, 1e-12), ("RK45", 1e-12, 1e-14), ("LSODA", 1e-12, 1e-14)]
    scan = []
    for method, rtol, atol in configs:
        t, _, _, d = run(30.0, 301, method, rtol, atol)
        t2, _, _, d2 = run(20.0, 401, method, rtol, atol)
        scan.append({
            "method": method, "rtol": rtol, "atol": atol,
            "main_max_rel_mismatch_vs_stored": rel_mismatch(d, ref_main),
            "main_max_rel_mismatch_t_le_12": rel_mismatch(d[t <= 12], ref_main[t <= 12]),
            "main_fitted_rate_14_29": fit_rate(t, d, 14, 29),
            "div_max_rel_mismatch_vs_stored": rel_mismatch(d2, ref_div),
            "div_max_rel_mismatch_t_le_12": rel_mismatch(d2[t2 <= 12], ref_div[t2 <= 12]),
        })
        print(scan[-1])

    # primary configuration as documented in the spec
    t, A, B, d = run(30.0, 301, "DOP853", 1e-13, 1e-15)
    rate = fit_rate(t, d, 14, 29)
    first_O1 = t[np.argmax(d >= 1.0)] if np.any(d >= 1.0) else None
    c = np.sqrt(BETA * (RHO - 1.0))
    J = np.array([[-SIG, SIG, 0.0], [1.0, -1.0, -c], [c, c, -BETA]])
    eig = np.linalg.eigvals(J)
    # lobe switches from a finely sampled reference trajectory
    tf = np.linspace(0.0, 30.0, 30001)
    Af = solve_ivp(lorenz, (0.0, 30.0), [1.0, 1.0, 1.0], method="DOP853", rtol=1e-13, atol=1e-15, t_eval=tf).y
    sw = tf[1:][np.diff(np.sign(Af[0])) != 0]

    # secondary run and candidate definitions of "lyapunov_estimate"
    t2, _, _, d2 = run(20.0, 401, "DOP853", 1e-13, 1e-15)
    cands = {
        "polyfit_all_t_0_20": fit_rate(t2, d2, 0, 20),
        "polyfit_t_1_20": fit_rate(t2, d2, 1, 20),
        "polyfit_t_5_20": fit_rate(t2, d2, 5, 20),
        "polyfit_t_10_20": fit_rate(t2, d2, 10, 20),
        "polyfit_t_14_20": fit_rate(t2, d2, 14, 20),
        "log_ratio_end_over_start_div_T": float(np.log(d2[-1] / d2[0]) / 20.0),
        "log_ratio_end_over_eps_div_T": float(np.log(d2[-1] / EPS) / 20.0),
        "polyfit_first_half_t_0_10": fit_rate(t2, d2, 0, 10),
        "polyfit_second_half_t_10_20": fit_rate(t2, d2, 10, 20),
    }
    ts = np.array(s_div["t"])
    lg = np.log(ref_div / EPS)
    cands_on_stored = {
        "lsq_through_origin_log(sep/eps)_t_0_20": float(np.sum(ts * lg) / np.sum(ts * ts)),
        "polyfit_last_half_of_samples": float(np.polyfit(ts[len(ts) // 2:], np.log(ref_div[len(ts) // 2:]), 1)[0]),
        "polyfit_log_sep_t_gt_0": fit_rate(ts[1:], ref_div[1:], 0.01, 20),
        "mean_of_log(sep/eps)/t_t_gt_0": float(np.mean(lg[1:] / ts[1:])),
        "window_scan_lo_to_20_closest_to_stored": min(
            ((lo, fit_rate(ts, ref_div, lo, 20)) for lo in np.arange(0, 19, 0.5)),
            key=lambda p: abs(p[1] - s_div["lyapunov_estimate"])),
        "polyfit_all_t_0_20": fit_rate(np.array(s_div["t"]), ref_div, 0, 20),
        "log_ratio_end_over_start_div_T": float(np.log(ref_div[-1] / ref_div[0]) / 20.0),
        "polyfit_t_14_20": fit_rate(np.array(s_div["t"]), ref_div, 14, 20),
        "polyfit_t_10_20": fit_rate(np.array(s_div["t"]), ref_div, 10, 20),
    }

    def at(tt):
        return float(d[int(round(tt / 0.1))])

    out = {
        "protocol": "Lorenz-63 (10,28,8/3); ICs (1,1,1),(1+1e-8,1,1); solve_ivp DOP853 rtol=1e-13 atol=1e-15; "
                    "t_eval=linspace(0,30,301); rate = polyfit slope of log(sep) on t in [14,29]",
        "eps": EPS,
        "t": t.tolist(),
        "diff": d.tolist(),
        "fitted_rate_14_to_29": rate,
        "stored_fitted_rate_14_to_29": s_main["fitted_rate_14_to_29"],
        "max_rel_mismatch_vs_stored_diff": rel_mismatch(d, ref_main),
        "separation_at": {str(tt): at(tt) for tt in (0.3, 1.0, 5.0, 10.0, 12.0, 13.0, 13.5, 14.0, 20.0, 25.0, 29.0, 29.2, 30.0)},
        "max_sep_t_1_to_13": float(np.max(d[(t >= 1) & (t <= 13)])),
        "max_sep_t_0_to_1": float(np.max(d[t <= 1])),
        "first_t_sep_ge_1": None if first_O1 is None else float(first_O1),
        "jacobian_eigenvalues_at_C_plus": [[float(e.real), float(e.imag)] for e in eig],
        "lobe_switch_times": [float(x) for x in sw[:12]],
        "divergence_run_candidates_rerun": cands,
        "divergence_run_candidates_on_stored_diff": cands_on_stored,
        "stored_divergence_lyapunov_estimate": s_div["lyapunov_estimate"],
        "divergence_run_max_rel_mismatch": rel_mismatch(d2, ref_div),
        "solver_scan": scan,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        if k not in ("t", "diff", "solver_scan"):
            print(k, ":", v)


if __name__ == "__main__":
    main()
