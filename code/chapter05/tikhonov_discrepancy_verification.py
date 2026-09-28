"""Tikhonov regularization and the discrepancy principle on the backward heat equation (Chapter 5,
section 5.3; Theorem 5.3; Figure 5.1, left panel).

  * Sine-mode representation on (0, 1): K = 40 modes, forward singular values
    s_k = exp(-nu k^2 pi^2 T), nu = 0.5, T = 0.05 (diagonal forward operator).
  * True coefficients p_1 = 1, p_5 = 0.5, others 0 (the Example A initial condition).
  * Data y = s * p; noise sigma = 0.01 max|y|, y_delta = y + sigma z with
    z = numpy.random.default_rng(0).standard_normal(40).
  * Tikhonov estimate p_alpha = s y_delta / (s^2 + alpha), alpha on logspace(-8, 0, 60).
  * Oracle: grid alpha minimizing ||p_alpha - p_true||_2. Discrepancy principle: delta = sigma
    sqrt(K), tau = 1.2, largest grid alpha with residual <= tau delta.
  * The "near-unregularized" error is Tikhonov with alpha = 1e-12; alpha = 0 overflows in float64
    because s_40 = exp(-394.8).
This protocol reproduces the stored original (tikhonov_discrepancy_verification.json) to round-off.

Usage:
    python code/chapter05/tikhonov_discrepancy_verification.py
Output: results/chapter05/tikhonov_discrepancy_verification_rerun.json
Runtime: under a second. Deterministic (fixed seed).
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter05"
OUT = RESULTS / "tikhonov_discrepancy_verification_rerun.json"
OLD = RESULTS / "tikhonov_discrepancy_verification.json"
NU, T, K, TAU = 0.5, 0.05, 40, 1.2


def main():
    k = np.arange(1, K + 1)
    s = np.exp(-NU * k ** 2 * np.pi ** 2 * T)
    p_true = np.zeros(K)
    p_true[0], p_true[4] = 1.0, 0.5
    y = s * p_true
    sigma = 0.01 * np.max(np.abs(y))
    z = np.random.default_rng(0).standard_normal(K)
    noise = sigma * z
    yd = y + noise
    alphas = np.logspace(-8, 0, 60)
    P = np.array([s * yd / (s ** 2 + a) for a in alphas])
    errs = np.linalg.norm(P - p_true, axis=1)
    res = np.linalg.norm(P * s - yd, axis=1)
    io = int(np.argmin(errs))
    delta = sigma * np.sqrt(K)
    idp = int(np.where(res <= TAU * delta)[0].max())
    naive_alpha = 1e-12
    naive_err = float(np.linalg.norm(s * yd / (s ** 2 + naive_alpha) - p_true))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        unreg = yd / s
        unreg_err = float(np.linalg.norm(unreg - p_true))
    pc_oracle = P[io]
    out = {
        "nu": NU, "T": T, "K": K, "sigma": float(sigma), "tau": TAU,
        "noise_model": "sigma = 0.01*max|y|, iid N(0,sigma^2) per mode, default_rng(0)",
        "noise_norm_actual": float(np.linalg.norm(noise)), "delta_used": float(delta),
        "noise_to_signal_norm_ratio": float(np.linalg.norm(noise) / np.linalg.norm(y)),
        "alphas": alphas.tolist(), "errs": errs.tolist(), "residuals": res.tolist(),
        "oracle_best_alpha": float(alphas[io]), "oracle_best_err": float(errs[io]),
        "discrepancy_alpha": float(alphas[idp]), "discrepancy_err": float(errs[idp]),
        "discrepancy_residual": float(res[idp]),
        "discrepancy_gap_rel": float(errs[idp] / errs[io] - 1),
        "naive_alpha": naive_alpha, "naive_err": naive_err,
        "unregularized_alpha0_err": unreg_err,
        "oracle_mode1": float(pc_oracle[0]), "oracle_mode5": float(pc_oracle[4]),
        "mode5_damping": float(s[4]), "mode5_data_contribution": float(y[4]),
        "max_amplification_bound_oracle": float(1 / (2 * np.sqrt(alphas[io]))),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    old = json.load(open(OLD))
    dev = max(np.max(np.abs(errs / np.array(old["errs"]) - 1)), np.max(np.abs(res / np.array(old["residuals"]) - 1)),
              abs(naive_err / old["naive_err"] - 1), abs(sigma / old["sigma"] - 1))
    print("max rel deviation vs stored:", dev)
    for kk in ["sigma", "noise_norm_actual", "delta_used", "noise_to_signal_norm_ratio", "oracle_best_alpha",
               "oracle_best_err", "discrepancy_alpha", "discrepancy_err", "discrepancy_residual",
               "discrepancy_gap_rel", "naive_err", "unregularized_alpha0_err", "oracle_mode1", "oracle_mode5",
               "mode5_damping", "mode5_data_contribution"]:
        print("%-28s %s" % (kk, out[kk]))
    print("err at smallest grid alpha 1e-8:", errs[0])


if __name__ == "__main__":
    main()
