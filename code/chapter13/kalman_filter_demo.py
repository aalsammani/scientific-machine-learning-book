"""Kalman filter for a damped oscillator observed in position only (Chapter 13, section 13.5;
Figure 13.1, right panel).

System (explicit-Euler discretization of x'' + 2 zeta omega0 x' + omega0^2 x = 0, dt = 0.1):
    x_{k+1} = A x_k + w_k,  A = [[1, dt], [-omega0^2 dt, 1 - 2 zeta omega0 dt]],  w_k ~ N(0, Q)
    y_k     = x_k[0] + v_k, v_k ~ N(0, R)
omega0 = 1, zeta = 0.05, Q = diag(1e-6, 1e-5), R = 0.05^2, 300 steps, true x_0 = (1, 0).
Filter initialized at (y_0, 0) with covariance diag(R, 1). The model-only forecast propagates the same
initial estimate with A and no observations. Errors are relative L2 errors over the whole trajectory
(position and, separately, the never-observed velocity). The same protocol is repeated over
N_SEEDS = 50 independent realizations (seed s = numpy default_rng(s)); seed 0 is stored in full for
the figure.

Usage:
    python code/chapter13/kalman_filter_demo.py
Output: results/chapter13/kalman_filter_study.json
Runtime: a few seconds. Deterministic (fixed seeds).
"""
import json, os
from pathlib import Path
import numpy as np

DT, OMEGA0, ZETA, NSTEP = 0.1, 1.0, 0.05, 300
Q = np.diag([1e-6, 1e-5]); R = 0.05 ** 2
A = np.array([[1.0, DT], [-OMEGA0 ** 2 * DT, 1.0 - 2 * ZETA * OMEGA0 * DT]])
H = np.array([[1.0, 0.0]])
N_SEEDS = 50


def run(seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((NSTEP, 2)); x[0] = [1.0, 0.0]
    for k in range(NSTEP - 1):
        x[k + 1] = A @ x[k] + rng.multivariate_normal([0, 0], Q)
    y = x[:, 0] + np.sqrt(R) * rng.standard_normal(NSTEP)
    xe = np.zeros((NSTEP, 2)); xm = np.zeros((NSTEP, 2))
    m = np.array([y[0], 0.0]); P = np.diag([R, 1.0])
    xe[0] = m; xm[0] = m
    for k in range(1, NSTEP):
        m = A @ m; P = A @ P @ A.T + Q                        # predict
        S = H @ P @ H.T + R; K = P @ H.T / S                 # update (Theorem 5.1)
        m = m + (K * (y[k] - H @ m)).ravel(); P = (np.eye(2) - K @ H) @ P
        xe[k] = m; xm[k] = A @ xm[k - 1]
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return dict(err_kalman_position=rel(xe[:, 0], x[:, 0]), err_raw_obs=rel(y, x[:, 0]),
                err_model_only=rel(xm[:, 0], x[:, 0]), err_kalman_velocity=rel(xe[:, 1], x[:, 1]),
                err_model_only_velocity=rel(xm[:, 1], x[:, 1])), (x, y, xe, xm)


def main():
    rows = []
    for s in range(N_SEEDS):
        r, traj = run(s)
        rows.append(r)
        if s == 0:
            x0, y0, xe0, xm0 = traj
    keys = rows[0].keys()
    summ = {k: dict(median=float(np.median([r[k] for r in rows])), min=float(np.min([r[k] for r in rows])),
                    max=float(np.max([r[k] for r in rows]))) for k in keys}
    frac = float(np.mean([r["err_kalman_position"] < min(r["err_raw_obs"], r["err_model_only"]) for r in rows]))
    fracv = float(np.mean([r["err_kalman_velocity"] < r["err_model_only_velocity"] for r in rows]))
    out = dict(protocol=__doc__, n_seeds=N_SEEDS, seed0=rows[0], summary=summ,
               fraction_kalman_better_than_both=frac, fraction_kalman_velocity_better_than_model_only=fracv,
               seed0_trajectory=dict(t=(DT * np.arange(NSTEP)).tolist(), x_true=x0.tolist(), y_obs=y0.tolist(),
                                     x_kalman=xe0.tolist(), x_model_only=xm0.tolist()))
    results = Path(__file__).resolve().parents[2] / "results" / "chapter13"
    os.makedirs(results, exist_ok=True)
    json.dump(out, open(results / "kalman_filter_study.json", "w"), indent=1)
    print("seed0", rows[0]); print(json.dumps(summ, indent=1)); print("fraction better than both", frac, "velocity better than model-only", fracv)


if __name__ == "__main__":
    main()
