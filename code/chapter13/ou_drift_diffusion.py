"""Drift and diffusion estimation for the Ornstein-Uhlenbeck process (Chapter 13, section 13.4;
Theorem 13.2; Figure 13.1, left panel)

    dX = -theta X dt + sigma dW,  theta = 0.5, sigma = 0.4,

observed at spacing dt in {0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4} over a record of fixed length
T = 2000; 20 records per dt (numpy default_rng(10000 i + r)).

The data are generated with the exact OU transition
    X_{n+1} = exp(-theta dt) X_n + sigma sqrt((1 - exp(-2 theta dt)) / (2 theta)) Z_n,
started from the stationary distribution, so no simulation error enters at any dt.
Estimators (Theorem 13.2):
    drift      theta_hat   = -sum X_n dX_n / (dt sum X_n^2)          (least-squares drift fit)
    diffusion  sigma_hat^2 = sum dX_n^2 / (n dt)                      (realized quadratic variation)
Their large-sample limits at fixed dt are (1 - exp(-theta dt))/dt and
sigma^2 (1 - exp(-theta dt)) / (theta dt), both of which tend to the true values as dt -> 0.

--legacy-em generates the data instead with the Euler-Maruyama scheme at the sampling step itself
(which is not a sample of the OU process at coarse dt). This is the pitfall discussed in
section 13.4: with such data the drift estimate looks unbiased at every dt.

Usage:
    python code/chapter13/ou_drift_diffusion.py
    python code/chapter13/ou_drift_diffusion.py --legacy-em

Outputs: results/chapter13/ou_drift_diffusion.json, or ou_drift_diffusion_legacy_em.json with
--legacy-em. Runtime: under a minute. Deterministic (fixed seeds).
"""
import argparse, json, os
from pathlib import Path
import numpy as np

THETA, SIGMA, T_TOTAL, N_REPS = 0.5, 0.4, 2000.0, 20
DTS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]


def simulate_fast(dt, rng, legacy_em=False):
    """AR(1) recursion x_{k+1} = a x_k + s z_k, vectorized with scipy.signal.lfilter."""
    from scipy.signal import lfilter
    n = int(round(T_TOTAL / dt))
    if legacy_em:
        a, s, x0 = 1.0 - THETA * dt, SIGMA * np.sqrt(dt), 0.0
        z = rng.standard_normal(n)
    else:
        a = np.exp(-THETA * dt)
        s = SIGMA * np.sqrt((1.0 - a * a) / (2.0 * THETA))
        x0 = rng.standard_normal() * SIGMA / np.sqrt(2.0 * THETA)
        z = rng.standard_normal(n)
    rest, _ = lfilter([1.0], [1.0, -a], s * z, zi=[a * x0])
    return np.concatenate([[x0], rest])


def estimate(x, dt):
    xn, dx = x[:-1], np.diff(x)
    theta_hat = -np.dot(xn, dx) / (dt * np.dot(xn, xn))
    sigma_hat = np.sqrt(np.dot(dx, dx) / (len(dx) * dt))
    return theta_hat, sigma_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-em", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = Path(__file__).resolve().parents[2] / "results" / "chapter13"
    out = args.out or str(results / ("ou_drift_diffusion_legacy_em.json" if args.legacy_em else "ou_drift_diffusion.json"))
    res = []
    for i, dt in enumerate(DTS):
        th, sg = [], []
        for r in range(N_REPS):
            rng = np.random.default_rng(10_000 * i + r)
            x = simulate_fast(dt, rng, args.legacy_em)
            a, b = estimate(x, dt)
            th.append(a); sg.append(b)
        th, sg = np.array(th), np.array(sg)
        lim_th = (1 - np.exp(-THETA * dt)) / dt
        lim_sg = SIGMA * np.sqrt((1 - np.exp(-THETA * dt)) / (THETA * dt))
        row = dict(dt=dt, n_increments=int(round(T_TOTAL / dt)),
                   theta_mean=th.mean(), theta_std=th.std(ddof=1), theta_bias_rel=th.mean() / THETA - 1,
                   sigma_mean=sg.mean(), sigma_std=sg.std(ddof=1), sigma_bias_rel=sg.mean() / SIGMA - 1,
                   theta_limit_exact=None if args.legacy_em else lim_th,
                   sigma_limit_exact=None if args.legacy_em else lim_sg)
        res.append(row)
        print(f"dt={dt:7.3f}  theta {th.mean():.4f} +- {th.std(ddof=1):.4f} (bias {100*(th.mean()/THETA-1):+.2f}%)"
              f"   sigma {sg.mean():.4f} +- {sg.std(ddof=1):.4f} (bias {100*(sg.mean()/SIGMA-1):+.2f}%)")
    meta = dict(theta_true=THETA, sigma_true=SIGMA, T_total=T_TOTAL, n_reps=N_REPS,
                data_generation="Euler-Maruyama at the sampling step (not a sample of the OU process)" if args.legacy_em
                else "exact OU transition, stationary start",
                theta_std_asymptotic_sqrt_2theta_over_T=float(np.sqrt(2 * THETA / T_TOTAL)),
                seeds="numpy default_rng(10000*i_dt + rep)", results=res)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(meta, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
