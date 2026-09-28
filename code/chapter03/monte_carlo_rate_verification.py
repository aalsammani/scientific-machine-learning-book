"""Monte Carlo O(1/n) variance rate (Chapter 3, section 3.9; Figure 3.1, right panel).

  * Integrand g(x) = sin^2(pi x) + 0.3 x on [0, 1]; exact integral 0.65; exact
    Var(g(U)) = 3/8 + 0.6/4 + 0.09/3 - 0.65^2 = 0.1325 for U ~ Uniform(0, 1).
  * For n in {100, 400, 1600, 6400, 25600}: 200 independent repetitions of the plain Monte Carlo
    estimator; report the empirical variance of the 200 estimates and n times it.
  * Primary run: numpy.random.default_rng(0), draws of shape (200, n), ddof = 0.
The generator and seed of the stored original (monte_carlo_rate_verification.json, used by
Figure 3.1) were not recorded. The primary run above reproduces the stored values exactly; a scan
over other generators and seeds, and the spread of n Var over 50 further seeds, are also recorded.

Usage:
    python code/chapter03/monte_carlo_rate_verification.py
Output: results/chapter03/monte_carlo_rate_verification_rerun.json
Runtime: about 1-2 minutes. Pseudo-random with fixed seeds (repeatable).
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter03"
STORED = RESULTS / "monte_carlo_rate_verification.json"
OUT = RESULTS / "monte_carlo_rate_verification_rerun.json"
NS = [100, 400, 1600, 6400, 25600]
REPS = 200
TRUE = 0.65
SIGMA2 = 3 / 8 + 0.6 * 0.25 + 0.09 / 3 - 0.65 ** 2


def g(x):
    return np.sin(np.pi * x) ** 2 + 0.3 * x


def experiment(gen, seed, layout, ddof, fresh_per_n):
    rows = []
    if gen == "default_rng":
        rng = np.random.default_rng(seed)
        draw = lambda shape: rng.random(shape)
    else:
        rng = np.random.RandomState(seed)
        draw = lambda shape: rng.rand(*shape) if isinstance(shape, tuple) else rng.rand(shape)
    for n in NS:
        if fresh_per_n:
            if gen == "default_rng":
                rng = np.random.default_rng(seed)
                draw = lambda shape, rng=rng: rng.random(shape)
            else:
                rng = np.random.RandomState(seed)
                draw = lambda shape, rng=rng: rng.rand(*shape) if isinstance(shape, tuple) else rng.rand(shape)
        if layout == "matrix":
            est = g(draw((REPS, n))).mean(axis=1)
        else:
            est = np.array([g(draw(n)).mean() for _ in range(REPS)])
        rows.append({"n": n, "empirical_var": float(np.var(est, ddof=ddof)), "mean_est": float(est.mean()),
                     "n_times_var": float(n * np.var(est, ddof=ddof))})
    return rows


def main():
    with open(STORED) as fh:
        st = json.load(fh)
    s_var = np.array([r["empirical_var"] for r in st["rows"]])
    s_mean = np.array([r["mean_est"] for r in st["rows"]])
    scan = []
    for gen in ("default_rng", "RandomState"):
        for seed in (0, 1, 7, 42, 123, 1234, 2024, 2025, 12345, 20240101):
            for layout in ("matrix", "loop"):
                for fresh in (False, True):
                    for ddof in (0, 1):
                        rows = experiment(gen, seed, layout, ddof, fresh)
                        v = np.array([r["empirical_var"] for r in rows])
                        m = np.array([r["mean_est"] for r in rows])
                        scan.append({"gen": gen, "seed": seed, "layout": layout, "fresh_rng_per_n": fresh,
                                     "ddof": ddof,
                                     "max_rel_diff_var": float(np.max(np.abs(v - s_var) / s_var)),
                                     "max_abs_diff_mean": float(np.max(np.abs(m - s_mean)))})
    scan.sort(key=lambda d: d["max_abs_diff_mean"])
    exact_hits = [d for d in scan if d["max_abs_diff_mean"] < 1e-12]
    primary = experiment("default_rng", 0, "matrix", 0, False)
    # distribution of n*Var over many independent seeds: what spread is "sampling noise"?
    nv = np.array([[r["n_times_var"] for r in experiment("default_rng", s, "matrix", 0, False)] for s in range(100, 150)])
    out = {"true_value": TRUE, "exact_sigma2": SIGMA2, "reps": REPS,
           "primary_run": {"generator": "default_rng(0)", "layout": "(200,n) matrix", "ddof": 0, "rows": primary},
           "stored_n_times_var": [float(n * v) for n, v in zip(NS, s_var)],
           "sd_of_n_times_var_over_50_seeds_per_n": nv.std(axis=0).tolist(),
           "mean_of_n_times_var_over_50_seeds_per_n": nv.mean(axis=0).tolist(),
           "theoretical_sd_of_sample_var_times_n_approx": float(SIGMA2 * np.sqrt(2 / (REPS - 1))),
           "bitwise_reproducing_configs": exact_hits,
           "closest_configs": scan[:5]}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
