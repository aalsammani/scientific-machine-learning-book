"""Calibration of a ten-member deep ensemble of physics-informed networks for Running Example A
(Chapter 14, section 14.3; Figure 14.1).

Protocol: heat equation of Chapter 1 (code/reference_problems/exA_heat.py: nu = 0.5, T = 1,
two-mode initial condition); each member is the 3 x 32 tanh network of Chapter 1
(code/chapter01/exA_three_ways_pytorch.py: same architecture, initialization and collocation
construction, n_r = 2000, n_b = n_0 = 100), trained with Adam (1500 iterations, learning rate 1e-3)
and no L-BFGS stage; members use seeds 0-9. The ensemble mean m and standard deviation s at 200
evaluation points drawn uniformly in [0, 1] x [0, T] (seed 999) define Gaussian intervals
m +- z_q s; empirical coverage of the exact solution is reported at five nominal levels
(50, 68, 80, 90, 95%). A second set of 200 points (seed 1999) shows the evaluation-set
variability, and a set of 2000 points (seed 2999) gives the coverage with small sampling error.

Usage:
    python code/chapter14/calibration_check.py
Output: results/chapter14/calibration_check_rerun.json
Runtime: about 1 minute with the ten members trained in parallel processes (one thread each).
Stochastic but seeded. Note: the protocol string stored in the committed result file states T = 0.2, while the
code takes T = 1 from exA_heat.py; see code/chapter14/README.md.
"""
import json, os, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import torch
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter01"))
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
torch.set_default_dtype(torch.float64)

N_MEMBERS, ADAM_ITERS, LR = 10, 1500, 1e-3
LEVELS = [0.5, 0.68, 0.8, 0.9, 0.95]
EVAL_SETS = [(999, 200), (1999, 200), (2999, 2000)]   # (seed, number of evaluation points)


def member(seed):
    torch.set_num_threads(1)
    import exA_three_ways_pytorch as ex
    g = torch.Generator().manual_seed(seed + 1000)
    xr = torch.rand(ex.N_R, generator=g); tr = torch.rand(ex.N_R, generator=g) * ex.T
    tb = torch.rand(ex.N_B, generator=g) * ex.T
    x0 = torch.rand(ex.N_0, generator=g); u0 = torch.tensor(ex.exact(x0.numpy(), 0.0))
    model = ex.MLP(ex.LAYERS, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(ADAM_ITERS):
        opt.zero_grad()
        l = ((ex.residual(model, xr, tr) ** 2).mean()
             + (model(torch.zeros(ex.N_B), tb) ** 2).mean() + (model(torch.ones(ex.N_B), tb) ** 2).mean()
             + ((model(x0, torch.zeros(ex.N_0)) - u0) ** 2).mean())
        l.backward(); opt.step()
    out = {}
    for es, npts in EVAL_SETS:
        rng = np.random.default_rng(es)
        xe, te = rng.uniform(0, 1, npts), rng.uniform(0, ex.T, npts)
        with torch.no_grad():
            out[es] = model(torch.tensor(xe), torch.tensor(te)).numpy().tolist()
    return seed, out


def main():
    from exA_heat import exact, T
    with Pool(N_MEMBERS) as p:
        res = dict(p.map(member, range(N_MEMBERS)))
    summary = {}
    for es, npts in EVAL_SETS:
        rng = np.random.default_rng(es)
        xe, te = rng.uniform(0, 1, npts), rng.uniform(0, T, npts)
        U = np.array([res[s][es] for s in range(N_MEMBERS)])
        m, s = U.mean(0), U.std(0, ddof=1)
        ue = exact(xe, te)
        cov = {str(q): float(np.mean(np.abs(ue - m) <= norm.ppf(0.5 + q / 2) * s)) for q in LEVELS}
        summary[f"eval_seed_{es}"] = dict(coverage=cov, mean_abs_error=float(np.mean(np.abs(ue - m))),
                                          mean_std=float(np.mean(s)),
                                          rel_l2_ensemble_mean=float(np.linalg.norm(ue - m) / np.linalg.norm(ue)))
        print(es, json.dumps(summary[f"eval_seed_{es}"]))
    out = dict(protocol=__doc__, n_members=N_MEMBERS, levels=LEVELS, **summary)
    json.dump(out, open(REPO_ROOT / "results" / "chapter14" / "calibration_check_rerun.json", "w"), indent=1)


if __name__ == "__main__":
    main()
