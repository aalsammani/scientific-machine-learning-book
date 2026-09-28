"""Diagnostics for the hard-constraint optimization difficulty (Chapter 7, section 7.7).

Retrains the hard-constraint ansatz of code/chapter06/constraint_ablation.py,
u = u0(x) + x(1-x) t N(x, t), with the same collocation seed (seed + 6000), and records per run:
  * the location (x, t) of the peak residual and the envelope x(1-x)t there;
  * Spearman correlations of |residual| with the envelope and with time on the evaluation grid;
  * gradient norms of the loss restricted to five time bins, at checkpoints during training;
  * relative errors of sine modes 1 and 5 at t = 0.02.
Adam (5000 iterations, learning rate 1e-3), then L-BFGS (at most 8000 iterations; the runs of
constraint_ablation.py allow 10000, so the final errors of the two scripts are not directly
comparable).

Usage:
    python code/chapter07/hard_constraint_diagnostics.py            # seeds 0-4, as in the book
    python code/chapter07/hard_constraint_diagnostics.py 3 4        # chosen seeds

Outputs: results/chapter07/ch08-hardconstraint-diag_seed{s}.json and *_history.json.
Runtime: about 3.5 minutes per seed on one core. Stochastic but seeded.
"""
import json, time, sys, os
from pathlib import Path
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import exact, NU, T, MODES  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter01"))
from exA_three_ways_pytorch import MLP, LAYERS, n_params  # noqa: E402

OUT = str(REPO_ROOT / "results" / "chapter07")
os.makedirs(OUT, exist_ok=True)
N_R = 2000
ADAM_ITERS = int(os.environ.get("ADAM_ITERS", 5000))
LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", 8000))
EVAL_X, EVAL_T = np.linspace(0, 1, 201), np.linspace(0, T, 101)


def u0_torch(x):
    return torch.sin(np.pi * x) + 0.5 * torch.sin(5 * np.pi * x)


def envelope(x, t):
    return x * (1 - x) * t


def residual_and_graph(u_fn, x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_fn(x, t)
    u_t, u_x = torch.autograd.grad(u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - NU * u_xx


def mode_coeffs(u_fn, t_val, K=8):
    xg = torch.linspace(0, 1, 401)
    with torch.no_grad():
        u = u_fn(xg, torch.full_like(xg, t_val))
    ks = torch.arange(1, K + 1, dtype=torch.float64)
    S = torch.sin(np.pi * ks[:, None] * xg[None, :])
    w = torch.full((401,), 1 / 400, dtype=torch.float64)
    w[0] = 0.5 / 400
    w[-1] = 0.5 / 400
    return (2 * (S * (u * w)[None, :]).sum(1)).tolist()


def exact_coeffs(t_val, K=8):
    c = np.zeros(K)
    for k, a in MODES:
        c[k - 1] = a * np.exp(-NU * (k * np.pi) ** 2 * t_val)
    return c


def run_hard_diag(seed):
    g = torch.Generator().manual_seed(seed + 6000)
    xr = torch.rand(N_R, generator=g)
    tr = torch.rand(N_R, generator=g) * T
    model = MLP(LAYERS, seed)

    def u_fn(x, t):
        return u0_torch(x) + envelope(x, t) * model(x, t)

    def loss_fn():
        r = residual_and_graph(u_fn, xr, tr)
        return (r ** 2).mean()

    # gradient-norm-by-time-bin diagnostic: compute, at three checkpoints, the norm of
    # d(loss)/d(theta) restricted to collocation points in each of 5 time bins
    def grad_norm_by_time_bin(n_bins=5):
        bins = np.linspace(0, T, n_bins + 1)
        norms = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (tr >= lo) & (tr < hi)
            if mask.sum() < 2:
                norms.append(float("nan"))
                continue
            xb, tb = xr[mask], tr[mask]
            r = residual_and_graph(u_fn, xb, tb)
            l = (r ** 2).mean()
            grads = torch.autograd.grad(l, list(model.parameters()), retain_graph=False, allow_unused=True)
            gn = float(np.sqrt(sum((g_.detach() ** 2).sum().item() for g_ in grads if g_ is not None)))
            norms.append(gn)
        return norms

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    t0 = time.perf_counter()
    for it in range(ADAM_ITERS):
        opt.zero_grad()
        l = loss_fn()
        l.backward()
        opt.step()
        if it % 500 == 0:
            gnb = grad_norm_by_time_bin()
            hist.append(dict(iter=it, phase="adam", loss=float(l.detach()), grad_norm_by_time_bin=gnb,
                              modes_t0=mode_coeffs(u_fn, 0.0), modes_t002=mode_coeffs(u_fn, 0.02)))
    adam_s = time.perf_counter() - t0

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS, max_eval=LBFGS_ITERS * 2,
                               tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50,
                               line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = loss_fn()
        l.backward()
        return l
    t0 = time.perf_counter()
    lbfgs.step(closure)
    lbfgs_s = time.perf_counter() - t0
    gnb = grad_norm_by_time_bin()
    hist.append(dict(iter=ADAM_ITERS + LBFGS_ITERS, phase="final", loss=float(loss_fn().detach()),
                      grad_norm_by_time_bin=gnb, modes_t0=mode_coeffs(u_fn, 0.0), modes_t002=mode_coeffs(u_fn, 0.02)))

    # final spatial residual and error maps, and envelope map, on the evaluation grid
    X, Tg = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
    Uex = exact(X, Tg)
    with torch.no_grad():
        U = u_fn(torch.tensor(X.ravel()), torch.tensor(Tg.ravel())).numpy().reshape(X.shape)
    e = U - Uex
    Xi, Ti = X[1:-1, 1:], Tg[1:-1, 1:]
    R = residual_and_graph(u_fn, torch.tensor(Xi.ravel()), torch.tensor(Ti.ravel())).detach().numpy().reshape(Xi.shape)
    Env = envelope(Xi, Ti)

    rel_l2 = float(np.linalg.norm(e) / np.linalg.norm(Uex))
    rel_residual_l2 = float(np.linalg.norm(R) / np.linalg.norm(Uex[1:-1, 1:]))

    # localization diagnostics
    peak_idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
    peak_x, peak_t = Xi[peak_idx], Ti[peak_idx]
    peak_env = Env[peak_idx]
    # correlation between |R| and 1/(envelope+eps) across the grid, and between |R| and time
    from scipy.stats import spearmanr
    rho_env, _ = spearmanr(np.abs(R).ravel(), Env.ravel())
    rho_t, _ = spearmanr(np.abs(R).ravel(), Ti.ravel())

    ex_t002 = exact_coeffs(0.02)
    final_modes = hist[-1]["modes_t002"]
    mode1_err = abs(final_modes[0] - ex_t002[0]) / ex_t002[0]
    mode5_err = abs(final_modes[4] - ex_t002[4]) / ex_t002[4]

    result = dict(seed=seed, rel_l2=rel_l2, rel_residual_l2=rel_residual_l2,
                  adam_s=adam_s, lbfgs_s=lbfgs_s, n_params=n_params(model),
                  peak_residual_abs=float(np.abs(R).max()), peak_x=float(peak_x), peak_t=float(peak_t),
                  peak_envelope=float(peak_env),
                  spearman_abs_residual_vs_envelope=float(rho_env), spearman_abs_residual_vs_time=float(rho_t),
                  mode1_rel_err_t002=float(mode1_err), mode5_rel_err_t002=float(mode5_err),
                  final_grad_norm_by_time_bin=hist[-1]["grad_norm_by_time_bin"])
    return result, hist


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    for seed in seeds:
        r, h = run_hard_diag(seed)
        json.dump(r, open(f"{OUT}/ch08-hardconstraint-diag_seed{seed}.json", "w"), indent=1)
        json.dump(h, open(f"{OUT}/ch08-hardconstraint-diag_seed{seed}_history.json", "w"))
        print("seed", seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                              if k != "final_grad_norm_by_time_bin"}, flush=True)
        print("  grad_norm_by_time_bin (5 bins, t=0..T):", [round(x, 6) if x == x else None for x in r["final_grad_norm_by_time_bin"]])
