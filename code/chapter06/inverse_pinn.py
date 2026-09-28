"""Inverse physics-informed network for Running Example A (Chapter 6, section 6.7): recover the
diffusivity nu (true value 0.5) from 20 noisy interior observations by adding a scalar nu_hat
(softplus-parameterized, initialized at 0.1) to the trainable parameters.

Specification: network and initialization of code/chapter01/exA_three_ways_pytorch.py (3 x 32 tanh,
float64, one thread); collocation n_r = 2000, n_b = 100, n_0 = 100 and 20 observation points drawn
with torch.Generator(seed + 3000); observation noise N(0, (SIGMA_REL max|u|)^2) from
numpy.random.default_rng(seed + 4000); unit weights on the residual, boundary, initial and data
losses; Adam (5000 iterations, learning rate 1e-3), then L-BFGS (default at most 5000 iterations,
strong-Wolfe line search). The stored result files do not record the L-BFGS limit that was used.

Usage (the noise level is set by an environment variable; seeds default to 0-4):
    SIGMA_REL=0.01 python code/chapter06/inverse_pinn.py
    SIGMA_REL=0.05 python code/chapter06/inverse_pinn.py 0 1 2 3 4
(On Windows PowerShell: $env:SIGMA_REL="0.05"; python code/chapter06/inverse_pinn.py)

Outputs: results/chapter06/ch07-e02_inverse_seed{s}_sigma{level}.json and *_history.json.
Runtime: about 2 minutes per run on one core. Stochastic but seeded.
"""
import json, time, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import exact, NU, T  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter01"))
from exA_three_ways_pytorch import MLP, LAYERS, n_params  # noqa: E402

OUT = str(REPO_ROOT / "results" / "chapter06")
os.makedirs(OUT, exist_ok=True)
N_R, N_B, N_0, N_OBS = 2000, 100, 100, 20
ADAM_ITERS = int(os.environ.get("ADAM_ITERS", 5000))
LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", 5000))
SIGMA_REL = float(os.environ.get("SIGMA_REL", 0.01))  # relative noise level
EVAL_X, EVAL_T = np.linspace(0, 1, 201), np.linspace(0, T, 101)


def residual(model, nu_raw, x, t):
    nu = torch.nn.functional.softplus(nu_raw)
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t, u_x = torch.autograd.grad(u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - nu * u_xx, nu


def run(seed):
    g = torch.Generator().manual_seed(seed + 3000)
    xr = torch.rand(N_R, generator=g)
    tr = torch.rand(N_R, generator=g) * T
    tb = torch.rand(N_B, generator=g) * T
    x0 = torch.rand(N_0, generator=g)
    u0 = torch.tensor(exact(x0.numpy(), 0.0))

    xo = torch.rand(N_OBS, generator=g)
    to_ = torch.rand(N_OBS, generator=g) * T
    u_clean = exact(xo.numpy(), to_.numpy())
    rng = np.random.default_rng(seed + 4000)
    noise = rng.normal(0, SIGMA_REL * np.abs(u_clean).max(), size=u_clean.shape)
    u_obs = torch.tensor(u_clean + noise)

    model = MLP(LAYERS, seed)
    nu_raw = torch.zeros(1, requires_grad=True)  # softplus(0) = log(2) =/= NU; see init note below
    # initialize so that softplus(nu_raw0) = 0.1 (a deliberately wrong starting guess, not NU=0.5)
    with torch.no_grad():
        target0 = 0.1
        nu_raw.copy_(torch.log(torch.expm1(torch.tensor(target0))))

    def loss_fn():
        r, nu = residual(model, nu_raw, xr, tr)
        lr = (r ** 2).mean()
        lb = (model(torch.zeros(N_B), tb) ** 2).mean() + (model(torch.ones(N_B), tb) ** 2).mean()
        l0 = ((model(x0, torch.zeros(N_0)) - u0) ** 2).mean()
        ld = ((model(xo, to_) - u_obs) ** 2).mean()
        return lr + lb + l0 + ld

    params = list(model.parameters()) + [nu_raw]
    opt = torch.optim.Adam(params, lr=1e-3)
    hist = []
    t0 = time.perf_counter()
    for it in range(ADAM_ITERS):
        opt.zero_grad()
        l = loss_fn()
        l.backward()
        opt.step()
        if it % 200 == 0:
            hist.append(dict(iter=it, loss=float(l.detach()),
                              nu_hat=float(torch.nn.functional.softplus(nu_raw).detach())))
    adam_s = time.perf_counter() - t0

    lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_ITERS, max_eval=LBFGS_ITERS * 2,
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

    nu_hat = float(torch.nn.functional.softplus(nu_raw).detach())
    hist.append(dict(iter=ADAM_ITERS + LBFGS_ITERS, loss=float(loss_fn().detach()), nu_hat=nu_hat))

    X, Tg = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
    Uex = exact(X, Tg)
    with torch.no_grad():
        U = model(torch.tensor(X.ravel()), torch.tensor(Tg.ravel())).numpy().reshape(X.shape)
    rel_l2 = float(np.linalg.norm(U - Uex) / np.linalg.norm(Uex))
    result = dict(seed=seed, nu_true=NU, nu_hat=nu_hat, nu_rel_error=abs(nu_hat - NU) / NU,
                  sigma_rel=SIGMA_REL, n_obs=N_OBS, rel_l2_field=rel_l2,
                  adam_s=adam_s, lbfgs_s=lbfgs_s, n_params=n_params(model) + 1)
    return result, hist


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    for seed in seeds:
        r, h = run(seed)
        tag = f"sigma{SIGMA_REL}"
        json.dump(r, open(f"{OUT}/ch07-e02_inverse_seed{seed}_{tag}.json", "w"), indent=1)
        json.dump(h, open(f"{OUT}/ch07-e02_inverse_seed{seed}_{tag}_history.json", "w"))
        print("seed", seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
