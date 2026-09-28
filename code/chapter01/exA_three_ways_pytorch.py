"""Running Example A three ways (PyTorch implementation): supervised regression and a
physics-informed network with the same specification as exA_three_ways.py (JAX).

Book: Chapter 1, section 1.8 (the PyTorch rows of Tables 1.3 and 1.4, section 1.8.5); the
physics-informed runs are also the forward result of Chapter 6, section 6.7. Other experiments
import MLP, residual and train from this module (Chapters 6, 7 and 14).

Specification: network [2, 32, 32, 32, 1], tanh, LeCun-normal initialization (torch.Generator
seeded with the model seed), 2241 parameters, float64, one thread; n_r = 2000, n_b = 100 per
boundary, n_0 = 100 points drawn with torch.Generator(seed + 1000) (physics-informed) or a random
subset of the FTCS data drawn with torch.Generator(seed + 2000) (regression); unit loss weights;
Adam (5000 iterations, learning rate 1e-3), then torch.optim.LBFGS (at most 10000 iterations,
history 50, strong-Wolfe line search).
Note: the output key lbfgs_iters counts loss evaluations (closure calls), which is at least the
number of L-BFGS iterations.

Usage:
    python code/chapter01/exA_three_ways_pytorch.py              # both methods, seeds 0-4
    python code/chapter01/exA_three_ways_pytorch.py pinn 0       # physics-informed network, seed 0

Outputs (results/chapter01/): ch01-e03_pinn_seed{s}_pytorch.json, ch01-e02_regression_seed{s}_pytorch.json,
environment_pytorch.json (rewritten on every call).
Runtime (recorded environment, one thread): about 210 s per physics-informed run and 55 s per
regression run. Stochastic but seeded; repeatable on a fixed platform and software stack.
"""
import json, time, sys, os, platform
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import exact, ftcs, NU, T, MODES  # noqa: E402

OUT = str(REPO_ROOT / "results" / "chapter01")
os.makedirs(OUT, exist_ok=True)
LAYERS = [2, 32, 32, 32, 1]
ADAM_ITERS = int(os.environ.get("ADAM_ITERS", 5000))
ADAM_LR = 1e-3
LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", 10000))
N_R, N_B, N_0 = 2000, 100, 100
TRACK_EVERY = 100
EVAL_X, EVAL_T = np.linspace(0, 1, 201), np.linspace(0, T, 101)
EXTRAP_T = np.linspace(T, 1.5 * T, 51)[1:]
TRACK_T = (0.0, 0.02, 0.1)


class MLP(nn.Module):
    def __init__(self, layers, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.linears = nn.ModuleList()
        for din, dout in zip(layers[:-1], layers[1:]):
            lin = nn.Linear(din, dout)
            with torch.no_grad():
                # LeCun normal: N(0, 1/fan_in), matching the JAX init exactly
                lin.weight.copy_(torch.randn(dout, din, generator=g) * (1.0 / din) ** 0.5)
                lin.bias.zero_()
            self.linears.append(lin)

    def forward(self, x, t):
        h = torch.stack([x, t], dim=-1)
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)[..., 0]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def residual(model, x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t, u_x = torch.autograd.grad(u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - NU * u_xx


def mode_coeffs(model, t_val, K=8):
    xg = torch.linspace(0, 1, 401)
    with torch.no_grad():
        u = model(xg, torch.full_like(xg, t_val))
    ks = torch.arange(1, K + 1, dtype=torch.float64)
    S = torch.sin(np.pi * ks[:, None] * xg[None, :])
    w = torch.full((401,), 1 / 400, dtype=torch.float64)
    w[0] = 0.5 / 400
    w[-1] = 0.5 / 400
    return (2 * (S * (u * w)[None, :]).sum(1)).tolist()


def track(model):
    return {f"t{t}": mode_coeffs(model, t) for t in TRACK_T}


def evaluate(model, walls):
    X, Tg = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
    Uex = exact(X, Tg)
    xt = torch.tensor(X.ravel())
    tt = torch.tensor(Tg.ravel())
    with torch.no_grad():
        U = model(xt, tt).numpy().reshape(X.shape)
    # warm-up, then repeated timing, so that one-time setup costs are excluded from inference cost
    with torch.no_grad():
        for _ in range(3):
            model(xt, tt)
    t0 = time.perf_counter()
    reps = 20
    with torch.no_grad():
        for _ in range(reps):
            model(xt, tt)
    inf_t = (time.perf_counter() - t0) / reps
    e = U - Uex
    Xi, Ti = X[1:-1, 1:], Tg[1:-1, 1:]
    R = residual(model, torch.tensor(Xi.ravel()), torch.tensor(Ti.ravel())).detach().numpy()
    bc = np.abs(U[[0, -1], :]).max()
    ic = np.linalg.norm(U[:, 0] - Uex[:, 0]) / np.linalg.norm(Uex[:, 0])
    Xe, Te = np.meshgrid(EVAL_X, EXTRAP_T, indexing="ij")
    with torch.no_grad():
        Ue = model(torch.tensor(Xe.ravel()), torch.tensor(Te.ravel())).numpy().reshape(Xe.shape)
    ex = exact(Xe, Te)
    out = dict(rel_l2=float(np.linalg.norm(e) / np.linalg.norm(Uex)),
               rel_linf=float(np.abs(e).max() / np.abs(Uex).max()),
               rel_residual_l2=float(np.linalg.norm(R) / np.linalg.norm(Uex[1:-1, 1:])),
               residual_linf=float(np.abs(R).max()),
               bc_linf=float(bc), ic_rel_l2=float(ic),
               extrap_l2_rel_to_domain_norm=float(np.linalg.norm(Ue - ex) / np.linalg.norm(Uex) * np.sqrt(Uex.size / ex.size)),
               extrap_abs_linf=float(np.abs(Ue - ex).max()),
               inference_s_per_20301_points_warmed_up=inf_t,
               n_params=n_params(model))
    out.update(walls)
    return out


def train(loss_fn, model, seed):
    hist = []
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    t0 = time.perf_counter()
    for it in range(ADAM_ITERS):
        opt.zero_grad()
        l = loss_fn(model)
        l.backward()
        opt.step()
        if it % TRACK_EVERY == 0:
            hist.append(dict(iter=it, phase="adam", loss=float(l.detach()), modes=track(model)))
    adam_wall = time.perf_counter() - t0

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS, max_eval=LBFGS_ITERS * 2,
                               tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")
    it_counter = [ADAM_ITERS]

    def closure():
        lbfgs.zero_grad()
        l = loss_fn(model)
        l.backward()
        it_counter[0] += 1
        if it_counter[0] % TRACK_EVERY == 0:
            hist.append(dict(iter=it_counter[0], phase="lbfgs", loss=float(l.detach()), modes=track(model)))
        return l
    t0 = time.perf_counter()
    lbfgs.step(closure)  # NOTE: torch.optim.LBFGS.step returns the loss at the FIRST closure
    lbfgs_wall = time.perf_counter() - t0  # evaluation, not the final one; do not use that return value for logging.
    final_loss = float(loss_fn(model).detach())
    hist.append(dict(iter=it_counter[0], phase="final", loss=final_loss, modes=track(model)))
    return dict(adam_s=adam_wall, lbfgs_s=lbfgs_wall, lbfgs_iters=it_counter[0] - ADAM_ITERS, lbfgs_final_train_loss=final_loss)


def run_pinn(seed):
    g = torch.Generator().manual_seed(seed + 1000)
    xr = torch.rand(N_R, generator=g)
    tr = torch.rand(N_R, generator=g) * T
    tb = torch.rand(N_B, generator=g) * T
    x0 = torch.rand(N_0, generator=g)
    u0 = torch.tensor(exact(x0.numpy(), 0.0))
    model = MLP(LAYERS, seed)

    def loss(model):
        lr = (residual(model, xr, tr) ** 2).mean()
        lb = (model(torch.zeros(N_B), tb) ** 2).mean() + (model(torch.ones(N_B), tb) ** 2).mean()
        l0 = ((model(x0, torch.zeros(N_0)) - u0) ** 2).mean()
        return lr + lb + l0
    walls = train(loss, model, seed)
    m = evaluate(model, walls)
    m["seed"] = seed
    return m, None


def run_regression(seed, fd_data):
    x_d, t_d, u_d = fd_data
    g = torch.Generator().manual_seed(seed + 2000)
    idx = torch.randperm(x_d.shape[0], generator=g)[:N_R]
    xs, ts, us = torch.tensor(x_d[idx]), torch.tensor(t_d[idx]), torch.tensor(u_d[idx])
    model = MLP(LAYERS, seed)

    def loss(model):
        return ((model(xs, ts) - us) ** 2).mean()
    walls = train(loss, model, seed)
    m = evaluate(model, walls)
    m["seed"] = seed
    return m, None


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which not in ("both", "pinn", "regression"):
        raise SystemExit("usage: python code/chapter01/exA_three_ways_pytorch.py [both|pinn|regression] [SEED ...]")
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2, 3, 4]
    nx = 81
    dx = 1 / (nx - 1)
    nt = int(np.ceil(T / (0.4 * dx ** 2 / NU)))
    xg, tg, U = ftcs(nx, nt)
    Xg, Tg = np.meshgrid(xg, tg, indexing="ij")
    mask = Tg > 0
    fd_data = (Xg[mask], Tg[mask], U.T[mask])
    env = dict(platform=platform.platform(), torch=torch.__version__, python=platform.python_version(),
               float64=True, layers=LAYERS, adam=dict(iters=ADAM_ITERS, lr=ADAM_LR), lbfgs_iters=LBFGS_ITERS,
               n_r=N_R, n_b=N_B, n_0=N_0, threads=torch.get_num_threads())
    json.dump(env, open(f"{OUT}/environment_pytorch.json", "w"), indent=1)
    for seed in seeds:
        for tag, fn in (("ch01-e03_pinn", run_pinn), ("ch01-e02_regression", lambda s: run_regression(s, fd_data))):
            if which != "both" and which not in tag:
                continue
            m, hist = fn(seed)
            json.dump(m, open(f"{OUT}/{tag}_seed{seed}_pytorch.json", "w"), indent=1)
            print(tag, seed, "pytorch",
                  {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()
                   if k in ("rel_l2", "rel_linf", "rel_residual_l2", "extrap_abs_linf", "adam_s", "lbfgs_s", "lbfgs_iters")},
                  flush=True)
