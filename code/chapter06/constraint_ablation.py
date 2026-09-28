"""Constraint and activation ablations on Running Example A (Chapter 6, section 6.9; the hard-
constraint runs are revisited in Chapter 7, section 7.7).

Variants (first argument), all with the network of code/chapter01/exA_three_ways_pytorch.py unless
stated, collocation drawn with torch.Generator(seed + 6000), Adam (5000 iterations, learning rate
1e-3), then L-BFGS (at most 10000 iterations):
    soft           residual + boundary + initial losses, unit weights (result prefix ch07-e04_soft);
    underweighted  as soft with boundary weight 0.001 (ch07-f01_underweighted);
    hard           u = u0(x) + x(1-x) t N(x, t), which satisfies the initial and boundary conditions
                   exactly; residual loss only (ch07-e04_hard; seeds 0-3 in the book);
    relu           ReLU activation with He initialization instead of tanh/LeCun (ch07-f04_relu;
                   seeds 0-1). The second derivative of a ReLU network vanishes almost everywhere.
The reported rel_residual_l2 is the residual on the held-out evaluation grid divided by the norm of
the exact solution there, not the training loss.

Usage:
    python code/chapter06/constraint_ablation.py soft 0
    python code/chapter06/constraint_ablation.py hard 1

Output: results/chapter06/{prefix}_seed{s}.json
Runtime: about 2.5-3.5 minutes per run on one core (relu about 0.5 minutes). Stochastic but seeded.
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
N_R, N_B, N_0 = 2000, 100, 100
ADAM_ITERS = int(os.environ.get("ADAM_ITERS", 5000))
LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", 10000))
EVAL_X, EVAL_T = np.linspace(0, 1, 201), np.linspace(0, T, 101)


class ReLUMLP(nn.Module):
    """Same layer sizes as MLP but with ReLU activation (ch07-f04)."""
    def __init__(self, layers, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.linears = nn.ModuleList()
        for din, dout in zip(layers[:-1], layers[1:]):
            lin = nn.Linear(din, dout)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(dout, din, generator=g) * (2.0 / din) ** 0.5)  # He init
                lin.bias.zero_()
            self.linears.append(lin)

    def forward(self, x, t):
        h = torch.stack([x, t], dim=-1)
        for lin in self.linears[:-1]:
            h = torch.relu(lin(h))
        return self.linears[-1](h)[..., 0]


def hard_ansatz(net_out, x, t, u0_vals):
    """u_theta(x,t) = u0(x) + x(1-x) t * N(x,t): exactly satisfies u(x,0)=u0(x) and u(0,t)=u(1,t)=0.
    u0_vals must be a differentiable function of x (native torch ops), not a detached constant,
    or automatic differentiation through the full ansatz silently drops the u0'(x), u0''(x)
    contributions to the residual."""
    return u0_vals + x * (1 - x) * t * net_out


def u0_torch(x):
    """Native-torch, differentiable initial condition, matching exA_heat.exact(x, 0.0)."""
    return torch.sin(np.pi * x) + 0.5 * torch.sin(5 * np.pi * x)


def residual_general(u_fn, x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_fn(x, t)
    u_t, u_x = torch.autograd.grad(u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - NU * u_xx


def evaluate(u_fn):
    X, Tg = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
    Uex = exact(X, Tg)
    with torch.no_grad():
        U = u_fn(torch.tensor(X.ravel()), torch.tensor(Tg.ravel())).numpy().reshape(X.shape)
    e = U - Uex
    bc = np.abs(U[[0, -1], :]).max()
    ic = np.linalg.norm(U[:, 0] - Uex[:, 0]) / np.linalg.norm(Uex[:, 0])
    Xi, Ti = X[1:-1, 1:], Tg[1:-1, 1:]
    R = residual_general(u_fn, torch.tensor(Xi.ravel()), torch.tensor(Ti.ravel())).detach().numpy()
    return dict(rel_l2=float(np.linalg.norm(e) / np.linalg.norm(Uex)),
                rel_linf=float(np.abs(e).max() / np.abs(Uex).max()),
                rel_residual_l2=float(np.linalg.norm(R) / np.linalg.norm(Uex[1:-1, 1:])),
                bc_linf=float(bc), ic_rel_l2=float(ic))


def train(loss_fn, params, adam_iters=ADAM_ITERS, lbfgs_iters=LBFGS_ITERS):
    opt = torch.optim.Adam(params, lr=1e-3)
    t0 = time.perf_counter()
    for it in range(adam_iters):
        opt.zero_grad()
        l = loss_fn()
        l.backward()
        opt.step()
    adam_s = time.perf_counter() - t0
    lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, max_eval=lbfgs_iters * 2,
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
    return adam_s, lbfgs_s


def run_hard(seed):
    g = torch.Generator().manual_seed(seed + 6000)
    xr = torch.rand(N_R, generator=g)
    tr = torch.rand(N_R, generator=g) * T
    model = MLP(LAYERS, seed)

    def u_fn(x, t):
        return hard_ansatz(model(x, t), x, t, u0_torch(x))

    def loss_fn():
        r = residual_general(u_fn, xr, tr)
        return (r ** 2).mean()

    adam_s, lbfgs_s = train(loss_fn, list(model.parameters()))
    m = evaluate(u_fn)
    m.update(seed=seed, tag="hard", adam_s=adam_s, lbfgs_s=lbfgs_s, n_params=n_params(model))
    return m


def run_soft(seed, boundary_weight=1.0):
    g = torch.Generator().manual_seed(seed + 6000)
    xr = torch.rand(N_R, generator=g)
    tr = torch.rand(N_R, generator=g) * T
    tb = torch.rand(N_B, generator=g) * T
    x0 = torch.rand(N_0, generator=g)
    u0 = torch.tensor(exact(x0.numpy(), 0.0))
    model = MLP(LAYERS, seed)

    def loss_fn():
        r = residual_general(model, xr, tr)
        lr = (r ** 2).mean()
        lb = (model(torch.zeros(N_B), tb) ** 2).mean() + (model(torch.ones(N_B), tb) ** 2).mean()
        l0 = ((model(x0, torch.zeros(N_0)) - u0) ** 2).mean()
        return lr + boundary_weight * lb + l0

    adam_s, lbfgs_s = train(loss_fn, list(model.parameters()))
    m = evaluate(model)
    m.update(seed=seed, tag=f"soft_bw{boundary_weight}", adam_s=adam_s, lbfgs_s=lbfgs_s, n_params=n_params(model))
    return m


def run_relu(seed):
    g = torch.Generator().manual_seed(seed + 6000)
    xr = torch.rand(N_R, generator=g)
    tr = torch.rand(N_R, generator=g) * T
    tb = torch.rand(N_B, generator=g) * T
    x0 = torch.rand(N_0, generator=g)
    u0 = torch.tensor(exact(x0.numpy(), 0.0))
    model = ReLUMLP(LAYERS, seed)

    def loss_fn():
        r = residual_general(model, xr, tr)
        lr = (r ** 2).mean()
        lb = (model(torch.zeros(N_B), tb) ** 2).mean() + (model(torch.ones(N_B), tb) ** 2).mean()
        l0 = ((model(x0, torch.zeros(N_0)) - u0) ** 2).mean()
        return lr + lb + l0

    adam_s, lbfgs_s = train(loss_fn, list(model.parameters()))
    m = evaluate(model)
    m.update(seed=seed, tag="relu", adam_s=adam_s, lbfgs_s=lbfgs_s, n_params=n_params(model))
    return m


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python code/chapter06/constraint_ablation.py {hard|soft|underweighted|relu} SEED")
    which = sys.argv[1]
    seed = int(sys.argv[2])
    if which == "hard":
        r = run_hard(seed)
        tag = "ch07-e04_hard"
    elif which == "soft":
        r = run_soft(seed, boundary_weight=1.0)
        tag = "ch07-e04_soft"
    elif which == "underweighted":
        r = run_soft(seed, boundary_weight=0.001)
        tag = "ch07-f01_underweighted"
    elif which == "relu":
        r = run_relu(seed)
        tag = "ch07-f04_relu"
    else:
        raise SystemExit("first argument must be hard, soft, underweighted or relu")
    json.dump(r, open(f"{OUT}/{tag}_seed{seed}.json", "w"), indent=1)
    print(tag, "seed", seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
