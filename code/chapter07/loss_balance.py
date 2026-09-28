"""Gradient-norm loss balancing on Running Example A (Chapter 7, section 7.5; Figure 7.1, right panel).

Both modes start from a deliberately underweighted boundary term, omega_b = 0.001:
    fixed     omega_b stays at 0.001;
    gradnorm  every 200 Adam iterations, omega_b is reset to ||grad L_r|| / ||grad L_b|| (norms of
              the gradients of the residual and boundary losses with respect to the network
              parameters); the weight is then frozen during L-BFGS.
Network and initialization of code/chapter01/exA_three_ways_pytorch.py (3 x 32 tanh, float64, one
thread); collocation n_r = 2000, n_b = 100, n_0 = 100 drawn with torch.Generator(seed + 7000);
Adam (5000 iterations, learning rate 1e-3), then L-BFGS (at most 8000 iterations, strong-Wolfe line
search). Seeds 0-2 in the book.

Usage:
    python code/chapter07/loss_balance.py gradnorm 0
    python code/chapter07/loss_balance.py fixed 0

Output: results/chapter07/lossbalance_{mode}_seed{s}.json
Runtime: about 2.5 minutes per run on one core. Stochastic but seeded.
"""
import torch, time, json, sys, os
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems")); sys.path.insert(0, str(REPO_ROOT / "code" / "chapter01"))
from exA_heat import exact, NU, T
from exA_three_ways_pytorch import MLP
import numpy as np

N_R, N_B, N_0 = 2000, 100, 100
EVAL_X, EVAL_T = np.linspace(0,1,201), np.linspace(0,T,101)
ADAM_ITERS = int(os.environ.get("ADAM_ITERS", 5000))
LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", 8000))

def residual(model, x, t):
    x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t, u_x = torch.autograd.grad(u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - NU * u_xx

def evaluate(model):
    X, Tg = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
    Uex = exact(X, Tg)
    with torch.no_grad():
        U = model(torch.tensor(X.ravel()), torch.tensor(Tg.ravel())).numpy().reshape(X.shape)
    e = U - Uex
    bc = np.abs(U[[0,-1],:]).max()
    return dict(rel_l2=float(np.linalg.norm(e)/np.linalg.norm(Uex)), bc_linf=float(bc))

def run(seed, mode, omega_b0=0.001):
    g = torch.Generator().manual_seed(seed+7000)
    xr = torch.rand(N_R, generator=g); tr = torch.rand(N_R, generator=g)*T
    tb = torch.rand(N_B, generator=g)*T
    x0 = torch.rand(N_0, generator=g); u0 = torch.tensor(exact(x0.numpy(), 0.0))
    model = MLP([2,32,32,32,1], seed)
    omega_b = torch.tensor(omega_b0)

    def parts():
        r = residual(model, xr, tr); lr = (r**2).mean()
        lb = (model(torch.zeros(N_B), tb)**2).mean() + (model(torch.ones(N_B), tb)**2).mean()
        l0 = ((model(x0, torch.zeros(N_0)) - u0)**2).mean()
        return lr, lb, l0

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0=time.perf_counter()
    for it in range(ADAM_ITERS):
        if mode == "gradnorm" and it % 200 == 0:
            lr, lb, l0 = parts()
            g_r = torch.autograd.grad(lr, model.parameters(), retain_graph=True, allow_unused=True)
            g_b = torch.autograd.grad(lb, model.parameters(), retain_graph=True, allow_unused=True)
            norm_r = sum((x**2).sum() for x in g_r if x is not None).sqrt()
            norm_b = sum((x**2).sum() for x in g_b if x is not None).sqrt()
            omega_b = (norm_r / (norm_b + 1e-12)).detach()
        opt.zero_grad()
        lr, lb, l0 = parts()
        loss = lr + omega_b*lb + l0
        loss.backward()
        opt.step()
    adam_s = time.perf_counter()-t0
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS, max_eval=LBFGS_ITERS*2,
                               tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(); lr,lb,l0 = parts(); loss = lr+omega_b*lb+l0; loss.backward(); return loss
    t0=time.perf_counter()
    lbfgs.step(closure)
    lbfgs_s = time.perf_counter()-t0
    m = evaluate(model); m.update(seed=seed, mode=mode, final_omega_b=float(omega_b), adam_s=adam_s, lbfgs_s=lbfgs_s)
    return m

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python code/chapter07/loss_balance.py {fixed|gradnorm} SEED")
    mode = sys.argv[1]; seed = int(sys.argv[2])
    if mode not in ("fixed", "gradnorm"):
        raise SystemExit("first argument must be fixed or gradnorm")
    r = run(seed, mode)
    json.dump(r, open(REPO_ROOT / "results" / "chapter07" / f"lossbalance_{mode}_seed{seed}.json", "w"), indent=1)
    print(mode, seed, r)
