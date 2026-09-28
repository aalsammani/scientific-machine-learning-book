"""Deep Ritz for the Poisson problem (Chapter 8, sections 8.3 and 8.9-8.10; Figure 8.1).

    -Laplace(u) = f on (0,1)^2, u = 0 on the boundary, f = 2 pi^2 sin(pi x) sin(pi y),
    exact solution u = sin(pi x) sin(pi y); minimum energy -pi^2/4.

Trial function u = x(1-x) y(1-y) N(x, y) (boundary condition exact), N an MLP [2, 32, 32, 32, 1]
with tanh and LeCun-normal initialization (seeded torch.Generator), float64, one thread. The Ritz
energy 0.5 |grad u|^2 - f u is estimated by the mean over n_r = 2000 uniform points drawn with
torch.Generator(seed + 100):
    fixed     one fixed point set for the whole run;
    resample  a fresh point set at every Adam step and every L-BFGS function evaluation.
Adam (3000 iterations, learning rate 1e-3), then L-BFGS (at most 3000 iterations). Relative L2
error on a 101 x 101 grid. Seeds 0-2 in the book.

Usage:
    python code/chapter08/deepritz.py fixed 0
    python code/chapter08/deepritz.py resample 0

Output: results/chapter08/deepritz_{mode}_seed{s}.json
Runtime: about 15-35 seconds per run on one core. Stochastic but seeded.
"""
import torch, time, json, sys, os
from pathlib import Path
import numpy as np
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

class MLP(torch.nn.Module):
    def __init__(self, layers, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.linears = torch.nn.ModuleList()
        for din,dout in zip(layers[:-1],layers[1:]):
            lin = torch.nn.Linear(din,dout)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(dout,din,generator=g)*(1.0/din)**0.5)
                lin.bias.zero_()
            self.linears.append(lin)
    def forward(self, xy):
        h = xy
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)[...,0]

def hard_ansatz(model, xy):
    x, y = xy[...,0], xy[...,1]
    return x*(1-x)*y*(1-y) * model(xy)

f = lambda x,y: 2*np.pi**2*torch.sin(np.pi*x)*torch.sin(np.pi*y)
u_exact = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
ADAM_ITERS = int(os.environ.get("ADAM_ITERS", 3000))
LBFGS_ITERS = int(os.environ.get("LBFGS_ITERS", 3000))

def run(seed, resample, n_r=2000):
    model = MLP([2,32,32,32,1], seed)
    gen = torch.Generator().manual_seed(seed+100)
    xy_fixed = torch.rand(n_r, 2, generator=gen)

    def energy(xy):
        xy_ = xy.clone().requires_grad_(True)
        u = hard_ansatz(model, xy_)
        grad_u = torch.autograd.grad(u.sum(), xy_, create_graph=True)[0]
        gradsq = (grad_u**2).sum(-1)
        fx = f(xy_[:,0], xy_[:,1])
        return 0.5*gradsq.mean() - (fx*u).mean()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0=time.perf_counter()
    for it in range(ADAM_ITERS):
        xy = torch.rand(n_r, 2, generator=gen) if resample else xy_fixed
        opt.zero_grad(); E = energy(xy); E.backward(); opt.step()
    adam_s=time.perf_counter()-t0

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS, max_eval=LBFGS_ITERS*2,
                               tolerance_grad=1e-14, tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad()
        xy = torch.rand(n_r, 2, generator=gen) if resample else xy_fixed
        E = energy(xy); E.backward(); return E
    t0=time.perf_counter()
    lbfgs.step(closure)
    lbfgs_s=time.perf_counter()-t0

    ex = np.linspace(0,1,101)
    X,Y = np.meshgrid(ex,ex,indexing="ij")
    with torch.no_grad():
        xyt = torch.tensor(np.stack([X.ravel(),Y.ravel()],axis=-1))
        U = hard_ansatz(model, xyt).numpy().reshape(X.shape)
    Uex = u_exact(X,Y)
    rel_l2 = float(np.linalg.norm(U-Uex)/np.linalg.norm(Uex))
    final_energy = float(energy(xy_fixed if not resample else torch.rand(n_r,2,generator=gen)))
    return dict(seed=seed, resample=resample, rel_l2=rel_l2, adam_s=adam_s, lbfgs_s=lbfgs_s,
                n_params=sum(p.numel() for p in model.parameters()), final_energy=final_energy,
                true_min_energy=-np.pi**2/4)

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("fixed", "resample"):
        raise SystemExit("usage: python code/chapter08/deepritz.py {fixed|resample} SEED")
    resample = sys.argv[1] == "resample"
    seed = int(sys.argv[2])
    r = run(seed, resample)
    tag = "resample" if resample else "fixed"
    json.dump(r, open(Path(__file__).resolve().parents[2] / "results" / "chapter08" / f"deepritz_{tag}_seed{seed}.json", "w"), indent=1)
    print(tag, seed, r)
