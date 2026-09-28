"""Universal differential equation for Lotka-Volterra (Running Example B; Chapter 10, section 10.2).

True system: x' = alpha x - beta x y, y' = delta x y - gamma y, (alpha, beta, delta, gamma) =
(1.5, 1, 1, 3), x(0) = (1, 1). The known linear terms (alpha x, -gamma y) are kept and the
interaction terms are replaced by a network N: R^2 -> R^2 (MLP [2, 16, 2], tanh, weights
N(0, 0.3^2) from a seeded torch.Generator, zero biases, 82 parameters, float64, one thread).
Training data: the true trajectory at 50 equally spaced times on [0, 10], computed with the same
fixed-step RK4 integrator (two substeps per interval, h = 10/98) that is used inside the training
loop, so the data contain no integration error relative to the model's own discretization (the
integrator's error with respect to the exact flow is not measured here). Loss: mean squared
trajectory error; Adam (600 iterations, learning rate 1e-2), then L-BFGS (at most 400 iterations,
learning rate 0.5, history 30, strong-Wolfe line search). The learned interaction term is evaluated
on the states visited by the trajectory and on a 20 x 20 grid over [0.2, 2]^2. Seeds 0 and 1 in the
book.

Usage:
    python code/chapter10/lv_ude.py 0
    python code/chapter10/lv_ude.py 1

Outputs: results/chapter10/lv_ude_seed{s}.json (and lv_ude_traj_seed0.npz for seed 0).
lv_ude_summary.json lists the two seeds' results together; it was assembled by hand from the
per-seed files. Runtime: about 70 seconds per seed on one core. Stochastic but seeded.
"""
import torch, numpy as np, time, json
from pathlib import Path
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

a_, b_, dd_, g_ = 1.5, 1.0, 1.0, 3.0

def lv_true(t, xy):
    x, y = xy[...,0], xy[...,1]
    return torch.stack([a_*x - b_*x*y, dd_*x*y - g_*y], -1)

def rk4_integrate(f, x0, t_eval):
    # fixed fine sub-stepping RK4 between the coarse evaluation points
    substeps = 2
    out = [x0]
    x = x0
    for i in range(len(t_eval)-1):
        t0, t1 = t_eval[i], t_eval[i+1]
        h = (t1-t0)/substeps
        for _ in range(substeps):
            k1 = f(t0, x)
            k2 = f(t0+h/2, x+h/2*k1)
            k3 = f(t0+h/2, x+h/2*k2)
            k4 = f(t0+h, x+h*k3)
            x = x + h/6*(k1+2*k2+2*k3+k4)
            t0 = t0+h
        out.append(x)
    return torch.stack(out, 0)

t_eval = torch.linspace(0, 10, 50)
x0 = torch.tensor([1.0, 1.0])
with torch.no_grad():
    true_traj = rk4_integrate(lv_true, x0, t_eval)

class Interaction(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(2,16), torch.nn.Tanh(), torch.nn.Linear(16,2))
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(torch.randn(*m.weight.shape, generator=g)*0.3)
                    m.bias.zero_()
    def forward(self, t, xy):
        x, y = xy[...,0], xy[...,1]
        known = torch.stack([a_*x, -g_*y], -1)
        return known + self.net(xy)

def train(seed, adam_iters=600, lbfgs_iters=400):
    model = Interaction(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    t0=time.perf_counter()
    for it in range(adam_iters):
        opt.zero_grad()
        pred = rk4_integrate(model, x0, t_eval)
        loss = ((pred-true_traj)**2).mean()
        loss.backward(); opt.step()
    adam_s=time.perf_counter()-t0
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=lbfgs_iters, max_eval=lbfgs_iters*2,
                               tolerance_grad=1e-13, tolerance_change=1e-15, history_size=30, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(); pred = rk4_integrate(model, x0, t_eval)
        loss=((pred-true_traj)**2).mean(); loss.backward(); return loss
    t0=time.perf_counter()
    lbfgs.step(closure)
    lbfgs_s=time.perf_counter()-t0
    with torch.no_grad():
        pred = rk4_integrate(model, x0, t_eval)
    rel_l2 = float(torch.norm(pred-true_traj)/torch.norm(true_traj))
    # evaluate the interaction term BOTH off a generic grid and directly on the trajectory's own
    # visited (x,y) values, since these can differ substantially (the trajectory need not visit
    # a naive default grid at all)
    xg, yg = torch.meshgrid(torch.linspace(0.2,2,20), torch.linspace(0.2,2,20), indexing="ij")
    grid = torch.stack([xg.ravel(), yg.ravel()], -1)
    with torch.no_grad():
        learned = model.net(grid).numpy()
    true_interaction = np.stack([(-b_*xg*yg).numpy().ravel(), (dd_*xg*yg).numpy().ravel()], -1)
    interaction_rel_l2_grid = float(np.linalg.norm(learned-true_interaction)/np.linalg.norm(true_interaction))

    with torch.no_grad():
        learned_on_traj = model.net(true_traj).numpy()
    xt, yt = true_traj[:,0].numpy(), true_traj[:,1].numpy()
    true_interaction_on_traj = np.stack([-b_*xt*yt, dd_*xt*yt], -1)
    interaction_rel_l2_traj = float(np.linalg.norm(learned_on_traj-true_interaction_on_traj)/np.linalg.norm(true_interaction_on_traj))

    return dict(seed=seed, rel_l2_trajectory=rel_l2,
                rel_l2_interaction_term_default_grid=interaction_rel_l2_grid,
                rel_l2_interaction_term_on_trajectory=interaction_rel_l2_traj,
                trajectory_x_range=[float(xt.min()),float(xt.max())], trajectory_y_range=[float(yt.min()),float(yt.max())],
                adam_s=adam_s, lbfgs_s=lbfgs_s, n_params=sum(p.numel() for p in model.parameters())), pred.detach().numpy(), true_traj.numpy()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python code/chapter10/lv_ude.py SEED")
    seed = int(sys.argv[1])
    r, pred, true = train(seed)
    results = Path(__file__).resolve().parents[2] / "results" / "chapter10"
    results.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(results / f"lv_ude_seed{seed}.json", "w"), indent=1)
    if seed == 0:
        np.savez(results / "lv_ude_traj_seed0.npz", t=t_eval.numpy(), pred=pred, true=true)
    print(r)
