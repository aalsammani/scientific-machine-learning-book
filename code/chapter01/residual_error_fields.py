"""Data for Figure 1.4 (Chapter 1, section 1.8.5): squared PDE residual and squared error of the
PyTorch physics-informed network (seed 0) on a 101 x 101 grid.

Retrains the seed-0 network of exA_three_ways_pytorch.py (about 3.5 minutes on one core) and stores
the fields; code/figures/make_fig1_4.py plots them without retraining.

Usage:
    python code/chapter01/residual_error_fields.py
Output: results/chapter01/fig15_fields.npz (arrays X, T, residual, error).
Seeded; the stored file was produced in the recorded environment and a rerun elsewhere agrees
only up to floating-point differences.
"""
import sys, time, json
import numpy as np
import torch
from pathlib import Path
from exA_three_ways_pytorch import MLP, LAYERS, N_R, N_B, N_0, T, ADAM_LR, train
from exA_heat import exact

from exA_three_ways_pytorch import residual

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

seed = 0
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

t0 = time.perf_counter()
walls = train(loss, model, seed)
print("trained in", time.perf_counter()-t0, "s", walls)

# evaluate on a fine grid for the residual and error fields
nx, nt = 101, 101
xg = np.linspace(0, 1, nx)
tg = np.linspace(0, T, nt)
Xg, Tg = np.meshgrid(xg, tg, indexing="ij")
Xf = torch.tensor(Xg.ravel(), requires_grad=True)
Tf = torch.tensor(Tg.ravel(), requires_grad=True)
res = residual(model, Xf, Tf).detach().numpy().reshape(nx, nt)
with torch.no_grad():
    u_pred = model(torch.tensor(Xg.ravel()), torch.tensor(Tg.ravel())).numpy().reshape(nx, nt)
u_true = exact(Xg, Tg)
err = u_pred - u_true

np.savez(Path(__file__).resolve().parents[2] / "results" / "chapter01" / "fig15_fields.npz", X=Xg, T=Tg, residual=res, error=err)
print("residual^2 range:", (res**2).min(), (res**2).max())
print("error^2 range:", (err**2).min(), (err**2).max())
