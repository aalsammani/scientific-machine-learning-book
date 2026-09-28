"""DeepONet for the Darcy family of Running Example D (Chapter 9, section 9.6; reused in Chapters 12,
14 and 15). Branch input: the five coefficients c; trunk input: the node coordinates (x, y).

Architecture: branch and trunk MLPs [., 64, 64, 32] with tanh, p = 32, a learned scalar bias;
weights N(0, 1/fan_in) from a torch.Generator seeded with the model seed, zero biases; 13057
parameters; float64; one thread.
Training: full-batch mean-squared error over all 441 nodes of the first n_train training pairs;
Adam (3000 iterations, learning rate 1e-3), then L-BFGS (at most 2000 iterations, history 50,
strong-Wolfe line search).
Two output protocols:
  * raw outputs: the network predicts u directly (the original protocol; stored under the directory
    name sweep_firstedition_protocol);
  * standardized outputs (--normalize): the network predicts (u - training mean field) / s, with s
    the standard deviation of u - mean over the training set, and the loss is computed in those
    units.
Metric: median over the 100 test pairs of the relative L2 error.

Usage (a relative --data path is taken relative to the repository root; --out and --state are
ordinary file paths, relative to the current directory):
    python code/chapter09/darcy/deeponet.py N_TRAIN SEED [--normalize] [--data PATH] [--out JSON] [--state PT]
    python code/chapter09/darcy/deeponet.py 200 0 --normalize

Runtime: about 1 minute per training on one core (recorded environment: 63 s). Deterministic for a
fixed seed on a fixed platform and software stack.
"""
import torch, numpy as np, time, json, sys
from pathlib import Path
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = "results/chapter09/darcy_dataset.npz"   # relative to REPO_ROOT


class DeepONet(torch.nn.Module):
    def __init__(self, k_dim, trunk_dim, p, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        def mlp(din, layers):
            mods = []; d0 = din
            for h in layers[:-1]:
                lin = torch.nn.Linear(d0, h)
                with torch.no_grad():
                    lin.weight.copy_(torch.randn(h, d0, generator=g) * (1.0 / d0) ** 0.5); lin.bias.zero_()
                mods += [lin, torch.nn.Tanh()]; d0 = h
            lin = torch.nn.Linear(d0, layers[-1])
            with torch.no_grad():
                lin.weight.copy_(torch.randn(layers[-1], d0, generator=g) * (1.0 / d0) ** 0.5); lin.bias.zero_()
            mods.append(lin)
            return torch.nn.Sequential(*mods)
        self.branch = mlp(k_dim, [64, 64, p])
        self.trunk = mlp(trunk_dim, [64, 64, p])
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, coeffs_batch, coords_batch):
        return self.branch(coeffs_batch) @ self.trunk(coords_batch).T + self.bias


def load(path=DEFAULT_DATA):
    d = np.load(REPO_ROOT / path)
    coeffs, U, X, Y = d["coeffs"], d["U"], d["X"], d["Y"]
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
    return coeffs, U, coords, int(d["n_train"]), int(d["n_test"])


class Normalized(torch.nn.Module):
    """Wraps a DeepONet so that it predicts u = u_mean + s * net(c, x), with the training-set mean
    field u_mean and scalar s = std of (u - u_mean) over the training set (standard output scaling)."""
    def __init__(self, net, u_mean, s):
        super().__init__()
        self.net = net
        self.register_buffer("u_mean", torch.as_tensor(u_mean)); self.register_buffer("s", torch.as_tensor(s))
    def forward(self, c, x):
        return self.u_mean + self.s * self.net(c, x)


def train(n_train_used, seed, data=DEFAULT_DATA, adam_iters=3000, lbfgs_iters=2000, normalize=False):
    coeffs, U, coords, n_train, n_test = load(data)
    assert n_train_used <= n_train
    ctrain = torch.tensor(coeffs[:n_train_used])
    utrain = torch.tensor(U[:n_train_used].reshape(n_train_used, -1))
    if normalize:
        u_mean = utrain.mean(0); s = (utrain - u_mean).std()
        model = Normalized(DeepONet(coeffs.shape[1], 2, 32, seed), u_mean, s)
        target = (utrain - u_mean) / s          # the loss is computed in normalized units
        fit = lambda: model.net(ctrain, coords_t)
    else:
        model = DeepONet(coeffs.shape[1], 2, 32, seed)
        target = utrain
        fit = lambda: model(ctrain, coords_t)
    coords_t = torch.tensor(coords)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for it in range(adam_iters):
        opt.zero_grad(); loss = ((fit() - target) ** 2).mean(); loss.backward(); opt.step()
    adam_s = time.perf_counter() - t0
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, max_eval=lbfgs_iters * 2,
                              tolerance_grad=1e-14, tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(); l = ((fit() - target) ** 2).mean(); l.backward(); return l
    t0 = time.perf_counter(); lbfgs.step(closure); lbfgs_s = time.perf_counter() - t0
    ctest = torch.tensor(coeffs[n_train:n_train + n_test])
    utest = U[n_train:n_train + n_test].reshape(n_test, -1)
    with torch.no_grad():
        pred_test = model(ctest, coords_t).numpy()
        train_mse = float(((model(ctrain, coords_t) - utrain) ** 2).mean())
    rel = np.linalg.norm(pred_test - utest, axis=1) / np.linalg.norm(utest, axis=1)
    res = dict(n_train_used=n_train_used, seed=seed, dataset=data, normalize=normalize, rel_l2_median=float(np.median(rel)),
               rel_l2_min=float(rel.min()), rel_l2_max=float(rel.max()), rel_l2_mean=float(rel.mean()),
               train_mse=train_mse, adam_s=adam_s, lbfgs_s=lbfgs_s,
               n_params=sum(p.numel() for p in model.parameters()))
    return res, model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n_train", type=int); ap.add_argument("seed", type=int)
    ap.add_argument("--data", default=DEFAULT_DATA); ap.add_argument("--out", default=None)
    ap.add_argument("--state", default=None); ap.add_argument("--normalize", action="store_true")
    a = ap.parse_args()
    out = a.out or str(REPO_ROOT / "results" / "chapter09" / f"deeponet_n{a.n_train}_seed{a.seed}.json")
    r, model = train(a.n_train, a.seed, a.data, normalize=a.normalize)
    json.dump(r, open(out, "w"), indent=1)
    if a.state:
        torch.save(model.state_dict(), a.state)
    print(r)
