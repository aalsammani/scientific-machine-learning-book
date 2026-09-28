"""Resolution-transfer experiment for the Fourier neural operator (Chapter 9, section 9.4; Table 9.1;
Figure 9.1, left panel).

For each seed: 400 training and 100 test inputs drawn with numpy.random.default_rng(seed) from the
training distribution of fno1d.py; training on a 64-point grid; FNO with 12 retained modes and
width 16 (13153 parameters); full-batch Adam, 2000 iterations, learning rate 1e-3, mean-squared
loss; evaluation of the median relative L2 test error at 32, 64, 128 and 256 points.

Usage:
    python code/chapter09/fno/fno_resolution_transfer.py 0 1 2 3 4

Output: results/chapter09/fno/resolution_transfer.json (entries for the given seeds are added or
replaced). Runtime: about 1-3 minutes per seed on one core. See fno1d.py for the reproducibility
caveat about the pointwise layers' initialization.
"""
import sys, torch, numpy as np, json, time, os
from pathlib import Path
from fno1d import FNO1d, sample_u0_coeffs, eval_on_grid, exact_propagate, MAX_MODE_TRAIN, NU, T
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

N_TRAIN_RES = 64
N_TRAIN, N_TEST = 400, 100
RESULTS = Path(__file__).resolve().parents[3] / "results" / "chapter09" / "fno"
OUT = RESULTS / "resolution_transfer.json"

def train_seed(seed, modes=12, width=16, adam_iters=2000):
    rng = np.random.default_rng(seed)
    coeffs_all = sample_u0_coeffs(N_TRAIN+N_TEST, MAX_MODE_TRAIN, rng)
    u0_train, _ = eval_on_grid(coeffs_all[:N_TRAIN], N_TRAIN_RES)
    u_train_true = exact_propagate(coeffs_all[:N_TRAIN], N_TRAIN_RES, T)
    model = FNO1d(modes=modes, width=width, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    u0_t, u_true_t = torch.tensor(u0_train), torch.tensor(u_train_true)
    t0 = time.perf_counter()
    for it in range(adam_iters):
        opt.zero_grad()
        loss = ((model(u0_t)-u_true_t)**2).mean()
        loss.backward(); opt.step()
    train_s = time.perf_counter()-t0
    return model, coeffs_all, train_s

def eval_resolution(model, coeffs_test, n_grid):
    u0, _ = eval_on_grid(coeffs_test, n_grid)
    u_true = exact_propagate(coeffs_test, n_grid, T)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(torch.tensor(u0)).numpy()
    inf_s = (time.perf_counter()-t0)/coeffs_test.shape[0]
    relerr = np.linalg.norm(pred-u_true, axis=1)/np.linalg.norm(u_true, axis=1)
    linf = np.abs(pred-u_true).max(axis=1)
    return dict(median_rel_l2=float(np.median(relerr)), max_rel_l2=float(relerr.max()),
                median_linf=float(np.median(linf)), inference_s_per_sample=inf_s)

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]]
    if not seeds:
        raise SystemExit("usage: python code/chapter09/fno/fno_resolution_transfer.py SEED [SEED ...]   (the book uses 0 1 2 3 4)")
    resolutions = [32, 64, 128, 256]
    RESULTS.mkdir(parents=True, exist_ok=True)
    existing = json.load(open(OUT)) if os.path.exists(OUT) else dict(resolutions=resolutions, results=[])
    for seed in seeds:
        model, coeffs_all, train_s = train_seed(seed)
        coeffs_test = coeffs_all[N_TRAIN:]
        per_res = {res: eval_resolution(model, coeffs_test, res) for res in resolutions}
        existing["results"] = [r for r in existing["results"] if r["seed"] != seed]
        existing["results"].append(dict(seed=seed, train_s=train_s, per_resolution=per_res,
                                         n_params=sum(p.numel() for p in model.parameters())))
        print(f"seed {seed}: train {train_s:.1f}s;", {r: round(per_res[r]['median_rel_l2'],5) for r in resolutions})
    existing["results"].sort(key=lambda r: r["seed"])
    existing["n_seeds"] = len(existing["results"])
    json.dump(existing, open(OUT, "w"), indent=1)
