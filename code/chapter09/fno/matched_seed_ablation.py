"""Matched-seed mode-truncation ablation (Chapter 9, section 9.5; Figure 9.2).

For each seed, a 4-mode and a 12-mode FNO (width 16) are trained on the same data (numpy
default_rng(seed)) with the protocol of fno_resolution_transfer.py, and the difference of their
median relative test errors at 64 points is recorded. The book uses seeds 1-4; these are fresh
training runs, so the 12-mode errors differ from the Table 9.1 entries for the same seeds (see the
initialization caveat in fno1d.py).

Usage:
    python code/chapter09/fno/matched_seed_ablation.py 1 2 3 4

Output: results/chapter09/fno/matched_seed_ablation.json (entries for the given seeds are added or
replaced). Runtime: about 2 minutes per seed on one core.
"""
import sys, torch, numpy as np, json, time, os
from pathlib import Path
from fno1d import FNO1d, sample_u0_coeffs, eval_on_grid, exact_propagate, MAX_MODE_TRAIN, NU, T
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

N_TRAIN_RES = 64
N_TRAIN, N_TEST = 400, 100
RESULTS = Path(__file__).resolve().parents[3] / "results" / "chapter09" / "fno"
OUT = RESULTS / "matched_seed_ablation.json"

def train_and_eval(seed, modes, adam_iters=2000):
    rng = np.random.default_rng(seed)
    coeffs_all = sample_u0_coeffs(N_TRAIN+N_TEST, MAX_MODE_TRAIN, rng)
    u0_train, _ = eval_on_grid(coeffs_all[:N_TRAIN], N_TRAIN_RES)
    u_train_true = exact_propagate(coeffs_all[:N_TRAIN], N_TRAIN_RES, T)
    model = FNO1d(modes=modes, width=16, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    u0_t, u_true_t = torch.tensor(u0_train), torch.tensor(u_train_true)
    t0 = time.perf_counter()
    for it in range(adam_iters):
        opt.zero_grad()
        loss = ((model(u0_t)-u_true_t)**2).mean()
        loss.backward(); opt.step()
    train_s = time.perf_counter()-t0
    coeffs_test = coeffs_all[N_TRAIN:]
    u0_test, _ = eval_on_grid(coeffs_test, N_TRAIN_RES)
    u_test_true = exact_propagate(coeffs_test, N_TRAIN_RES, T)
    with torch.no_grad():
        pred = model(torch.tensor(u0_test)).numpy()
    relerr = np.linalg.norm(pred-u_test_true, axis=1)/np.linalg.norm(u_test_true, axis=1)
    return dict(median_rel_l2=float(np.median(relerr)), mean_rel_l2=float(np.mean(relerr)),
                train_s=train_s, n_params=sum(p.numel() for p in model.parameters()))

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]]
    if not seeds:
        raise SystemExit("usage: python code/chapter09/fno/matched_seed_ablation.py SEED [SEED ...]   (the book uses 1 2 3 4)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    existing = json.load(open(OUT)) if os.path.exists(OUT) else dict(seeds=[], results=[])
    for seed in seeds:
        r4 = train_and_eval(seed, modes=4)
        r12 = train_and_eval(seed, modes=12)
        delta = r4["median_rel_l2"] - r12["median_rel_l2"]
        existing["results"] = [r for r in existing["results"] if r["seed"] != seed]
        existing["results"].append(dict(seed=seed, modes4=r4, modes12=r12, delta=delta))
        print(f"seed {seed}: 4-mode={r4['median_rel_l2']*100:.3f}% 12-mode={r12['median_rel_l2']*100:.3f}% delta={delta*100:.3f}pp")
    existing["results"].sort(key=lambda r: r["seed"])
    existing["seeds"] = [r["seed"] for r in existing["results"]]
    json.dump(existing, open(OUT, "w"), indent=1)
