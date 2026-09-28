"""Serial timing for the break-even analysis (Chapter 9, section 9.6; Chapter 15, section 15.2).
Run alone on an otherwise idle machine.

Measures, on one thread:
  * the classical solve cost per query (assembly + sparse direct solve, n = 21; median of five
    repetitions of 500 solves);
  * the DeepONet training cost (n = 200, seed 0, standardized outputs);
  * the DeepONet evaluation cost per query, single-query (median of 2000) and batched
    (1000 queries per call, median of 50);
  * fit and evaluation cost of the quadratic least-squares response surface in c.
Break-even query count N* = (200 c_s + c_t) / (c_s - c_e), where c_s is the solve cost per query
(so 200 c_s is the cost of generating the training data), c_t the training cost and c_e the
evaluation cost per query; there is no break-even when c_e >= c_s.

Usage:
    python code/chapter09/darcy/break_even_timing.py
Output: results/chapter09/break_even_timing.json
Runtime: about 1.5 minutes. Timings are hardware and software specific and are not portable.
"""
import sys, time, json, platform, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import darcy_data
from deeponet import train
torch.set_num_threads(1)

R = Path(__file__).resolve().parents[3] / "results" / "chapter09"
d = np.load(f"{R}/darcy_dataset.npz")
coeffs, U = d["coeffs"], d["U"]
out = dict(machine=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
           torch=torch.__version__, numpy=np.__version__, threads=1, precision="float64")

# classical solve: median over 5 repetitions of 500 solves
reps = []
for r in range(5):
    t0 = time.perf_counter()
    for i in range(500):
        darcy_data.darcy_solve(coeffs[i])
    reps.append((time.perf_counter() - t0) / 500)
out["classical_per_solve_s"] = float(np.median(reps)); out["classical_per_solve_reps_s"] = reps
print("classical per solve:", reps)

# DeepONet training (n = 200, seed 0, normalized protocol)
res, model = train(200, 0, normalize=True)
out["deeponet_n200_seed0"] = res
out["deeponet_training_s"] = res["adam_s"] + res["lbfgs_s"]
print("training:", out["deeponet_training_s"], "test median", res["rel_l2_median"])

# DeepONet evaluation, one query (5 coefficients -> 441 nodal values), median of 2000
X, Y = d["X"], d["Y"]
coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
c = torch.tensor(coeffs[450:451]); ts = []
with torch.no_grad():
    for _ in range(200): model(c, coords)
    for _ in range(2000):
        t0 = time.perf_counter(); model(c, coords); ts.append(time.perf_counter() - t0)
out["deeponet_eval_per_query_s"] = float(np.median(ts))
print("deeponet eval per query:", np.median(ts))
# batched evaluation: 1000 queries in one call, per-query cost = total / 1000 (median of 50)
cb = torch.tensor(np.random.default_rng(5).uniform(-1, 1, (1000, 5))); tb = []
with torch.no_grad():
    for _ in range(5): model(cb, coords)
    for _ in range(50):
        t0 = time.perf_counter(); model(cb, coords); tb.append((time.perf_counter() - t0) / 1000)
out["deeponet_eval_per_query_batched1000_s"] = float(np.median(tb))
print("deeponet eval per query, batch of 1000:", np.median(tb))

# quadratic response surface (least squares in the 5 coefficients), fitted on 200 samples
iu = np.triu_indices(5)
def feats(C):
    return np.hstack([np.ones((len(C), 1)), C, (C[:, :, None] * C[:, None, :])[:, iu[0], iu[1]]])
t0 = time.perf_counter()
W, *_ = np.linalg.lstsq(feats(coeffs[:200]), U[:200].reshape(200, -1), rcond=None)
out["quadratic_fit_s"] = time.perf_counter() - t0
ts = []
for i in range(2000):
    t0 = time.perf_counter(); feats(coeffs[450:451]) @ W; ts.append(time.perf_counter() - t0)
out["quadratic_eval_per_query_s"] = float(np.median(ts))
pred = feats(coeffs[400:]) @ W; ut = U[400:].reshape(100, -1)
out["quadratic_n200_test_median"] = float(np.median(np.linalg.norm(pred - ut, axis=1) / np.linalg.norm(ut, axis=1)))

cs, ce, ct = out["classical_per_solve_s"], out["deeponet_eval_per_query_s"], out["deeponet_training_s"]
out["break_even_queries_deeponet_single_query_eval"] = (200 * cs + ct) / (cs - ce) if cs > ce else None
ceb = out["deeponet_eval_per_query_batched1000_s"]
out["break_even_queries_deeponet_batched_eval"] = (200 * cs + ct) / (cs - ceb)
out["break_even_queries_quadratic"] = (200 * cs + out["quadratic_fit_s"]) / (cs - out["quadratic_eval_per_query_s"])
print(json.dumps({k: v for k, v in out.items() if k != "deeponet_n200_seed0"}, indent=1))
json.dump(out, open(f"{R}/break_even_timing.json", "w"), indent=1)
