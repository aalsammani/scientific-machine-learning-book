"""Distribution shift for the Darcy DeepONet of Chapter 9 (Chapter 14, section 14.5; Figure 14.2;
Chapter 9, section 9.8).

In-family inputs: a = 1 + 0.08 sum_{k=1..5} c_k sin(k pi x) cos(k pi y)  (Chapter 9 test set).
Out-of-family inputs: same form with modes 6..10, c = default_rng(123).uniform(-1, 1, (100, 5)).
The DeepONet's branch network receives only the coefficient vector c, so an out-of-family field
is necessarily presented to it as the in-family field with the same coefficients. Reported:
  * DeepONet error, in family and out of family (5 seeds, standardized outputs, n = 200; the
    stored model states results/chapter09/sweep_normalized/state_n200_seed{s}.pt);
  * the error of an exact in-family surrogate on the out-of-family set, i.e. the relative
    difference between G(a_{6..10}(c)) and G(a_{1..5}(c)), a floor that no surrogate whose input is
    the coefficient vector c can beat;
  * the error of the constant (training-mean) predictor in and out of family.

Usage:
    python code/chapter14/distribution_shift.py
Output: results/chapter14/deeponet_distribution_shift.json
Runtime: a few seconds. Deterministic.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter09" / "darcy"))
import darcy_data
from deeponet import DeepONet, Normalized
torch.set_default_dtype(torch.float64)

R9 = REPO_ROOT / "results" / "chapter09"
d = np.load(f"{R9}/darcy_dataset.npz")
coeffs, U, X, Y = d["coeffs"], d["U"], d["X"], d["Y"]
coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
c_in, U_in = coeffs[400:500], U[400:500].reshape(100, -1)
c_out = np.random.default_rng(123).uniform(-1, 1, (100, 5))
U_out = np.stack([darcy_data.darcy_solve(c, first_mode=6)[3].ravel() for c in c_out])
U_in_same_c = np.stack([darcy_data.darcy_solve(c, first_mode=1)[3].ravel() for c in c_out])

def rel(P, T):
    return np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)
def summ(r):
    return dict(median=float(np.median(r)), min=float(r.min()), max=float(r.max()))

out = {}
out["exact_in_family_surrogate_on_out_of_family"] = summ(rel(U_in_same_c, U_out))
mean = U[:200].reshape(200, -1).mean(0)
out["mean_predictor"] = dict(in_family=summ(rel(mean[None], U_in)), out_of_family=summ(rel(mean[None], U_out)))
out["deeponet_normalized_n200"] = {}
for s in range(5):
    sd = torch.load(f"{R9}/sweep_normalized/state_n200_seed{s}.pt")
    m = Normalized(DeepONet(5, 2, 32, 0), sd["u_mean"], sd["s"]); m.load_state_dict(sd)
    with torch.no_grad():
        p_in = m(torch.tensor(c_in), coords).numpy(); p_out = m(torch.tensor(c_out), coords).numpy()
    ri, ro = rel(p_in, U_in), rel(p_out, U_out)
    out["deeponet_normalized_n200"][s] = dict(in_family=summ(ri), out_of_family=summ(ro),
                                               degradation_factor_of_medians=float(np.median(ro) / np.median(ri)))
    print(s, out["deeponet_normalized_n200"][s])
print("exact in-family surrogate on shifted set:", out["exact_in_family_surrogate_on_out_of_family"])
print("mean predictor:", out["mean_predictor"])
json.dump(out, open(REPO_ROOT / "results" / "chapter14" / "deeponet_distribution_shift.json", "w"), indent=1)
