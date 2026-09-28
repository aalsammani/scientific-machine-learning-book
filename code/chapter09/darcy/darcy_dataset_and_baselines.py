"""Darcy dataset and non-learned baselines (Chapter 9, section 9.6; Chapter 14, section 14.4).

  1. Check that the archived DeepONet state trained with the legacy (first-order) solver reproduces
     its test error on the archived legacy test set (results/chapter09/legacy_first_order_solver/).
  2. Generate the dataset with the verified solver: 500 coefficient draws from default_rng(0),
     identical to the draws of the legacy dataset (asserted).
  3. Quantify the difference between legacy and corrected data (about 14%, Chapter 14, section
     14.4, Instance 1) and compute baselines on the 100 test pairs for training-set sizes
     25, 50, 100, 200, 400:
       * the training-mean field;
       * affine least squares in the coefficients c;
       * quadratic least squares in c (only when n > 21, the number of quadratic features).

Usage:
    python code/chapter09/darcy/darcy_dataset_and_baselines.py
Outputs: results/chapter09/darcy_dataset.npz and results/chapter09/regeneration_prepare.json.
Runtime: a few seconds (wall-clock entries are machine specific). Deterministic.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import darcy_data
from deeponet import DeepONet

R = Path(__file__).resolve().parents[3] / "results" / "chapter09"
LEG = R / "legacy_first_order_solver"   # archived dataset and DeepONet state from the legacy solver

out = {}
# 1. legacy reproduction check
leg = np.load(f"{LEG}/darcy_dataset.npz")
coeffs_l, U_l, X, Y = leg["coeffs"], leg["U"], leg["X"], leg["Y"]
coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
m = DeepONet(5, 2, 32, 0); m.load_state_dict(torch.load(f"{LEG}/deeponet_state_n200.pt"))
with torch.no_grad():
    p = m(torch.tensor(coeffs_l[400:500]), coords).numpy()
ut = U_l[400:500].reshape(100, -1)
rel = np.linalg.norm(p - ut, axis=1) / np.linalg.norm(ut, axis=1)
out["legacy_state_on_legacy_test_median"] = float(np.median(rel))
print("stored n=200 state on legacy test set, median rel L2:", np.median(rel), "(published 0.02264)")

# 2. dataset with the verified solver (identical coefficient draws)
rng = np.random.default_rng(0)
coeffs = rng.uniform(-1, 1, (500, 5))
assert np.allclose(coeffs, coeffs_l), "coefficient draws differ from the archived legacy dataset"
A = np.zeros((500, 21, 21)); U = np.zeros((500, 21, 21))
t0 = time.perf_counter()
for i in range(500):
    X, Y, a, u = darcy_data.darcy_solve(coeffs[i]); A[i] = a; U[i] = u
gen = time.perf_counter() - t0
np.savez(f"{R}/darcy_dataset.npz", coeffs=coeffs, A=A, U=U, X=X, Y=Y, n_train=400, n_test=100, gen_time=gen, boundary="corrected")
out["generation_time_500_s"] = gen
print("regenerated 500 solves in", gen, "s")

# 3. legacy vs corrected data, and baseline predictors on the corrected test set
dl = np.linalg.norm((U_l - U).reshape(500, -1), axis=1) / np.linalg.norm(U.reshape(500, -1), axis=1)
out["legacy_vs_corrected_rel_diff_median"] = float(np.median(dl)); out["legacy_vs_corrected_rel_diff_range"] = [float(dl.min()), float(dl.max())]
Ut = U[400:500].reshape(100, -1); nt = np.linalg.norm(Ut, axis=1)
X_, Y_, a1, u_a1 = darcy_data.darcy_solve(np.zeros(5))
out["baseline_a_equal_1"] = float(np.median(np.linalg.norm(Ut - u_a1.ravel(), axis=1) / nt))
spread = np.linalg.norm(Ut - Ut.mean(0), axis=1) / nt
out["test_set_spread_about_its_mean_median"] = float(np.median(spread))
base = {}
for n in (25, 50, 100, 200, 400):
    Utr = U[:n].reshape(n, -1); C = coeffs[:n]
    mean = Utr.mean(0)
    r_mean = np.linalg.norm(Ut - mean, axis=1) / nt
    Phi = np.hstack([np.ones((n, 1)), C])                    # affine in c
    W, *_ = np.linalg.lstsq(Phi, Utr, rcond=None)
    r_lin = np.linalg.norm(Ut - np.hstack([np.ones((100, 1)), coeffs[400:]]) @ W, axis=1) / nt
    def quad(Cm):
        iu = np.triu_indices(5)
        return np.hstack([np.ones((len(Cm), 1)), Cm, (Cm[:, :, None] * Cm[:, None, :])[:, iu[0], iu[1]]])
    r_quad = None
    if n > 21:
        Wq, *_ = np.linalg.lstsq(quad(C), Utr, rcond=None)
        r_quad = np.linalg.norm(Ut - quad(coeffs[400:]) @ Wq, axis=1) / nt
    base[n] = dict(mean_predictor_median=float(np.median(r_mean)), affine_ls_median=float(np.median(r_lin)),
                   affine_ls_range=[float(r_lin.min()), float(r_lin.max())],
                   quadratic_ls_median=None if r_quad is None else float(np.median(r_quad)),
                   quadratic_ls_range=None if r_quad is None else [float(r_quad.min()), float(r_quad.max())])
    print(n, base[n])
out["baselines_by_n_train"] = base
# the same baselines on the legacy data (to interpret results obtained with the legacy solver)
Utl = U_l[400:500].reshape(100, -1); ntl = np.linalg.norm(Utl, axis=1)
out["legacy_test_spread_about_train200_mean_median"] = float(np.median(np.linalg.norm(Utl - U_l[:200].reshape(200, -1).mean(0), axis=1) / ntl))
Phi = np.hstack([np.ones((200, 1)), coeffs_l[:200]]); W, *_ = np.linalg.lstsq(Phi, U_l[:200].reshape(200, -1), rcond=None)
out["legacy_affine_ls_n200_median"] = float(np.median(np.linalg.norm(Utl - np.hstack([np.ones((100, 1)), coeffs_l[400:]]) @ W, axis=1) / ntl))
print({k: v for k, v in out.items() if k != "baselines_by_n_train"})
json.dump(out, open(f"{R}/regeneration_prepare.json", "w"), indent=1)
