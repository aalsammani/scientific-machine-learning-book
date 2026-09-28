"""Practical identifiability (Fisher information) and surrogate-error contamination for the Darcy
family of Chapter 9 (Chapter 12, sections 12.4 and 12.6; Figure 12.2; Chapter 15, section 15.2).

Protocol:
  * true coefficients c_true = default_rng(42).uniform(-1, 1, 5);
  * 15 sparse observation nodes drawn next from the same generator (without replacement);
  * noise standard deviation sigma = 0.02 max|u(c_true)|;
  * Fisher information F = sigma^-2 J^T J with J the central-difference sensitivity (step 1e-4) at
    c_true, for the 15 sparse nodes and for all 441 nodes; condition number over 50 random
    placements of the 15 nodes (default_rng(7));
  * dense single-coefficient inversion: all 441 nodes observed, c_1 unknown, c_2..c_5 fixed at their
    true values; estimate by bounded scalar minimization of the sum of squared residuals on [-1, 1]
    with either the finite-difference solver ("classical") or a trained DeepONet ("surrogate") as
    the forward map; M = 200 noise realizations (default_rng(1000 + m)) give bias, standard
    deviation and RMSE. One further realization (a fresh seed-42 generator after the five
    coefficient draws; key first_edition_realization) is the single realization of the original
    study and is reported separately.
The two surrogates of the book are the n = 200, seed 0 DeepONets of Chapter 9 trained with
standardized outputs (well trained) and with raw outputs (poorly trained); they are the defaults.
--legacy uses the legacy first-order boundary treatment of the Darcy solver and reproduces the
numbers originally obtained with that solver.

Usage (relative --surrogate paths are taken relative to the repository root; --out is an ordinary
file path):
    python code/chapter12/darcy_inversion.py
    python code/chapter12/darcy_inversion.py --surrogate results/chapter09/sweep_normalized/state_n200_seed0.pt

Output: results/chapter12/darcy_inversion_dense.json (with --legacy:
results/chapter12/darcy_inversion_legacy_solver.json unless --out is given).
Runtime: under a minute on one core. Deterministic given the fixed seeds.
"""
import sys, json, argparse, os
from pathlib import Path
import numpy as np, torch
from scipy.optimize import minimize_scalar
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter09" / "darcy"))
import darcy_data
from deeponet import DeepONet, Normalized
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--legacy", action="store_true")
ap.add_argument("--surrogate", action="append", default=[])
ap.add_argument("--M", type=int, default=200)
ap.add_argument("--out", default=None)
args = ap.parse_args()
if not args.surrogate:
    args.surrogate = ["results/chapter09/sweep_normalized/state_n200_seed0.pt",
                      "results/chapter09/sweep_firstedition_protocol/state_n200_seed0.pt"]
if args.out is None:
    args.out = str(REPO_ROOT / "results" / "chapter12" / ("darcy_inversion_legacy_solver.json" if args.legacy else "darcy_inversion_dense.json"))
boundary = "legacy" if args.legacy else "corrected"

def U(c):
    return darcy_data.darcy_solve(c, boundary=boundary)[3].ravel()

rng = np.random.default_rng(42)
c_true = rng.uniform(-1, 1, 5)
X, Y, a, u_true = darcy_data.darcy_solve(c_true, boundary=boundary)
u_true = u_true.ravel()
sigma = 0.02 * np.abs(u_true).max()
pts = rng.choice(441, 15, replace=False)
# the original dense experiment drew its noise from a fresh generator: seed 42, after c_true
rng_dense = np.random.default_rng(42); rng_dense.uniform(-1, 1, 5)
noise0 = rng_dense.normal(0, sigma, 441)
coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
out = dict(boundary=boundary, c_true=c_true.tolist(), sigma_obs=float(sigma), sparse_nodes=pts.tolist())

# ---- Fisher information (sparse 15 nodes, and all 441 nodes) ----
def jac(nodes, h=1e-4):
    return np.stack([(U(c_true + h*e)[nodes] - U(c_true - h*e)[nodes]) / (2*h) for e in np.eye(5)], 1)
Jall = np.stack([(U(c_true + 1e-4*e) - U(c_true - 1e-4*e)) / 2e-4 for e in np.eye(5)], 1)
for name, nodes in (("sparse15", pts), ("dense441", np.arange(441))):
    F = Jall[nodes].T @ Jall[nodes] / sigma**2
    ev, V = np.linalg.eigh(F)
    out[f"fim_{name}"] = dict(eigenvalues=ev.tolist(), condition_number=float(ev[-1]/ev[0]),
                              eigenvectors_columns=V.tolist(),
                              cramer_rao_std_along_eigvecs=(1/np.sqrt(ev)).tolist(),
                              cramer_rao_std_coordinates=np.sqrt(np.diag(np.linalg.inv(F))).tolist())
    print(name, "eig", np.round(ev, 3), "kappa", round(ev[-1]/ev[0], 1), "CR std", np.round(1/np.sqrt(ev), 3))
    print("   eigvec of smallest:", np.round(V[:, 0], 2), " largest:", np.round(V[:, -1], 2))
kappas = []
prng = np.random.default_rng(7)
for k in range(50):
    nodes = prng.choice(441, 15, replace=False)
    ev = np.linalg.eigvalsh(Jall[nodes].T @ Jall[nodes] / sigma**2)
    kappas.append(float(ev[-1]/ev[0]))
out["fim_sparse15_condition_number_over_50_placements"] = dict(min=min(kappas), median=float(np.median(kappas)), max=max(kappas))
print("kappa over placements: min/median/max", min(kappas), np.median(kappas), max(kappas))

# ---- dense single-coefficient inversion ----
def classical_fit(y):
    def loss(c0):
        c = c_true.copy(); c[0] = c0
        return np.sum((U(c) - y)**2)
    return minimize_scalar(loss, bounds=(-1, 1), method="bounded", options=dict(xatol=1e-8)).x

def load_surrogate(path):
    sd = torch.load(REPO_ROOT / path)
    if any(k.startswith("net.") for k in sd):
        m = Normalized(DeepONet(5, 2, 32, 0), sd["u_mean"], sd["s"])
    else:
        m = DeepONet(5, 2, 32, 0)
    m.load_state_dict(sd); m.eval()
    return m

def surrogate_fit(m, y):
    def loss(c0):
        c = c_true.copy(); c[0] = c0
        with torch.no_grad():
            p = m(torch.tensor(c)[None], coords).numpy().ravel()
        return np.sum((p - y)**2)
    return minimize_scalar(loss, bounds=(-1, 1), method="bounded", options=dict(xatol=1e-8)).x

noises = [noise0] + [np.random.default_rng(1000 + m).normal(0, sigma, 441) for m in range(args.M)]
ys = [u_true + n for n in noises]
cl = np.array([classical_fit(y) for y in ys])
out["classical"] = dict(first_edition_realization=float(cl[0]), abs_error_first_edition_realization=float(abs(cl[0]-c_true[0])),
                        bias=float(cl[1:].mean() - c_true[0]), std=float(cl[1:].std()),
                        rmse=float(np.sqrt(np.mean((cl[1:] - c_true[0])**2))))
print("classical", out["classical"])
F1 = Jall[:, 0] @ Jall[:, 0] / sigma**2
out["cramer_rao_std_c1_dense"] = float(1/np.sqrt(F1))
print("Cramer-Rao std for c1 alone (dense):", 1/np.sqrt(F1))
out["surrogates"] = {}
for path in args.surrogate:
    m = load_surrogate(path)
    with torch.no_grad():
        field_err = float(np.linalg.norm(m(torch.tensor(c_true)[None], coords).numpy().ravel() - u_true) / np.linalg.norm(u_true))
    sr = np.array([surrogate_fit(m, y) for y in ys])
    rec = dict(field_relerr_at_c_true=field_err, first_edition_realization=float(sr[0]),
               abs_error_first_edition_realization=float(abs(sr[0]-c_true[0])),
               bias=float(sr[1:].mean() - c_true[0]), std=float(sr[1:].std()),
               rmse=float(np.sqrt(np.mean((sr[1:] - c_true[0])**2))),
               rmse_ratio_to_classical=float(np.sqrt(np.mean((sr[1:] - c_true[0])**2)) / out["classical"]["rmse"]),
               fraction_at_bound=float(np.mean(np.abs(sr[1:]) > 0.999)))
    out["surrogates"][path] = rec
    print(path, rec)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print("written", args.out)
