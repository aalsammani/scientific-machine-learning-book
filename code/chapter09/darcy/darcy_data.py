"""Variable-coefficient Darcy solver and dataset generator (Running Example D; Chapter 9,
section 9.6, reused in Chapters 12, 14 and 15; Appendix D).

Problem: -div(a grad u) = f on (0,1)^2, u = 0 on the boundary, f = 1, with the five-parameter
coefficient family a(x, y) = 1 + (0.4/5) sum_{k=1..5} c_k sin(k pi x) cos(k pi y), c_k ~ U(-1, 1).

Discretization: node-centered five-point finite differences on the n x n interior nodes
x_i = i h (i = 1..n, h = 1/(n+1)); n = 21 for the dataset. The conductance on a face between two
interior nodes is the harmonic mean of the nodal coefficients; on a face between an interior node
and the boundary it is the coefficient at the interior node. Sparse direct solve (SciPy spsolve).
Verified by code/chapter09/darcy/darcy_manufactured_verification.py (second order).

boundary="legacy" uses the conductance 2 a on boundary faces, an inconsistent treatment (it places
the boundary at distance h/2 instead of h) that reduces the scheme to first order. It is kept only
to reproduce the verification record discussed in Chapter 14, section 14.4 (Instance 1); every
result in the book uses the default boundary="corrected".

Usage:
    python code/chapter09/darcy/darcy_data.py
Output: results/chapter09/darcy_dataset.npz (500 coefficient draws from numpy default_rng(0);
the first 400 are training data, the last 100 test data). darcy_dataset_and_baselines.py writes the
same dataset. Runtime: under a second. Deterministic.
"""
import numpy as np, time, os
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

RESULTS = Path(__file__).resolve().parents[3] / "results" / "chapter09"

N_GRID = 21   # solver resolution used for the operator-learning dataset
K = 5         # dimension of the coefficient parameterization
AMP = 0.4     # a = 1 + (AMP/K) sum_k c_k sin(k pi x) cos(k pi y)


def grid(n):
    h = 1.0 / (n + 1)
    xs = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return h, X, Y


def make_a(coeffs, X, Y, first_mode=1):
    """Coefficient field; modes first_mode..first_mode+K-1 (first_mode=1 is the training family)."""
    a = np.ones_like(X)
    for k in range(len(coeffs)):
        m = first_mode + k
        a = a + AMP / K * coeffs[k] * np.sin(np.pi * m * X) * np.cos(np.pi * m * Y)
    return a


def assemble(a, h, boundary="corrected"):
    """Sparse matrix of the 5-point discretization of -div(a grad .) with zero Dirichlet data.
    a is the (n, n) array of nodal coefficient values, indexed [i, j] <-> (x_i, y_j)."""
    n = a.shape[0]
    bf = {"corrected": 1.0, "legacy": 2.0}[boundary]
    hx = 2.0 / (1.0 / a[:-1, :] + 1.0 / a[1:, :])   # faces between (i,j) and (i+1,j)
    hy = 2.0 / (1.0 / a[:, :-1] + 1.0 / a[:, 1:])   # faces between (i,j) and (i,j+1)
    ae = np.empty_like(a); ae[:-1, :] = hx; ae[-1, :] = bf * a[-1, :]
    aw = np.empty_like(a); aw[1:, :] = hx;  aw[0, :] = bf * a[0, :]
    an = np.empty_like(a); an[:, :-1] = hy; an[:, -1] = bf * a[:, -1]
    as_ = np.empty_like(a); as_[:, 1:] = hy; as_[:, 0] = bf * a[:, 0]
    idx = np.arange(n * n).reshape(n, n)
    rows = [idx.ravel()]; cols = [idx.ravel()]; vals = [((ae + aw + an + as_) / h**2).ravel()]
    rows.append(idx[:-1, :].ravel()); cols.append(idx[1:, :].ravel()); vals.append((-hx / h**2).ravel())
    rows.append(idx[1:, :].ravel()); cols.append(idx[:-1, :].ravel()); vals.append((-hx / h**2).ravel())
    rows.append(idx[:, :-1].ravel()); cols.append(idx[:, 1:].ravel()); vals.append((-hy / h**2).ravel())
    rows.append(idx[:, 1:].ravel()); cols.append(idx[:, :-1].ravel()); vals.append((-hy / h**2).ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return A.tocsr()


def solve_field(a, f, h, boundary="corrected"):
    """Solve with nodal coefficient array a and nodal forcing array f (or a scalar f)."""
    n = a.shape[0]
    b = np.broadcast_to(np.asarray(f, dtype=float), (n, n)).ravel().copy()
    return spsolve(assemble(a, h, boundary), b).reshape(n, n)


def darcy_solve(coeffs, n=N_GRID, f_val=1.0, first_mode=1, boundary="corrected"):
    h, X, Y = grid(n)
    a = make_a(coeffs, X, Y, first_mode)
    return X, Y, a, solve_field(a, f_val, h, boundary)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_train, n_test = 400, 100
    coeffs_all = rng.uniform(-1, 1, (n_train + n_test, K))
    A_all = np.zeros((n_train + n_test, N_GRID, N_GRID))
    U_all = np.zeros((n_train + n_test, N_GRID, N_GRID))
    t0 = time.perf_counter()
    for i in range(n_train + n_test):
        X, Y, a, u = darcy_solve(coeffs_all[i])
        A_all[i] = a; U_all[i] = u
    gen_time = time.perf_counter() - t0
    print("data generation time for", n_train + n_test, "solves:", gen_time, "s; per solve:", gen_time / (n_train + n_test))
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(RESULTS / "darcy_dataset.npz",
             coeffs=coeffs_all, A=A_all, U=U_all, X=X, Y=Y, n_train=n_train, n_test=n_test,
             gen_time=gen_time, boundary="corrected")
