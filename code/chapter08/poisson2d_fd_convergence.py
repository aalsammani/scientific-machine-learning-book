"""Five-point finite differences for the Poisson problem of the Deep Ritz example (Chapter 8,
section 8.9, classical comparator for code/chapter08/deepritz.py).

    -Laplace(u) = f on (0,1)^2, u = 0 on the boundary, f = 2 pi^2 sin(pi x) sin(pi y),
    exact solution u = sin(pi x) sin(pi y).

Standard five-point Laplacian on n x n interior nodes (h = 1/(n+1)), n = 15, 31, 63, 127; sparse
direct solve; relative discrete L2 error at the interior nodes; observed order between successive
grids.

Provenance: the stored file data/reference/exA/poisson2d_fd_verification.json (quoted in section 8.9)
was produced by a script that was not preserved. This script was written for the public repository
to regenerate it; it reproduces the stored errors to round-off (compare with git diff after
running).

Usage:
    python code/chapter08/poisson2d_fd_convergence.py
Output: data/reference/exA/poisson2d_fd_verification.json
Runtime: under a second. Deterministic.
"""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "reference" / "exA" / "poisson2d_fd_verification.json"


def solve(n):
    h = 1.0 / (n + 1)
    xs = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    T1 = diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]) / h ** 2
    A = (kron(T1, identity(n)) + kron(identity(n), T1)).tocsc()
    f = 2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    u = spsolve(A, f.ravel()).reshape(n, n)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return h, float(np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact))


def main():
    rows = []
    for n in (15, 31, 63, 127):
        h, err = solve(n)
        rows.append(dict(n=n, h=h, rel_l2=err))
    order = [float(np.log(rows[i - 1]["rel_l2"] / rows[i]["rel_l2"]) / np.log(rows[i - 1]["h"] / rows[i]["h"]))
             for i in range(1, len(rows))]
    with open(OUT_FILE, "w") as fh:
        json.dump(dict(rows=rows, observed_order=order), fh, indent=1)
    for r in rows:
        print(r)
    print("observed order:", [round(o, 4) for o in order])


if __name__ == "__main__":
    main()
