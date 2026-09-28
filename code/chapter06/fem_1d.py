"""One-dimensional P1 finite elements for Running Example A: consistent mass matrix, Crank-Nicolson
time stepping with dt = h. Verified against the exact solution by refinement, independently of the
finite-difference verification in code/reference_problems/exA_heat.py.

Book: Chapter 3, section 3.8 (observed orders) and Chapter 6, section 6.10 (Table 6.1, mesh-based
comparator for the physics-informed network).

Usage:
    python code/chapter06/fem_1d.py            # errors and observed orders
    python code/chapter06/fem_1d.py --timed    # the same errors with wall-clock times per mesh

Outputs: data/reference/exA/fem_verification.json (default) or
data/reference/exA/fem_verification_timed.json (--timed). The stored timed file was produced by a
variant of this script that was not preserved; --timed reproduces its errors, while its wall-clock
times are machine specific. Runtime: about a second. Deterministic.
"""
import numpy as np
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import exact, NU, T, initial  # noqa: E402


def assemble_1d_P1(nx):
    """Stiffness K and consistent mass M on a uniform mesh of nx nodes on (0,1),
    homogeneous Dirichlet eliminated (interior nodes only, size nx-2)."""
    x = np.linspace(0, 1, nx)
    h = x[1] - x[0]
    n = nx - 2  # interior DOFs
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    # element stiffness [[1,-1],[-1,1]]/h, element mass h/6*[[2,1],[1,2]]
    for e in range(nx - 1):
        i, j = e - 1, e  # local-to-global for interior dof indexing (node e and e+1)
        ke = np.array([[1, -1], [-1, 1]]) / h
        me = h / 6 * np.array([[2, 1], [1, 2]])
        for a_local, a_global in ((0, e - 1), (1, e)):
            if a_global < 0 or a_global >= n:
                continue
            for b_local, b_global in ((0, e - 1), (1, e)):
                if b_global < 0 or b_global >= n:
                    continue
                K[a_global, b_global] += ke[a_local, b_local]
                M[a_global, b_global] += me[a_local, b_local]
    return x, h, K, M


def fem_cn(nx, nt, nu=NU, T_=T):
    from scipy.linalg import solve
    x, h, K, M = assemble_1d_P1(nx)
    dt = T_ / nt
    u0 = initial(x[1:-1])
    A_lhs = M + 0.5 * dt * nu * K
    A_rhs = M - 0.5 * dt * nu * K
    U = np.zeros((nt + 1, nx))
    U[0, 1:-1] = u0
    u = u0.copy()
    for n in range(nt):
        rhs = A_rhs @ u
        u = solve(A_lhs, rhs)
        U[n + 1, 1:-1] = u
    t = np.linspace(0, T_, nt + 1)
    return x, t, U


if __name__ == "__main__":
    import json
    timed = "--timed" in sys.argv[1:]
    rows = []
    for nx in (11, 21, 41, 81, 161, 321):
        h = 1 / (nx - 1)
        nt = int(round(T / h))  # dt = h, matches the Crank-Nicolson-FD verification protocol
        t0 = time.perf_counter()
        x, t, U = fem_cn(nx, nt)
        wall = time.perf_counter() - t0
        X, Tg = np.meshgrid(x, t, indexing="ij")
        Uex = exact(X, Tg)
        e = U.T - Uex
        rel_l2 = float(np.linalg.norm(e) / np.linalg.norm(Uex))
        rows.append(dict(nx=nx, nt=nt, h=h, rel_l2=rel_l2, wall_s=wall) if timed else dict(nx=nx, nt=nt, h=h, rel_l2=rel_l2))
    hs = np.array([r["h"] for r in rows])
    es = np.array([r["rel_l2"] for r in rows])
    order = (np.log(es[1:] / es[:-1]) / np.log(hs[1:] / hs[:-1])).tolist()
    if timed:
        out = dict(method="P1 FEM CN dt=h", rows=rows)
        out_file = REPO_ROOT / "data" / "reference" / "exA" / "fem_verification_timed.json"
    else:
        out = dict(method="P1 FEM, consistent mass, Crank-Nicolson dt=h", rows=rows,
                   observed_order_l2=order, expected_order=2.0)
        out_file = REPO_ROOT / "data" / "reference" / "exA" / "fem_verification.json"
    with open(out_file, "w") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print(r)
    print("observed order:", [round(o, 3) for o in order])
