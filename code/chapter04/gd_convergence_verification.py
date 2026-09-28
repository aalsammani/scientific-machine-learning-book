"""Gradient-descent bounds versus observed behavior on a quadratic (Chapter 4, section 4.5,
Theorems 4.4 and 4.5; Figure 4.1).

  * float64 NumPy, legacy global generator np.random.seed(0), d = 20.
  * Q = orthogonal factor of QR(randn(20, 20)); A = Q diag(linspace(1, 50, 20)) Q^T, so mu = 1,
    L = 50, condition number 50.
  * b = randn(20); f(x) = 0.5 x^T A x - b^T x; x* = A^{-1} b; x0 = 3 randn(20).
  * Gradient descent with step 1/L = 0.02.
  * Theorem 4.4 check: min_{k<K} ||grad f(x_k)||^2 against 2L (f(x0) - f*) / K.
  * Theorem 4.5 check: ||x_k - x*||^2 against (1 - mu/L)^k ||x0 - x*||^2; the sharper factor
    ((L - mu)/(L + mu))^k is also recorded.
This protocol reproduces the stored original (gd_convergence_verification.json, plotted in
Figure 4.1) to round-off; the script prints the maximum relative deviation.

Usage:
    python code/chapter04/gd_convergence_verification.py
Output: results/chapter04/gd_convergence_verification_rerun.json
Runtime: under a second. Deterministic.
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter04"
OUT = RESULTS / "gd_convergence_verification_rerun.json"
D, MU, L = 20, 1.0, 50.0
ETA = 1.0 / L
K_LIST = [1, 10, 50, 100, 250, 500]
k_LIST = [0, 20, 50, 100, 200, 299]


def main():
    np.random.seed(0)
    Q = np.linalg.qr(np.random.randn(D, D))[0]
    A = Q @ np.diag(np.linspace(MU, L, D)) @ Q.T
    b = np.random.randn(D)
    x0 = 3.0 * np.random.randn(D)
    xs = np.linalg.solve(A, b)

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    fstar = f(xs)
    n_it = max(max(K_LIST), max(k_LIST) + 1)
    x = x0.copy()
    g2, d2 = [], []
    for _ in range(n_it + 1):
        g = A @ x - b
        g2.append(float(g @ g))
        d2.append(float((x - xs) @ (x - xs)))
        x = x - ETA * g
    g2 = np.array(g2)
    gap0 = f(x0) - fstar

    convex = []
    for K in K_LIST:
        m = float(g2[:K].min())
        bnd = 2 * L * gap0 / K
        convex.append({"K": K, "min_grad_sq": m, "bound": bnd, "bound_over_observed": bnd / m})
    strong = []
    for k in k_LIST:
        strong.append({
            "k": k,
            "actual": d2[k],
            "predicted": (1 - MU / L) ** k * d2[0],
            "predicted_sharp_(L-mu)/(L+mu)": ((L - MU) / (L + MU)) ** k * d2[0],
        })
    res = {
        "protocol": "legacy np.random.seed(0); Q=qr(randn(20,20)); A=Q diag(linspace(1,50,20)) Q^T; "
                    "b=randn(20); x0=3*randn(20); eta=1/L; float64",
        "L": L, "mu": MU, "eta": ETA,
        "f0_minus_fstar": float(gap0), "dist0_sq": d2[0],
        "convex_check": convex,
        "strongly_convex_check": strong,
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    # compare with stored file
    with open(RESULTS / "gd_convergence_verification.json") as fh:
        old = json.load(fh)
    mx = 0.0
    for a, o in zip(convex, old["convex_check"]):
        mx = max(mx, abs(a["min_grad_sq"] / o["min_grad_sq"] - 1), abs(a["bound"] / o["bound"] - 1))
    for a, o in zip(strong, old["strongly_convex_check"]):
        mx = max(mx, abs(a["actual"] / o["actual"] - 1), abs(a["predicted"] / o["predicted"] - 1))
    print("max relative deviation from stored JSON:", mx)
    for c in convex:
        print("K=%4d  min|g|^2=%.6e  bound=%.6e  ratio=%.3e" % (c["K"], c["min_grad_sq"], c["bound"], c["bound_over_observed"]))
    for s in strong:
        print("k=%4d  |x-x*|^2=%.6e  bound=%.6e" % (s["k"], s["actual"], s["predicted"]))


if __name__ == "__main__":
    main()
