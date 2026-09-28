"""Classical RK4 order verification (Chapter 3, section 3.6; Figure 3.1, left panel).

  * y' = -2 t y, y(0) = 1, exact solution exp(-t^2); fixed-step RK4 from t = 0 to t = 1.
  * Step sizes h = 0.2, 0.1, 0.05, 0.025, 0.0125; error |y_N - exp(-1)|.
  * Observed order between successive rows = log2(err(h) / err(h/2)).
The default variant (t accumulated as t + h) reproduces the stored original
(rk4_order_verification.json) bitwise.

Usage:
    python code/chapter03/rk4_order_verification.py
Output: results/chapter03/rk4_order_verification_rerun.json
Runtime: under a second. Deterministic.
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "chapter03"
STORED = RESULTS / "rk4_order_verification.json"
OUT = RESULTS / "rk4_order_verification_rerun.json"


def f(t, y):
    return -2.0 * t * y


def rk4(h, T=1.0, accumulate_t=True):
    """accumulate_t=True: t <- t + h each step (reproduces the stored file bitwise);
    False: t = (i+1) h (differs from the stored file only at round-off level)."""
    n = int(round(T / h))
    t, y = 0.0, 1.0
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h if accumulate_t else (i + 1) * h
    return y


def table(accumulate_t):
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    exact = np.exp(-1.0)
    rows = [{"h": h, "err": float(abs(rk4(h, accumulate_t=accumulate_t) - exact))} for h in hs]
    orders = [float(np.log2(rows[i]["err"] / rows[i + 1]["err"])) for i in range(len(rows) - 1)]
    return rows, orders


def main():
    rows, orders = table(True)
    rows_b, orders_b = table(False)
    with open(STORED) as fh:
        st = json.load(fh)
    rel = [abs(r["err"] - s["err"]) / s["err"] for r, s in zip(rows, st["rows"])]
    rel_b = [abs(r["err"] - s["err"]) / s["err"] for r, s in zip(rows_b, st["rows"])]
    out = {"problem": "y'=-2ty, y(0)=1, exact=exp(-t^2), error at t=1", "rows": rows,
           "observed_order": orders, "expected_order": 4.0,
           "stored_observed_order": st["observed_order"],
           "max_rel_diff_err_vs_stored": max(rel),
           "variant_t_equals_(i+1)h": {"rows": rows_b, "observed_order": orders_b,
                                       "max_rel_diff_err_vs_stored": max(rel_b)}}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
