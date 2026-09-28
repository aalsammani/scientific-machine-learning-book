# Chapter 3: Numerical Foundations and Scientific Computing

| Book item | Script | Stored results | Cost | Randomness |
|---|---|---|---|---|
| §3.3: FTCS stability threshold (`r = 0.4, 0.5, 0.6`; blow-up of order `1e176` after 1333 steps; `max|g| = 1.40`) | `ftcs_stability_demo.py` | `results/chapter03/ftcs_stability_demo.json`, `ftcs_stability_demo_rerun.json` | FAST | deterministic |
| §3.5: aliasing of `sin(12x)` on 16 points to wavenumber 4 | `aliasing_demo.py` | `results/chapter03/aliasing_demo.json`, `aliasing_demo_rerun.json` | FAST | deterministic |
| §3.6 and Figure 3.1 (left): RK4 orders 3.90, 3.99, 4.00, 4.00 | `rk4_order_verification.py` | `results/chapter03/rk4_order_verification.json`, `_rerun.json` | FAST | deterministic |
| §3.7: Robertson kinetics, Radau vs BDF (`1.6e-12`), RK45 function evaluations | `../reference_problems/refs_B_to_G.py exE` | `data/reference/exE/verification.json`, `reference.npz` | FAST (about 10 s) | deterministic |
| §3.8: Crank-Nicolson manufactured-solution orders 1.96-1.99 | `../reference_problems/exA_heat.py` | `data/reference/exA/verification.json` (`cn_mms`) | FAST | deterministic |
| §3.8: P1 finite-element orders | `../chapter06/fem_1d.py` | `data/reference/exA/fem_verification.json` | FAST | deterministic |
| §3.9 and Figure 3.1 (right): Monte Carlo `n Var` = 0.130, 0.123, 0.138, 0.126, 0.122 | `monte_carlo_rate_verification.py` | `results/chapter03/monte_carlo_rate_verification.json`, `_rerun.json` | FAST (about 30 s) | seeded (`default_rng(0)`) |

```bash
python code/chapter03/ftcs_stability_demo.py
python code/chapter03/aliasing_demo.py
python code/chapter03/rk4_order_verification.py
python code/chapter03/monte_carlo_rate_verification.py
python code/reference_problems/refs_B_to_G.py exE
python code/figures/make_fig3_1.py
```

Provenance. The files without the `_rerun` suffix are the stored original results (the RK4 and
Monte Carlo files are plotted in Figure 3.1). The scripts that produced them were not preserved;
the scripts here reconstruct the protocols documented in their docstrings and write `_rerun`
files. The RK4, aliasing and Monte Carlo reruns reproduce the stored values exactly. The FTCS
blow-up at `r = 0.6` is seeded only by round-off, so its leading digits depend on floating-point
details: the rerun gives `2.88e176` after 1333 steps on the 41-point grid, and no tested variant
reproduces the stored `3.25e176` digits (the book states the order of magnitude and the
implementation dependence).

§3.7 compares Radau and RK45 at the same relative tolerance `1e-6`; the absolute tolerances differ
(`atol = 1e-12` for Radau, `1e-10` for RK45; see `refs_B_to_G.py`).
