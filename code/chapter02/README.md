# Chapter 2: Differential Equations, Dynamical Systems, and Scientific Models

| Book item | Script | Stored results (`results/chapter02/`) | Cost | Randomness |
|---|---|---|---|---|
| §2.3 and Figure 2.1: separation of two Lorenz-63 trajectories started `1e-8` apart; fitted rate 0.89; separation checkpoints; eigenvalues at C+; lobe-switch time | `lorenz_sensitivity.py` | `lorenz_sensitivity.json` (plotted), `lorenz_sensitivity_rerun.json` (reproduction and the quantities quoted in §2.3) | FAST (about 10 s) | deterministic |

Configuration: `sigma = 10, rho = 28, beta = 8/3`; initial conditions `(1, 1, 1)` and
`(1 + 1e-8, 1, 1)`; SciPy `solve_ivp` with DOP853, `rtol = 1e-13`, `atol = 1e-15`; output every
0.1 on `[0, 30]`; growth rate = least-squares slope of `log(separation)` on `t in [14, 29]`.

```bash
python code/chapter02/lorenz_sensitivity.py
python code/figures/make_fig2_1.py
```

Provenance. `lorenz_sensitivity.json` is the stored run plotted in Figure 2.1; the script that
produced it was not preserved. `lorenz_sensitivity.py` reconstructs the protocol, reproduces the
fitted rate exactly, and writes the eigenvalues, lobe-switch times and separation checkpoints
quoted in §2.3 to `lorenz_sensitivity_rerun.json`. Its separation curve differs from the stored one
by at most about `2e-4` relative, so the absolute tolerance of the stored run is not confirmed; the
script records a solver scan. `results_lorenz_divergence.json` is a second stored run used only by
that scan; its numbers are not quoted in the book.

A trajectory of a chaotic system is a reference only up to a finite horizon; see §2.3 and
Appendix D (Example G).
