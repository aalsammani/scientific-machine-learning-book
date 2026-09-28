# Chapter 1: What Is Scientific Machine Learning?

Computations of sections 1.8 (the heat equation three ways) and 1.9 (frequency-dependent
learning). All use Running Example A, `u_t = 0.5 u_xx` on `(0,1) x (0,1]` with zero Dirichlet data
and `u(x,0) = sin(pi x) + 0.5 sin(5 pi x)`. The exact solution is available in closed form
([`code/reference_problems/exA_heat.py`](../reference_problems/exA_heat.py)), so every error below
is measured against the exact solution.

| Book item | Script | Stored results (`results/chapter01/`) | Cost | Randomness |
|---|---|---|---|---|
| §1.8.2 reference verification; Table 1.4 FD orders | `../reference_problems/exA_heat.py` | `data/reference/exA/verification.json` | FAST | deterministic |
| Table 1.3 (FD row, JAX rows), Table 1.5, §1.8.5 | `exA_three_ways.py` | `ch01-e01_fd.json`, `ch01-f02_timeblind.json`, `ch01-e02_regression_seed{0-4}.json`, `ch01-e03_pinn_seed{0-4}.json` (+ `_history.json`), `environment.json` | MODERATE (about 14 min for 5 seeds, both methods) | seeded |
| Table 1.3 (PyTorch rows), Table 1.4, §1.8.5 | `exA_three_ways_pytorch.py` | `ch01-e0{2_regression,3_pinn}_seed{0-4}_pytorch.json`, `environment_pytorch.json` | MODERATE (about 22 min) | seeded |
| Figure 1.4 data | `residual_error_fields.py` | `fig15_fields.npz` | MODERATE (one training, about 3.5 min) | seeded |
| §1.9 controls (gradient descent, linear model) | `spectral_bias_control.py` | `ch01-f01_control.json` | MODERATE | seeded |
| §1.9 control (Adam, three amplitude configurations) | `exA_spectral_control.py` | `ch01-f01_spectral_control.json` | MODERATE | seeded |
| §1.8.5 and §1.9 derived summaries | none preserved (see below) | `ch01_final_summary_both_frameworks.json`, `ch01-e02_e03_t0_decomposition.json`, `ch01-f01_spectral_bias_summary.json`, `ch01-f01_spectral_bias_analysis.json`, `ch01-f01_spectral_modes_summary.json` | | |

Figures: `code/figures/make_fig1_1.py` (schematic), `make_fig1_3_comparison.py`, `make_fig1_4.py`,
`make_fig1_5.py`, `make_fig1_6.py` (chapter dependency graph). Figure 1.2 is a hand-drawn
conceptual diagram with no generating script.

## Configuration

Both implementations use the same specification: network `[2, 32, 32, 32, 1]`, tanh, LeCun-normal
initialization, 2241 parameters, float64; `n_r = 2000` residual, `n_b = 100` points per boundary,
`n_0 = 100` initial points; unit loss weights; Adam (5000 iterations, learning rate `1e-3`) followed
by L-BFGS (at most 10000 iterations); seeds 0-4. The regression network is trained on 2000 points
of the verified FTCS solution (81 points, `r = 0.4`) at `t > 0`. The constants are defined at the
top of each script.

The JAX and PyTorch versions draw their random numbers from different generators, so the two
frameworks give different (statistically comparable) results for the same seed; the book reports
both.

## Reproducing

```bash
python code/chapter01/exA_three_ways.py              # needs requirements-jax.txt
python code/chapter01/exA_three_ways_pytorch.py
python code/chapter01/residual_error_fields.py
python code/chapter01/spectral_bias_control.py       # needs requirements-jax.txt
python code/chapter01/exA_spectral_control.py        # needs requirements-jax.txt
python code/figures/make_fig1_3_comparison.py        # figures from stored results only
```

Rerunning overwrites the stored results in `results/chapter01/`; use `git diff` to compare and
`git checkout -- results/chapter01` to restore.

## Notes

- **Derived summaries.** The five summary files listed above were produced by analysis steps that
  were not preserved as scripts. Every per-seed quantity they aggregate is in the per-seed files,
  which the scripts above produce; `make_fig1_3_comparison.py` and `make_fig1_5.py` recompute the
  plotted medians from the per-seed files and only cross-check against the summaries.
- **`lbfgs_iters` in the PyTorch result files** counts loss evaluations (closure calls), not L-BFGS
  iterations.
- **Timings** (Table 1.3) were measured on the machine described in
  [`docs/COMPUTATIONAL_ENVIRONMENT.md`](../../docs/COMPUTATIONAL_ENVIRONMENT.md) and are not portable.
- **§1.9 wording.** The stored linear-model control (`ch01-f01_control.json`, key
  `linear_sine_basis_gd`) reaches a relative error of `1.05e-6` at iteration 20 and round-off level
  by iteration 60; the ratios "17" and "32" quoted in §1.9 are not stored in any result file (a
  recomputation from the stored medians gives about 17 and 31). See
  [`docs/KNOWN_ISSUES.md`](../../docs/KNOWN_ISSUES.md).
