# Chapter 8: Weak, Variational, and Structure-Preserving Learning

Problem: `-Laplace(u) = 2 pi^2 sin(pi x) sin(pi y)` on the unit square with zero boundary values;
exact solution `u = sin(pi x) sin(pi y)`, minimum Ritz energy `-pi^2/4`.

| Book item | Script and invocation | Stored results | Cost per run | Randomness |
|---|---|---|---|---|
| §8.3 and Figure 8.1: Deep Ritz with a fixed collocation set (errors 137%, 864%, 207%; energies below the true minimum) | `deepritz.py fixed SEED`, seeds 0-2 | `results/chapter08/deepritz_fixed_seed{0-2}.json` | FAST (about 35 s) | seeded |
| §8.3 and Figure 8.1: Deep Ritz with resampled collocation (1.4%, 1.7%, 0.47%) | `deepritz.py resample SEED`, seeds 0-2 | `results/chapter08/deepritz_resample_seed{0-2}.json` | FAST (about 15 s) | seeded |
| §8.9-§8.10: five-point finite differences, orders 2.00, errors `3.2e-3` ... `5.0e-5` on 15^2 ... 127^2 interior nodes | `poisson2d_fd_convergence.py` | `data/reference/exA/poisson2d_fd_verification.json` | FAST | deterministic |

`deepritz_summary.json` lists the six per-seed results together; it was assembled from the
per-seed files.

```bash
python code/chapter08/deepritz.py fixed 0
python code/chapter08/deepritz.py resample 0
python code/chapter08/poisson2d_fd_convergence.py
python code/figures/make_fig8_1.py
```

Configuration: trial function `x(1-x) y(1-y) N(x, y)` with `N` a 3 x 32 tanh network (LeCun
normal), `n_r = 2000` uniform points (`torch.Generator(seed + 100)`), Adam 3000 iterations
(learning rate `1e-3`) then L-BFGS at most 3000 iterations, float64, one thread.

Provenance. The script that produced `poisson2d_fd_verification.json` was not preserved;
`poisson2d_fd_convergence.py` was written for this repository and reproduces the stored errors to
round-off (largest relative difference `7e-12`). The file is stored with the other reference-solver
verifications in `data/reference/exA/`; the problem is the two-dimensional Poisson problem above,
not Example A.
