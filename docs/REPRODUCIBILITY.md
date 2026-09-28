# Reproducibility

## Two kinds of reproducibility

- **Bitwise reproducibility**: a rerun produces the same floating-point numbers as the stored
  result.
- **Scientific reproducibility**: a rerun produces results that support the same statements (the
  same orders of convergence, the same qualitative failure, errors of the same size and spread),
  even when the digits differ.

This repository aims at scientific reproducibility for everything and documents where bitwise
reproducibility was observed. It does not promise bitwise reproducibility of neural-network
training across machines, operating systems, library versions or thread counts.

## Environment

Python 3.14 with the pinned versions in [`requirements.txt`](../requirements.txt); the optional JAX
scripts additionally need [`requirements-jax.txt`](../requirements-jax.txt). Everything runs on a
CPU; no GPU is used or needed. See [`COMPUTATIONAL_ENVIRONMENT.md`](COMPUTATIONAL_ENVIRONMENT.md) for
the machines on which the stored results were produced.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt       # Windows: .venv\Scripts\python.exe
.venv/bin/python -m pip install -r requirements-jax.txt   # optional
.venv/bin/python -m pytest tests                          # about 10 seconds
```

## How the stored results were verified for this release

In a fresh virtual environment built from `requirements.txt` and `requirements-jax.txt` (Windows 11,
Python 3.14.3, CPU only), from a working directory outside the repository:

1. **Figures.** All 23 figure scripts were run on the stored results. Every regenerated PNG was
   identical, pixel for pixel, to the figure used in the book.
2. **Classical and statistical computations.** Every script of Chapters 2-5, the reference-problem
   scripts, the finite-element, RBF and five-point comparators, the Darcy verification and
   baselines, the SIR/SEIR verification, the SINDy sweep, the Darcy inversion, the Ornstein-Uhlenbeck
   and Kalman studies and the distribution-shift study were rerun, and every numeric entry of each
   output was compared with the stored file.
   - Almost all outputs were identical bitwise.
   - The ODE reference trajectories of Examples B, E, F and G, which were originally computed on
     Linux with an earlier SciPy, agree to at most `6e-10` relative. The tolerance-study diagnostics
     at round-off level (invariant drift at `rtol = 1e-12`) differ in their leading digits, as
     expected for round-off-level quantities.
   - The Kansa RBF errors agree to `5e-5` relative (an ill-conditioned least-squares solve).
   - The Crank-Nicolson verification agrees to `1e-11` relative.
3. **Learned methods.** One run of each of the following was repeated:
   - the Darcy DeepONet (`n = 200`, seed 0) and the ten-member calibration ensemble: bitwise
     identical, having been produced on the same machine;
   - the resampled Deep Ritz run: identical to round-off;
   - Deep Ritz with fixed collocation, the Lotka-Volterra UDE, the `omega_r` sweep, the JAX
     residual-bound check and the FNO experiments: statistically consistent, with differences
     documented in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), section 3.
4. **Stored numbers against the book.** `tests/test_stored_results_match_book.py` checks that the
   stored files contain the values printed in the book, at the printed precision.

The expensive DeepONet sweep (50 trainings) and the full five-seed physics-informed runs of
Chapters 1 and 6 were not rerun in full for this release.

## Seeds and determinism

- Every stochastic experiment sets its seeds explicitly (the one exception is the initialization of
  the FNO pointwise layers, described below): NumPy `default_rng(seed)` (or the legacy
  `np.random.seed` where noted), `torch.Generator().manual_seed(...)` for network initialization and
  collocation points, and `jax.random.PRNGKey(seed)` in the JAX scripts. The offsets used to
  decorrelate experiments (for example `seed + 1000` for collocation points) are in each script.
- All learned-method scripts use float64. PyTorch scripts set `torch.set_num_threads(1)` because
  multithreaded reductions change the order of floating-point additions.
- Two known sources of run-to-run variation remain:
  - the FNO scripts initialize their pointwise layers from PyTorch's unseeded global stream;
  - FNO training varies between runs even with an identical process history (see
    `code/chapter09/README.md`).
- The hard-constraint physics-informed runs (§6.9, §7.7) are sensitive enough that reruns of a seed
  can give very different errors; the book discusses this.

## Seed policy of the book

Learned-method results are reported over five seeds (median and range) unless the text states
otherwise; smaller numbers are used for diagnostics whose effect is large compared with the
seed-to-seed spread:

| Experiment | Seeds |
|---|---|
| Lotka-Volterra UDE (§10.2) | 2 |
| Loss balancing (§7.5), Deep Ritz (§8.3), Burgers at `nu = 0.001` | 3 |
| Hard-constraint ablation (§6.9) | 4 |
| FNO mode ablation (§9.5) | 4 |
| ReLU ablation | 2 |
| Underweighted boundary loss | 1 |

Each script runs one seed per call, except where a seed list is accepted. The medians and ranges
in the book are over the stored per-seed files listed in [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md).
Aggregation is done by the figure scripts, the sweep script `deeponet_sweep.py`, or (for Chapter 1)
by the stored summary files.

## Reference solutions

A numerical solution is used as a reference only when it is verified and much more accurate than
the error measured against it (Appendix C of the book):

- **Example A:** an exact finite Fourier series.
- **Example C:** a Cole-Hopf solution evaluated by Gauss-Hermite quadrature, cross-checked against
  an independent method-of-lines solution that converges to it at second order (difference `8e-5`
  at the finest grid).
- **Example D (Darcy):** verified by three manufactured solutions (orders 2.00, 2.00, 1.95) and a
  nested-grid comparison (about 0.2% at the 21 x 21 data resolution). This is a verified reference
  for the discrete data, not the exact PDE solution.
- **FNO heat semigroup:** exact.
- **ODE examples (B, E, F, G):** tolerance studies (changes between successive tolerances) and, for
  Robertson, agreement of two independent implicit methods. For Lorenz-63 a trajectory is a
  reference only up to a finite horizon.

## Regenerating figures and tables

Figures: `python code/figures/make_figN_M.py` (one script per figure; stored results only).
Tables 1.3-1.5, 6.1 and 9.1 are typeset in the book from the stored files named in
`EXPERIMENT_INDEX.md`; no table is generated automatically.

## Computational cost

| Scope | Time on one CPU core |
|---|---|
| All figures | about 1.5 minutes (Figure 1.6 alone about 1 minute) |
| All FAST scripts | a few minutes |
| One physics-informed or DeepONet training | 0.5-4 minutes |
| Each chapter's full set of learned-method runs | 10-60 minutes |
| Full DeepONet sweep | about 1 hour serially |

Nothing requires a GPU.
