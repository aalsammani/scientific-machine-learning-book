# Chapter 9: Operator Learning

Two groups of experiments: a Fourier neural operator (FNO) for the periodic heat semigroup
(sections 9.3-9.5, `fno/`) and a DeepONet with classical and non-learned baselines for
two-dimensional Darcy flow (sections 9.6 and 9.8, `darcy/`). The Darcy models and data are reused in
Chapters 12, 14 and 15.

## Fourier neural operator (`fno/`)

Operator `u0 -> u(., 1)` for `u_t = 0.05 u_xx` on `[0, 2 pi)`, periodic; exact reference (each
Fourier mode decays by `exp(-nu k^2 T)`), so no reference discretization error enters. Inputs have
random Fourier coefficients on modes 1-8 with amplitude `1/m` for mode `m` (see `fno1d.py`); 400
training and 100 test inputs per seed; training on a 64-point grid; full-batch Adam, 2000
iterations, learning rate `1e-3`; float64.

| Book item | Script | Stored results (`results/chapter09/fno/`) | Cost | Randomness |
|---|---|---|---|---|
| §9.4, Table 9.1, Figure 9.1 (left): resolution transfer, 12 modes, width 16, five seeds, evaluation at 32-256 points | `fno_resolution_transfer.py 0 1 2 3 4` | `resolution_transfer.json` | MODERATE (about 1-3 min per seed) | seeded data; not reproducible run to run (below) |
| §9.3 aliasing sweep (8-64 points); §9.5 shifted inputs (modes 9-16) and a 4-mode model; Figure 9.1 (center, right) | `fno_aliasing_and_shift.py` | `failure_ood.json` | MODERATE (about 3 min) | as above |
| §9.5, Figure 9.2: matched-seed 4-mode versus 12-mode ablation, seeds 1-4 | `matched_seed_ablation.py 1 2 3 4` | `matched_seed_ablation.json` | MODERATE (about 2-4 min per seed) | as above |

**Reproducibility of the FNO numbers.** FNO training in this implementation is *not* bitwise
reproducible, even on one machine. Two causes are known:

- The pointwise (Conv1d) layers are initialized from PyTorch's global random stream, which the
  scripts never seed, so a model depends on how many models were created before it in the same
  process.
- In the reference environment of this repository, seven separate training runs of the identical
  seed-0 configuration, each in a fresh process, gave test errors between 0.27% and 0.73%. The run-to-run variation is therefore not explained by
  initialization order alone; a plausible, unverified source is floating-point nondeterminism in the
  FFT, amplified over 2000 optimizer steps.

The stored Table 9.1 values (0.23-0.34% over five seeds) are one realization. What does reproduce:

- in every run, the error is constant across evaluation resolutions to three significant figures;
- the seed-to-seed spread exceeds the resolution-to-resolution spread by more than two orders of
  magnitude;
- the failure-mode results are reproduced in magnitude. A rerun gave aliasing errors of 20.6%
  (8 points) and 5.8% (12 points), against 20.5% and 5.6% in the book; a shifted-input error of
  2,872% (book: 3,032%); and a 4-mode model error of 3.7% (book: 3.5%).

As stated in §9.5, the ablation models are fresh training runs, so their 12-mode errors differ from
the Table 9.1 entries for the same seed labels.

## DeepONet and baselines for Darcy flow (`darcy/`)

`-div(a grad u) = 1` on `(0,1)^2`, `u = 0` on the boundary,
`a = 1 + 0.08 sum_{k=1..5} c_k sin(k pi x) cos(k pi y)`, `c_k ~ U(-1, 1)`; 21 x 21 interior grid;
500 coefficient draws (`default_rng(0)`), the first 400 for training (the size-`n` training set is
the first `n`), the last 100 for testing.

| Book item | Script | Stored results (`results/chapter09/`) | Cost | Randomness |
|---|---|---|---|---|
| §9.6, Appendix D, Chapter 14 §14.4: solver verification (orders 2.00, 2.00, 1.95; about 0.2% from the PDE solution at 21 x 21; legacy boundary treatment first order, about 14%) | `darcy_manufactured_verification.py` | `darcy_verification/manufactured_solution_convergence.json` | FAST | deterministic |
| §9.6: dataset (500 solves in 0.46 s); baselines: training mean 1.3%, affine least squares 0.06-0.07%, quadratic least squares 0.002-0.006% | `darcy_dataset_and_baselines.py` | `darcy_dataset.npz`, `regeneration_prepare.json` | FAST | seeded (`default_rng(0)`) |
| §9.6, Figure 9.3: DeepONet data-cost sweep, 5 sizes x 5 seeds x 2 output protocols (raw: 39/7.3/1.9/1.8/1.8%; standardized: 0.34/0.27/0.20/0.20/0.18%) | `deeponet_sweep.py` (calls `deeponet.py`) | `sweep_firstedition_protocol/`, `sweep_normalized/` (per-run JSON; model states for `n = 200`), `deeponet_sweep_summary.json` | EXPENSIVE serially (about 1 h; about 10 min with 14 parallel workers) | seeded; deterministic on a fixed platform |
| §9.6, Chapter 15 §15.2: timings and break-even query counts (classical solve 0.94 ms; training 63 s; `N*` about `6.7e4` batched, none unbatched; quadratic surface about 200) | `break_even_timing.py` | `break_even_timing.json` | MODERATE (about 1.5 min) | timings machine specific |
| §9.8: distribution shift | `code/chapter14/distribution_shift.py` | `results/chapter14/deeponet_distribution_shift.json` | FAST | deterministic |

```bash
python code/chapter09/darcy/darcy_manufactured_verification.py
python code/chapter09/darcy/darcy_dataset_and_baselines.py
python code/chapter09/darcy/deeponet.py 200 0 --normalize --out my_run.json   # one training (about 1 min)
python code/chapter09/darcy/deeponet_sweep.py      # skips runs whose output file exists
python code/chapter09/darcy/break_even_timing.py   # run alone on an idle machine
python code/figures/make_fig9_3.py
```

**Reproducibility of the DeepONet numbers.** DeepONet training is deterministic for a fixed seed on
a fixed platform and software stack: retraining `n = 200`, seed 0, standardized outputs in the
reference environment reproduced the stored result bitwise. The stored `n = 200` model states are
included (about 0.1 MB each) because Chapters 12 and 14 evaluate them.

**Output protocols and names.** The raw-output runs are stored under `sweep_firstedition_protocol/`
and the summary key `firstedition_protocol`. The name refers to the original training protocol
(unscaled outputs), and is kept because stored files and downstream scripts use it. Figure 9.3 labels
the two protocols "unscaled" and "scaled outputs"; the text calls them raw and standardized.

**Legacy solver archive.** `results/chapter09/legacy_first_order_solver/` contains the dataset and
DeepONet results obtained with the earlier, first-order boundary treatment of the solver (the
defect analyzed in Chapter 14, §14.4). It is kept as part of the verification record:
`darcy_dataset_and_baselines.py` uses it to quantify the legacy data error (about 14%) and to check
that the archived model reproduces its originally reported error. No other number in the book comes
from this archive.

**Verification details.** The orders 2.00, 2.00 and 1.95 are the max-norm orders at the finest
refinement level (M1, M3, M2); the grid-L2 orders are also stored. The "about 0.2%" difference
between the 21 x 21 data and the PDE solution rests on three coefficient draws compared with nested
351 x 351 solutions.
