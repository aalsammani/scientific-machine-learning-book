# Known issues and limitations

This file records every discrepancy found between the book text, the code and the stored results
while this repository was prepared, together with the limits of what can be reproduced. The book and
the stored results were not changed to make them agree; each item gives the evidence so that a
reader can check it. Items are grouped by kind and ordered by chapter.

## 1. Statements in the book that differ from the code or the stored results

None of these changes a main conclusion of the book; most concern a detail of description or
rounding. The §7.7 item weakens the stated evidence for run-to-run irreproducibility of the
hard-constraint training (the quoted runs are not identical configurations, and one value is not
stored), although the four-seed spread of §6.9 (3.6-64.1%) remains.

| Location | Book text | Code and stored results |
|---|---|---|
| §1.9 | The linear sine-basis control reaches relative errors "below `10^-6` within 20 iterations". | At iteration 20 the stored error is `1.053e-6`; it is `1.1e-12` at iteration 40 and at round-off level by iteration 60 (`results/chapter01/ch01-f01_control.json`, key `linear_sine_basis_gd`). |
| §1.9 | Median error ratios 17 (physics-informed network) and 32 (regression). | Not stored in any result file; a recomputation from the stored per-seed medians gives about 17 and 31. |
| §1.9 | "Four controls ... with five seeds each." | The linear-model control is one deterministic run; the gradient-descent control covers the original and swapped amplitudes (not equal amplitudes). |
| §1.9 | The `k = 5` mode drops below 10% after the L-BFGS phase begins. | For the JAX regression network, seed 2, it never drops below 10% at `t = 0` (`ch01-f01_spectral_bias_summary.json`). |
| §3.7 | Radau and RK45 compared "at the same tolerances (relative `10^-6`)". | The relative tolerances agree; the absolute tolerances are `1e-12` (Radau) and `1e-10` (RK45) in `code/reference_problems/refs_B_to_G.py`. |
| §6.8 | The Burgers network has "four hidden layers of 48 units rather than three layers of 32". | The network is `[2, 48, 48, 48, 1]`, three hidden layers of 48 units (4897 parameters, as stored in `results/chapter06/ch07-e03_burgers_seed*.json`); checked by `tests/test_derivatives_and_models.py`. |
| §6.9 | A hard constraint with a non-differentiable initial condition "reproduces the time-blind failure of Chapter 1 (295% relative error)". | No stored result for this variant exists; 295% is the time-blind value of Table 1.5. |
| §6.9 | ReLU networks, "all else unchanged", with residual loss 0.10-0.18. | The ReLU runs use He initialization; 0.10-0.18 is the residual on the held-out evaluation grid divided by the norm of the exact solution (`rel_residual_l2`), not the training loss. |
| §6.10 | "133-210 seconds of training per seed". | 133 s is the median of the JAX runs and 210 s the median of the PyTorch runs of Chapter 1; the §6.10 accuracy figures are the PyTorch ones, whose per-seed range is 199-222 s (`results/chapter06/ch07-e01_pareto_data.json`). |
| §6.10 | Refining both implicit methods to 321 points gives `8.9e-4`. | P1 finite elements: `8.88e-4`; Crank-Nicolson finite differences: `8.60e-4` (`data/reference/exA/verification.json`). The stated factor of about 7 holds for both. |
| Table 6.1 | Classical methods at "`10^3`-`10^4`x lower cost". | With about 210 s for the network and 0.009-0.14 s for the classical methods the ratio is about `1.4e3` to `2.3e4`. |
| §7.2 | The bounds were evaluated "on trained hard- and soft-constraint networks from Chapter 6". | `code/chapter07/residual_bound_check.py` trains its own JAX networks (1500 Adam steps); its hard variant imposes the boundary conditions exactly and the initial condition as a penalty, unlike Chapter 6's ansatz. The bound values and their validity match the text. |
| §7.5 | The fixed-weight runs are "independent reruns of Chapter 6's configuration". | `loss_balance.py` allows 8000 L-BFGS iterations (Chapter 6: 10000) and uses its own collocation seed (`seed + 7000`). |
| §7.7 | "Repeating the identical training configuration ... 3.58%, 2.99%, and 12.6%." | 3.58% is `results/chapter06/ch07-e04_hard_seed1.json` (L-BFGS limit 10000); 12.6% is `results/chapter07/ch08-hardconstraint-diag_seed1.json` (limit 8000), so the two are not identical configurations; 2.99% is in no stored file. |
| §7.7 | Envelope correlations `-0.46, -0.41, -0.40, -0.50, -0.57`; time correlations `-0.45, -0.37, ...`. | Stored seed-0 envelope correlation `-0.4546` rounds to `-0.45`; stored seed-1 time correlation `-0.3647` rounds to `-0.36`. |
| §9.4 | Inputs "scaled by a `1/(k+1)` envelope". | In `fno1d.py` the factor is written `1/(k+1)` with a zero-based index `k`, so mode `m` is scaled by `1/m`; the shifted inputs of §9.5 use `1/(m+1)` with the true mode number. |
| §9.5 | "The trained model from §9.4 was evaluated on inputs ...". | `fno_aliasing_and_shift.py` retrains the seed-0 configuration (in-distribution error 0.27%, against 0.23% in Table 9.1). |
| §9.6, App. D | Observed orders 2.00, 2.00, 1.95 (norm not named). | These are max-norm orders; the grid-L2 orders are 2.00, 1.97, 2.01 (M1, M2, M3). The "about 0.2%" data error rests on three coefficient draws. |
| §10.3, Ex. B.2 | One observation of `I` leaves a one-parameter family of admissible parameters. | The seven family members stored in `sir_nonidentifiability_verification.json` are all flagged inadmissible; admissible members exist only in a narrow band of `gamma` below `1/7`, which the grid does not sample. |
| §10.4 | The perturbed trajectory continues "with norm between about 0.014 and 0.016". | The stored norm is 0.030 at `t = 1`, 0.018 at `t = 2`, and between 0.01553 and 0.01569 for `3 <= t <= 15` (`results/chapter10/closure_instability.json`); it never reaches 0.014. |
| §12.5 | Joint loss about `6.8e-3` at `omega_r = 10`. | The boundary loss is not stored; the stored terms give `omega_r L_r + L_data = 6.85e-3`, a lower bound. |
| App. D | Example B reference: DOP853 at `rtol = 1e-12`. | The Lotka-Volterra trajectory used to train and score the §10.2 UDE is computed with the model's fixed-step RK4 integrator, not with the DOP853 reference; its integration error is not reported. |
| App. D | Example G reference at `rtol = 1e-13`. | The §11.5 SINDy data use `rtol = 1e-12`, as Chapter 11 states. |

## 2. Results whose producing code was not preserved

| Stored result | Used in | Status |
|---|---|---|
| `results/chapter10/closure_instability.json` | §10.4, Figure 10.1 (bottom) | No producing script; the stored file does not record the form of the bump function, so the experiment cannot be rerun. The figure is regenerated from the stored file. |
| `results/chapter06/ch07-e01_pareto_data.json` | §6.10, Table 6.1 (network cost range) | Aggregate of the Chapter 1 PyTorch runs and the classical comparators; no producing script. Its internal note uses an obsolete figure label. The underlying per-run files are present. |
| Chapter 1 summaries: `ch01_final_summary_both_frameworks.json`, `ch01-e02_e03_t0_decomposition.json`, `ch01-f01_spectral_bias_summary.json`, `ch01-f01_spectral_bias_analysis.json`, `ch01-f01_spectral_modes_summary.json` | §1.8.5, §1.9 | Derived from the per-seed files by analysis steps that were not preserved. The figure scripts recompute the plotted medians from the per-seed files. |
| `results/chapter10/lv_ude_summary.json`, `results/chapter07/lossbalance_summary.json`, `results/chapter08/deepritz_summary.json` | §10.2, §7.5, §8.3 | Collections of the per-seed files; no separate producer needed. |
| Original runs behind `*_rerun.json` (Chapters 2-5), `ntk_spectrum.json` (Chapter 7), `poisson2d_fd_verification.json`, `fem_verification_timed.json` | Chapters 2-8 | The original scripts were not preserved; the scripts in this repository reconstruct the protocols. They reproduce the stored values exactly or to round-off, except for the implementation-dependent FTCS blow-up digits (§3.3) and the Lorenz separation curve, which agrees to about `2e-4` relative. |

## 3. Reproducibility limits of learned-method experiments

- **Bitwise reproducibility is not claimed** for neural-network training across machines or
  software versions. Results produced in the first recorded environment (Linux, one CPU; see
  [`COMPUTATIONAL_ENVIRONMENT.md`](COMPUTATIONAL_ENVIRONMENT.md)) are reproduced in the reference
  environment of this repository only statistically. Checked examples:
  - Deep Ritz with fixed collocation, seed 0: 212% against the stored 137%. Both runs show the same
    failure: the energy falls below the true minimum. (The resampled run, seed 0, agreed to
    round-off.)
  - Lotka-Volterra UDE, seed 0: trajectory error 0.88% against 0.99%.
  - `omega_r` sweep: the recovered `nu` agrees to four digits.
  - JAX residual-bound check: agrees to about `5e-5` relative, and both bounds hold.
- **FNO training (Chapter 9) is not reproducible run to run** even on one machine: seven separate
  runs of the seed-0 configuration gave test errors between 0.27% and 0.73%, against the stored
  0.23%. Resolution invariance, the seed-versus-resolution comparison and the failure-mode results
  reproduce; see [`code/chapter09/README.md`](../code/chapter09/README.md).
- **Hard-constraint physics-informed training** (§6.9, §7.7) is sensitive to the run; the book
  discusses this.
- **Deterministic on the reference machine:** the Darcy DeepONet (retraining `n = 200`, seed 0,
  standardized outputs) and the calibration ensemble reproduced their stored values bitwise. The
  classical and statistical computations reproduced bitwise or to the tolerances listed in
  [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (the NTK reproduction bitwise; the Crank-Nicolson
  verification to `1e-11`, the ODE reference trajectories to `6e-10`, the Kansa errors to `5e-5`),
  apart from the implementation-dependent FTCS blow-up digits and the Lorenz separation curve noted
  in section 2.
- **Timings** (Chapters 1, 4, 6, 9, 15) are specific to the machine on which they were measured.

## 4. Metadata inside stored result files

Stored files are kept byte for byte as produced, including metadata that no longer describes the
public layout:

- **Internal paths** such as `experiments/ch09new/results/darcy_dataset.npz`, and experiment
  identifiers such as `ch07-e03` that do not equal the final chapter numbers.
- **Platform strings** of the machines used.
- **A calibration protocol string.** `results/chapter14/calibration_check_rerun.json` stores a
  `protocol` string that states `T = 0.2` and names a development path. The code uses `T = 1`, and
  rerunning the current script reproduces every stored number exactly, so the stored numbers
  correspond to `T = 1`, as the book states.
- **Key names.** Some keys, such as `first_edition_realization` in the Chapter 12 results and the
  directory `sweep_firstedition_protocol`, name the original single-realization study or training
  protocol.

## 5. Relation between the published code and the stored results

The published scripts compute the same quantities, with the same algorithms, parameters and seeds,
as the runs that produced the stored results. Differences that a reader comparing the two may notice:

- **Output locations and names.** Default input and output locations are resolved relative to the
  repository. The metadata stored inside older result files still shows the layout in which they were
  produced (section 4). A few scripts carry descriptive names (for example
  `darcy_dataset_and_baselines.py`, `fno_resolution_transfer.py`, `residual_bound_check.py`) and
  write result files with the names listed in the chapter READMEs.
- **Example C self-check.** `refs_B_to_G.py exC` compares the 100- and 150-node Gauss-Hermite rules
  with the 200-node rule, which is the comparison recorded in the stored
  `data/reference/exC/verification.json`. NumPy's `hermgauss` overflows at 400 nodes.
- **Example D.** It is verified by `code/chapter09/darcy/darcy_manufactured_verification.py`, not by
  `refs_B_to_G.py`.
- **Two stored files are regenerated by options or scripts that postdate them:**
  - `fem_verification_timed.json` by `fem_1d.py --timed`;
  - `poisson2d_fd_verification.json` by `poisson2d_fd_convergence.py`, to `7e-12` relative.
- **Rerunning a seed.** `fno_resolution_transfer.py` and `matched_seed_ablation.py` replace an
  existing entry for that seed rather than appending a duplicate.
- **Argument checking.** Scripts that take arguments print a usage line when called with missing or
  unknown ones.
