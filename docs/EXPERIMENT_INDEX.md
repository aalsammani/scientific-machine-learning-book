# Experiment index

From a section, table or figure of the book to the code, configuration and stored results in this
repository. Paths are relative to the repository root. "Configuration" names where the parameters are defined: every experiment is configured
by the constants at the top of its script (documented in the script's docstring) plus the
command-line arguments shown; there are no separate configuration files.

Cost classes (one CPU core): **F** fast (under 1 minute), **M** moderate (1-30 minutes),
**E** expensive (over 30 minutes). Randomness: **D** deterministic, **S** stochastic with fixed seeds.
Details and caveats are in the chapter READMEs (`code/chapterNN/README.md`) and in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Part I: Foundations

| Ch. | Section | Experiment | Main script | Configuration | Stored results | Figure / Table | Notes |
|---|---|---|---|---|---|---|---|
| 1 | §1.8.2 | Verification of the Example A reference (exact solution; FTCS and Crank-Nicolson orders; manufactured solution) | `code/reference_problems/exA_heat.py` | script constants | `data/reference/exA/verification.json` | Table 1.4 | F, D |
| 1 | §1.8.3-§1.8.5 | Heat equation three ways: FD, regression network, physics-informed network; five seeds; JAX | `code/chapter01/exA_three_ways.py` | constants; args `[pinn\|regression] [seeds]` | `results/chapter01/ch01-e01_fd.json`, `ch01-e02_regression_seed{0-4}.json`, `ch01-e03_pinn_seed{0-4}.json` (+ `_history.json`) | Table 1.3, Figure 1.3 | M, S; needs JAX |
| 1 | §1.8.3-§1.8.5 | Same, PyTorch | `code/chapter01/exA_three_ways_pytorch.py` | constants; same args | `results/chapter01/*_pytorch.json` | Tables 1.3-1.4, Figure 1.3 | M, S |
| 1 | §1.8.4 | Time-blind baselines | `code/chapter01/exA_three_ways.py` | constants | `results/chapter01/ch01-f02_timeblind.json` | Table 1.5 | F, D; needs JAX (written by the JAX script) |
| 1 | §1.8.5 | Seed variability; errors with and without `t = 0` | per-seed files above | | `results/chapter01/ch01_final_summary_both_frameworks.json`, `ch01-e02_e03_t0_decomposition.json` | | derived summaries; producing analysis not preserved |
| 1 | §1.8.5 | Squared residual and error fields, PyTorch seed 0 | `code/chapter01/residual_error_fields.py` | constants | `results/chapter01/fig15_fields.npz` | Figure 1.4 | M, S |
| 1 | §1.9 | Mode learning curves | JAX histories above | | `results/chapter01/ch01-e03_pinn_seed{0-4}_history.json`, `ch01-f01_spectral_*.json` | Figure 1.5 | summaries derived, see chapter README |
| 1 | §1.9 | Controls: gradient descent and linear sine-basis model | `code/chapter01/spectral_bias_control.py` | constants | `results/chapter01/ch01-f01_control.json` | | M, S; needs JAX |
| 1 | §1.9 | Control: Adam with swapped and equal amplitudes | `code/chapter01/exA_spectral_control.py` | constants | `results/chapter01/ch01-f01_spectral_control.json` | | M, S; needs JAX |
| 1 | §1.2, §1.10 | Schematics | `code/figures/make_fig1_1.py`, `make_fig1_6.py` | constants | | Figures 1.1, 1.6 | Figure 1.2 has no script |
| 2 | §2.3 | Lorenz-63 separation of nearby trajectories; rate 0.89 | `code/chapter02/lorenz_sensitivity.py` | constants | `results/chapter02/lorenz_sensitivity.json` (plotted), `_rerun.json` | Figure 2.1 | F, D; reconstruction of an unpreserved script |
| 3 | §3.3 | FTCS stability threshold | `code/chapter03/ftcs_stability_demo.py` | constants | `results/chapter03/ftcs_stability_demo.json`, `_rerun.json` | | F, D |
| 3 | §3.5 | Aliasing of `sin(12x)` on 16 points | `code/chapter03/aliasing_demo.py` | constants | `results/chapter03/aliasing_demo.json`, `_rerun.json` | | F, D |
| 3 | §3.6 | RK4 order verification | `code/chapter03/rk4_order_verification.py` | constants | `results/chapter03/rk4_order_verification.json`, `_rerun.json` | Figure 3.1 (left) | F, D |
| 3 | §3.7 | Robertson kinetics: Radau vs BDF, RK45 cost | `code/reference_problems/refs_B_to_G.py exE` | constants | `data/reference/exE/` | | F, D |
| 3 | §3.8 | Crank-Nicolson manufactured solution; P1 finite elements | `code/reference_problems/exA_heat.py`; `code/chapter06/fem_1d.py` | constants | `data/reference/exA/verification.json`, `fem_verification.json` | | F, D |
| 3 | §3.9 | Monte Carlo `O(1/n)` variance | `code/chapter03/monte_carlo_rate_verification.py` | constants | `results/chapter03/monte_carlo_rate_verification.json`, `_rerun.json` | Figure 3.1 (right) | F, S |
| 4 | §4.5 | Gradient descent vs Theorems 4.4-4.5 | `code/chapter04/gd_convergence_verification.py` | constants | `results/chapter04/gd_convergence_verification.json`, `_rerun.json` | Figure 4.1 | F, S (fixed seed) |
| 4 | §4.6 | Reverse-mode AD cost ratio | `code/chapter04/ad_cost_scaling.py` | constants | `results/chapter04/ad_cost_scaling.json`, `_rerun.json` | | M; timings not portable |
| 5 | §5.3 | Tikhonov regularization, discrepancy principle | `code/chapter05/tikhonov_discrepancy_verification.py` | constants | `results/chapter05/tikhonov_discrepancy_verification.json`, `_rerun.json` | Figure 5.1 (left) | F, S (fixed seed) |
| 5 | §5.4 | Two-compartment structural non-identifiability | `code/chapter05/two_compartment_nonidentifiability.py` | constants | `results/chapter05/two_compartment_nonidentifiability.json`, `_rerun.json` | Figure 5.1 (right) | F, D |
| 5 | §5.7 | Continuous adjoint vs finite differences | `code/chapter05/adjoint_vs_fd_verification.py` | constants | `results/chapter05/adjoint_vs_fd_verification.json`, `_rerun.json` | | F, D |

## Part II: Learning Solutions and Operators

| Ch. | Section | Experiment | Main script | Configuration | Stored results | Figure / Table | Notes |
|---|---|---|---|---|---|---|---|
| 6 | §6.4 | Residual as two AD passes (schematic) | `code/figures/make_fig6_1.py` | | | Figure 6.1 | |
| 6 | §6.7 | Forward problem, Example A | `code/chapter01/exA_three_ways_pytorch.py pinn` | constants | `results/chapter01/ch01-e03_pinn_seed{0-4}_pytorch.json` | | M, S |
| 6 | §6.7 | Inverse problem: recover `nu` (noise 1%, 5%) | `code/chapter06/inverse_pinn.py` | constants; env `SIGMA_REL`; seeds | `results/chapter06/ch07-e02_inverse_seed{0-4}_sigma{0.01,0.05}.json` | | M, S |
| 6 | §6.8 | Viscous Burgers, `nu = 0.01/pi` | `code/chapter06/burgers_pinn.py e03 SEED` | constants; args | `results/chapter06/ch07-e03_burgers_seed{0-4}.json` | | M, S; architecture note in KNOWN_ISSUES |
| 6 | §6.8 | Burgers reference: Cole-Hopf vs method of lines | `code/reference_problems/refs_B_to_G.py exC` | constants | `data/reference/exC/` | | M, D |
| 6 | §6.9 | Sparse collocation (`n_r = 200`) | `code/chapter06/burgers_pinn.py f02 SEED` | args | `results/chapter06/ch07-f02_burgers_seed{0-4}.json` | | F-M, S |
| 6 | §6.8-§6.9 | Burgers at `nu = 0.001` | `code/chapter06/burgers_pinn.py f03 SEED` | args | `results/chapter06/ch07-f03_burgers_seed{0-2}.json` | | M, S |
| 6 | §6.9 | Underweighted boundary loss; soft baseline | `code/chapter06/constraint_ablation.py {underweighted,soft} 0` | args | `results/chapter06/ch07-f01_underweighted_seed0.json`, `ch07-e04_soft_seed0.json` | | M, S |
| 6 | §6.9 | Hard constraint | `code/chapter06/constraint_ablation.py hard SEED` | args | `results/chapter06/ch07-e04_hard_seed{0-3}.json` | | M, S; not reproducible run to run (§7.7) |
| 6 | §6.9 | ReLU activation | `code/chapter06/constraint_ablation.py relu SEED` | args | `results/chapter06/ch07-f04_relu_seed{0,1}.json` | | F, S |
| 6 | §6.10 | Classical comparators: FTCS, Crank-Nicolson, P1 FE, Kansa RBF; cost-accuracy comparison | `code/reference_problems/exA_heat.py`, `code/chapter06/fem_1d.py [--timed]`, `code/chapter06/kansa_rbf.py` | constants | `data/reference/exA/verification.json`, `fem_verification*.json`, `results/chapter06/kansa_convergence.json`, `results/chapter06/ch07-e01_pareto_data.json` | Table 6.1 | F, D; the PINN cost aggregate has no preserved producer |
| 7 | §7.2 | Numerical check of Theorem 7.1's bounds | `code/chapter07/residual_bound_check.py` | constants | `results/chapter07/ch08-t81_check.json` | | M, S; needs JAX |
| 7 | §7.3 | Burgers at `nu = 0.001` | see Chapter 6 | | `results/chapter06/ch07-f03_burgers_seed{0-2}.json` | | |
| 7 | §7.4 | Residual NTK spectrum at initialization; ten-seed sweep | `code/chapter07/ntk_spectrum.py` | constants | `results/chapter07/ntk_spectrum.json` (plotted), `ntk_spectrum_rerun.json` | Figure 7.1 (left) | F, D |
| 7 | §7.5 | Gradient-norm loss balancing | `code/chapter07/loss_balance.py {fixed,gradnorm} SEED` | args | `results/chapter07/lossbalance_{fixed,gradnorm}_seed{0-2}.json` | Figure 7.1 (right) | M, S |
| 7 | §7.7 | Hard-constraint diagnostics | `code/chapter07/hard_constraint_diagnostics.py` | constants; seeds | `results/chapter07/ch08-hardconstraint-diag_seed{0-4}.json` | | M, S; see KNOWN_ISSUES |
| 8 | §8.3 | Deep Ritz, fixed vs resampled collocation | `code/chapter08/deepritz.py {fixed,resample} SEED` | args | `results/chapter08/deepritz_{fixed,resample}_seed{0-2}.json` | Figure 8.1 | F, S |
| 8 | §8.9-§8.10 | Five-point finite differences for the same Poisson problem | `code/chapter08/poisson2d_fd_convergence.py` | constants | `data/reference/exA/poisson2d_fd_verification.json` | | F, D; script written to regenerate the stored file |
| 9 | §9.3 | Aliasing at coarse resolution | `code/chapter09/fno/fno_aliasing_and_shift.py` | constants | `results/chapter09/fno/failure_ood.json` | Figure 9.1 (center) | M, S; not bitwise reproducible |
| 9 | §9.4 | FNO resolution transfer, five seeds | `code/chapter09/fno/fno_resolution_transfer.py 0 1 2 3 4` | constants; seeds | `results/chapter09/fno/resolution_transfer.json` | Table 9.1, Figure 9.1 (left) | M, S; not bitwise reproducible |
| 9 | §9.5 | Distribution shift and mode truncation | `code/chapter09/fno/fno_aliasing_and_shift.py` | constants | `results/chapter09/fno/failure_ood.json` | Figure 9.1 (right) | M, S |
| 9 | §9.5 | Matched-seed mode ablation, seeds 1-4 | `code/chapter09/fno/matched_seed_ablation.py 1 2 3 4` | seeds | `results/chapter09/fno/matched_seed_ablation.json` | Figure 9.2 | M, S |
| 9 | §9.6 | Darcy solver verification (orders 2.00/2.00/1.95; 0.2%) | `code/chapter09/darcy/darcy_manufactured_verification.py` | constants | `results/chapter09/darcy_verification/manufactured_solution_convergence.json` | | F, D |
| 9 | §9.6 | Dataset; mean, affine and quadratic baselines | `code/chapter09/darcy/darcy_dataset_and_baselines.py` | constants | `results/chapter09/darcy_dataset.npz`, `regeneration_prepare.json` | Figure 9.3 | F, D |
| 9 | §9.6 | DeepONet sweep: 5 sizes x 5 seeds x 2 protocols | `code/chapter09/darcy/deeponet_sweep.py` (runs `deeponet.py`) | constants in `deeponet.py` | `results/chapter09/sweep_normalized/`, `sweep_firstedition_protocol/`, `deeponet_sweep_summary.json` | Figure 9.3 | E serially, S (deterministic per seed on a fixed platform) |
| 9 | §9.6 | Timing and break-even query count | `code/chapter09/darcy/break_even_timing.py` | constants | `results/chapter09/break_even_timing.json` | | M; timings not portable |
| 9 | §9.8 | DeepONet under distribution shift | `code/chapter14/distribution_shift.py` | constants | `results/chapter14/deeponet_distribution_shift.json` | | F, D |

## Part III: Learning Dynamics and Hidden Physics

| Ch. | Section | Experiment | Main script | Configuration | Stored results | Figure / Table | Notes |
|---|---|---|---|---|---|---|---|
| 10 | §10.2 | Universal differential equation, Lotka-Volterra, two seeds | `code/chapter10/lv_ude.py SEED` | constants; seed | `results/chapter10/lv_ude_seed{0,1}.json`, `lv_ude_traj_seed0.npz` | | M, S |
| 10 | §10.3 | SIR non-identifiability (Theorem 10.1) | `code/chapter10/sir_seir_nonidentifiability.py` | constants | `results/chapter10/sir_nonidentifiability_verification.json`, `sir_trajectories.npz` | Figure 10.1 (top) | F, D |
| 10 | §10.4 | Closure with localized perturbation: offline-online gap | not preserved | partial, in chapter README | `results/chapter10/closure_instability.json` | Figure 10.1 (bottom) | cannot be rerun; see KNOWN_ISSUES |
| 11 | §11.4-§11.5 | SINDy on Lorenz-63, noise sweep, 50 realizations; library coherence | `code/chapter11/sindy_noise_sweep.py` | constants | `results/chapter11/sindy_noise_sweep.json` | Figure 11.1 | F, S |
| 12 | §12.4 | Fisher information, 15 and 441 nodes; 50 placements | `code/chapter12/darcy_inversion.py` | constants; `--surrogate` | `results/chapter12/darcy_inversion_dense.json` | Figure 12.2 (left) | F, S (fixed seeds) |
| 12 | §12.5 | Flexible-network trap, `omega_r` sweep | `code/chapter12/omega_r_sweep.py` | constants | `results/chapter12/omega_r_sweep.json` | Figure 12.1 | M, S |
| 12 | §12.6 | Surrogate-error contamination, 200 noise realizations | `code/chapter12/darcy_inversion.py` | constants; `--surrogate` | `results/chapter12/darcy_inversion_dense.json` | Figure 12.2 (right) | F, S (fixed seeds) |

## Part IV: Reliability, Applications, and Frontiers

| Ch. | Section | Experiment | Main script | Configuration | Stored results | Figure / Table | Notes |
|---|---|---|---|---|---|---|---|
| 13 | §13.4 | Ornstein-Uhlenbeck drift and diffusion vs sampling interval (Theorem 13.2) | `code/chapter13/ou_drift_diffusion.py` | constants | `results/chapter13/ou_drift_diffusion.json` | Figure 13.1 (left) | F, S |
| 13 | §13.4 | The same estimators on Euler-Maruyama data at the sampling step | `code/chapter13/ou_drift_diffusion.py --legacy-em` | constants | `results/chapter13/ou_drift_diffusion_legacy_em.json` | | F, S |
| 13 | §13.5 | Kalman filter, damped oscillator, 50 realizations | `code/chapter13/kalman_filter_demo.py` | constants | `results/chapter13/kalman_filter_study.json` | Figure 13.1 (right) | F, S |
| 14 | §14.3 | Calibration of a ten-member ensemble | `code/chapter14/calibration_check.py` | constants | `results/chapter14/calibration_check_rerun.json` | Figure 14.1 | M, S |
| 14 | §14.4 | Four instances of flawed comparison | Chapter 9 Darcy verification and baselines; Chapter 6 comparisons | | `results/chapter09/...`, `data/reference/exA/...` | | reuse |
| 14 | §14.5 | Distribution shift for the Darcy DeepONet | `code/chapter14/distribution_shift.py` | constants | `results/chapter14/deeponet_distribution_shift.json` | Figure 14.2 | F, D |
| 15 | §15.1 | SEIR non-identifiability (Theorem 15.1) | `code/chapter10/sir_seir_nonidentifiability.py` | constants | `results/chapter15/seir_nonidentifiability_verification.json`, `seir_trajectories.npz` | Figure 15.1 | F, D |
| 15 | §15.2 | Subsurface-flow case study (break-even, contamination, Fisher information) | Chapter 9 and 12 scripts | | `results/chapter09/break_even_timing.json`, `results/chapter12/darcy_inversion_dense.json` | | reuse |
| 16 | all | No computational experiments; §16.1 summarizes results of Chapters 7, 9, 12 and 14 | | | | | |

Appendix D's reference problems are listed in [`../data/README.md`](../data/README.md); Appendix C
states the reporting standard that the experiments follow.
