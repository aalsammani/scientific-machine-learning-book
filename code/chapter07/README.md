# Chapter 7: Why PINNs Work—and Why They Fail

Result files in `results/chapter07/` include identifiers of the form `ch08-...`; these were
assigned when the experiments were run and do not denote Chapter 8.

| Book item | Script and invocation | Stored results (`results/chapter07/`) | Cost | Randomness |
|---|---|---|---|---|
| §7.2: numerical check of the residual-to-error bounds of Theorem 7.1 (hard and soft forms, two seeds, early training) | `residual_bound_check.py` (JAX) | `ch08-t81_check.json` | MODERATE (a few minutes) | seeded |
| §7.4 and Figure 7.1 (left): residual NTK spectrum at initialization, 80 points, seed 0, `t = 0.1`; seed sweep over ten seeds | `ntk_spectrum.py` | `ntk_spectrum.json` (plotted), `ntk_spectrum_rerun.json` (reproduction, resolved spectrum, seed sweep) | FAST (under a minute) | deterministic |
| §7.5 and Figure 7.1 (right): gradient-norm loss balancing from `omega_b = 0.001` | `loss_balance.py {fixed,gradnorm} SEED`, seeds 0-2 | `lossbalance_{fixed,gradnorm}_seed{0-2}.json` | MODERATE (about 2.5 min per run) | seeded |
| §7.7: hard-constraint diagnostics (peak-residual location, Spearman correlations with envelope and time, per-mode errors) | `hard_constraint_diagnostics.py` (seeds 0-4) | `ch08-hardconstraint-diag_seed{0-4}.json` (+ `_history.json`) | MODERATE (about 3.5 min per seed) | seeded; see note |
| §7.3: Burgers at `nu = 0.001` | `code/chapter06/burgers_pinn.py f03 SEED` | `results/chapter06/ch07-f03_burgers_seed{0-2}.json` | MODERATE | seeded |

`lossbalance_summary.json` lists the six per-seed results together; it was assembled from the
per-seed files.

```bash
python code/chapter07/ntk_spectrum.py
python code/chapter07/loss_balance.py gradnorm 0
python code/chapter07/hard_constraint_diagnostics.py 0 1 2 3 4
python code/chapter07/residual_bound_check.py          # needs requirements-jax.txt
python code/figures/make_fig7_1.py
```

## Notes

- **NTK spectrum provenance.** The stored spectrum `ntk_spectrum.json` has no preserved producing
  script and does not record its configuration. `ntk_spectrum.py` identifies it (reference network,
  seed 0, time slice `t = 0.1`), reproduces it (first eight eigenvalues to about `1e-10` relative),
  and computes the resolved part of the spectrum from the singular values of the Jacobian.
  Eigenvalues below the rounding floor, including the negative ones, carry no information about
  frequency; Figure 7.1 plots only the twelve largest.
- **Residual-bound check (§7.2).** The networks are trained inside `residual_bound_check.py` (JAX,
  1500 Adam steps); its "hard" variant imposes the boundary conditions exactly
  (`u = x(1-x) N`) and treats the initial condition as a penalty. The bound holds for any
  sufficiently smooth trial function, so the training state does not affect the validity of the
  check.
- **Hard-constraint reruns (§7.7).** `hard_constraint_diagnostics.py` allows at most 8000 L-BFGS
  iterations, whereas `code/chapter06/constraint_ablation.py hard` allows 10000. §7.7 compares the
  seed-1 errors 3.58% (`results/chapter06/ch07-e04_hard_seed1.json`) and 12.6%
  (`ch08-hardconstraint-diag_seed1.json`) as repetitions of the same configuration; because of the
  different iteration limits the two runs are not identical configurations, and the third value
  quoted there (2.99%) is not in any stored result file. See
  [`docs/KNOWN_ISSUES.md`](../../docs/KNOWN_ISSUES.md).
- **Loss balancing.** `loss_balance.py` uses at most 8000 L-BFGS iterations and collocation seed
  `seed + 7000`; the weight is updated every 200 Adam iterations and frozen during L-BFGS.
