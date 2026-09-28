# Chapter 12: Inverse Scientific Machine Learning

| Book item | Script | Stored results (`results/chapter12/`) | Cost | Randomness |
|---|---|---|---|---|
| §12.4, Figure 12.2 (left): Fisher information of the five-coefficient Darcy inversion (15 nodes: eigenvalues 0.064 ... 5.175, condition number 80; 50 placements: 33-422; 441 nodes: condition number 50) | `darcy_inversion.py` | `darcy_inversion_dense.json` (`fim_*`) | FAST (under a minute) | seeded (`default_rng(42)`, `default_rng(7)`) |
| §12.6, Figure 12.2 (right): recovery of `c_1` from 441 noisy observations, 200 realizations: classical RMSE 0.088 (Cramer-Rao 0.084), well-trained surrogate 0.093, poorly trained surrogate 0.242 | `darcy_inversion.py` | `darcy_inversion_dense.json` (`classical`, `surrogates`) | same run | seeded (`default_rng(1000 + m)`) |
| §12.5, Figure 12.1: joint estimation of `nu` and a network, residual weight `omega_r` from 0 to 10 | `omega_r_sweep.py` | `omega_r_sweep.json` | MODERATE (about 2 min) | seeded |

The two surrogates are the `n = 200`, seed-0 DeepONets of Chapter 9, trained with standardized
outputs (`results/chapter09/sweep_normalized/state_n200_seed0.pt`) and with raw outputs
(`results/chapter09/sweep_firstedition_protocol/state_n200_seed0.pt`). They are the defaults of
`darcy_inversion.py`, so no DeepONet is retrained. Figures: `code/figures/make_fig12_1.py`,
`make_fig12_2.py`.

```bash
python code/chapter12/darcy_inversion.py
python code/chapter12/omega_r_sweep.py
python code/figures/make_fig12_1.py
python code/figures/make_fig12_2.py
```

## Notes

- The keys `first_edition_realization` in `darcy_inversion_dense.json` refer to the single noise
  realization of the original single-realization study (a fresh seed-42 generator); the book's
  numbers are the 200-realization statistics.
- `darcy_inversion.py --legacy` uses the first-order boundary treatment of the Darcy solver that is
  discussed in Chapter 14, §14.4; it writes `darcy_inversion_legacy_solver.json` and is not used for
  any number in the book.
- `omega_r_sweep.py` uses its own heat-equation setting with `T = 0.2` and 40 observations. The
  stored file records the data and residual losses but not the boundary loss, so the joint loss
  quoted in §12.5 (about `6.8e-3` at `omega_r = 10`) can be checked only as
  `omega_r L_r + L_data = 6.85e-3`, a lower bound. A rerun in the reference environment of this
  repository reproduced the recovered `nu` values to four digits.
