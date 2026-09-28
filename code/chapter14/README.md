# Chapter 14: Verification, Validation, and Trustworthy SciML

| Book item | Script | Stored results | Cost | Randomness |
|---|---|---|---|---|
| §14.3, Figure 14.1: calibration of a ten-member deep ensemble of physics-informed networks for Example A (coverage 61/69/73/78/83% at nominal 50/68/80/90/95%, 2000 points) | `calibration_check.py` | `results/chapter14/calibration_check_rerun.json` | MODERATE (about 1 min with ten parallel processes) | seeded |
| §14.5, Figure 14.2 (and Chapter 9, §9.8): distribution shift for the Darcy DeepONet (in family 0.15-0.23%, out of family 1.32-1.35%; exact in-family operator 1.37%; training-mean field 0.42%) | `distribution_shift.py` | `results/chapter14/deeponet_distribution_shift.json` | FAST | deterministic (stored model states) |
| §14.4, Instance 1: the Darcy solver's first-order boundary defect (order 1 instead of 2; about 14% data error) | `code/chapter09/darcy/darcy_manufactured_verification.py`, `darcy_dataset_and_baselines.py` | `results/chapter09/darcy_verification/`, `results/chapter09/regeneration_prepare.json`, `results/chapter09/legacy_first_order_solver/` | FAST | deterministic |
| §14.4, Instances 2-4 | reuse the results of Chapter 9 (§9.6) and Chapter 6 (§6.10) | see those chapters | | |

`distribution_shift.py` loads the five stored `n = 200` DeepONet states of Chapter 9
(`results/chapter09/sweep_normalized/state_n200_seed{0-4}.pt`), so it does not retrain.
Figures: `code/figures/make_fig14_1.py`, `make_fig14_2.py`.

```bash
python code/chapter14/calibration_check.py
python code/chapter14/distribution_shift.py
python code/figures/make_fig14_1.py
python code/figures/make_fig14_2.py
```

## Note on the calibration protocol string

`calibration_check_rerun.json` contains a `protocol` string copied from the script's docstring at
the time of the run. That string states `T = 0.2` and names a development path; the code, however,
takes `T = 1` from `code/reference_problems/exA_heat.py`, and a rerun of the current script in the
reference environment reproduces every stored number exactly. The stored numbers therefore
correspond to `T = 1`, as stated in the book; only the metadata string is wrong. The result file is
kept unmodified.
