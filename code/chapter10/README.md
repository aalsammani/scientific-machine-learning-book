# Chapter 10: Neural Differential Equations and Hybrid Mechanistic Models

| Book item | Script and invocation | Stored results (`results/chapter10/`) | Cost | Randomness |
|---|---|---|---|---|
| §10.2: universal differential equation for Lotka-Volterra (trajectory error 0.99% / 0.96%; interaction error 5.9% / 5.0% on the visited states, 430% / 146% on a grid) | `lv_ude.py SEED`, seeds 0 and 1 | `lv_ude_seed{0,1}.json`, `lv_ude_traj_seed0.npz`, `lv_ude_summary.json` | MODERATE (about 70 s per seed) | seeded |
| §10.3, Theorem 10.1, Figure 10.1 (top): SIR non-identifiability from incidence; identity `1.4e-17`, forward check `3e-12`, two-time recovery of `gamma` | `sir_seir_nonidentifiability.py` | `sir_nonidentifiability_verification.json`, `sir_trajectories.npz` | FAST | deterministic |
| §10.4, Figure 10.1 (bottom): closure with a localized perturbation; offline `L2(mu)` error against bump radius; online behavior of the perturbed system | none preserved | `closure_instability.json` | | |

The same `sir_seir_nonidentifiability.py` also writes the SEIR results of Chapter 15
(`results/chapter15/`). Figure 10.1: `code/figures/make_fig10_1.py`.

```bash
python code/chapter10/lv_ude.py 0
python code/chapter10/lv_ude.py 1
python code/chapter10/sir_seir_nonidentifiability.py
python code/figures/make_fig10_1.py
```

## Notes

- **Closure experiment (§10.4).** The script that produced `closure_instability.json` was not
  preserved, and the stored file does not record every setting (in particular the form of the bump
  function); it is published as a stored result so that Figure 10.1 can be regenerated, but the
  experiment itself cannot be rerun from this repository. Partial configuration: closure
  `g(x) = 0.1 (x1^3, x2^3)`, perturbation matrix `E = [[1.2, 0.3], [-0.3, 1.2]]`, offline error by
  Monte Carlo over 20000 points from `N(0, 0.3 I)` (seed 0), RK45 at `rtol = 1e-10`, initial
  condition `(0.05, 0)`, horizon `[0, 15]` (read from the stored file). The norm of the stored
  perturbed trajectory is 0.030 at `t = 1`, 0.018 at `t = 2` and 0.0157 at `t = 3`, and stays
  between 0.01553 and 0.01569 for `3 <= t <= 15`; §10.4 describes it as "between about 0.014 and
  0.016". See
  [`docs/KNOWN_ISSUES.md`](../../docs/KNOWN_ISSUES.md).
- **Lotka-Volterra data.** The "true" trajectory used for training and evaluation is computed with
  the same fixed-step RK4 integrator (two substeps per interval) that the UDE uses; its error with
  respect to the exact flow is not measured. The stored seed results were computed in the first
  recorded environment (see [`docs/COMPUTATIONAL_ENVIRONMENT.md`](../../docs/COMPUTATIONAL_ENVIRONMENT.md)).
  A rerun of seed 0 in the reference environment of this repository gave trajectory error 0.88%,
  interaction error 5.0% on the visited states and 381% on the grid (stored: 0.99%, 5.9%, 430%):
  the same qualitative result, different digits.
- **SIR single-observation family.** For one observation of `I` the script evaluates members of the
  remaining family on a grid of seven `gamma` values in `[0.13, 0.16]`; all seven stored members are
  flagged inadmissible (the admissible members lie in a narrow band of `gamma` just below `1/7`,
  which this grid does not sample). The two-observation recovery of `gamma` is exact to the solver
  tolerance.
