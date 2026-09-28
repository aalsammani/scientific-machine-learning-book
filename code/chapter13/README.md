# Chapter 13: Uncertainty, Stochasticity, and Data Assimilation

| Book item | Script | Stored results (`results/chapter13/`) | Cost | Randomness |
|---|---|---|---|---|
| §13.4, Theorem 13.2, Figure 13.1 (left): drift and diffusion estimation for the Ornstein-Uhlenbeck process from data at spacing `dt` (drift bias -2.7% at `dt = 0.1`, -57% at `dt = 4`; diffusion -1.2%, -35%) | `ou_drift_diffusion.py` | `ou_drift_diffusion.json` | FAST | seeded (`default_rng(10000 i + r)`) |
| §13.4: the same estimators applied to Euler-Maruyama data generated at the sampling step (the pitfall discussed in the text) | `ou_drift_diffusion.py --legacy-em` | `ou_drift_diffusion_legacy_em.json` | FAST | seeded |
| §13.5, Figure 13.1 (right): Kalman filter for a damped oscillator observed in position; 50 realizations (position error 2.1% vs 7.1% raw and 5.3% model only; better than both in 48/50; velocity better than model only in 26/50) | `kalman_filter_demo.py` | `kalman_filter_study.json` | FAST | seeded (`default_rng(s)`, `s = 0..49`) |

```bash
python code/chapter13/ou_drift_diffusion.py
python code/chapter13/ou_drift_diffusion.py --legacy-em
python code/chapter13/kalman_filter_demo.py
python code/figures/make_fig13_1.py
```

The OU data are generated with the exact transition density from the stationary distribution, so no
simulation error enters at any `dt`; `theta = 0.5`, `sigma = 0.4`, record length `T = 2000`,
20 records per `dt`. The Kalman system is the explicit-Euler discretization of
`x'' + 2 zeta omega0 x' + omega0^2 x = 0` with `omega0 = 1`, `zeta = 0.05`, `dt = 0.1`,
`Q = diag(1e-6, 1e-5)`, `R = 0.05^2`, 300 steps.
