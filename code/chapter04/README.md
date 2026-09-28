# Chapter 4: Learning, Approximation, Optimization, and Differentiation

| Book item | Script | Stored results (`results/chapter04/`) | Cost | Randomness |
|---|---|---|---|---|
| §4.5 and Figure 4.1: gradient descent on a 20-dimensional quadratic (condition number 50) against the bounds of Theorems 4.4 and 4.5 | `gd_convergence_verification.py` | `gd_convergence_verification.json` (plotted), `_rerun.json` | FAST | seeded (`np.random.seed(0)`) |
| §4.6: cost of a reverse-mode gradient relative to a forward evaluation, network `[d, 256, 1]`, `d = 10 ... 1e5` | `ad_cost_scaling.py` | `ad_cost_scaling.json` (first run quoted in §4.6), `ad_cost_scaling_rerun.json` (repeated single-thread sweeps) | MODERATE (1-3 min) | timings vary between runs and machines |

```bash
python code/chapter04/gd_convergence_verification.py
python code/chapter04/ad_cost_scaling.py
python code/figures/make_fig4_1.py
```

Provenance. `gd_convergence_verification.py` reproduces the stored original to round-off.
`ad_cost_scaling.json` is the stored first run (ratios 6.4, 6.0, 4.7, 2.4, 11.3); the value 11.3 at
`d = 1e5` did not recur in repeated single-thread sweeps (`ad_cost_scaling_rerun.json`, key
`params_1thread`), and §4.6 reports it as an outlier. Timing ratios are hardware and software
dependent; a rerun on a different machine gives different numbers.
