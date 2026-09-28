# Chapter 11: Discovering Governing Equations and Hidden Physics

| Book item | Script | Stored results | Cost | Randomness |
|---|---|---|---|---|
| §11.4: mutual coherence of the degree-2 monomial library on a Lorenz-63 trajectory (`z`-`z^2` 0.988; centered maximum 0.955) | `sindy_noise_sweep.py` | `results/chapter11/sindy_noise_sweep.json` (`coherence_top5`, `coherence_top5_centered`) | FAST | deterministic |
| §11.5, Figure 11.1: SINDy (STLSQ) under noise levels 0-50%, 50 realizations per level; spurious terms (medians 1/1/3/8/14/17), runs missing a true term (0/0/2/4/12/26%) | `sindy_noise_sweep.py` | same file (`levels`) | FAST (a few seconds) | seeded (`default_rng(0..49)`) |

```bash
python code/chapter11/sindy_noise_sweep.py
python code/figures/make_fig11_1.py
```

Configuration: Lorenz-63 (`sigma = 10, rho = 28, beta = 8/3`), `x(0) = (1, 1, 1)`, 1000 samples at
spacing 0.01, DOP853 reference (`rtol 1e-12, atol 1e-14`); additive Gaussian noise with standard
deviation (noise level) x (standard deviation of the clean component); derivatives by
`numpy.gradient`; ten monomials of degree at most 2; threshold 0.1, 10 refit iterations.

The Chapter 11 reference trajectory uses `rtol = 1e-12`; the Example G reference solution of
Appendix D (`code/reference_problems/refs_B_to_G.py`) additionally uses `rtol = 1e-13`.
