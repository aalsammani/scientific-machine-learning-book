# Chapter 5: Inverse Problems, Regularization, and Identifiability

| Book item | Script | Stored results (`results/chapter05/`) | Cost | Randomness |
|---|---|---|---|---|
| §5.3 and Figure 5.1 (left): Tikhonov regularization and the discrepancy principle for the backward heat equation (near-unregularized error 2,027; oracle 0.501; discrepancy principle 0.503) | `tikhonov_discrepancy_verification.py` | `tikhonov_discrepancy_verification.json` (plotted), `_rerun.json` | FAST | seeded (`default_rng(0)`) |
| §5.4 and Figure 5.1 (right): two-compartment structural non-identifiability (Theorem 5.4) | `two_compartment_nonidentifiability.py` | `two_compartment_nonidentifiability.json` (plotted), `_rerun.json` | FAST | deterministic |
| §5.7: continuous adjoint gradient versus centered finite differences (Theorem 5.7) | `adjoint_vs_fd_verification.py` | `adjoint_vs_fd_verification.json`, `_rerun.json` | FAST | deterministic |

```bash
python code/chapter05/tikhonov_discrepancy_verification.py
python code/chapter05/two_compartment_nonidentifiability.py
python code/chapter05/adjoint_vs_fd_verification.py
python code/figures/make_fig5_1.py
```

Provenance. The files without the `_rerun` suffix are the stored original results. The scripts
here reconstruct the protocols (documented in their docstrings) and reproduce the stored values to
round-off; the `_rerun` files also contain the diagnostics quoted in the text (mode-5 damping and
data contribution in §5.3, the `x2` difference in §5.4, the refined-grid comparison in §5.7, which
shows that the adjoint/finite-difference discrepancy comes from the trapezoidal quadrature).

The "near-unregularized" error in §5.3 is Tikhonov with `alpha = 1e-12`: the unregularized inverse
(`alpha = 0`) overflows in double precision.
