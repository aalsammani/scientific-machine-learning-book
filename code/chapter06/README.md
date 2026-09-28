# Chapter 6: Physics-Informed Neural Networks: Formulation and Computation

Result files in `results/chapter06/` carry experiment identifiers of the form `ch07-...`; these are
identifiers assigned when the experiments were run and do not denote Chapter 7 (see
[`docs/EXPERIMENT_INDEX.md`](../../docs/EXPERIMENT_INDEX.md)).

| Book item | Script and invocation | Stored results | Cost per run | Randomness |
|---|---|---|---|---|
| §6.7 forward problem (0.65%, range 0.22-0.69%) | `code/chapter01/exA_three_ways_pytorch.py pinn` | `results/chapter01/ch01-e03_pinn_seed{0-4}_pytorch.json` | MODERATE (about 3.5 min) | seeded |
| §6.7 inverse problem: recover `nu` from 20 observations, noise 1% and 5% | `SIGMA_REL=0.01 python inverse_pinn.py` and `SIGMA_REL=0.05 python inverse_pinn.py` (seeds 0-4) | `ch07-e02_inverse_seed{0-4}_sigma{0.01,0.05}.json` (+ `_history.json`) | MODERATE (about 2 min) | seeded |
| §6.8 viscous Burgers, `nu = 0.01/pi` (3.1-11.6%) | `burgers_pinn.py e03 SEED`, seeds 0-4 | `ch07-e03_burgers_seed{0-4}.json` | MODERATE (about 1.5 min) | seeded |
| §6.8-§6.9 and §7.3: Burgers with `nu = 0.001` (17, 23, 42%) | `burgers_pinn.py f03 SEED`, seeds 0-2 | `ch07-f03_burgers_seed{0-2}.json` | MODERATE | seeded |
| §6.9 sparse collocation, `n_r = 200` (23-58%) | `burgers_pinn.py f02 SEED`, seeds 0-4 | `ch07-f02_burgers_seed{0-4}.json` | FAST-MODERATE | seeded |
| §6.9 loss imbalance (boundary weight 0.001) | `constraint_ablation.py underweighted 0` and `soft 0` | `ch07-f01_underweighted_seed0.json`, `ch07-e04_soft_seed0.json` | MODERATE (about 3.5 min) | seeded |
| §6.9 hard constraint (3.6, 16.7, 17.0, 64.1%) | `constraint_ablation.py hard SEED`, seeds 0-3 | `ch07-e04_hard_seed{0-3}.json` | MODERATE (about 2.5 min) | seeded; see note |
| §6.9 ReLU activation (294-296%) | `constraint_ablation.py relu SEED`, seeds 0-1 | `ch07-f04_relu_seed{0,1}.json` | FAST (about 0.5 min) | seeded |
| §6.10 and Table 6.1: FTCS and Crank-Nicolson finite differences | `code/reference_problems/exA_heat.py` | `data/reference/exA/verification.json` | FAST | deterministic |
| §6.10 and Table 6.1: P1 finite elements (Crank-Nicolson, `dt = h`) | `fem_1d.py` and `fem_1d.py --timed` | `data/reference/exA/fem_verification.json`, `fem_verification_timed.json` | FAST | deterministic |
| §6.10 and Table 6.1: Kansa RBF collocation | `kansa_rbf.py` | `kansa_convergence.json` | FAST | deterministic |
| §6.10 and Table 6.1: physics-informed network cost range | none preserved | `ch07-e01_pareto_data.json` (aggregate of the Chapter 1 runs) | | |
| §6.8 reference: Cole-Hopf vs method of lines (`8e-5`) | `code/reference_problems/refs_B_to_G.py exC` | `data/reference/exC/verification.json`, `reference_nu_0.01_over_pi.npz` | MODERATE (about 4 min) | deterministic |

Figure 6.1 is a schematic (`code/figures/make_fig6_1.py`).

## Configuration

All scripts are float64 and single-threaded; the constants are at the top of each script and in
its docstring. Example A runs use the network of Chapter 1 (3 x 32 tanh, LeCun normal). The Burgers
network is `[2, 48, 48, 48, 1]` (three hidden layers of 48 units, 4897 parameters), Adam 2500
iterations then L-BFGS at most 3000 iterations, `n_r = 2000`, `n_b = n_0 = 200`. Collocation seeds
are offset per experiment (`seed + 3000` inverse, `seed + 5000` Burgers, `seed + 6000` constraint
ablations) so that experiments do not share point sets.

```bash
SIGMA_REL=0.01 python code/chapter06/inverse_pinn.py          # PowerShell: $env:SIGMA_REL="0.01"
python code/chapter06/burgers_pinn.py e03 0
python code/chapter06/constraint_ablation.py hard 0
python code/chapter06/fem_1d.py
python code/chapter06/kansa_rbf.py
```

Each learned-method script runs one seed per call (except `inverse_pinn.py`, which loops over the
seeds given, default 0-4); the book's medians and ranges are over the seeds listed in the table.

## Notes

- **Burgers architecture.** The code and the stored results (`n_params = 4897`) use three hidden
  layers of 48 units; §6.8 of the book describes the network as "four hidden layers of 48 units".
  See [`docs/KNOWN_ISSUES.md`](../../docs/KNOWN_ISSUES.md).
- **Burgers reference at `nu = 0.001`.** The `f03` runs are scored against the Cole-Hopf quadrature
  with 200 nodes, whose accuracy was verified only at `nu = 0.01/pi`. Their stored
  `rel_residual_l2` is evaluated with `nu = 0.01/pi` and is not meaningful for these runs.
- **Inverse problem L-BFGS limit.** The script default is 5000 iterations; the stored results do
  not record the limit that was used.
- **Hard-constraint runs are not reproducible run to run** in the sense discussed in §7.7; rerunning
  the same seed can give a substantially different error.
- **ReLU runs** use He initialization; the reported `rel_residual_l2` is the residual on the
  held-out evaluation grid divided by the norm of the exact solution there.
- **Timings** in §6.10 and Table 6.1 are machine specific (see
  [`docs/COMPUTATIONAL_ENVIRONMENT.md`](../../docs/COMPUTATIONAL_ENVIRONMENT.md)).
- **Cole-Hopf quadrature check.** `refs_B_to_G.py exC` compares the 100- and 150-node Gauss-Hermite
  rules with the 200-node rule used for the reference (NumPy's `hermgauss` overflows at 400 nodes).
  A rerun reproduced the stored method-of-lines rows quoted in §6.8 to about `1e-8` relative.
