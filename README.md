# Scientific Machine Learning
## Mathematical Foundations, Methods, and Limitations

**Abdallah A. Alsammani**

This repository contains the code, computational experiments, stored results, and reproducibility
materials accompanying the book *Scientific Machine Learning: Mathematical Foundations, Methods,
and Limitations*.

- Book (arXiv): ARXIV_URL_PENDING
- Archived version of this repository (Zenodo): ZENODO_DOI_PENDING

## 1. About the book

The book develops scientific machine learning from its mathematical foundations and treats it
alongside the classical numerical methods it builds on and competes with (finite differences, finite
elements, spectral methods, ODE and SDE solvers, adjoints, regularized inversion, reduced models and
system identification). It is organized around the object that is learned (a solution, an operator,
a vector field, a governing equation, a physical parameter, a closure, a posterior) and around the
sources of error: approximation, sampling, discretization, optimization, model form, measurement
noise and surrogate error. Its computational examples are reported, as far as each allows, with
verified reference solutions, several random seeds, classical baselines and stated failure modes.

It has 16 chapters in four parts:

- **Part I, Foundations:**
  - 1 What Is Scientific Machine Learning?
  - 2 Differential Equations, Dynamical Systems, and Scientific Models
  - 3 Numerical Foundations and Scientific Computing
  - 4 Learning, Approximation, Optimization, and Differentiation
  - 5 Inverse Problems, Regularization, and Identifiability
- **Part II, Learning Solutions and Operators:**
  - 6 Physics-Informed Neural Networks: Formulation and Computation
  - 7 Why PINNs Work—and Why They Fail
  - 8 Weak, Variational, and Structure-Preserving Learning
  - 9 Operator Learning
- **Part III, Learning Dynamics and Hidden Physics:**
  - 10 Neural Differential Equations and Hybrid Mechanistic Models
  - 11 Discovering Governing Equations and Hidden Physics
  - 12 Inverse Scientific Machine Learning
- **Part IV, Reliability, Applications, and Frontiers:**
  - 13 Uncertainty, Stochasticity, and Data Assimilation
  - 14 Verification, Validation, and Trustworthy SciML
  - 15 Scientific Case Studies
  - 16 Frontiers and Open Problems

Chapters 1-15 contain computations; Chapter 16 has none.

## 2. Repository contents

```
code/
  reference_problems/   verified reference solutions of the running examples (Appendix D)
  chapter01/ ... chapter14/
                        experiment scripts; each directory has a README mapping book sections to
                        scripts, settings, stored results, cost and randomness
  figures/              one script per book figure, reading stored results only
results/chapter01/ ... chapter15/
                        stored results behind every number, table and figure (JSON, NPZ, PT)
data/reference/         reference solutions and their verification records
figures/chapter01/ ... chapter15/
                        the book's figures (PNG and PDF)
tests/                  scientific tests (pytest)
docs/                   experiment index, reproducibility, environment, data provenance,
                        known issues, licensing review
```

Chapter 15's computations are produced by the Chapter 10, 9 and 12 scripts; their stored results are
in `results/chapter15/` and in the Chapter 9 and 12 result directories.

## 3. Quick start

```bash
git clone REPOSITORY_URL_PENDING
cd scientific-machine-learning-book
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # Linux CPU-only torch: see section 4
pytest tests                               # about 10 seconds

python code/chapter09/darcy/darcy_manufactured_verification.py   # verified Darcy solver, a few seconds
python code/chapter05/tikhonov_discrepancy_verification.py       # Tikhonov / discrepancy principle
python code/figures/make_fig9_3.py                               # a figure from stored results
git status                                 # shows which stored files a script rewrote
```

Scripts can be run from any directory; their default inputs and outputs are located relative to the
repository. Running a script rewrites its stored result (or figure) in place; with deterministic
scripts such as the two above, the rewritten numbers are identical and only metadata (timings,
platform strings, descriptive text) may change. `git checkout -- results data figures` restores the stored
versions.

## 4. Installation

- Python 3.14 and the pinned packages in [`requirements.txt`](requirements.txt) (NumPy, SciPy,
  SymPy, Matplotlib, PyTorch CPU, pytest) form the tested reference environment.
- Four scripts (Chapter 1's JAX implementation and controls, and Chapter 7's residual-bound check)
  also need JAX and Optax: `pip install -r requirements-jax.txt`.
- On Linux, `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu` gives a
  CPU-only PyTorch. No GPU is needed.
- A conda alternative is [`environment.yml`](environment.yml).

Details: [`docs/COMPUTATIONAL_ENVIRONMENT.md`](docs/COMPUTATIONAL_ENVIRONMENT.md).

## 5. Reproducing experiments

Each `code/chapterNN/README.md` lists, for every computational result of the chapter:

- the section, table or figure;
- the script and its arguments;
- the stored result files;
- the cost;
- whether the result is deterministic.

The pattern is:

```bash
python code/chapter06/burgers_pinn.py e03 0            # one seed of one experiment
python code/chapter11/sindy_noise_sweep.py              # a complete study
python code/figures/make_fig11_1.py                     # its figure, from the stored results
```

As in the quick start, running an experiment script overwrites the corresponding stored result;
`git diff` shows what changed.

Cost classes (one CPU core):

| Class | Time | Examples |
|---|---|---|
| Fast | under 1 minute | all classical, statistical and verification computations; the figure scripts; Deep Ritz; SINDy; Kalman; Ornstein-Uhlenbeck |
| Moderate | 1-30 minutes | physics-informed network trainings (0.5-4 minutes per seed), the FNO experiments, the UDE, the calibration ensemble |
| Expensive | over 30 minutes | the full DeepONet sweep (about 1 hour serially; parallelized over processes); rerunning every seed of every learned-method experiment of Chapter 1 or Chapter 6 (each set is moderate on its own) |

All learned-method experiments, and the statistical computations (Monte Carlo, noise realizations,
random test problems), use random numbers from explicit seeds; the one exception is the
initialization of the FNO pointwise layers (Chapter 9). The classical solvers and verifications are
deterministic.

## 6. Experiment index

[`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md) maps every computational section, table and
figure of the book to its script, configuration, stored results and cost.

## 7. Computational environment

The stored results were produced in two recorded environments: a single-CPU Linux container and a
Windows desktop, both CPU only. The repository's pinned environment is the second one. See
[`docs/COMPUTATIONAL_ENVIRONMENT.md`](docs/COMPUTATIONAL_ENVIRONMENT.md). Timings in the book are
specific to the stated machines.

## 8. Data

All data are generated by the code in this repository; no external dataset is used. Reference
solutions of the running examples are in [`data/reference/`](data/reference/) (described in
[`data/README.md`](data/README.md)); the Darcy dataset is
`results/chapter09/darcy_dataset.npz`. Provenance of every dataset:
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## 9. Reproducibility notes

- **Figures.** In the reference environment, every figure script regenerates its figure pixel for
  pixel.
- **Classical and statistical computations.** Every rerun reproduced the stored results bitwise,
  to round-off, or within the tolerances stated in `docs/REPRODUCIBILITY.md` (the two exceptions,
  an implementation-dependent FTCS blow-up magnitude and a Lorenz separation curve reconstructed to
  `2e-4`, are explained in `docs/KNOWN_ISSUES.md`).
- **Neural-network training.** It is reproduced scientifically, not bitwise, across machines. FNO
  training (Chapter 9) varies from run to run even on one machine.
- **Missing producers.** A few stored results have no preserved producing script; the closure
  experiment of §10.4 is the only computation that cannot be rerun.
- **Discrepancies with the book.** Small differences between the book text and the code or stored
  results are listed with evidence in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

Details: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 10. Citation

Please cite the book and, if you use the code or results, this repository. Citation metadata is in
[`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button reads it.

```
Abdallah A. Alsammani. Computational Materials for Scientific Machine Learning: Mathematical
Foundations, Methods, and Limitations. Version 1.0.0. ZENODO_DOI_PENDING.
```

## 11. License

The code and the other contents of this repository are released under the MIT License; see
[`LICENSE`](LICENSE) and [`docs/LICENSING_REVIEW.md`](docs/LICENSING_REVIEW.md). The book itself is
distributed separately under CC BY 4.0.

## 12. Reporting issues

Please report errors in the code, the stored results or the documentation, and discrepancies with
the book, through the issue tracker of this repository. Include:

- the script and command;
- the environment (operating system, Python and package versions);
- the output you expected and the output you observed.
