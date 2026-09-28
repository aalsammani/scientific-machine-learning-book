# Computational environment

## Environments in which the stored results were produced

The book's results were computed in two environments.

| | Environment 1 | Environment 2 |
|---|---|---|
| Machine | Linux x86-64 container, 1 vCPU, 3 GB RAM, no GPU | x86-64 desktop, Windows 11, CPU only (20 logical CPUs), no GPU; timings on one thread |
| Python | 3.12 | 3.14.3 |
| Packages | NumPy 2.4.4, SciPy 1.17.1, JAX 0.11.1, jaxlib 0.11.1, Optax 0.2.8 (recorded for the JAX runs); PyTorch 2.14.0 (a CUDA build, run on the CPU) | NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0, PyTorch 2.14.0+cpu, Matplotlib 3.11.2 |
| Used for | Chapter 1 experiments and reference solutions; Chapters 6-8 learned-method experiments; Chapter 9 FNO; Chapter 10 UDE and closure; Chapter 12 `omega_r` sweep | Chapters 2-5 reproduction scripts; Chapter 7 NTK script; Chapter 9 Darcy verification, dataset, DeepONet sweep, baselines and timing; Chapters 10/15 SIR/SEIR verification; Chapter 11 SINDy sweep; Chapter 12 Darcy inversion; Chapter 13; Chapter 14; all figures |

Precision is float64 for all learned methods. Wall-clock times reported in the book were measured
in the environment named with them and are not portable to other hardware or software. For
Environment 1 the recorded versions are Python 3.12, NumPy, SciPy, JAX and Optax (above) and PyTorch
2.14.0+cu130 (`results/chapter01/environment_pytorch.json`); the versions of other packages used
there are not recorded.

## Reference environment of this repository

The versions pinned in [`requirements.txt`](../requirements.txt) (and
[`requirements-jax.txt`](../requirements-jax.txt)) are those of Environment 2, plus pytest. They were
installed from PyPI into a fresh virtual environment (Windows 11, Python 3.14.3); the test suite
and the reproduction checks described in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) were run in it.

| Package | Version | Needed for |
|---|---|---|
| Python | 3.14 | all |
| numpy | 2.5.3 | all |
| scipy | 1.18.1 | ODE solvers, sparse solves, optimization, statistics |
| sympy | 1.14.0 | Darcy manufactured solutions |
| matplotlib | 3.11.2 | figures |
| torch | 2.14.0 (CPU) | PyTorch experiments |
| pytest | 9.1.1 | tests |
| jax, jaxlib | 0.11.1 | optional: Chapter 1 JAX scripts, Chapter 7 residual-bound check |
| optax | 0.2.8 | optional: as above |

Minimum versions are not established. The code uses `numpy.trapezoid` (NumPy 2.0 or later) and
`scipy.integrate.cumulative_simpson` (SciPy 1.12 or later). Other recent versions will likely run the
code, but reproducing the stored numbers bitwise is expected only with the pinned versions, and not
always across operating systems (see [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), section 3).

## Installation notes

- A plain `pip install torch==2.14.0` from PyPI installs a CUDA-enabled build on Linux. For a smaller
  CPU-only install use
  `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`. GPU execution is
  neither used nor tested.
- `environment.yml` creates a conda environment that installs the same pinned packages with pip.
- No compiler or system library is needed beyond what the wheels provide.
