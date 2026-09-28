# Stored results

The numerical results behind the book's printed numbers, tables and figures, organized by the
chapter in which they are reported (`chapter01/` ... `chapter15/`; Chapter 16 has no computations).
They are the files produced when the experiments were run for the book, stored unmodified.

- **Formats.** JSON for scalar results, per-seed metrics and training histories; NPZ (NumPy) for
  fields, trajectories and datasets; PT (PyTorch `state_dict`) for the Chapter 9 DeepONet models that
  Chapters 12 and 14 reuse. No file is larger than 4 MB.
- **Where each file comes from.** See [`docs/EXPERIMENT_INDEX.md`](../docs/EXPERIMENT_INDEX.md) and
  the README in the corresponding `code/chapterNN/` directory. Figures are generated from these files
  by the scripts in `code/figures/`, without retraining any model.
- **File-name prefixes.** Many files carry experiment identifiers such as `ch07-e03` or `ch08-...`.
  These identifiers were assigned when the experiments were run and do not always equal the chapter
  number in the final book (for example, `results/chapter06/ch07-e03_burgers_seed0.json` belongs to
  Chapter 6). The directory, not the prefix, gives the chapter.
- **`*_rerun.json` files** (Chapters 2-5, 7) are outputs of reproduction scripts written after the
  original runs; the files without the suffix are the originals. The chapter READMEs say which file
  holds which printed number. The exception is `chapter14/calibration_check_rerun.json`, which has no
  unsuffixed counterpart; it is the result reported in §14.3.
- **Names containing `first_edition` or `firstedition`** (for example `sweep_firstedition_protocol/`)
  refer to the original DeepONet training protocol (raw, unscaled outputs; Chapter 9) and to the
  single noise realization of the original inversion study (Chapter 12); both are part of the
  results reported in the book.
- **Metadata inside the files** (paths such as `experiments/ch09new/results/...`, platform strings,
  a protocol string in `chapter14/calibration_check_rerun.json`) reflects the environment in which the
  files were produced and is left unchanged; see [`docs/KNOWN_ISSUES.md`](../docs/KNOWN_ISSUES.md).
- **`chapter09/legacy_first_order_solver/`** holds results obtained with a defective boundary
  treatment of the Darcy solver, kept as part of the verification record (Chapter 14, §14.4).

Rerunning a script overwrites the corresponding stored file. With the repository under version
control, `git diff` shows the change and `git checkout -- results/` restores the stored results.
