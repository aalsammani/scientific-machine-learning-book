# Figures

The 24 figures of the book, as PNG (the version used in the book) and PDF, organized by chapter.

Every figure except Figure 1.2 (the learned-object taxonomy, a conceptual diagram) is produced by a
script in [`code/figures/`](../code/figures/) named after the figure number (`make_fig9_1.py` makes
Figure 9.1; `make_fig1_3_comparison.py` makes Figure 1.3). The scripts read only the stored results
in `results/` and `data/`; none retrains a model. Figures 1.1, 1.6 and 6.1 are schematics drawn from
constants in their scripts.

In the reference environment (see [`docs/COMPUTATIONAL_ENVIRONMENT.md`](../docs/COMPUTATIONAL_ENVIRONMENT.md)),
every script regenerates its PNG pixel for pixel. Regenerated PDFs differ from the stored ones in
their embedded creation date. With other versions of Matplotlib or its fonts, the regenerated
figures show the same data but may differ in rendering.

```bash
python code/figures/make_fig9_1.py      # writes figures/chapter09/F9.1_*.png and .pdf
```
