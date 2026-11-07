# docs/

Source for the MkDocs Material documentation site published at `sergimichi.github.io/vectorzip` (deployed by `.github/workflows/docs.yml`). Config is in the repo root `mkdocs.yml`.

## Files

- **`index.md`** — Landing page: tagline, install command, and quickstart showing `VectorZipModel` as a drop-in `SentenceTransformer` replacement, plus notes on the universal wrapping capabilities.
- **`how-it-works.md`** — Architecture & methodology writeup. Explains Post-Hoc Parametric Transfer, why raw neural embeddings lack spatial coordinate ordering, and the two-stage `dct` pipeline: correlation-based TSP coordinate reordering (`1 − |corr|` distance + nearest-neighbor, 2-opt, and Or-opt local search) and orthonormal DCT-II projection. Includes LaTeX math (rendered via `javascripts/mathjax.js`).
- **`benchmarks.md`** — Empirical benchmark tables and trade-off analysis: projection throughput, CPU vector search latency (384D vs. 64D), calibration state footprint (`dct` vs. `pca`), in-distribution vs. out-of-distribution generalization, and capability transfer from high-capacity models.
- **`api.md`** — API reference page. Auto-renders `vectorzip.compressor` docstrings via `mkdocstrings` (configured in `mkdocs.yml`).
- **`javascripts/mathjax.js`** — MathJax loader so the LaTeX equations in `how-it-works.md` render in the docs site.
