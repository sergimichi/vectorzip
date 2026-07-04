# docs/

Source for the MkDocs Material documentation site published at `sergimichi.github.io/vectorzip` (deployed by `.github/workflows/docs.yml`). Config is in the repo root `mkdocs.yml`.

## Files

- **`index.md`** — Landing page: tagline, install command, and quickstart showing `VectorZipModel` as a drop-in `SentenceTransformer` replacement, plus notes on the universal wrapping capabilities.
- **`how-it-works.md`** — Architecture & methodology writeup. Explains the Same-Family Parametric Transfer Hypothesis, why raw NLP embeddings resist spectral compression (no inherent dimension ordering), and the two-stage flagship DCT pipeline: TSP dimensional reordering (covariance → `1−|corr|` distance → nearest-neighbor + 2-opt) and orthonormal DCT-II projection with energy compaction. Includes LaTeX math (rendered via `javascripts/mathjax.js`).
- **`benchmarks.md`** — Published results tables: fidelity vs. compression ratio, CPU search latency (the headline 6.02× speedup), out-of-distribution domain generalization (PCA vs VectorZip), and downstream RAG accuracy transfer (compressed high-capacity model vs. native small model).
- **`api.md`** — API reference page. Auto-renders `vectorzip.compressor` docstrings via `mkdocstrings` (configured in `mkdocs.yml`).
- **`javascripts/mathjax.js`** — MathJax loader so the LaTeX equations in `how-it-works.md` render in the docs site.
