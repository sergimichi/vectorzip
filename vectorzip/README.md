# vectorzip/

The installable Python package. Only two real modules — everything ships from here.

## Files

- **`__init__.py`** — Package entry point. Exports the public API (`VectorZip`, `VectorZipModel`, `compress`) and sets `__version__ = "0.1.0"`. Keep these three names stable; they are the documented surface.
- **`compressor.py`** — All compression logic. Contains:
  - `RandomProjectionCompressor` — Gaussian random projection with Gram-Schmidt orthonormalization (the `rp` backend).
  - `ProductQuantizer` — subvector k-means → `uint8` codes (the `pq` backend; uses sklearn `KMeans`, lazy-imported).
  - `ScalarQuantizer` — per-column min/max scaling to `uint8` (the `sq8` backend and the `*+sq8` hybrid layer).
  - `VectorZip` — the main estimator. sklearn-style `fit` / `transform` / `fit_transform` / `inverse_transform`. Dispatches to the backends above by `method` string (`dct`, `pca`, `rp`, `pq`, `sq8`, `matryoshka`/`mrl`, plus `*+sq8` hybrids). Holds the flagship DCT pipeline: `_solve_tsp` (nearest-neighbor + 2-opt over a `1−|corr|` distance matrix) and DCT-II projection via `scipy.fftpack.dct`.
  - `VectorZipModel` — drop-in wrapper around any encoder (`encode` / `embed_documents` / `get_text_embeddings` / `embed` / callable). Auto-calibrates on first `encode()`, serializes calibration to pure JSON via `save_pretrained` / `from_pretrained`.
  - `compress(X, ...)` — one-line functional `fit_transform` helper. `decompress()` intentionally raises `NotImplementedError`.
- **`default_corpus.py`** — A large (~3000 lines) built-in list of diverse generic sentences (`DEFAULT_CORPUS`) used for auto-calibration when the user does not supply their own corpus. Domain-agnostic so the fitted TSP/PCA basis generalizes.
