# benchmarks/

Performance and downstream-quality benchmark scripts. All scripts add `../..` to `sys.path` to import the local `vectorzip` package (dev-time, not installed-package tests). Most pull embedding models via `sentence_transformers` and datasets via HuggingFace `datasets`, and write a `results.json` next to the script.

## Top-level scripts (headline results → feed `docs/benchmarks.md` and README)

- **`run_benchmarks.py`** — Core performance benchmark: fidelity (MSE, cosine retention) and projection latency across model/dimension scales (96D → 768D targets at 4× ratio). Uses a high-performance TSP nearest-neighbor solver. Produces the fidelity table in the docs.
- **`run_baselines.py`** — Baseline comparison harness: re-implements competitor compressors (random projection, PCA, k-means/PQ, raw SQ8) alongside VectorZip for apples-to-apples comparison. Output: `baseline_comparison_results.json`.
- **`search_latency_test.py`** — The headline CPU search-latency benchmark: 100k-vector DB, 1000 random queries, 384D (raw/SQ8) vs 64D (VectorZip/PCA). Produces the 6.02× speedup number. Output: `search_latency_results.json`.
- **`spanish_benchmark.py`** — Spanish Multilingual Sovereignty benchmark (homonym disambiguation): BGE-M3 1024 compressed to 384 vs. native BGE-Small 384. Validates multilingual capability preservation.
- **`long_context_benchmark.py`** — Long-context PDF retrieval benchmark: BGE-M3 1024 compressed to 384 vs. native BGE-Small 384 (8K context window preservation).
- **`qwen_spanish_benchmark.py`** — "Star" benchmark: Qwen2.5-Large 1536 compressed to 512 vs. native Qwen2.5-Small 512 on Spanish semantic matching.

## Numbered ablation suites (`<N>_<name>/`)

Each folder has a `run.py` and a committed `results.json`.

- **`1_encoding_latency_penalty/`** — Measures the pure transform-only latency cost of compressing 1024D → 384D (VectorZip vs PCA), separating fitting cost from inference cost.
- **`2_default_corpus_bias/`** — Tests whether auto-calibration on the internal generic `DEFAULT_CORPUS` introduces bias vs. fitting on the target domain.
- **`3_tsp_scalability/`** — TSP solver scalability and reconstruction cosine across corpus sizes / dimensionality (uses a medical flashcards dataset).
- **`4_linear_compression_limit/`** — Measures embedding "smoothness" (mean squared difference of adjacent components) to probe how far linear/DCT compression can go.
- **`5_pca_vs_vectorzip_same_corpus/`** — PCA vs. VectorZip reconstruction cosine when fit and tested on the **same** corpus (in-distribution).
- **`6_pca_vs_vectorzip_different_corpus/`** — PCA vs. VectorZip when fit on one corpus and tested on a different one (cross-corpus generalization).
- **`7_cross_dataset_generalization/`** — The OOD benchmark: fit on Wikipedia, test on PubMed. Source of the "1.9× more robust to domain shift than PCA" claim.
- **`8_search_quality_preservation/`** — Ranking-quality preservation: NDCG@10 and Recall@k of compressed-vector retrieval vs. ground-truth retrieval.
- **`9_downstream_rag_hit_rate/`** — End-to-end RAG Hit-Rate@3 with a real retriever + LLM, comparing compressed vs. uncompressed vs. native-small embeddings.
- **`10_training_overhead_scaling/`** — Calibration (fit) time scaling of VectorZip vs. PCA as corpus size grows.
