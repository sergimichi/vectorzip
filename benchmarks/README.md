# benchmarks/

Reproducible performance, latency, and retrieval-quality benchmarks for `vectorzip`. Organized into three groups:

- **`library/`** — Core throughput, search latency, baseline comparison, and capability transfer scripts.
- **`results/`** — Committed JSON outputs produced by the benchmark suites.
- **`1_..10_<name>/`** — Numbered ablation suites, each containing a standalone `run.py` script and its committed `results.json`.

All scripts automatically resolve the repository root on `sys.path` so they can be executed from any working directory.

## `library/` — Core & Capability Benchmarks

- **`run_benchmarks.py`** — Measures projection latency, Mean Squared Error (MSE), and cosine similarity retention across embedding scales ($384\text{D} \to 96\text{D}$ up to $3072\text{D} \to 768\text{D}$ at a $4\times$ compression ratio).
- **`run_baselines.py`** — Apples-to-apples comparison of `VectorZip` (`dct`), `PCA`, `RandomProjectionCompressor` (`rp`), `ProductQuantizer` (`pq`), and `ScalarQuantizer` (`sq8`) across in-distribution (Wikipedia) and out-of-distribution (Medical, Code, Legal) corpora. Output: `results/baseline_comparison_results.json`.
- **`search_latency_test.py`** — CPU vector search latency benchmark over 100,000 normalized document vectors and 1,000 queries, comparing 384D baseline search against 64D compressed search ($6.02\times$ throughput speedup). Output: `results/search_latency_results.json`.
- **`spanish_benchmark.py`** — Multilingual homonym disambiguation demo comparing `BAAI/bge-m3` ($1024\text{D}$ compressed to $384\text{D}$) against native `BAAI/bge-small-en-v1.5` ($384\text{D}$).
- **`long_context_benchmark.py`** — Needle-in-a-haystack long-context retrieval demo (~1,150 words, beyond the 512-token window of small encoders) comparing `BAAI/bge-m3` compressed to $384\text{D}$ against native `BAAI/bge-small-en-v1.5` ($384\text{D}$).
- **`qwen_spanish_benchmark.py`** — Multilingual semantic matching demo comparing `Alibaba-NLP/gte-Qwen2-1.5B-instruct` ($1536\text{D}$ compressed to $512\text{D}$) against native `Qwen/Qwen2.5-0.5B-Instruct` ($512\text{D}$).

## Numbered Ablation Suites (`1_..10_<name>/`)

Each directory contains a standalone `run.py` and its committed `results.json`:

- **`1_encoding_latency_penalty/`** — Isolates the pure transform latency overhead of compressing $1024\text{D} \to 384\text{D}$ (>120,000 vectors/sec on CPU) relative to transformer inference time.
- **`2_default_corpus_bias/`** — Compares calibration state memory and disk footprints between `PCA` (~1.5 MB dense float matrix) and `VectorZip` `dct` (~4.9 KB JSON permutation array — $314\times$ smaller).
- **`3_tsp_scalability/`** — Evaluates in-distribution vs. out-of-distribution generalization drop when calibrating on generic text and evaluating on specialized medical text.
- **`4_linear_compression_limit/`** — Quantifies coordinate signal smoothness (Mean Squared Difference between adjacent dimensions) before and after TSP coordinate reordering.
- **`5_pca_vs_vectorzip_same_corpus/`** — Evaluates reconstruction cosine similarity across target dimensions ($K \in \{256, 128, 96, 64, 32, 16\}$) when calibrated and tested on the same domain.
- **`6_pca_vs_vectorzip_different_corpus/`** — Evaluates reconstruction cosine similarity across target dimensions when calibrated on generic text and evaluated on medical text.
- **`7_cross_dataset_generalization/`** — Multi-domain generalization benchmark (calibrated on generic text; tested on Medical, Source Code, and Legal corpora at $K \in \{96, 64\}$).
- **`8_search_quality_preservation/`** — Evaluates nearest-neighbor ranking preservation (`nDCG@10`, `Recall@5`, `Recall@10`) of compressed vectors relative to uncompressed ground-truth search.
- **`9_downstream_rag_hit_rate/`** — Measures end-to-end RAG Top-3 Hit Rate at $4\times$ compression ($384\text{D} \to 96\text{D}$).
- **`10_training_overhead_scaling/`** — Measures `.fit()` calibration time scaling across corpus sizes ($M \in \{1000, 2000, 5000, 10000\}$ vectors at $1024\text{D}$).
