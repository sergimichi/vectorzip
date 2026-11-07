# Empirical Benchmarks & Trade-Off Analysis

All benchmarks in this section are reproducible using the scripts in [`benchmarks/`](https://github.com/sergimichi/vectorzip/tree/main/benchmarks). Rather than claiming a single universal winner, VectorZip provides multiple compression backends so engineers can select the optimal point on the **speed, storage, and in-domain vs. out-of-distribution (OOD) fidelity** Pareto frontier.

---

## 1. Projection Throughput & Spectral Fidelity

Measured on 1,000 correlated synthetic vectors per dimensional scale at a **4.0x compression ratio** (`benchmarks/library/run_benchmarks.py`), evaluating reconstruction Mean Squared Error (MSE), cosine similarity retention, and transform latency:

| Original $\to$ Target Dims | Compression Ratio | Reconstruction MSE | Cosine Retention | Transform Latency (1,000 Vectors) |
| :--- | :---: | :---: | :---: | :---: |
| **384D $\to$ 96D** (MiniLM / BGE-Small scale) | 4.0x | `0.000301` | **99.97%** | `~6.3 ms` |
| **768D $\to$ 192D** (BGE-Base / Nomic scale) | 4.0x | `0.000310` | **99.97%** | `~12.3 ms` |
| **1024D $\to$ 256D** (BGE-Large / BGE-M3 scale) | 4.0x | `0.001628` | **99.84%** | `~8.9 ms` |
| **1536D $\to$ 384D** (GTE-Qwen2 / OpenAI-Small scale) | 4.0x | `0.001392` | **99.87%** | `~13.0 ms` |
| **3072D $\to$ 768D** (OpenAI-Large scale) | 4.0x | `0.000470` | **99.95%** | `~33.8 ms` |

At $1024\text{D} \to 384\text{D}$ (`benchmarks/1_encoding_latency_penalty`), pure transform throughput exceeds **120,000 vectors/sec** on CPU (`~8.1 ms` per 1,000 vectors), adding less than `0.015 ms` per sentence to end-to-end transformer encoding.

---

## 2. Vector Search Latency on CPU (384D vs. 64D)

Why combine dimensionality reduction with scalar quantization (`dct+sq8` or `pca+sq8`) instead of applying 8-bit quantization (`sq8`) alone?

We benchmarked exact inner-product top-10 search over **100,000 normalized document vectors** and **1,000 queries** on CPU (`benchmarks/library/search_latency_test.py`):

| Vector Representation | Mean Query Latency | Search Throughput | Speedup |
| :--- | :---: | :---: | :---: |
| **384D Baseline** | `4.48 ms` | **223.0 QPS** | 1.00x |
| **64D Reduced (`dct` / `pca` / `rp` / `matryoshka`)** | **`0.75 ms`** | **1,341.6 QPS** | **6.02x** |

!!! note "Dimensionality Drives Search Speed"
    The **6.02x speedup** is a direct result of reducing inner-product complexity from $D=384$ to $K=64$ ($384 / 64 = 6.0$) and applies equally to any dimensionality reduction backend (`dct`, `pca`, `rp`, or `matryoshka`). Pairing dimensionality reduction with `+sq8` yields both the **6.0x compute speedup** and a **24x memory reduction** ($1,536\text{ bytes} \to 64\text{ bytes}$).

---

## 3. Calibration State Footprint: `dct` vs. `pca`

When deploying calibrated compressors across edge workers or serverless functions, serialization size matters (`benchmarks/2_default_corpus_bias`):

| Metric ($1024\text{D} \to 384\text{D}$) | PCA (`pca`) | VectorZip Spectral (`dct`) | Reduction Factor |
| :--- | :---: | :---: | :---: |
| **Learned Parameters** | $1024 \times 384$ float matrix + mean | $1024$ integer permutation indices | — |
| **Serialized Size** | `1,545.2 KB` (Pickle) | **`4.9 KB` (Pure JSON)** | **314x smaller** |

---

## 4. In-Distribution vs. Out-of-Distribution Trade-Offs

To evaluate how different backends behave on real neural embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384D), we calibrated compressors on a generic corpus and tested both in-distribution and across specialized out-of-distribution (OOD) domains (`benchmarks/3_tsp_scalability`, `benchmarks/7_cross_dataset_generalization`, `benchmarks/8_search_quality_preservation`, `benchmarks/9_downstream_rag_hit_rate`):

### Reconstruction Cosine Similarity ($384\text{D} \to 64\text{D}$, 6x Compression)

| Backend | Generic (In-Distribution) | Medical (OOD) | Code (OOD) | Relative ID $\to$ Medical Drop |
| :--- | :---: | :---: | :---: | :---: |
| **PCA (`pca`)** | **0.8782** | **0.4925** | 0.3924 | -43.9% relative (-0.3858 abs) |
| **Spectral DCT (`dct`)** | 0.6313 | 0.4280 | **0.4269** | **-32.2% relative (-0.2033 abs)** |
| **Random Projection (`rp`)** | 0.4049 | 0.4082 | 0.4170 | ~0.0% (data-independent) |

### Retrieval Ranking Preservation ($384\text{D} \to 96\text{D}$, 4x Compression)

| Backend | RAG Top-3 Hit Rate (`9_downstream_rag_hit_rate`) | Ranking nDCG@10 (`8_search_quality_preservation`) | Recall@10 |
| :--- | :---: | :---: | :---: |
| **Uncompressed 384D Baseline** | 98.0% | 1.000 | 1.00 |
| **PCA (`pca`, 96D)** | **96.0%** | 0.400 | **0.31** |
| **Spectral DCT (`dct`, 96D)** | **96.0%** | **0.445** | 0.29 |

**Engineering Guidelines:**

* **Use `"pca"` (or `"pca+sq8"`)** when you have a representative in-domain calibration sample and want maximum variance retention and reconstruction cosine similarity on that domain.
* **Use `"dct"` (or `"dct+sq8"`)** when you need an ultra-compact JSON calibration state (~5 KB), higher top-rank nDCG@10 preservation, or resilience against severe domain shifts (such as uncalibrated source code or cross-lingual corpora).
* **Use `"matryoshka"`** when your embedding backbone was explicitly trained with Matryoshka Representation Learning (MRL).

---

## 5. High-Capacity Compression vs. Native Small Models

Compressing a large, capable model (`BAAI/bge-m3`, 1024D, 567M parameters, 8K token window, 100+ languages) to 384D allows you to fit within the exact vector database footprint of a small native model (`BAAI/bge-small-en-v1.5`, 384D, 33M parameters, 512 token window) while retaining capabilities that small models lack (`benchmarks/library/`):

* **Long-Context Retrieval (`long_context_benchmark.py`)**: On ~1,150-word documents where the target fact appears past the 512-token mark (middle or end of document), native `BGE-Small` (384D) fails due to context truncation, whereas `BGE-M3` compressed to 384D via VectorZip retrieves the correct document across all positions.
* **Multilingual Disambiguation (`spanish_benchmark.py`, `qwen_spanish_benchmark.py`)**: On Spanish polysemy disambiguation (e.g., *banco* as financial institution vs. bench vs. shoal of fish; *gato* as hydraulic jack vs. feline), compressing `BGE-M3` ($1024\text{D} \to 384\text{D}$) or `GTE-Qwen2-1.5B` ($1536\text{D} \to 512\text{D}$) preserves 100% top-1 disambiguation accuracy where smaller native models misclassify homonyms.
