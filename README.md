# VectorZip

[![PyPI version](https://badge.fury.io/py/vectorzip.svg)](https://pypi.org/project/vectorzip/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**VectorZip** is a lightweight, unified Python library for post-hoc embedding compression in Retrieval-Augmented Generation (RAG) and semantic search pipelines. It compresses dense floating-point embeddings by **4x to 48x** via orthonormal spectral projections, low-rank linear projections, and 8-bit quantization—reducing vector database memory footprints and accelerating brute-force / HNSW distance calculations without retraining or fine-tuning the underlying encoder.

---

## Resources

* **Documentation & Website**: [https://sergimichi.github.io/vectorzip/](https://sergimichi.github.io/vectorzip/)
* **Architecture & Mathematical Foundations**: [https://sergimichi.github.io/vectorzip/how-it-works/](https://sergimichi.github.io/vectorzip/how-it-works/)
* **Empirical Benchmarks**: [https://sergimichi.github.io/vectorzip/benchmarks/](https://sergimichi.github.io/vectorzip/benchmarks/)
* **API Reference**: [https://sergimichi.github.io/vectorzip/api/](https://sergimichi.github.io/vectorzip/api/)
* **Source Repository**: [https://github.com/sergimichi/vectorzip](https://github.com/sergimichi/vectorzip)

---

## Installation

```bash
pip install vectorzip
```

---

## Compression Backends & Engineering Trade-offs

VectorZip exposes six compression backends and multiplicative hybrid modes (`*+sq8`) under a unified scikit-learn-style API (`VectorZip`) and a drop-in model wrapper (`VectorZipModel`):

| Compression Backend | Method String | Target Use Case | Storage Reduction | Calibration State Size (1024D $\to$ 384D) | Key Trade-off |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **TSP-Reordered DCT** | `"dct"` | Cross-domain / OOD generalization | 2x – 16x | **~5 KB** (1D permutation) | Universal basis avoids overfitting to calibration domain covariance |
| **Principal Component Analysis** | `"pca"` | Closed-domain / In-distribution | 2x – 16x | ~1.5 MB ($N \times K$ matrix) | Optimal variance retention in-domain; sensitive to distribution shift |
| **Gaussian Random Projection** | `"rp"` | Data-free / Zero-calibration | 2x – 16x | ~1.5 MB ($N \times K$ matrix) | Data-oblivious JL distance preservation |
| **Matryoshka Slicing** | `"matryoshka"` | MRL-trained models (e.g., Nomic, OpenAI) | 2x – 24x | **0 KB** (stateless slice) | Zero compute overhead; requires an MRL-trained backbone |
| **Product Quantization** | `"pq"` | High-ratio storage compression | 8x – 64x | ~8 KB (codebook centroids) | Non-linear vector quantization into `uint8` subvector codes |
| **Scalar Quantization** | `"sq8"` | Precision reduction (`float32` $\to$ `uint8`) | 4x | ~8 KB (per-dim min/max) | Near-lossless (>99.9% cosine retention); no dimension reduction |
| **Hybrid (`*+sq8`)** | `"dct+sq8"`, `"pca+sq8"`, etc. | Extreme memory constraints | **8x – 48x** | ~13 KB (`dct+sq8`) | Combines $D/K$ dimension reduction with 4x `uint8` quantization |

*Note on Search Speedup: Reducing embedding dimensionality from $D$ to $K$ (e.g., $384\text{D} \to 64\text{D}$) reduces exact inner-product FLOPs proportionally, yielding an empirical **6.02x CPU search throughput speedup** (1,341 QPS vs. 223 QPS over 100,000 vectors) across all dimension-reducing backends (`dct`, `pca`, `rp`, `matryoshka`).*

---

## Quickstart

### 1. Drop-In Model Wrapper (`VectorZipModel`)

`VectorZipModel` wraps `SentenceTransformer`, LangChain, LlamaIndex, or custom embedding callables, handling calibration, compression, and pure-JSON state serialization:

```python
from vectorzip import VectorZipModel

# Wrap an encoder, specify target dimensions, and choose a backend
model = VectorZipModel("BAAI/bge-m3", n_components=128, method="dct+sq8")

# Calibrate on a representative corpus and encode documents into uint8 vectors
documents = [
    "VectorZip enables post-hoc embedding compression.",
    "Dimensionality reduction accelerates vector similarity search.",
]
model.fit(documents)
compressed_docs = model.encode(documents)

# Save pure-JSON calibration state (no pickle required)
model.save_pretrained("./vectorzip_calibrated_model")
```

In production, load the pre-calibrated configuration to compress incoming queries deterministically:

```python
from vectorzip import VectorZipModel

production_model = VectorZipModel.from_pretrained("./vectorzip_calibrated_model")
query_vector = production_model.encode(["Fast vector similarity search"])
```

### 2. Low-Level Estimator API (`VectorZip`)

Use `VectorZip` directly on pre-computed NumPy embedding matrices:

```python
import numpy as np
from vectorzip import VectorZip

embeddings = np.random.randn(1000, 1024).astype(np.float32)

vz = VectorZip(n_components=256, method="dct")
compressed = vz.fit_transform(embeddings)      # Shape: (1000, 256)
reconstructed = vz.inverse_transform(compressed)  # Shape: (1000, 1024)
```

---

## Repository Structure

```text
vectorzip/       # Installable Python package (core estimators & universal model wrapper)
tests/           # Unit, integration, thread-safety, and numerical edge-case test suites
benchmarks/
  library/       # Benchmark scripts for throughput, baseline comparisons, and RAG demos
  results/       # JSON benchmark outputs
  1_..10_*/      # Reproducible ablation studies (latency, memory footprint, OOD, ranking)
docs/            # MkDocs Material documentation site source
scripts/         # Pre-publish verification and build pipeline
```

---

## Running Tests

```bash
python tests/test_vectorzip.py
python tests/test_extreme_edge_cases.py
```

---

## License

VectorZip is released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).
