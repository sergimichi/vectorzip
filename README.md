# VectorZip

[![PyPI version](https://badge.fury.io/py/vectorzip.svg)](https://pypi.org/project/vectorzip/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

VectorZip is a high-performance Python library designed to optimize vector database storage and query latency in Retrieval-Augmented Generation (RAG) pipelines. By applying post-hoc mathematical projections to high-dimensional embedding outputs, VectorZip shrinks vector footprints by **4x to 48x** and accelerates CPU search speeds by up to **6x**, with negligible loss in semantic retrieval fidelity.

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

## Compression Modes & Performance Matrix

VectorZip integrates multiple compression backends under a single unified, easy-to-use API. Choose the method that best matches your target model and database constraints:

| Compression Mode | Method String | Target Model Type | Compression Factor | Search Speedup (CPU) | Retrieval Fidelity |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Spectral DCT** (Flagship) | `"dct"` | Standard / Non-Matryoshka | 4x - 12x | **6.02x** | **High** |
| **Matryoshka Slicing** | `"matryoshka"` | Native Matryoshka models | 4x - 24x | **6.02x** | **High** |
| **Principal Component Analysis** | `"pca"` | Closed-domain restricted | 4x - 12x | **6.02x** | **High** (Local) |
| **Product Quantization** | `"pq"` | General dense vectors | 8x - 16x | 4.20x | Moderate |
| **Scalar Quantization** | `"sq8"` | General dense vectors | 4x | 1.0x (Baseline) | **High** |
| **Hybrid Spectral+SQ8** | `"dct+sq8"` | Standard / Non-Matryoshka | **12x - 48x** | **6.02x** | **High** |

*Note: Search Speedup is measured over a database of 100,000 document vectors on CPU compared to raw high-dimensional scalar quantized (SQ8) search.*

---

## 1-Minute Quickstart

VectorZip provides a high-level model wrapper, `VectorZipModel`, that acts as a direct, drop-in replacement for standard `SentenceTransformer` models, automating offline calibration and online query compression:

### 1. Calibrate and Compress Database Index
```python
from vectorzip import VectorZipModel

# Wrap your model, specify target dimensions, and choose your method
model = VectorZipModel("BAAI/bge-m3", n_components=128, method="dct+sq8")

# Calibrate and compress your corpus (calibration runs automatically on first batch)
documents = ["VectorZip enables high-performance compression.", "Vector search is now 6x faster."]
compressed_embeddings = model.encode(documents)

# Save the calibrated JSON configuration for production
model.save_pretrained("./vectorzip_calibrated_model")
```

### 2. Compress Online Queries in Production
```python
from vectorzip import VectorZipModel

# Load the pre-calibrated JSON configuration in your production environment
production_model = VectorZipModel.from_pretrained("./vectorzip_calibrated_model")

# Compress query embedding instantly to uint8 for database search
query_vector = production_model.encode(["Fast vector search"])
```

---

## License

VectorZip is released under the Apache 2.0 License.
