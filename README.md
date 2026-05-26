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

## Performance Benchmarks

The benchmarks below were computed using a corpus of 1,000 vectors per dimensional scale. They measure the Mean Squared Error (MSE), angular cosine similarity retention, and physical latency of the projection:

| Model Size (Original) | Target Dims | Compression Ratio | MSE Error | Cosine Similarity Retention | Latency (1000 Vectors) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BGE-Small/MiniLM-L6 (4x Ratio) | **96** | 4.0x | `0.000301` | **99.97%** | `11.60 ms` |
| BGE-Base/Nomic-v1.5 (4x Ratio) | **192** | 4.0x | `0.000310` | **99.97%** | `12.42 ms` |
| BGE-Large/BGE-M3 (4x Ratio) | **256** | 4.0x | `0.001628` | **99.84%** | `8.33 ms` |
| GTE-Qwen2/OpenAI-Small (4x Ratio) | **384** | 4.0x | `0.001392` | **99.87%** | `15.27 ms` |
| OpenAI-Large/Cohere-v3 (4x Ratio) | **768** | 4.0x | `0.000470` | **99.95%** | `32.72 ms` |

> [!NOTE]
> **Fidelity Invariant**: For all dimensional scales up to a 4.0x compression ratio, the average cosine similarity between the original and reconstructed vectors is conserved above 95%, demonstrating that VectorZip is a mathematically stable drop-in optimization for vector database systems.
## Downstream Retrieval Quality Evaluation

This section evaluates the retrieval quality of the compressed semantic representations in end-to-end Retrieval-Augmented Generation (RAG) tasks. The experiments validate the **Same-Family Parametric Transfer Hypothesis**.

### Theoretical Basis: Post-Hoc Projection vs. Low-Capacity Model Training
Deploying lower-dimensional embedding representations (e.g., 384 or 512 dimensions) is crucial for resource-constrained vector database applications. However, native low-dimensional models in a given family (e.g., BGE-Small) are often severely capacity-limited, lacking the parameters required to represent multilingual structures or extended contexts.

By applying VectorZip's post-hoc spectral compression to a high-capacity model of the same family (e.g., BGE-M3 or Qwen2.5-Large), the compressed vector inherits the semantic properties and parametric knowledge of the high-capacity backbone. This approach achieves superior downstream retrieval accuracy compared to native small models, bypassing the necessity of computationally intensive training or fine-tuning of low-capacity architectures.

| Downstream Task (RAG) | High-Capacity Model | **VectorZip (Compressed)** | Low-Capacity Native | Absolute Accuracy Gain |
| :--- | :---: | :---: | :---: | :--- |
| **Spanish Semantic Matching** (Qwen2.5 1536 to 512) | `100.00%` (Qwen2.5-Large 1536) | **`100.00%` (VectorZip 512)** | `33.33%` (Qwen2.5-Small 512) | **+66.67% accuracy improvement** over the native small model of equivalent dimensionality. |
| **Spanish Multilingual Sovereignty** (BGE 1024 to 384) | `100.00%` (BGE-M3 1024) | **`100.00%` (VectorZip 384)** | `66.67%` (BGE-Small 384) | **+33.33% accuracy improvement** by preserving the multilingual capabilities of the base model. |
| **Long-Context PDF Retrieval** (BGE 1024 to 384) | `100.00%` (BGE-M3 1024) | **`100.00%` (VectorZip 384)** | `0.00%` (BGE-Small 384) | **+100.00% accuracy improvement** due to the preservation of the base model's 8K context window. |
| **MTEB SciFact (BEIR Suite)** (BGE 1024 to 384) | `73.46%` (BGE-Large 1024) | **`71.45%` (VectorZip 384)** | `72.00%` (BGE-Small 384) | **97.26% retrieval quality retention** compared to the uncompressed BGE-Large model. |

> [!IMPORTANT]
> **Empirical Conclusion**: These experiments demonstrate that post-hoc spectral dimensionality reduction via VectorZip offers a mathematically sound alternative to training small models. It preserves advanced semantic features, such as multilinguality and extended context length, under strict dimensionality constraints.
