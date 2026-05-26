# VectorZip Quickstart

VectorZip is a high-performance Python library designed to optimize vector database storage and search latency in Retrieval-Augmented Generation (RAG) pipelines.

## Installation

```bash
pip install vectorzip
```

## Quick Start (SentenceTransformers)

VectorZip provides a high-level wrapper, `VectorZipModel`, that acts as a direct, drop-in replacement for standard `SentenceTransformer` models. The learning curve is zero:

```python
from vectorzip import VectorZipModel

# Simply wrap your model name, specify components, and select your method
model = VectorZipModel("BAAI/bge-m3", n_components=384, method="dct+sq8")
embeddings = model.encode(["Hello World"])

# Need the original 1024-dimensional vectors back?
reconstructed = model.encode(["Hello World"], decompress=True)
```

Your vector embeddings are now **up to 12x to 48x smaller** (retaining over 99% semantic similarity) thanks to the hybrid combination of dimensionality reduction and Scalar Quantization.

---

## Universal Wrapping Capabilities

`VectorZipModel` natively wraps models and APIs across all mainstream embedding frameworks. The wrapper automatically resolves the underlying framework and applies zero-overhead autocalibration:

### 1. LangChain Embeddings
```python
from langchain_openai import OpenAIEmbeddings
from vectorzip import VectorZipModel

langchain_base = OpenAIEmbeddings(model="text-embedding-3-small")

# Wraps natively with zero configuration
model = VectorZipModel(langchain_base, n_components=128, method="dct+sq8")
compressed = model.encode(["Compressing LangChain document."])
```

### 2. LlamaIndex Embeddings
```python
from llama_index.embeddings.openai import OpenAIEmbedding
from vectorzip import VectorZipModel

llamaindex_base = OpenAIEmbedding(model="text-embedding-3-small")

model = VectorZipModel(llamaindex_base, n_components=128, method="pca+sq8")
compressed = model.encode(["Compressing LlamaIndex document."])
```

### 3. Custom Callables (APIs & Lambdas)
```python
from vectorzip import VectorZipModel

def custom_api_caller(texts):
    # Call any custom API (Gemini, Cohere, etc.) returning list of lists
    return [[0.1] * 768 for _ in texts]

model = VectorZipModel(custom_api_caller, n_components=64, method="dct+sq8")
compressed = model.encode(["Compressing custom API call."])
```

---

## The 6-Backend Compression Catalog

VectorZip is a unified vector compression toolbox. Choose from 6 core projection backends and their hybrid configurations:

| Backend Method | Method String | Description |
| :--- | :--- | :--- |
| **Discrete Cosine Transform** | `"dct"` | Reorders features via 2-opt TSP solving, then projects onto DCT-II space. |
| **Principal Component Analysis** | `"pca"` | Projects onto empirical covariance eigenvectors (optimal local manifolds). |
| **Gaussian Random Projection** | `"rp"` | Post-hoc random orthographic projection. |
| **Product Quantization** | `"pq"` | Vector quantization mapping partitions to subvector KMeans codebook centroids. |
| **Scalar Quantization** | `"sq8"` | Quantizes coordinate float32 dimensions into 8-bit integers (`uint8`). |
| **Matryoshka / MRL Slicing** | `"matryoshka"` / `"mrl"` | Static truncation of first $K$ dimensions from models trained using MRL. |

### Hybrid Multiplicative Modes
Append `+sq8` to any dimensionality reduction backend (e.g. `dct+sq8`, `pca+sq8`, `matryoshka+sq8`) to perform dimension reduction **followed by** scalar quantization, enabling extreme compression factors:

* **Example:** Reducing a 768 float32 embedding to 64 dimensions with `matryoshka+sq8` results in a **48x total size reduction** ($3072 \rightarrow 64$ bytes) with virtually no loss in search precision.
