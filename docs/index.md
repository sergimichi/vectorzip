# VectorZip Quickstart

**VectorZip** is a lightweight, unified Python library designed to reduce vector database storage footprints and accelerate similarity search latency in Retrieval-Augmented Generation (RAG) pipelines.

## Installation

```bash
pip install vectorzip
```

## Quick Start (`SentenceTransformers`)

VectorZip provides a high-level wrapper, `VectorZipModel`, that acts as a direct drop-in replacement for `SentenceTransformer` models:

```python
from vectorzip import VectorZipModel

# Wrap your model, specify target components, and select a compression backend
model = VectorZipModel("BAAI/bge-m3", n_components=384, method="dct+sq8")
embeddings = model.encode(["Hello World"])

# Reconstruct approximate vectors in the original 1024D space when needed
reconstructed = model.encode(["Hello World"], decompress=True)
```

By combining dimensionality reduction ($1024\text{D} \to 384\text{D}$ or $128\text{D}$) with 8-bit scalar quantization (`+sq8`), vector footprints shrink by **10x to 48x** while preserving downstream retrieval quality.

---

## Universal Framework Wrapping

`VectorZipModel` natively wraps models and APIs across mainstream embedding frameworks. It automatically detects the underlying encoding interface (`encode`, `embed_documents`, `get_text_embeddings`, `embed`, or any Python callable):

### 1. LangChain Embeddings
```python
from langchain_openai import OpenAIEmbeddings
from vectorzip import VectorZipModel

langchain_base = OpenAIEmbeddings(model="text-embedding-3-small")

model = VectorZipModel(langchain_base, n_components=128, method="dct+sq8")
compressed = model.encode(["Compressing a LangChain document."])
```

### 2. LlamaIndex Embeddings
```python
from llama_index.embeddings.openai import OpenAIEmbedding
from vectorzip import VectorZipModel

llamaindex_base = OpenAIEmbedding(model="text-embedding-3-small")

model = VectorZipModel(llamaindex_base, n_components=128, method="pca+sq8")
compressed = model.encode(["Compressing a LlamaIndex document."])
```

### 3. Custom Callables (APIs & Lambdas)
```python
from vectorzip import VectorZipModel

def custom_api_caller(texts):
    # Call any custom embedding endpoint returning a 2D array or list of lists
    return [[0.1] * 768 for _ in texts]

model = VectorZipModel(custom_api_caller, n_components=64, method="dct+sq8")
compressed = model.encode(["Compressing custom API embeddings."])
```

---

## The 6-Backend Compression Catalog

VectorZip provides six core compression backends and their multiplicative hybrid configurations:

| Backend Method | Method String | Description |
| :--- | :--- | :--- |
| **Discrete Cosine Transform** | `"dct"` | Reorders dimensions via correlation-based TSP local search, then projects onto an orthonormal DCT-II basis. |
| **Principal Component Analysis** | `"pca"` | Projects onto the top-$K$ empirical covariance eigenvectors (optimal in-distribution variance retention). |
| **Gaussian Random Projection** | `"rp"` | Data-independent orthonormal random projection preserving pairwise distances via the Johnson-Lindenstrauss lemma. |
| **Product Quantization** | `"pq"` | Partitions vectors into subvectors and quantizes each to nearest $k$-means codebook centroids (`uint8`). |
| **Scalar Quantization** | `"sq8"` | Per-dimension min-max uniform quantization from `float32` into 8-bit unsigned integers (`uint8`). |
| **Matryoshka / MRL Slicing** | `"matryoshka"` / `"mrl"` | Stateless prefix slicing of the first $K$ coordinates for models trained with Matryoshka Representation Learning. |

### Hybrid Multiplicative Modes (`*+sq8`)
Append `+sq8` to any dimensionality reduction backend (`dct+sq8`, `pca+sq8`, `rp+sq8`, `matryoshka+sq8`) to apply dimensionality reduction **followed by** 8-bit scalar quantization:

* **Example:** Reducing a 768D `float32` embedding ($3,072\text{ bytes}$) to 64 dimensions with `+sq8` produces a $64\text{-byte}$ `uint8` vector—a **48x total memory reduction**.
