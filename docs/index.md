# VectorZip Quickstart

VectorZip is a high-performance Python library designed to optimize vector database storage and search latency in Retrieval-Augmented Generation (RAG) pipelines.

## Installation

```bash
pip install vectorzip
```

## Quick Start (Drop-in Wrapper)

VectorZip provides a high-level wrapper, `VectorZipModel`, that acts as a direct, drop-in replacement for standard `SentenceTransformer` models. The learning curve is zero:

### Before: Standard SentenceTransformers
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(["Hello World"])
```

### After: VectorZip (4x smaller embeddings)
Simply swap the import, add the `n_components` target, and VectorZip handles the complex spectral calibration under the hood automatically:

```python
from vectorzip import VectorZipModel

model = VectorZipModel("BAAI/bge-m3", n_components=384)
embeddings = model.encode(["Hello World"])

# Need the original 1024-dimensional vectors back?
reconstructed = model.encode(["Hello World"], decompress=True)
```

And that's it! Your vector embeddings are now 4x smaller but retain over 99% of their semantic meaning.
