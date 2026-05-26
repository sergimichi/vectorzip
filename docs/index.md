# VectorZip Quickstart

VectorZip is a high-performance Python library designed to optimize vector database storage and search latency in Retrieval-Augmented Generation (RAG) pipelines.

## Installation

```bash
pip install vectorzip
```

## Quick Start (Drop-in Wrapper)

VectorZip provides a high-level wrapper, `VectorZipModel`, that acts as a direct, drop-in replacement for standard `SentenceTransformer` models. It automates vector compression and calibration transparently:

```python
from vectorzip import VectorZipModel

# 1. Instantiate the model wrapper 
# (e.g., project 1024-dimensional BGE-M3 down to 384 dimensions)
model = VectorZipModel("BAAI/bge-m3", n_components=384)

# 2. Encode sentences directly 
# (automatically calibrates on the first batch and returns 384-dim embeddings)
compressed_embeddings = model.encode([
    "The European Central Bank reduced interest rates.",
    "A school of tropical fish swam quickly."
])

# 3. Retrieve high-dimensional reconstructed embeddings if needed
reconstructed_embeddings = model.encode(
    ["The European Central Bank reduced interest rates."],
    decompress=True
)
```

And that's it! Your vector embeddings are now 4x smaller but retain over 99% of their semantic meaning.
