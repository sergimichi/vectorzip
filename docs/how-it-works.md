# The Architecture of VectorZip

Deploying lower-dimensional embedding representations (e.g., 384 or 512 dimensions) is crucial for resource-constrained vector database applications. However, native low-dimensional models in a given family (e.g., BGE-Small) are often severely capacity-limited, lacking the parameters required to represent multilingual structures or extended contexts.

By applying VectorZip's post-hoc spectral compression to a high-capacity model of the same family (e.g., BGE-M3 or Qwen2.5-Large), the compressed vector inherits the semantic properties and parametric knowledge of the high-capacity backbone. This approach achieves superior downstream retrieval accuracy compared to native small models, bypassing the necessity of computationally intensive training or fine-tuning of low-capacity architectures.

## The Two-Stage Compression Process

VectorZip optimizes vector compression through a two-stage spectral alignment:

### 1. Dimensional Reordering (TSP)
NLP dense vectors do not possess a natural ordering like time-series or images, meaning high-frequency noise is arbitrarily scattered across dimensions. 
VectorZip solves this by formulating dimension ordering as a **Traveling Salesperson Problem (TSP)** over the empirical covariance matrix. By clustering highly correlated dimensions together, we artificially maximize "signal smoothness."

### 2. Spectral Projection (DCT-II)
Once the signal is "smoothed", it becomes highly susceptible to spectral compression. We project the reordered signals using the orthonormal Type-II Discrete Cosine Transform (DCT-II). Because the signal is smooth, the DCT concentrates semantic information almost entirely into the low-frequency coefficients, which are then truncated to the target dimensionality.

Because the transformation is purely linear and orthonormal before truncation, cosine similarity is remarkably preserved.
