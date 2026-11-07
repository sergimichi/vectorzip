# Architecture & Methodology

VectorZip is a unified embedding compression library designed to bridge the gap between high-capacity contextual encoders (such as `BAAI/bge-m3` or `GTE-Qwen2`) and the memory and latency budgets of production vector databases.

This page details the mathematical foundations and engineering trade-offs of the algorithms implemented in VectorZip, with a focus on our **TSP-Reordered Discrete Cosine Transform (`dct`)** pipeline.

---

## 1. The Capacity vs. Dimensionality Trade-Off

In Retrieval-Augmented Generation (RAG) architectures, reducing embedding dimensionality directly lowers RAM consumption and accelerates inner-product distance computations. A common practice is to deploy small native models (e.g., `BGE-Small-en-v1.5` at 384 dimensions, 33M parameters).

However, small native models have limited parameter capacity, often lacking the multilingual coverage and long-context window (e.g., 8,192 tokens) of larger backbones.

VectorZip leverages **Post-Hoc Parametric Transfer**:
> *By applying post-hoc dimensionality reduction to a high-capacity encoder (e.g., 1024D $\to$ 384D), the compressed representation retains the multilingual alignment and long-context reasoning of the large backbone while fitting within the storage and latency budget of a small model.*

---

## 2. The Dimension Ordering Problem in Neural Embeddings

Unlike audio waveforms or natural images, **dense neural embeddings have no inherent spatial or frequency ordering**. The coordinate sequence $[d_1, d_2, \dots, d_N]$ in an embedding vector is an arbitrary permutation determined by weight initialization during pre-training.

When viewed as a 1D discrete signal across coordinates, a raw embedding vector behaves like high-frequency white noise. Applying a frequency transform such as the Discrete Cosine Transform (DCT) directly to unordered coordinates yields poor energy compaction in the low-frequency coefficients.

VectorZip addresses this via a two-stage pipeline: **correlation-based TSP coordinate reordering** followed by **orthonormal DCT-II projection**.

---

## 3. Stage 1: TSP Coordinate Reordering

To maximize smoothness across adjacent coordinates before applying a spectral transform, VectorZip reorders embedding dimensions so that correlated features are placed next to one another.

We formulate coordinate ordering as a **Traveling Salesperson Problem (TSP)** over the $N$ feature dimensions:

1. **Standardized Correlation Matrix**: Given a calibration matrix $X \in \mathbb{R}^{M \times N}$, each dimension is standardized to zero mean and unit variance to compute the scale-invariant empirical correlation matrix $R \in \mathbb{R}^{N \times N}$.
2. **Correlation Distance Metric**: The distance between dimensions $i$ and $j$ is defined as:
   \[ D_{i,j} = 1 - |R_{i,j}| = 1 - \frac{|\Sigma_{i,j}|}{\sqrt{\Sigma_{i,i} \cdot \Sigma_{j,j}}} \]
3. **Heuristic Local Search**: We construct an initial tour using a vectorized Nearest-Neighbor greedy heuristic, then refine the permutation $P$ via **2-opt** segment reversals and **Or-opt** segment relocations (relocating contiguous blocks of length 1–3).

The resulting permutation $P$ is a 1D integer array of length $N$. Unlike PCA—which requires storing and multiplying a dense $N \times K$ floating-point matrix (~1.5 MB for $1024 \times 384$)—the TSP permutation serializes to **under 5 KB in pure JSON**.

---

## 4. Stage 2: Orthonormal DCT-II Projection

Once the input vector $x \in \mathbb{R}^N$ is reordered via permutation $P$ into $x^{(P)}$, VectorZip projects the signal using the orthonormal **Type-II Discrete Cosine Transform (DCT-II)**:

\[ C_k = \sqrt{\frac{2}{N}} \, \alpha_k \sum_{n=0}^{N-1} x^{(P)}_n \cos \left[ \frac{\pi}{N} \left( n + \frac{1}{2} \right) k \right], \quad k = 0, \dots, N-1 \]

where $\alpha_0 = \frac{1}{\sqrt{2}}$ and $\alpha_k = 1$ for $k > 0$.

### Truncation and Inverse Reconstruction
To compress the vector to $K$ components ($K < N$), we retain the first $K$ low-frequency coefficients $[C_0, C_1, \dots, C_{K-1}]$. When reconstructing back to $\mathbb{R}^N$ via `inverse_transform`, VectorZip pads the truncated tail with zeros, optionally applies a raised-cosine smooth taper near the cutoff frequency $K$ to suppress Gibbs ringing artifacts, applies the Inverse DCT-II (`idct`), and inverts the permutation $P^{-1}$.

### Orthonormality & Domain Generalization
Because the DCT-II basis functions are fixed, data-independent cosine waveforms rather than corpus-specific covariance eigenvectors:

* **In-Distribution (Closed Domain)**: PCA achieves higher variance retention and reconstruction cosine similarity on the exact distribution it was calibrated on (by the Eckart–Young–Mirsky theorem).
* **Out-of-Distribution (Domain Shift)**: PCA projections can overfit to dominant variance directions of the calibration corpus. Because DCT projects onto a fixed orthonormal basis after coordinate ordering, it exhibits lower relative sensitivity to domain shifts between calibration and deployment corpora.
