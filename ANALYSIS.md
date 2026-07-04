# VectorZip DCT Compression — Paper Feasibility Analysis

A thorough audit of the code, benchmark claims, and academic novelty of the TSP-reordered DCT embedding compression method, with recommendations for turning it into a publishable paper.

---

## Code & Claims Audit

### Critical Issues Found

**1. TSP distance metric: docs vs. code mismatch**

The docs (`docs/how-it-works.md:40`) claim the TSP distance is correlation-based:

```
D_{i,j} = 1 - |Σ_{i,j}| / sqrt(Σ_{i,i} · Σ_{j,j})
```

But the actual code (`compressor.py:339-342`) computes **squared Euclidean distance** between dimension columns:

```python
mean_sq = np.mean(X ** 2, axis=0)
dot_product = np.dot(X.T, X) / M
dist_matrix = mean_sq[:, np.newaxis] + mean_sq[np.newaxis, :] - 2 * dot_product
```

This is `||x_i - x_j||²`, NOT `1 - |correlation|`. The docs describe a different algorithm than what's implemented. **This must be reconciled before publishing.**

**2. The "6.02x speedup" is not DCT-specific**

`search_latency_test.py` uses **synthetic random vectors** (not real embeddings) and just compares 384D vs 64D brute-force dot products. The 6.02× is simply 384/64 ≈ 6, the dimensional ratio. This speedup applies to **any** dimensionality reduction method (PCA, RP, etc.). The README reuses this number for every method row, which is misleading.

**3. OOD "robustness" claim is misleading**

The docs claim DCT is "1.9× more robust to domain shifts" with a "20.3% drop" vs PCA's "38.6% drop". But the actual results (`7_cross_dataset_generalization/results.json`) show:

| | In-Distribution (Wiki) | Medical OOD | Absolute OOD Performance |
|---|---|---|---|
| PCA (64D) | 0.878 | 0.492 | **0.492** |
| DCT (64D) | 0.631 | 0.428 | **0.428** |

**PCA is better in absolute OOD performance.** The "percentage drop" framing favors DCT only because it starts from a much lower base (0.631 vs 0.878). A reviewer will catch this immediately.

**4. DCT is consistently worse than PCA in reconstruction fidelity**

`5_pca_vs_vectorzip_same_corpus/results.json`:

| Target Dims | PCA CSR | DCT CSR | Delta |
|---|---|---|---|
| 256 | 0.999 | 0.916 | -0.084 |
| 128 | 0.985 | 0.787 | -0.198 |
| 96 | 0.969 | 0.734 | -0.235 |
| 64 | 0.936 | 0.664 | -0.272 |
| 32 | 0.859 | 0.554 | -0.306 |
| 16 | 0.771 | 0.425 | -0.346 |

PCA dominates at every compression level, and the gap **widens** as compression increases.

**5. Product Quantization also outperforms DCT**

`baseline_comparison_results.json` shows PQ achieves 0.910 in-distribution CSR vs DCT's 0.631, and 0.475 medical OOD vs DCT's 0.428.

**6. Downstream benchmarks are too small for statistical significance**

- `spanish_benchmark.py`: **3 queries, 6 documents** → "100% accuracy"
- `long_context_benchmark.py`: **1 query, 3 documents** per test → "100% accuracy"
- `qwen_spanish_benchmark.py`: **3 queries, 6 documents** → "100% accuracy"

These sample sizes cannot support publication claims.

**7. Search quality is mixed**

`8_search_quality_preservation/results.json` (96D, 50 queries):

- Recall@5: PCA 0.26 vs DCT 0.22 (PCA wins)
- Recall@10: PCA 0.31 vs DCT 0.29 (PCA wins)
- NDCG@10: PCA 0.40 vs DCT 0.44 (DCT wins)

---

## Novelty Assessment (Literature Search)

**The TSP dimension reordering + DCT-II pipeline for post-hoc embedding compression appears genuinely novel.** No published paper combines these two ideas for this purpose.

### Closest related work:

| Paper | Venue | What it does | Key difference |
|---|---|---|---|
| **AxisTour** (Yamagiwa et al.) | EMNLP 2024 | TSP over embedding axes for interpretability | No DCT; merges axes instead of spectral truncation |
| **SCDTour** (Aida & Bollegala) | EMNLP 2025 | TSP axis ordering for semantic change detection | No spectral transform; goal is interpretability |
| **Salama et al.** | 2025 | DWT for embedding compression (50-93% reduction) | No dimension reordering; doesn't address the ordering problem |
| **SpecTemp** (Li et al.) | SIGIR 2026 | Adaptive spectral scaling on eigenvalues | Operates on eigenspectrum, not DCT; no reordering |
| **Almarwani et al.** | EMNLP 2019 | DCT for *constructing* sentence embeddings | Not post-hoc compression |
| **MRL** (Kusupati et al.) | NeurIPS 2022 | Training-time nested embeddings | Requires retraining |
| **Matryoshka-Adaptor** (Yoon et al.) | EMNLP 2024 | Post-hoc adapter to create MRL embeddings | Requires training an adapter |
| **PCA-RAG** | 2025 | PCA + PQ pipeline, 48-96× compression | No spectral reordering |

### Key gap this work fills:

The **dimension ordering problem** — the insight that embedding dimensions are unordered artifacts of weight initialization, making naive spectral transforms fail — is not addressed in any prior work. Salama et al. apply DWT directly without reordering and never discuss this. The TSP reordering to create a smooth pseudo-signal before DCT is the novel contribution.

---

## Recommendations for the Paper

### What's honestly publishable:

1. **The TSP reordering insight** — identifying that embedding dimensions are unordered and that spectral transforms fail without reordering is a genuine contribution
2. **The multilingual/capacity argument** — compressing a 1024D multilingual model (BGE-M3) to 384D preserves capabilities that a native 384D monolingual model (BGE-Small) lacks. This is the strongest selling point and aligns with the stated motivation.
3. **The long-context preservation** — compressing a model with 8K context to 384D retains context awareness that a 512-token native small model loses
4. **Training-free post-hoc compression** — unlike MRL/Matryoshka-Adaptor, no training or fine-tuning needed
5. **Orthonormality preservation** — DCT-II preserves dot products and norms, so cosine similarity is approximately invariant

### What needs fixing before submission:

1. **Reconcile the distance metric** — either implement the correlation-based distance from the docs, or update the docs to match the Euclidean distance in the code. This is a reviewer red flag.
2. **Run on standard benchmarks** — MTEB/BEIR suite (SciFact, NFCorpus, FiQA, MS MARCO, NQ) with proper metrics (nDCG@10, Recall@k). Current benchmarks use synthetic data and tiny samples.
3. **Be honest about absolute vs. relative OOD performance** — don't frame "smaller percentage drop" as superiority when absolute OOD performance is lower than PCA. Instead, frame it as "the DCT basis is universal (not dataset-fitted), so it degrades more gracefully relative to its in-distribution performance."
4. **Drop or qualify the 6.02× speedup claim** — this is just the dimensional ratio, not DCT-specific. Any method reducing 384→64 gets this.
5. **Scale up the multilingual/long-context benchmarks** — 3 queries/6 documents is not publishable. Use MIRACL (multilingual retrieval) or a proper Spanish retrieval dataset.
6. **Compare against all relevant baselines** — PCA, PQ, random projection, MRL (if models support it), SpecTemp, and naive truncation. The current code has these backends but benchmarks don't compare them all systematically.
7. **Ablation study** — show DCT without TSP reordering vs. DCT with TSP reordering to demonstrate the reordering's contribution. The `4_linear_compression_limit/results.json` partially does this (raw original order vs TSP order) but needs to be formalized.
8. **Use proper statistical testing** — report confidence intervals or run significance tests

### Suggested paper structure:

1. **Introduction** — the capacity bottleneck of small models (multilingual, long-context)
2. **Related Work** — MRL, PCA-RAG, SpecTemp, AxisTour/SCDTour, DWT compression
3. **Method** — TSP reordering (with CORRECT distance metric) + DCT-II truncation
4. **Theoretical motivation** — energy compaction, orthonormality, cosine similarity preservation
5. **Experiments** — MTEB/BEIR benchmarks, multilingual (MIRACL), long-context, OOD generalization
6. **Ablations** — TSP vs no-TSP, distance metric choice, compression ratio sweep
7. **Analysis** — when DCT wins vs loses vs PCA, the universality-vs-fidelity tradeoff

### Honest framing of the contribution:

The paper's thesis should be: *"Post-hoc spectral compression via TSP-reordered DCT enables using high-capacity multilingual/long-context models under tight dimensional budgets, preserving capabilities that native small models lack — at the cost of lower absolute reconstruction fidelity than PCA."* This is honest, novel, and practically useful.

---

## Bottom line

The **idea is novel and publishable**. The **execution and benchmarking need significant work** before it can survive peer review. The biggest risks are: (1) the docs/code metric mismatch, (2) PCA outperforming DCT in absolute terms on every reconstruction benchmark, and (3) tiny sample sizes on the headline multilingual/long-context claims. Fix those three things, run on BEIR/MTEB, and you have a credible paper — especially if positioned around the multilingual/capacity-preservation angle rather than "better than PCA at compression."
