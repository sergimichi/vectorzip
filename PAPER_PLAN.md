# VectorZip DCT Paper — Research Plan

**Goal**: Produce a paper competitive with top-tier venues (ACL, EMNLP, SIGIR, NeurIPS).
**Timeline**: 4-5 weeks (part-time), 2-3 weeks (full-time).

---

## Phase 1: Experimental Rigor (Week 1-2)

### 1.1 Standardized Benchmarks

**BEIR Suite** (the gold standard for retrieval evaluation):
- SciFact (scientific papers)
- NFCorpus (nutrition/medical)
- FiQA (financial Q&A)
- MS MARCO (web search, passage ranking)
- Natural Questions (open-domain QA)
- ArguAna (argument retrieval)
- TREC-COVID (COVID literature)

**Metrics** (per BEIR convention):
- nDCG@10 (primary)
- Recall@5, Recall@100
- MRR@10

**Compression levels**: K ∈ {full/2, full/4, full/8, full/16, full/32, 64, 32, 16}

### 1.2 Models to Test (minimum 5)

| Model | Dims | Params | Why |
|---|---|---|---|
| BGE-M3 | 1024 | 567M | Multilingual, long-context flagship |
| GTE-Qwen2-1.5B | 1536 | 1.5B | Largest, Chinese+English |
| BGE-Large-en-v1.5 | 1024 | 335M | Standard large English |
| all-MiniLM-L6-v2 | 384 | 23M | Standard small (baseline) |
| Nomic-Embed-v2 | 768 | 137M | MRL-native (compare against native MRL) |
| E5-Mistral-7B | 4096 | 7B | If compute allows — largest decoder-based |

### 1.3 Baselines to Compare (minimum 6)

| Baseline | Type | Implementation |
|---|---|---|
| **Raw (no compression)** | None | Ground truth |
| **PCA** | Linear, data-driven | sklearn PCA |
| **Random Projection** | Linear, data-free | Gaussian RP |
| **Product Quantization** | Quantization | FAISS or custom |
| **Naive truncation** | Slicing | First K dims |
| **Matryoshka (native)** | Training-time | For MRL models (Nomic, E5) |
| **SpecTemp** | Spectral scaling | From SIGIR 2026 repo |
| **DWT (Salama 2025)** | Wavelet | If code available |
| **DCT (ours)** | TSP + spectral | VectorZip |

### 1.4 OOD Generalization Protocol

For each BEIR dataset:
1. **Calibrate** on a generic corpus (Wikipedia or DEFAULT_CORPUS)
2. **Evaluate** on the BEIR dataset (domain-shifted)
3. Also calibrate on the BEIR dataset itself (in-distribution) for comparison
4. Report both ID and OOD numbers

This gives a 7×5×8×2 = **560 cell table** (7 datasets × 5 models × 8 compression levels × 2 conditions).

### 1.5 Statistical Significance

- Run each experiment **5 times** with different random seeds (for RP, PQ)
- Report **mean ± std** for all metrics
- Run **paired t-test** or **Wilcoxon signed-rank** for DCT vs PCA on each dataset
- Report **p-values** in tables

### 1.6 Multilingual Evaluation

**MIRACL** (multilingual retrieval, 18 languages):
- Test on: Spanish, French, German, Chinese, Arabic, Japanese, Hindi
- Calibrate on English Wikipedia
- Compare: compressed BGE-M3 vs native BGE-Small (the capacity argument)
- Metric: nDCG@10

### 1.7 Long-Context Evaluation

- Create documents of varying length (512, 1024, 2048, 4096, 8192 tokens)
- Place critical facts at beginning, middle, end
- Compress BGE-M3 (8K context) to 384D
- Compare against BGE-Small (512 token limit)
- Scale: minimum 100 queries per position

---

## Phase 2: Theoretical Analysis (Week 2-3)

### 2.1 Why Does DCT Work Despite Low Correlation?

This is the biggest open question. Possible explanations to investigate:

**Hypothesis A: TSP preserves local geometry**
- TSP doesn't need high absolute correlation — it needs *relative* correlation structure
- Even |corr| = 0.001 between neighbors might be enough to preserve local ranking
- Test: measure how much local geometry (k-NN overlap) changes after TSP vs random permutation

**Hypothesis B: DCT preserves pairwise distances better than PCA under truncation**
- DCT is orthonormal → preserves dot products exactly for retained coefficients
- PCA is also orthonormal but selects components by variance, not by retrieval relevance
- Test: measure pairwise cosine similarity distortion for top-k neighbors

**Hypothesis C: PCA overfits to dominant variance directions that are irrelevant for retrieval**
- PCA preserves the directions of maximum variance, which may be domain-specific noise
- DCT's frequency truncation is domain-agnostic → preserves a different subspace
- Test: compare what PCA components capture vs what DCT coefficients capture (feature attribution)

**Hypothesis D: Energy compaction isn't the right metric**
- DCT only captures 5.8% of variance in 64 components vs PCA's 66.8%
- But retrieval doesn't need variance preservation — it needs *neighborhood* preservation
- Test: measure neighborhood preservation (R@10 overlap) vs variance captured

### 2.2 Formal Properties to Prove

- **Orthonormality preservation**: DCT-II preserves dot products and norms for retained coefficients (trivial to show, but state formally)
- **Universal basis property**: DCT basis is independent of calibration data (contrast with PCA)
- **TSP permutation generalization**: The learned permutation is a structural property of the model, not the corpus (needs empirical validation)
- **Bounds on cosine similarity distortion**: Derive how much cosine sim can change under DCT truncation (relate to energy in truncated coefficients)

### 2.3 Ablation Studies

| Ablation | What it tests |
|---|---|
| DCT with TSP vs DCT without TSP | Value of dimension reordering |
| DCT with correlation-TSP vs Euclidean-TSP | Distance metric choice |
| DCT with taper vs DCT hard truncation | Tapering effect |
| DCT first-K vs DCT best-K coefficients | Coefficient selection strategy |
| DCT calibrated on N=100 vs N=1000 vs N=10000 | Calibration corpus size sensitivity |
| DCT calibrated on domain A, tested on domain B vs C | Cross-domain generalization |
| TSP permutation from model A applied to model B | Is the permutation model-specific or universal? |

---

## Phase 3: Additional Experiments (Week 3)

### 3.1 Storage & Latency Analysis

| Metric | Measure |
|---|---|
| Compressed vector size (bytes) | Per method, per K |
| Serialization size (calibration JSON) | DCT vs PCA vs PQ |
| Search latency (QPS over 100K, 1M vectors) | Brute force CPU + FAISS |
| Calibration time | Fit time per method |
| Transform time | Per-vector compression time |

### 3.2 The Capacity Argument (key selling point)

Compare:
1. **BGE-M3 (1024D) compressed to 384D** via DCT
2. **BGE-Small (384D) native**

On:
- Multilingual retrieval (MIRACL — Spanish, Chinese, Arabic)
- Long-context retrieval (8K vs 512 token limit)
- Domain-specific retrieval (BEIR)

Hypothesis: compressed large model > native small model at same dimensionality.

### 3.3 Hybrid DCT+SQ8 vs PCA+SQ8

- Compare quantized versions
- Measure compression ratio vs retrieval quality tradeoff
- DCT+SQ8 should achieve 12-48× compression

### 3.4 Scaling Laws

- How does DCT performance change with model dimensionality (384 → 768 → 1024 → 1536 → 4096)?
- Is there a "sweet spot" where DCT starts beating PCA in-distribution?
- Plot: NDCG vs K/D ratio for different model sizes

---

## Phase 4: Paper Writing (Week 4-5)

### 4.1 Structure

```
1. Introduction (1 page)
   - Problem: small models lack capacity for multilingual/long-context
   - Solution: post-hoc compression of large models
   - Limitation of PCA: overfits to calibration corpus
   - Our contribution: TSP-reordered DCT — OOD-robust, training-free

2. Related Work (1 page)
   - Post-hoc embedding compression: PCA, PQ, RP
   - Training-time: MRL, Matryoshka-Adaptor
   - Spectral methods: SpecTemp, DWT (Salama)
   - Dimension ordering: AxisTour, SCDTour, WordTour
   - Position our work: first to combine TSP reordering + DCT for post-hoc compression

3. Method (1.5 pages)
   3.1 The dimension ordering problem
   3.2 TSP reordering (correlation-based distance, NN + 2-opt + Or-opt)
   3.3 DCT-II projection with truncation
   3.4 Orthonormality and cosine similarity preservation
   3.5 Complexity analysis

4. Theoretical Analysis (1 page)
   4.1 Why DCT is OOD-robust (universal basis argument)
   4.2 Why DCT can outperform PCA for retrieval (neighborhood vs variance)
   4.3 Bounds on similarity distortion

5. Experiments (3 pages)
   5.1 Setup (models, datasets, metrics)
   5.2 Main results: BEIR (ID and OOD)
   5.3 Multilingual results (MIRACL)
   5.4 Long-context results
   5.5 The capacity argument (compressed large vs native small)
   5.6 Ablation studies
   5.7 Storage and latency

6. Analysis (0.5 pages)
   - When DCT wins vs loses
   - Scaling laws
   - The TSP permutation is model-specific (transferability)

7. Conclusion (0.5 pages)

Appendix:
   - Full BEIR tables
   - Statistical significance tests
   - Hyperparameter sensitivity
```

### 4.2 Target Venues (in order of fit)

| Venue | Deadline (typical) | Fit |
|---|---|---|
| **SIGIR 2027** | Jan 2027 | Best fit — retrieval-focused |
| **EMNLP 2026** | June 2026 | Good — NLP embeddings |
| **ACL 2027** | Feb 2027 | Good — general NLP |
| **NeurIPS 2026** | May 2026 | Possible — if theoretical contribution is strong |
| **ECIR 2027** | Sep 2026 | Safety venue — retrieval |

### 4.3 Key Figures to Produce

1. **Main result figure**: NDCG@10 vs compression ratio (K/D), DCT vs PCA vs PQ, on 3+ datasets, ID and OOD
2. **OOD collapse figure**: PCA NDCG drops sharply under domain shift, DCT stays stable
3. **Capacity argument figure**: compressed BGE-M3 vs native BGE-Small on multilingual
4. **Ablation bar chart**: TSP vs no-TSP, correlation vs Euclidean distance
5. **Scaling law figure**: DCT advantage grows with model dimensionality
6. **t-SNE visualization**: 2D projection of DCT-compressed vs PCA-compressed embeddings, colored by domain

### 4.4 Code & Reproducibility

- Clean repo with experiment scripts
- `requirements.txt` with exact versions
- Config files for each experiment
- Script to reproduce all tables: `python reproduce_all.py`
- Upload calibration configs + pre-computed embeddings for reproducibility
- Consider a HuggingFace Space demo

---

## Phase 5: Pre-Submission Checklist

- [ ] All BEIR datasets evaluated (7+)
- [ ] 5+ models tested
- [ ] 6+ baselines compared
- [ ] Statistical significance (p-values, confidence intervals)
- [ ] Multilingual evaluation (MIRACL, 5+ languages)
- [ ] Long-context evaluation (100+ queries per position)
- [ ] Ablation studies complete (7+ ablations)
- [ ] Theoretical analysis: why DCT works explained
- [ ] All claims cross-checked against data
- [ ] Code reproducible
- [ ] Figures publication-quality (matplotlib, not screenshots)
- [ ] Tables formatted for LaTeX
- [ ] Paper ≤ 8 pages (long paper) or ≤ 4 pages (short paper)
- [ ] Anonymous submission (no author names in draft)
- [ ] Related work complete (no missing citations)
- [ ] Limitations section honest

---

## Immediate Next Steps

1. Install BEIR evaluation framework: `pip install beir`
2. Write benchmark script: `benchmarks/beir_evaluation.py`
3. Run BGE-M3 on SciFact + NFCorpus as first validation
4. Analyze results — if DCT wins on real BEIR data, proceed to full suite
5. If DCT loses on BEIR, investigate why (different from synthetic tests)
