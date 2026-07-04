# Research Diary — TSP-DCT Embedding Compression Paper

**Target**: SIGIR 2027 (January 2027 deadline)
**Method name in paper**: "Spectral Compression" or "TSP-DCT" (NOT "VectorZip" — anonymized)
**Started**: July 2026

---

## 1. Core Thesis

> *Post-hoc spectral compression via TSP-reordered DCT enables deploying high-capacity embedding models under tight dimensional budgets. The TSP permutation captures model-intrinsic dimension structure rather than corpus-specific variance, resulting in OOD retrieval robustness where data-driven methods like PCA collapse — without retraining or fine-tuning.*

---

## 2. Key Findings So Far

### 2.1 DCT OOD > PCA OOD (confirmed)

| Model | Dataset | K | DCT OOD | PCA OOD | DCT retención | PCA retención |
|---|---|---|---|---|---|---|
| BGE-M3 | SciFact | 64 | 0.406 | 0.284 | **121%** | 48% |
| BGE-M3 | NFCorpus | 64 | 0.158 | 0.097 | **117%** | 44% |
| BGE-M3 | ArguAna | 32 | 0.341 | 0.239 | **99%** | 44% |
| BGE-Large | SciFact | 64 | 0.596 | 0.518 | **106%** | 71% |
| BGE-Large | NFCorpus | 64 | 0.226 | 0.186 | **106%** | 65% |
| MiniLM | SciFact | 48 | 0.511 | 0.454 | **109%** | 76% |
| MiniLM | NFCorpus | 16 | 0.065 | 0.058 | **104%** | 50% |

**Conclusión**: DCT retiene 92-121% OOD. PCA colapsa a 28-76%. Consistente en 3 modelos y 4 datasets.

### 2.2 DCT OOD > DCT ID (model-intrinsic structure)

DCT calibrado en corpus genérico funciona **mejor** que calibrado en el propio dataset. Esto sugiere que la permutación TSP captura estructura del modelo, no del corpus.

| Model | Dataset | K | DCT ID | DCT OOD | DCT OOD/ID |
|---|---|---|---|---|---|
| BGE-M3 | SciFact | 64 | 0.335 | 0.406 | **121%** |
| BGE-M3 | NFCorpus | 64 | 0.136 | 0.158 | **117%** |
| BGE-Large | SciFact | 64 | 0.562 | 0.596 | **106%** |

### 2.3 Patrón por nivel de compresión

| Compresión | DCT gana OOD | PCA gana OOD |
|---|---|---|
| Alta (K=32/16, ~32x) | **10/12** | 1 |
| Media (K=64/24-48, ~8-16x) | **7/12** | 5 |
| Baja (K=128+, ~4-8x) | 4/12 | 8 |

DCT domina a compresión alta. PCA domina a baja compresión.

### 2.4 FiQA es outlier

PCA le gana a DCT en FiQA en casi todos los K. Hipótesis: FiQA es financiero, estructura diferente. **Investigar por qué.**

### 2.5 RP (Random Projection) es competitivo OOD

RP también es una base universal → también es OOD-robusto. A K=32 RP a veces le gana a DCT. **Necesitamos explicar por qué DCT > RP.**

---

## 3. Teoría — Preguntas Abiertas

### 3.1 ¿Por qué DCT funciona si la correlación entre dimensiones es ~0?

**Estado**: ✅ Respondido — ver Sección 6.3

**Datos**:
- BGE-M3: mean |corr| = 0.0002, max |corr| = 0.0011
- MiniLM: mean |corr| = 0.0002, max |corr| = 0.0012
- Las dimensiones son esencialmente ortogonales (isotropía)
- TSP no debería encontrar nada que reordenar
- Pero DCT con TSP funciona mejor que DCT sin TSP

**Hallazgo clave**: TSP **sí** encuentra estructura (255% más correlación adyacente que random), pero la compactación de energía del DCT **no mejora** con TSP. La ventaja de DCT no viene de energy compaction sino de **neighborhood preservation**.

**El insight central**: PCA captura 10x más varianza que DCT (63% vs 6% a K=64), pero DCT preserva más vecinos bajo OOD. **Varianza ≠ calidad de retrieval.** PCA optimiza el objetivo equivocado para retrieval OOD.

### 3.2 ¿Por qué DCT OOD > DCT ID?

**Estado**: ✅ Respondido — ver Sección 6.3

**Hipótesis confirmada**: La permutación TSP captura estructura del modelo, no del corpus. Un corpus más diverso (genérico) muestrea mejor el espacio de activaciones del modelo.

**Hallazgo adicional**: La permutación de BGE-Large aplicada a BGE-M3 funciona **mejor** que la nativa de BGE-M3 (+0.041 nDCG@10 a K=64). Esto sugiere que la permutación captura estructura general de los modelos de embedding, no solo de un modelo específico.

**Tests realizados**:
- [x] Calibrar con corpus de diferentes tamaños (50-1000)
- [x] Comparar permutaciones TSP entre modelos (Kendall τ = 0.0002, 8% overlap)
- [x] Transferir permutación del modelo A al modelo B (cross-model funciona mejor en 3/4 casos)

### 3.3 ¿Por qué DCT > RP si ambos son bases universales?

**Estado**: ✅ Respondido — la respuesta es que **no siempre es mejor**

**Hallazgo**: DCT no le gana consistentemente a RP. De hecho, RP es mejor en varios casos:

| Dataset | K | DCT+TSP | DCT (no TSP) | RP | DCT vs RP |
|---|---|---|---|---|---|
| SciFact | 64 | 0.452 | 0.442 | **0.485** | RP gana |
| SciFact | 32 | 0.222 | 0.166 | **0.256** | RP gana |
| NFCorpus | 64 | 0.133 | 0.133 | 0.132 | ≈ |
| FiQA | 64 | 0.378 | 0.443 | 0.397 | DCT(no TSP) gana |
| **ArguAna** | 64 | **0.550** | 0.513 | 0.474 | **DCT+TSP gana** |
| ArguAna | 32 | 0.366 | 0.369 | 0.371 | ≈ |

**Conclusión honesta**: RP es sorprendentemente competitivo. DCT+TSP solo le gana a RP consistentemente en ArguAna. En SciFact, RP gana. El TSP gain es inconsistente — ayuda en SciFact (+0.056 a K=32) y ArguAna (+0.037) pero **perjudica** en FiQA (-0.052).

**Implicación para el paper**: No podemos claimar "DCT > RP". Tenemos que ser honestos. El argumento principal es DCT vs PCA (OOD robustness), no DCT vs RP. RP debe incluirse como baseline fuerte y mencionar que es competitivo OOD por la misma razón (base universal).

**TSP gain**: El reordenamiento TSP ayuda en algunos datasets pero no en otros. Necesita más investigación para entender cuándo ayuda.

### 3.5 ¿Por qué FiQA no favorece a DCT?

**Estado**: ✅ Respondido

**Hallazgo**: FiQA tiene mayor concentración de varianza en PCA (top-5 = 16.4%) vs otros datasets (14.0%). Cuando la varianza está más concentrada, PCA la aprovecha mejor.

| Dataset | PCA var@64 | DCT var@64 | PCA top-5 |
|---|---|---|---|
| SciFact | 0.539 | 0.061 | 14.0% |
| NFCorpus | 0.558 | 0.061 | 14.5% |
| **FiQA** | **0.587** | 0.062 | **16.4%** |
| ArguAna | 0.588 | 0.060 | 14.0% |

FiQA no es que DCT funcione mal, sino que PCA funciona mejor ahí porque la estructura de varianza favorece su enfoque.

---

## 4. Experimentos Completados

### 4.1 BEIR Benchmark (parcial)

| Modelo | SciFact | NFCorpus | FiQA | ArguAna |
|---|---|---|---|---|
| BGE-M3 (1024D) | ✅ | ✅ | ✅ | ✅ |
| MiniLM (384D) | ✅ | ✅ | ✅ | ✅ |
| BGE-Large (1024D) | ✅ | ✅ | ✅ | ✅ |

**Config**: 2000 docs sample, 200 queries, 6 métodos, 5 ratios, OOD + ID, 1 seed
**Results file**: `benchmarks/results_deep_benchmark.json`

### 4.2 Tests sintéticos anteriores

- OOD retrieval con datos sintéticos smooth: DCT 7.4x mejor que PCA a K=32
- Rank preservation: DCT preserva 2.6x más del top-10 original que PCA OOD
- Correlación entre dims: ~0.0002 (isotropía confirmada)

### 4.3 Ablation: TSP vs no-TSP (parcial)

| Test | K=64 | K=32 |
|---|---|---|
| DCT con TSP (correlación) | 0.406 | 0.197 |
| DCT sin TSP | 0.407 | 0.205 |
| TSP gain | -0.001 | -0.008 |

**Problema**: En SciFact/BGE-M3, TSP no ayuda. Pero en datos sintéticos smooth, TSP da +0.72. **Necesitamos entender cuándo TSP ayuda y cuándo no.**

---

## 5. Experimentos Pendientes

### 5.1 Críticos (bloquean SIGIR submit)

- [ ] **3+ datasets más**: TREC-COVID, NQ, SciDocs
- [ ] **2+ modelos más**: GTE-Qwen2-1.5B (1536D), E5-Mistral-7B (4096D)
- [ ] **5 seeds + p-values**: significancia estadística
- [ ] **Datasets completos** (no 2000 sample) — al menos para el modelo principal
- [ ] **SpecTemp baseline**: implementar comparación
- [ ] **Teoría**: al menos una de las hipótesis de 3.1 validada

### 5.2 Altamente recomendados

- [ ] **Ablation: TSP vs no-TSP** en más datasets/modelos
- [ ] **Ablation: correlación-TSP vs Euclidean-TSP**
- [ ] **Ablation: taper vs hard truncation**
- [ ] **Ablation: tamaño de corpus de calibración** (100, 500, 1000, 5000)
- [ ] **Transferibilidad**: permutación del modelo A aplicada al modelo B
- [ ] **MIRACL multilingüe**: español, francés, chino, árabe, japonés
- [ ] **Long-context**: 100+ queries por posición (beginning, middle, end)

### 5.3 Nice to have

- [ ] **PQ baseline** en BEIR
- [ ] **MRL baseline** (Nomic, E5 con Matryoshka nativo)
- [ ] **DWT baseline** (Salama 2025)
- [ ] **Scaling law figure**: DCT vs PCA por dimensionalidad del modelo
- [ ] **t-SNE visualization**: DCT-compressed vs PCA-compressed, colored by domain
- [ ] **Storage analysis**: tamaño de serialización DCT vs PCA
- [ ] **Latency analysis**: QPS sobre 100K y 1M vectores

---

## 6. Análisis Teórico — Progreso

### 6.1 Propiedades formales conocidas

| Propiedad | Demostrable | Estado |
|---|---|---|
| DCT-II es ortonormal | ✅ Trivial | Listo para escribir |
| Orthonormality → preserva dot products | ✅ Trivial | Listo |
| Universal basis (no data-dependent) | ✅ Por definición | Listo |
| TSP permutation es una bijection | ✅ Trivial | Listo |
| Energy compaction para señales smooth | ✅ Conocido (JPEG) | Citar |
| Bound de distorsión de cosine sim | ⚠️ Derivar | Pendiente |

### 6.3 Hallazgo teórico clave (3 julio 2026)

**El insight central del paper**: *Variance ≠ Retrieval Quality*

| K | PCA var | DCT var | PCA ID neigh | DCT ID neigh | PCA OOD neigh | DCT OOD neigh |
|---|---|---|---|---|---|---|
| 256 | 93.8% | 24.8% | 0.530 | 0.448 | 0.522 (98%) | 0.466 (104%) |
| 128 | 80.4% | 12.4% | 0.444 | 0.254 | 0.248 (56%) | **0.306 (120%)** |
| 64 | 63.0% | 6.1% | 0.326 | 0.130 | 0.112 (34%) | **0.174 (134%)** |
| 32 | 46.3% | 2.9% | 0.198 | 0.060 | 0.062 (31%) | **0.080 (133%)** |

**Tres observaciones clave**:

1. **PCA captura 10x más varianza** (63% vs 6% a K=64) pero DCT preserva más vecinos OOD
2. **PCA colapsa OOD**: retiene 31-56% de vecinos ID. DCT retiene 104-134% (mejor OOD que ID)
3. **DCT OOD > DCT ID**: la permutación TSP del corpus genérico es mejor que la del dominio específico

**Explicación teórica**:

- PCA optimiza **varianza global** del corpus de calibración → cuando el dominio cambia, la varianza que capturó ya no es relevante
- DCT usa una **base universal** (cosenos) → la proyección no depende del corpus → estable bajo domain shift
- TSP encuentra estructura relativa (255% más correlación adyacente que random) aunque la correlación absoluta sea ~0.0003. No mejora energy compaction, pero mejora la estructura del subespacio para retrieval

**Implicación para el paper**: El argumento no es "DCT comprime mejor" sino "DCT comprime el subespacio correcto para retrieval OOD". PCA optimiza el objetivo equivocado (varianza) para retrieval bajo domain shift.

### 6.4 Transferibilidad de TSP (3 julio 2026)

**Hallazgo**: La permutación TSP de BGE-Large aplicada a BGE-M3 funciona **mejor** que la nativa (+0.041 nDCG@10 a K=64). Pero las permutaciones son completamente diferentes (Kendall τ = 0.0002, 8% overlap).

**⚠️ CONDICIÓN IMPORTANTE**: Solo se ha probado entre BGE-M3 y BGE-Large — **ambos de BAAI, ambos 1024D**. No pudimos transferir a MiniLM porque tiene diferente dimensionalidad (384D vs 1024D).

**Limitación**: No sabemos si la transferibilidad funciona entre familias diferentes (ej. BAAI vs Microsoft vs Alibaba). Necesitamos modelos de familias distintas con la misma dimensionalidad para confirmar.

**Estructura de correlación similar entre modelos**:
- BGE-M3:    |corr| mean=0.000078, max=0.000568
- BGE-Large: |corr| mean=0.000078, max=0.000452
- MiniLM:    |corr| mean=0.000079, max=0.000511

Los tres modelos tienen **exactamente la misma distribución de correlación** (mean ~0.00008, 0% de pares con |corr|>0.001). Esto sugiere que la isotropía es una propiedad universal de los embeddings modernos, no específica de un modelo.

**Tests pendientes CRÍTICOS**:
- [ ] Transferir TSP entre modelos de **familias diferentes** con **misma dimensionalidad** (ej. BGE-Large 1024D vs E5-Mistral 4096D no funciona — necesitan misma D)
- [ ] Posibles pares: BGE-M3 (1024D) vs otro modelo 1024D de familia diferente
- [ ] Si la transferibilidad solo funciona dentro de la misma familia → el argumento es "arquitectura-specific" no "universal"

---

## 7. Estructura del Paper (borrador)

```
1. Introduction (1 page)
2. Related Work (1 page)
   - PCA, PQ, RP for embedding compression
   - MRL, Matryoshka-Adaptor
   - Spectral methods: SpecTemp, DWT
   - Dimension ordering: AxisTour, SCDTour
3. Method (1.5 pages)
   3.1 Dimension ordering problem
   3.2 TSP reordering (correlation-based)
   3.3 DCT-II projection with truncation
   3.4 Complexity analysis
4. Theoretical Analysis (1 page)
   4.1 OOD robustness of universal basis
   4.2 Cosine similarity distortion bound
   4.3 Model-intrinsic structure argument
5. Experiments (3 pages)
   5.1 Setup
   5.2 Main results: BEIR (ID and OOD)
   5.3 OOD retention analysis
   5.4 Ablation studies
   5.5 Multilingual evaluation
6. Analysis (0.5 pages)
   - When DCT wins vs loses
   - Why DCT OOD > DCT ID
   - Comparison with RP
7. Conclusion (0.5 pages)
```

---

## 8. Cronograma

| Semana | Tarea | Estado |
|---|---|---|
| Sem 1 (jul) | BEIR 4 datasets, 3 modelos | ✅ Done |
| Sem 2 (jul) | Teoría: hipótesis 3.1, bound 3.4 | ⏳ En curso |
| Sem 3 (jul) | 3 datasets más + 2 modelos más (RunPod) | ⏳ Pendiente |
| Sem 4 (jul) | MIRACL multilingüe + ablations | ⏳ Pendiente |
| Sem 5-6 (ago) | SpecTemp baseline + statistical significance | ⏳ Pendiente |
| Sem 7-8 (ago) | Escritura primer draft | ⏳ Pendiente |
| Sem 9 (sep) | Revision interna + pulir | ⏳ Pendiente |
| Sem 10 (oct) | Segundo draft + feedback | ⏳ Pendiente |
| Nov-dic | Pulir + figures + arXiv | ⏳ Pendiente |
| Ene 2027 | Submit SIGIR | ⏳ Pendiente |

---

## 9. Decisiones Pendientes

- [ ] Nombre del método en el paper (¿"Spectral Compression"? ¿"TSP-DCT"?)
- [ ] ¿Incluir PQ en main results o solo en appendix?
- [ ] ¿Short paper (4p) o long paper (9p)?
- [ ] ¿Enviar a arXiv antes o después del submit?
- [ ] ¿Qué modelos descartar si hay limitaciones de tiempo?

---

## 10. Notas y Observaciones

- La isotropía de embeddings modernos es un hallazgo importante por sí solo
- DCT OOD > DCT ID es contraintuitivo y valioso como "highlight" del paper
- RP es el rival más peligroso — también es universal y OOD-robusto
- FiQA outlier necesita explicación o mención honesta en limitations
- El argumento "compress large multilingual model > train small model" es el selling point práctico
- Anonymous.4open.science para compartir código en review
- NO mencionar "VectorZip" en el paper

---

## 11. Análisis del enemigo: Random Projection (RP)

### 11.1 Qué es RP

Random Projection multiplica el embedding por una matriz aleatoria Gaussiana. No aprende nada de los datos. Es "data-oblivious" — ignora la estructura del corpus.

### 11.2 Por qué es OOD-robusto (mismo motivo que DCT)

RP no depende del corpus → no puede overfitting a un dominio → estable bajo domain shift. **Es nuestro rival por la misma razón que DCT.**

### 11.3 Por qué RP es peligroso como baseline

| Ventaja RP | Problema para nosotros |
|---|---|
| Sin entrenamiento | DCT tampoco necesita entrenamiento real (solo TSP) |
| Base universal | DCT también es base universal |
| JL lemma da garantía teórica de distancia | DCT no tiene garantía equivalente |
| OOD-robusto | DCT también es OOD-robusto |
| Más rápido (sin TSP) | DCT tarda 7.5s en TSP |
| Implementado en sklearn | DCT está en nuestra librería |

### 11.4 Debilidades de RP (nuestros argumentos)

1. **No aprovecha estructura**: RP ignora toda la correlación entre dimensiones. DCT+TSP la aprovecha (aunque sea ~0.0002)
2. **No descorrelaciona**: RP no produce componentes independientes → peor para PQ posterior
3. **Preserva distancias, no neighborhoods**: JL lemma garantiza distancias globales, pero retrieval necesita preservar vecinos locales
4. **Cosine similarity subóptimo**: la literatura confirma que RP no preserva cosine tan bien como PCA en algunos casos
5. **Sin criterio de parada**: RP no dice cuántas dimensiones conservar. PCA tiene explained variance, DCT tiene energy compaction

### 11.5 Lo que NO podemos claimar

- ❌ "DCT es mejor que RP en OOD" — los datos muestran que RP es competitivo
- ❌ "DCT aprovecha estructura que RP ignora" — la ventaja es inconsistente (FiQA perjudica)

### 11.6 Lo que SÍ podemos claimar

- ✅ "DCT logra OOD robustness comparable a RP sin sacrificar tanto el rendimiento ID"
- ✅ "DCT+TSP supera a RP en algunos datasets (ArguAna +0.08)" pero no consistentemente
- ✅ "DCT se combina mejor con SQ8 que RP" (pendiente verificar)

### 11.7 Estrategia para el paper

**No posicionar DCT como "mejor que RP".** Posicionar DCT como:
1. "Mejor que PCA en OOD" (argumento principal)
2. "Comparable a RP en OOD pero con ventajas prácticas" (secundario)
3. "RP es un baseline fuerte que también es OOD-robusto — ambos son bases universales"

**En las tablas**: incluir RP como baseline honesto. Si RP le gana a DCT en algunos datasets, decirlo. Los reviewers respetan la honestidad.

## 12. Hallazgo clave: Ensemble DCT+RP (3 julio 2026)

### El descubrimiento

Promediar los cosine scores de DCT y RP produce un ensemble que **gana en 24/24 casos** (3 modelos × 4 datasets × 2 K values).

### Resultados

**Ensemble wins: 24/24 (100%)**
- Mejora promedio vs mejor método individual: **+0.096 nDCG@10**
- Mejora promedio vs PCA: **+0.161 nDCG@10**

### Tabla principal

| Modelo | Dataset | K | PCA | DCT | RP | **ENS** | ENS vs PCA |
|---|---|---|---|---|---|---|---|
| BGE-M3 | SciFact | 32 | 0.120 | 0.222 | 0.256 | **0.496** | +0.376 |
| BGE-M3 | ArguAna | 64 | 0.436 | 0.550 | 0.474 | **0.577** | +0.141 |
| MiniLM | SciFact | 32 | 0.405 | 0.450 | 0.477 | **0.669** | +0.264 |
| BGE-Large | ArguAna | 32 | 0.443 | 0.524 | 0.527 | **0.693** | +0.250 |

### Por qué funciona

DCT y RP capturan información complementaria:
- DCT capta estructura de frecuencias (baja frecuencia = información general)
- RP capta direcciones aleatorias (preserva distancias globales via JL lemma)
- Ambos se equivocan en cosas diferentes → promediar cancela los errores
- Similar a un random forest: cada método es débil individualmente pero juntos son fuertes

### Implicación para el paper

**Esto cambia el framing completamente.** El paper ya no es "DCT vs RP" — es:

> *"Spectral-Random Ensemble Compression: combining universal-basis methods (DCT + RP) for OOD-robust retrieval that outperforms both individually and PCA."*

### Lo que falta validar

- [ ] Ensemble con SQ8 (ambos métodos cuantizados) — HECHO: no mejora a igual storage
- [ ] Ensemble con más de 2 métodos — pendiente
- [ ] Ensemble con pesos optimizados — pendiente
- [ ] Ensemble en datasets completos — pendiente
- [ ] Significancia estadística — pendiente
- [x] Comparar contra PCA+RP ensemble — DCT+RP > PCA+RP (+0.12) — DCT es necesario
- [x] Coste computacional — 2x storage, 2x search

### Conclusión sobre ensemble

El ensemble DCT+RP promediando scores **gana 24/24 a igual K** pero **pierde a igual storage** porque usa 2x espacio. Cuando PCA+SQ8 recibe el mismo budget (K=128 vs ENS K=64×2), PCA+SQ8 gana en 3/4 datasets.

### Híbridos estructurales (mismo K, mismo storage)

Se probaron 14 híbridos diferentes:

| Idea | Qué hace | Resultado |
|---|---|---|
| H1 | K/2 DCT + K/2 RP concatenados | Peor que individual |
| H2 | DCT → RP (RP sobre coeficientes DCT) | Mejor híbrido, gana a 256B |
| H5 | RP → DCT | Igual que RP solo |
| H6 | DCT → PCA (PCA en espacio DCT) | Gana en FiQA BGE-Large |
| H7 | RP sobre primeros 2K DCT coefs | Gana en ArguAna BGE-M3 |
| H8 | DCT → whiten → RP | No mejora |
| H9 | Weighted DCT → RP | Peor |
| H10 | DCT → PCA (más formal) | Gana en FiQA MiniLM |
| H12 | Top-K DCT por varianza | Gana en SciFact MiniLM |
| H14 | RP(D/2) → DCT(K) | Gana en SciFact BGE-M3, NFCorpus BGE-Large |

**Ningún híbrido domina consistentemente.** Cada uno gana en casos específicos pero ninguno supera a RP o DCT en todos los datasets.

### Lo que aprendimos

1. **Los embeddings modernos son isotrópicos** (|corr| ~0.0002) — no hay mucha estructura que explotar
2. **RP es sorprendentemente fuerte** porque JL lemma garantiza preservación de distancias
3. **DCT es mejor que PCA en OOD** pero no mejor que RP
4. **Los híbridos no superan consistentemente** a los métodos individuales
5. **El ensemble promedio funciona a igual K pero no a igual storage**

### Implicación final para el paper

El paper **no puede** claimar "nuestro método es el mejor". Tiene que claimar:

1. **DCT es OOD-robusto vs PCA** — este es el argumento principal que se sostiene
2. **La isotropía de los embeddings es un hallazgo importante** — explica por qué DCT no comprime mejor
3. **Variance ≠ retrieval quality** — PCA captura 10x más varianza pero DCT preserva más vecinos OOD
4. **DCT+RP ensemble** como bonus (gana a igual K, no a igual storage)
5. **Híbridos estructurales** como exploración (H2 y H14 prometedores en casos específicos)

---

## 13. Estado del campo y posición del paper (3 julio 2026)

### 13.1 El gap que nadie ha cubierto

**No existe ningún paper que estudie OOD robustness DE compresión de embeddings.** Esto es un gap claro en la literatura 2025-2026:

- **OOD en retrieval**: existe (Bao et al. EMNLP 2025, Liu et al. ECIR 2025) — pero no estudian compresión
- **Compresión de embeddings**: muy activo (MRL, PQ, PCA, quantization) — pero nadie estudia OOD
- **OOD + compresión**: **cero papers encontrados**

### 13.2 SOTA actual de compresión

| Método | Tipo | Requiere training | OOD estudiado? |
|---|---|---|---|
| MRL (Kusupati 2022) | Training-time | ✅ Sí | ❌ |
| Matryoshka-Adaptor | Post-hoc adapter | ✅ Sí | ❌ |
| SMEC (EMNLP 2025) | Post-hoc adapter | ✅ Sí | ❌ |
| DIVE (May 2026) | Self-supervised adapter | ✅ Sí | ❌ |
| PCA + float8 (Huerga 2025) | Post-hoc | ❌ No | ❌ |
| CoRECT (Oct 2025) | Benchmark | — | ❌ |
| SpecTemp (SIGIR 2026) | Spectral scaling | ❌ No | ❌ |
| **Nuestro DCT** | Post-hoc | ❌ No | ✅ **Sí** |

### 13.3 Nuestra posición

**El paper se posiciona en el único gap que nadie ha tocado: OOD robustness de compresión de embeddings.**

Title: *"Does Embedding Compression Preserve Retrieval Quality Under Distribution Shift? A Study of Spectral vs Data-Driven Methods"*

Contribuciones:
1. **Primer estudio de OOD robustness en compresión de embeddings** — el gap principal
2. **DCT como método OOD-robusto** — retiene 100-121% OOD vs PCA 30-50%
3. **Insight: variance ≠ retrieval quality** — PCA captura 10x más varianza pero DCT preserva más vecinos OOD
4. **Isotropía universal** — los 3 modelos tienen idéntica distribución de correlación (~0.00008)
5. **TSP encuentra estructura oculta** — 255% mejor que random con corr ~0
6. **Comparación exhaustiva** — DCT, RP, PCA, B32, B64, B128, SQ8 variants, ensemble
7. **Practical: training-free, JSON-serializable, works with any encoder**

### 13.4 Competidores directos

| Paper | Qué hace | Nuestra ventaja |
|---|---|---|
| SMEC (EMNLP 2025) | MRL mejorado | No requiere training |
| DIVE (May 2026) | Adapter self-supervised | No requiere adapter ni training |
| SpecTemp (SIGIR 2026) | Spectral scaling en eigenvalues | Nosotros usamos DCT (frecuencia directa) + estudiamos OOD |
| Huerga (2025) | PCA + float8 | Nosotros estudiamos OOD, no solo ID |
| CoRECT (2025) | Benchmark de compresión | Nosotros proponemos método + analizamos OOD |

## 14. Plan de dos papers (3 julio 2026)

### Paper 1 — SIGIR 2027 (enviar enero 2027)

**Title**: *"Does Embedding Compression Preserve Retrieval Quality Under Distribution Shift? A Study of Spectral vs Data-Driven Methods"*

**Pregunta central**: ¿Sobrevive la compresión al domain shift?

**Incluye**:
- DCT+TSP vs PCA en OOD (argumento principal)
- "Variance ≠ retrieval quality" (insight teórico)
- Isotropía universal como hallazgo empírico
- Comprimido BGE-M3 > BGE-Small nativo multilingüe (killer result)
- BEIR 7+ datasets, 5+ modelos
- TSP encuentra estructura oculta (255% mejor que random)
- OOD retention: DCT 100-121% vs PCA 30-50%

**NO incluye** (guardar para Paper 2):
- Ensemble DCT+RP
- Block-DCT (B64, B128, B32)
- Cross-model TSP transfer
- Híbridos estructurales (H2, H14, etc.)
- SQ8-first pipeline

**Historia**: "PCA colapsa OOD. DCT no. Esto importa porque en RAG tus datos cambian de dominio. Y si comprimes un modelo multilingüe grande, le ganas al pequeño nativo."

### Paper 2 — EMNLP 2026 o ACL 2027 (6 meses después)

**Title**: *"Universal Structure in Embedding Spaces: TSP Reveals Model-Intrinsic Dimension Organization"*

**Pregunta central**: ¿Existe estructura universal en los embeddings?

**Incluye**:
- Cross-model TSP transfer (permutación de A funciona en B)
- Isotropía universal como propiedad formal
- Block-DCT adaptativo (B64, B128, B32)
- Ensemble DCT+RP (gana a igual K)
- Híbridos estructurales (H2 DCT→RP, H14 RP→DCT, etc.)
- SQ8-first pipeline
- "Estructura model-intrinsic vs corpus-specific"

**Historia**: "Los embeddings tienen estructura oculta que es compartida entre modelos. TSP la descubre. Se puede explotar con block-DCT y ensembles para mejorar compresión."

### Por qué esta división es coherente

- Paper 1 responde **"qué pasa"** (DCT sobrevive OOD, PCA no)
- Paper 2 responde **"por qué pasa"** (estructura universal, TSP, block-DCT)
- Paper 1 es empírico + análisis OOD
- Paper 2 es estructural + análisis teórico profundo
- No hay overlap — cada paper tiene su propia contribución
- Paper 1 establece el problema, Paper 2 profundiza en la solución

---

## 15. Resultados multilingües + análisis de novedad (3 julio 2026)

### 15.1 Resultados multilingües preliminares

| Idioma | BGE-Small nativo (384D) | BGE-M3 DCT→128D | BGE-M3 DCT→64D |
|---|---|---|---|
| Inglés (native) | 100% | **100%** | **100%** |
| Español (OOD) | 90% | **100%** (+10) | 90% (+0) |
| Francés (OOD) | 80% | **100%** (+20) | 90% (+10) |

BGE-M3 comprimido a 128D le gana a BGE-Small nativo en los 3 idiomas.

### 15.2 Análisis de novedad — Paper 1

**¿Está cubierto el "OOD + compression"?**
- Zuo & Khashabi (ACL 2026) — "Embedding Compression Improves Domain Adaptation" — usa PCA en dominio objetivo. **Diferente a nosotros**: ellos necesitan datos del dominio objetivo (domain adaptation), nosotros no (domain generalization con corpus genérico)
- Bao et al. (EMNLP 2025) — OOD en retrieval, no compression
- **Nadie ha comparado comprimido multilingüe vs nativo pequeño en retrieval**

**¿Está cubierto "comprimir grande > pequeño nativo"?**
- Singh et al. (2023) — destilación multilingüe → monolingüe, pero para NLP general, no retrieval
- Gurgurov et al. (2025) — compresión de mBERT, no retrieval
- **No hay paper que compare "comprimir modelo multilingüe grande vs entrenar modelo monolingüe pequeño" para retrieval**

**Conclusión**: Paper 1 es novel. La diferencia con Zuo & Khashabi es clave:
- Ellos: PCA en dominio objetivo = **domain adaptation** (necesitas datos del dominio)
- Nosotros: DCT en corpus genérico = **domain generalization** (no necesitas datos del dominio)

### 15.3 Análisis de novedad — Paper 2

**¿Está cubierto "estructura universal en embeddings"?**
- Huh et al. (ICML 2024) — "Platonic Representation Hypothesis" — modelos convergen a representaciones compartidas
- Jha et al. (2025) — "vec2vec" — traduce embeddings entre modelos sin datos pareados, 0.96 cosine
- Nuestro cross-model TSP encaja con esta línea pero desde ángulo diferente: estructura de **dimensiones** (permutación), no traducción de **espacios**

**Conclusión**: Paper 2 es novel pero tiene related work fuerte. Necesita distinguirse de Platonic Representation Hypothesis y vec2vec.

### 15.4 Pruebas necesarias para Paper 1 (SIGIR)

**Experimentos principales**:
- [x] DCT vs PCA OOD en BEIR (4 datasets, 3 modelos)
- [ ] 3+ datasets más (TREC-COVID, NQ, SciDocs) — RunPod
- [ ] 2+ modelos más (GTE-Qwen2, E5-Mistral) — RunPod
- [ ] MIRACL multilingüe (es, fr, de, zh, ar, ja) — RunPod
- [ ] Comprimido BGE-M3 vs BGE-Small nativo en MIRACL — RunPod
- [ ] 5 seeds con p-values
- [ ] SpecTemp baseline
- [ ] Ablation: TSP vs no-TSP, calibration corpus size
- [ ] Datasets completos (no 2000 sample)

**Análisis teórico**:
- [x] Variance ≠ retrieval quality
- [x] TSP encuentra estructura oculta
- [x] Isotropía universal
- [ ] Bound formal de distorsión de cosine similarity
- [ ] Explicación formal de por qué DCT OOD > DCT ID

**Figuras**:
- [ ] nDCG@10 vs compression ratio (DCT vs PCA vs RP), ID y OOD
- [ ] OOD retention bar chart (DCT vs PCA)
- [ ] Comprimido vs nativo small en multilingüe
- [ ] Variance captured vs neighbor preservation
- [ ] Ablation: TSP gain
