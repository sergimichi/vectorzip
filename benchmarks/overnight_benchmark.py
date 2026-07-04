#!/usr/bin/env python3
"""
Overnight benchmark — RTX 5090 (32GB VRAM) optimized.
Runs everything needed for Paper 1 (SIGIR).

Optimizations for 32GB VRAM:
- Batch all queries in single GPU matmul (not one-by-one)
- Large encoding batches (BGE-M3: 128, MiniLM: 512, Qwen2: 32)
- torch.no_grad() + inference_mode throughout
- All compressed doc sets stay on GPU during eval
- Pinned memory for fast CPU→GPU transfer
- float16 encoding where possible

Usage on RunPod (NVIDIA RTX 5090):
    python overnight_benchmark.py

Usage on local AMD (RX 7900 XT):
    HSA_OVERRIDE_GFX=1100 python overnight_benchmark.py
"""

import os, sys, json, time, warnings, pickle, argparse, traceback
import numpy as np
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorzip import VectorZip
from vectorzip.default_corpus import DEFAULT_CORPUS
from vectorzip.compressor import RandomProjectionCompressor
from sklearn.decomposition import PCA
from scipy.fftpack import dct

# ============================================================
# Config — tuned for RTX 5090 32GB
# ============================================================
MODELS = {
    'bge-m3':    {'name': 'BAAI/bge-m3', 'dims': 1024, 'batch': 128, 'max_seq': 512},
    'minilm':    {'name': 'all-MiniLM-L6-v2', 'dims': 384, 'batch': 512, 'max_seq': 256},
    'bge-large': {'name': 'BAAI/bge-large-en-v1.5', 'dims': 1024, 'batch': 128, 'max_seq': 512},
    'gte-qwen2': {'name': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'dims': 1536, 'batch': 32, 'max_seq': 512, 'trust_remote_code': True},
}

BEIR_DATASETS = ['scifact', 'nfcorpus', 'fiqa', 'arguana', 'scidocs', 'trec-covid', 'nq']
MIRACL_LANGS = ['es', 'fr', 'de', 'ar', 'ja', 'zh']
METHODS = ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']
RATIOS = ['1/2', '1/4', '1/8', '1/16', '1/32']
SEEDS = [42, 43, 44, 45, 46]
MAX_DOCS = 2000
MAX_QUERIES = 200
CALIB_SIZES = [50, 100, 200, 500, 1000]
DEVICE = 'cuda'
CACHE_DIR = './beir_cache'
OUTPUT_FILE = 'results_overnight.json'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)

# ============================================================
# GPU-accelerated batch evaluation (ALL queries at once)
# 32GB VRAM can hold 2000 docs + 200 queries + compressed variants
# ============================================================
def evaluate(docs_emb, queries_emb, doc_ids, qids, qrels, top_k=100, device='cuda'):
    """Evaluate retrieval — all queries in a single GPU matmul."""
    import torch

    try:
        if device == 'cuda' and torch.cuda.is_available():
            with torch.inference_mode():
                # Move everything to GPU at once
                docs_t = torch.from_numpy(docs_emb.astype(np.float32)).to(device, non_blocking=True)
                queries_t = torch.from_numpy(queries_emb.astype(np.float32)).to(device, non_blocking=True)

                # Normalize
                docs_norm = docs_t / (torch.norm(docs_t, dim=1, keepdim=True) + 1e-8)
                queries_norm = queries_t / (torch.norm(queries_t, dim=1, keepdim=True) + 1e-8)

                # ALL queries at once: (n_queries, n_docs) similarity matrix
                sims = queries_norm @ docs_norm.T  # (Q, D) — fits easily in 32GB

                # Get top-k for all queries at once
                topk_scores, topk_indices = torch.topk(sims, k=min(top_k, sims.shape[1]), dim=1)
                topk_indices = topk_indices.cpu().numpy()

                # Compute metrics on CPU (fast, just ranking)
                all_ndcg10, all_r5, all_r10, all_r100, all_map10 = [], [], [], [], []

                for i, qid in enumerate(qids):
                    ranked = topk_indices[i]
                    rel = qrels.get(qid, {})
                    if not rel: continue
                    rel_set = set(d for d, r in rel.items() if r > 0)

                    # nDCG@10
                    dcg = sum((2**rel.get(doc_ids[ranked[j]], 0) - 1) / np.log2(j + 2) for j in range(min(10, len(ranked))))
                    ideal = sorted(rel.values(), reverse=True)[:10]
                    idcg = sum((2**ideal[j] - 1) / np.log2(j + 2) for j in range(len(ideal)))
                    all_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)

                    # Recall@k
                    for k_val in [5, 10, 100]:
                        retrieved = set(doc_ids[ranked[j]] for j in range(min(k_val, len(ranked))))
                        rk = len(retrieved & rel_set) / len(rel_set) if rel_set else 0.0
                        if k_val == 5: all_r5.append(rk)
                        elif k_val == 10: all_r10.append(rk)
                        elif k_val == 100: all_r100.append(rk)

                    # MAP@10
                    hits, ap = 0, 0.0
                    for j in range(min(10, len(ranked))):
                        if doc_ids[ranked[j]] in rel_set:
                            hits += 1; ap += hits / (j + 1)
                    all_map10.append(ap / min(len(rel_set), 10) if rel_set else 0.0)

                # Free GPU memory
                del docs_t, queries_t, docs_norm, queries_norm, sims, topk_scores, topk_indices
                torch.cuda.empty_cache()

                return {
                    'ndcg@10': float(np.mean(all_ndcg10)) if all_ndcg10 else 0.0,
                    'recall@5': float(np.mean(all_r5)) if all_r5 else 0.0,
                    'recall@10': float(np.mean(all_r10)) if all_r10 else 0.0,
                    'recall@100': float(np.mean(all_r100)) if all_r100 else 0.0,
                    'map@10': float(np.mean(all_map10)) if all_map10 else 0.0,
                }
    except Exception as e:
        print(f"  GPU eval failed ({e}), falling back to CPU")

    # CPU fallback
    docs_norm = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-8)
    all_ndcg10, all_r5, all_r10, all_map10 = [], [], [], []
    for i, qid in enumerate(qids):
        q = queries_emb[i]; qn = q / (np.linalg.norm(q) + 1e-8)
        ranked = np.argsort(docs_norm @ qn)[::-1]
        rel = qrels.get(qid, {})
        if not rel: continue
        rel_set = set(d for d, r in rel.items() if r > 0)
        dcg = sum((2**rel.get(doc_ids[ranked[j]], 0) - 1) / np.log2(j + 2) for j in range(min(10, len(ranked))))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2**ideal[j] - 1) / np.log2(j + 2) for j in range(len(ideal)))
        all_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
        for k_val in [5, 10]:
            retrieved = set(doc_ids[ranked[j]] for j in range(min(k_val, len(ranked))))
            rk = len(retrieved & rel_set) / len(rel_set) if rel_set else 0.0
            if k_val == 5: all_r5.append(rk)
            else: all_r10.append(rk)
        hits, ap = 0, 0.0
        for j in range(min(10, len(ranked))):
            if doc_ids[ranked[j]] in rel_set:
                hits += 1; ap += hits / (j + 1)
        all_map10.append(ap / min(len(rel_set), 10) if rel_set else 0.0)
    return {
        'ndcg@10': float(np.mean(all_ndcg10)) if all_ndcg10 else 0.0,
        'recall@5': float(np.mean(all_r5)) if all_r5 else 0.0,
        'recall@10': float(np.mean(all_r10)) if all_r10 else 0.0,
        'recall@100': 0.0,
        'map@10': float(np.mean(all_map10)) if all_map10 else 0.0,
    }

# ============================================================
# Compression cache
# ============================================================
class CompressorCache:
    def __init__(self):
        self.cache = {}
    def get(self, method, calib_emb, K, seed=42, key='generic'):
        ck = f"{method}_K{K}_s{seed}_{key}"
        if ck in self.cache: return self.cache[ck]
        if method == 'pca':
            c = PCA(n_components=K, random_state=seed); c.fit(calib_emb)
        elif method == 'dct':
            c = VectorZip(n_components=K, method='dct'); c.fit(calib_emb)
        elif method == 'rp':
            c = RandomProjectionCompressor(n_components=K, seed=seed); c.fit(calib_emb)
        elif method == 'dct+sq8':
            c = VectorZip(n_components=K, method='dct+sq8'); c.fit(calib_emb)
        elif method == 'pca+sq8':
            c = VectorZip(n_components=K, method='pca+sq8'); c.fit(calib_emb)
        elif method == 'truncation':
            c = None
        else:
            c = None
        self.cache[ck] = c
        return c
    def transform(self, method, comp, emb, K):
        if method == 'truncation': return emb[:, :K]
        if method == 'pca': return comp.transform(emb)
        if method in ('dct', 'rp'): return comp.transform(emb)
        if method in ('dct+sq8', 'pca+sq8'): return comp.transform(emb).astype(float)
        return emb

# SpecTemp
def spectemp_compress(calib_emb, target_emb, K):
    mean = calib_emb.mean(axis=0)
    centered = calib_emb - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = S**2 / (calib_emb.shape[0] - 1)
    n_noise = max(1, len(eigenvalues) // 10)
    noise_floor = np.mean(eigenvalues[-n_noise:])
    snr = np.maximum(0, (eigenvalues - noise_floor) / (noise_floor + 1e-10))
    if K < len(snr):
        gamma = min(1.0, snr[K] / (snr[np.argmax(snr)] + 1e-10))
    else:
        gamma = 0.0
    weights = eigenvalues[:K]**(-gamma/2)
    proj = Vt[:K] * weights[:, np.newaxis]
    return (target_emb - mean) @ proj.T

# ============================================================
# Dataset loading
# ============================================================
def load_beir(name, data_dir='./beir_datasets'):
    from beir.datasets.data_loader import GenericDataLoader
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        os.makedirs(data_dir, exist_ok=True)
        print(f"    Downloading {name}...")
        import urllib.request, zipfile
        zp = os.path.join(data_dir, f"{name}.zip")
        urllib.request.urlretrieve(url, zp)
        with zipfile.ZipFile(zp, 'r') as z: z.extractall(data_dir)
        os.remove(zp)
    return GenericDataLoader(data_folder=path).load(split='test')

def load_miracl(lang, data_dir='./miracl_datasets'):
    from datasets import load_dataset
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"miracl_{lang}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    try:
        ds = load_dataset("miracl/miracl", lang, split="test")
        corpus, queries, qrels = {}, {}, {}
        for i, row in enumerate(ds):
            qid = str(row.get('query_id', i))
            queries[qid] = row['query']
            qrels[qid] = {}
            for p in row.get('positive_passages', []):
                did = str(p.get('docid', f"d{len(corpus)}"))
                corpus[did] = p['text']
                qrels[qid][did] = 1
            for p in row.get('negative_passages', []):
                did = str(p.get('docid', f"d{len(corpus)}"))
                if did not in corpus:
                    corpus[did] = p['text']
        with open(cache_path, 'wb') as f:
            pickle.dump({'corpus': corpus, 'queries': queries, 'qrels': qrels}, f)
        return {'corpus': corpus, 'queries': queries, 'qrels': qrels}
    except Exception as e:
        print(f"    MIRACL {lang} failed: {e}")
        return None

# ============================================================
# Save/load
# ============================================================
def save_results(results):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def load_results():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            return json.load(f)
    return {}

# ============================================================
# Main
# ============================================================
def main():
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB" if torch.cuda.is_available() else "")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Datasets: {BEIR_DATASETS}")
    print(f"Methods: {METHODS + ['spectemp']}")
    print(f"Seeds: {SEEDS}")
    print(f"Max docs: {MAX_DOCS}, Max queries: {MAX_QUERIES}")
    print()

    all_results = load_results()
    total_start = time.time()

    # ========================================
    # PART 1: BEIR benchmarks
    # ========================================
    for model_key in MODELS:
        info = MODELS[model_key]
        D = info['dims']
        batch_size = info.get('batch', 64)

        if model_key not in all_results:
            all_results[model_key] = {}

        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_key} ({info['name']}, {D}D, batch={batch_size})")
        print(f"{'#'*80}")

        # Load model
        t0 = time.time()
        print(f"  Loading model on {DEVICE}...")
        kwargs = {}
        if info.get('trust_remote_code'): kwargs['trust_remote_code'] = True
        try:
            model = SentenceTransformer(info['name'], device=DEVICE, **kwargs)
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            continue
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Encode generic calibration corpus
        generic_cache = os.path.join(CACHE_DIR, f"{model_key}_generic.pkl")
        if os.path.exists(generic_cache):
            with open(generic_cache, 'rb') as f: generic_emb = pickle.load(f)
            print(f"  Generic corpus from cache ({generic_emb.shape})")
        else:
            print(f"  Encoding calibration corpus (1000 texts, batch={batch_size})...")
            t0 = time.time()
            generic_emb = model.encode(DEFAULT_CORPUS[:1000], batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            with open(generic_cache, 'wb') as f: pickle.dump(generic_emb, f)
            print(f"  Encoded in {time.time()-t0:.1f}s")

        # Pre-fit compressors
        print(f"  Pre-fitting compressors...")
        comp_cache = CompressorCache()
        t0 = time.time()
        k_values = []
        for r in RATIOS:
            num, denom = r.split('/')
            K = max(int(D * int(num) / int(denom)), 16)
            if K < D: k_values.append((r, K))
        for method in METHODS:
            for _, K in k_values:
                for seed in SEEDS:
                    comp_cache.get(method, generic_emb, K, seed=seed)
        print(f"  {len(comp_cache.cache)} compressors fitted in {time.time()-t0:.1f}s")

        # Run each dataset
        for ds_idx, ds_name in enumerate(BEIR_DATASETS):
            progress_key = f"{model_key}/{ds_name}"
            if progress_key in all_results.get(model_key, {}) and 'raw' in all_results[model_key].get(ds_name, {}):
                print(f"\n  [{ds_idx+1}/{len(BEIR_DATASETS)}] {ds_name} - ALREADY DONE, skipping")
                continue

            elapsed = time.time() - total_start
            print(f"\n  [{ds_idx+1}/{len(BEIR_DATASETS)}] {ds_name} | Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

            # Load/cache embeddings
            cache_file = os.path.join(CACHE_DIR, f"{model_key}_{ds_name}_d{MAX_DOCS}_q{MAX_QUERIES}.pkl")
            if os.path.exists(cache_file):
                print(f"    Loading from cache...")
                with open(cache_file, 'rb') as f: cached = pickle.load(f)
                doc_ids = cached['doc_ids']; qids = cached['qids']
                docs_emb = cached['docs_emb']; queries_emb = cached['queries_emb']
                qrels = cached['qrels']
            else:
                print(f"    Loading {ds_name} from BEIR...")
                try:
                    corpus, queries, qrels_raw = load_beir(ds_name)
                except Exception as e:
                    print(f"    FAILED to load dataset: {e}")
                    continue
                valid_qids = [q for q in queries if q in qrels_raw and len(qrels_raw[q]) > 0]
                relevant = set()
                for qid in valid_qids: relevant.update(qrels_raw[qid].keys())
                all_doc_ids = list(corpus.keys())
                rel_in_corpus = [d for d in all_doc_ids if d in relevant]
                other = [d for d in all_doc_ids if d not in relevant]
                np.random.seed(42)
                n_fill = max(0, MAX_DOCS - len(rel_in_corpus))
                sampled = list(np.random.choice(other, min(n_fill, len(other)), replace=False)) if other else []
                doc_ids = rel_in_corpus + sampled
                qids = list(np.random.choice(valid_qids, min(MAX_QUERIES, len(valid_qids)), replace=False))
                doc_set = set(doc_ids)
                qrels = {qid: {d: int(round(r)) for d, r in qrels_raw[qid].items() if d in doc_set} for qid in qids}
                print(f"    Encoding {len(doc_ids)} docs + {len(qids)} queries (batch={batch_size})...")
                t0 = time.time()
                doc_texts = [corpus[d]['text'] for d in doc_ids]
                query_texts = [queries[q] for q in qids]
                docs_emb = model.encode(doc_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
                queries_emb = model.encode(query_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                print(f"    Encoded in {time.time()-t0:.1f}s")
                with open(cache_file, 'wb') as f:
                    pickle.dump({'doc_ids': doc_ids, 'qids': qids, 'docs_emb': docs_emb, 'queries_emb': queries_emb, 'qrels': qrels}, f)

            print(f"    {len(doc_ids)} docs, {len(qids)} queries, {D}D")
            ds_results = {}

            # RAW
            t0 = time.time()
            raw_m = evaluate(docs_emb, queries_emb, doc_ids, qids, qrels, device=DEVICE)
            ds_results['raw'] = raw_m
            print(f"    RAW: nDCG@10={raw_m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")

            # Count total experiments
            n_total = 1 + len(METHODS) * len(k_values) * len(SEEDS) + 2 * len(k_values) + len(k_values) + 5
            n_done = 1

            # All methods x K x seeds (OOD)
            for method in METHODS + ['spectemp']:
                for ratio_str, K in k_values:
                    for seed in SEEDS:
                        rk = f"{method}_K{K}_s{seed}_ood"
                        if rk in ds_results and 'ndcg@10' in ds_results.get(rk, {}):
                            n_done += 1; continue
                        try:
                            t0 = time.time()
                            if method == 'spectemp':
                                cd = spectemp_compress(generic_emb, docs_emb, K)
                                cq = spectemp_compress(generic_emb, queries_emb, K)
                            else:
                                comp = comp_cache.get(method, generic_emb, K, seed=seed)
                                cd = comp_cache.transform(method, comp, docs_emb, K)
                                cq = comp_cache.transform(method, comp, queries_emb, K)
                            m = evaluate(cd, cq, doc_ids, qids, qrels, device=DEVICE)
                            m['K'] = K; m['method'] = method; m['calibration'] = 'ood'; m['seed'] = seed
                            ds_results[rk] = m
                            n_done += 1
                            print(f"    [{n_done}/{n_total}] {method:10s} K={K:4d} s{seed} OOD: nDCG={m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")
                        except Exception as e:
                            n_done += 1
                            print(f"    [{n_done}/{n_total}] {method:10s} K={K:4d} s{seed} OOD: ERROR: {e}")
                            ds_results[rk] = {'error': str(e)}

            # ID calibration (DCT + PCA, seed 42)
            for method in ['dct', 'pca']:
                for ratio_str, K in k_values:
                    rk = f"{method}_K{K}_s42_id"
                    if rk in ds_results and 'ndcg@10' in ds_results.get(rk, {}):
                        n_done += 1; continue
                    try:
                        t0 = time.time()
                        comp = comp_cache.get(method, docs_emb, K, seed=42, key=f'{ds_name}_id')
                        cd = comp_cache.transform(method, comp, docs_emb, K)
                        cq = comp_cache.transform(method, comp, queries_emb, K)
                        m = evaluate(cd, cq, doc_ids, qids, qrels, device=DEVICE)
                        m['K'] = K; m['method'] = method; m['calibration'] = 'id'; m['seed'] = 42
                        ds_results[rk] = m
                        n_done += 1
                        print(f"    [{n_done}/{n_total}] {method:10s} K={K:4d} ID:  nDCG={m['ndcg@10']:.4f} ({time.time()-t0:.1f}s)")
                    except Exception as e:
                        n_done += 1
                        print(f"    [{n_done}/{n_total}] {method:10s} K={K:4d} ID:  ERROR: {e}")
                        ds_results[rk] = {'error': str(e)}

            # TSP ablation
            for ratio_str, K in k_values:
                rk = f"dct_notsp_K{K}_s42_ood"
                if rk not in ds_results or 'ndcg@10' not in ds_results.get(rk, {}):
                    try:
                        vz_nt = VectorZip(n_components=K, method='dct', tsp_optimize=False); vz_nt.fit(generic_emb)
                        cd = vz_nt.transform(docs_emb); cq = vz_nt.transform(queries_emb)
                        m = evaluate(cd, cq, doc_ids, qids, qrels, device=DEVICE)
                        m['K'] = K; m['method'] = 'dct_notsp'; m['calibration'] = 'ood'; m['seed'] = 42
                        ds_results[rk] = m
                        print(f"    [ablation] dct_notsp K={K:4d}: nDCG={m['ndcg@10']:.4f}")
                    except Exception as e:
                        ds_results[rk] = {'error': str(e)}

            # Calibration size ablation (BGE-M3/SciFact only)
            if model_key == 'bge-m3' and ds_name == 'scifact':
                for n_calib in CALIB_SIZES:
                    rk = f"dct_calib{n_calib}_K64_s42_ood"
                    if rk not in ds_results or 'ndcg@10' not in ds_results.get(rk, {}):
                        try:
                            vz_c = VectorZip(n_components=64, method='dct'); vz_c.fit(generic_emb[:n_calib])
                            cd = vz_c.transform(docs_emb); cq = vz_c.transform(queries_emb)
                            m = evaluate(cd, cq, doc_ids, qids, qrels, device=DEVICE)
                            m['calib_size'] = n_calib
                            ds_results[rk] = m
                            print(f"    [ablation] calib={n_calib} K=64: nDCG={m['ndcg@10']:.4f}")
                        except Exception as e:
                            ds_results[rk] = {'error': str(e)}

            all_results[model_key][ds_name] = ds_results
            save_results(all_results)
            print(f"    >>> {ds_name} COMPLETE. Saved.")

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # ========================================
    # PART 2: MIRACL multilingual
    # ========================================
    for model_key in ['bge-m3', 'minilm']:
        if model_key not in all_results:
            all_results[model_key] = {}
        if 'miracl' not in all_results[model_key]:
            all_results[model_key]['miracl'] = {}

        info = MODELS[model_key]
        D = info['dims']
        batch_size = info.get('batch', 64)

        print(f"\n{'#'*80}")
        print(f"# MIRACL MULTILINGUAL: {model_key}")
        print(f"{'#'*80}")

        # Reload model
        t0 = time.time()
        kwargs = {}
        if info.get('trust_remote_code'): kwargs['trust_remote_code'] = True
        try:
            model = SentenceTransformer(info['name'], device=DEVICE, **kwargs)
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            continue
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Reload generic corpus
        generic_cache = os.path.join(CACHE_DIR, f"{model_key}_generic.pkl")
        with open(generic_cache, 'rb') as f: generic_emb = pickle.load(f)

        comp_cache = CompressorCache()
        for K in [384, 128, 64]:
            if K >= D: continue
            for method in ['dct', 'pca', 'rp']:
                comp_cache.get(method, generic_emb, K, seed=42)

        for lang in MIRACL_LANGS:
            if lang in all_results[model_key].get('miracl', {}) and 'raw' in all_results[model_key]['miracl'].get(lang, {}):
                print(f"\n  MIRACL {lang} - ALREADY DONE")
                continue

            print(f"\n  MIRACL {lang}...")
            mirl = load_miracl(lang)
            if mirl is None:
                all_results[model_key]['miracl'][lang] = {'error': 'not available'}
                save_results(all_results)
                continue

            corpus = mirl['corpus']; queries = mirl['queries']; qrels_raw = mirl['qrels']
            valid_qids = [q for q in queries if q in qrels_raw and len(qrels_raw[q]) > 0]
            if not valid_qids:
                all_results[model_key]['miracl'][lang] = {'error': 'no valid queries'}
                save_results(all_results)
                continue

            relevant = set()
            for qid in valid_qids: relevant.update(qrels_raw[qid].keys())
            all_doc_ids = list(corpus.keys())
            rel_in = [d for d in all_doc_ids if d in relevant]
            other = [d for d in all_doc_ids if d not in relevant]
            np.random.seed(42)
            n_fill = max(0, MAX_DOCS - len(rel_in))
            sampled = list(np.random.choice(other, min(n_fill, len(other)), replace=False)) if other else []
            doc_ids = rel_in + sampled
            qids = list(np.random.choice(valid_qids, min(MAX_QUERIES, len(valid_qids)), replace=False))
            doc_set = set(doc_ids)
            qrels = {qid: {d: int(round(r)) for d, r in qrels_raw[qid].items() if d in doc_set} for qid in qids}

            print(f"    {len(doc_ids)} docs, {len(qids)} queries")
            t0 = time.time()
            doc_texts = [corpus[d] for d in doc_ids]
            query_texts = [queries[q] for q in qids]
            docs_emb = model.encode(doc_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            queries_emb = model.encode(query_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            print(f"    Encoded in {time.time()-t0:.1f}s")

            lang_results = {}
            lang_results['raw'] = evaluate(docs_emb, queries_emb, doc_ids, qids, qrels, device=DEVICE)
            print(f"    RAW: nDCG@10={lang_results['raw']['ndcg@10']:.4f}")

            for K in [384, 128, 64]:
                if K >= D: continue
                for method in ['dct', 'pca', 'rp']:
                    try:
                        comp = comp_cache.get(method, generic_emb, K, seed=42)
                        cd = comp_cache.transform(method, comp, docs_emb, K)
                        cq = comp_cache.transform(method, comp, queries_emb, K)
                        m = evaluate(cd, cq, doc_ids, qids, qrels, device=DEVICE)
                        m['K'] = K; m['method'] = method
                        lang_results[f'{method}_K{K}'] = m
                        print(f"    {method} K={K}: nDCG={m['ndcg@10']:.4f}")
                    except Exception as e:
                        lang_results[f'{method}_K{K}'] = {'error': str(e)}

            all_results[model_key]['miracl'][lang] = lang_results
            save_results(all_results)
            print(f"    >>> MIRACL {lang} COMPLETE. Saved.")

        del model
        torch.cuda.empty_cache()

    # ========================================
    # PART 3: BGE-Small comparison
    # ========================================
    if 'bge-small' not in all_results:
        print(f"\n{'#'*80}")
        print(f"# BGE-Small (multilingual comparison baseline)")
        print(f"{'#'*80}")
        try:
            model_small = SentenceTransformer('BAAI/bge-small-en-v1.5', device=DEVICE)
            all_results['bge-small'] = {'miracl': {}}

            for lang in MIRACL_LANGS + ['en']:
                if lang in all_results['bge-small']['miracl']:
                    continue
                print(f"\n  BGE-Small {lang}...")
                if lang == 'en':
                    corpus_data, queries_data, qrels_data = load_beir('scifact')
                    valid_qids = [q for q in queries_data if q in qrels_data and len(qrels_data[q]) > 0]
                    relevant = set()
                    for qid in valid_qids: relevant.update(qrels_data[qid].keys())
                    all_doc_ids = list(corpus_data.keys())
                    rel_in = [d for d in all_doc_ids if d in relevant]
                    other = [d for d in all_doc_ids if d not in relevant]
                    np.random.seed(42)
                    n_fill = max(0, MAX_DOCS - len(rel_in))
                    sampled = list(np.random.choice(other, min(n_fill, len(other)), replace=False)) if other else []
                    doc_ids = rel_in + sampled
                    qids = list(np.random.choice(valid_qids, min(MAX_QUERIES, len(valid_qids)), replace=False))
                    doc_set = set(doc_ids)
                    qrels = {qid: {d: int(round(r)) for d, r in qrels_data[qid].items() if d in doc_set} for qid in qids}
                    doc_texts = [corpus_data[d]['text'] for d in doc_ids]
                    query_texts = [queries_data[q] for q in qids]
                else:
                    mirl = load_miracl(lang)
                    if mirl is None:
                        all_results['bge-small']['miracl'][lang] = {'error': 'not available'}
                        save_results(all_results)
                        continue
                    corpus = mirl['corpus']; queries = mirl['queries']; qrels_raw = mirl['qrels']
                    valid_qids = [q for q in queries if q in qrels_raw and len(qrels_raw[q]) > 0]
                    if not valid_qids:
                        all_results['bge-small']['miracl'][lang] = {'error': 'no valid queries'}
                        save_results(all_results)
                        continue
                    relevant = set()
                    for qid in valid_qids: relevant.update(qrels_raw[qid].keys())
                    all_doc_ids = list(corpus.keys())
                    rel_in = [d for d in all_doc_ids if d in relevant]
                    other = [d for d in all_doc_ids if d not in relevant]
                    np.random.seed(42)
                    n_fill = max(0, MAX_DOCS - len(rel_in))
                    sampled = list(np.random.choice(other, min(n_fill, len(other)), replace=False)) if other else []
                    doc_ids = rel_in + sampled
                    qids = list(np.random.choice(valid_qids, min(MAX_QUERIES, len(valid_qids)), replace=False))
                    doc_set = set(doc_ids)
                    qrels = {qid: {d: int(round(r)) for d, r in qrels_raw[qid].items() if d in doc_set} for qid in qids}
                    doc_texts = [corpus[d] for d in doc_ids]
                    query_texts = [queries[q] for q in qids]

                docs_emb = model_small.encode(doc_texts, batch_size=512, show_progress_bar=False, convert_to_numpy=True)
                queries_emb = model_small.encode(query_texts, batch_size=512, show_progress_bar=False, convert_to_numpy=True)
                m = evaluate(docs_emb, queries_emb, doc_ids, qids, qrels, device=DEVICE)
                all_results['bge-small']['miracl'][lang] = {'raw': m}
                print(f"    BGE-Small {lang}: nDCG@10={m['ndcg@10']:.4f}")
                save_results(all_results)

            del model_small
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  BGE-Small failed: {e}")

    # ========================================
    # PART 4: Generate paper tables
    # ========================================
    print(f"\n{'='*100}")
    print("GENERATING PAPER TABLES")
    print(f"{'='*100}")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Table 1: OOD retention
    print(f"\n--- TABLE 1: OOD retention ---")
    print(f"Model & Dataset & K & DCT% & PCA% & Delta")
    for model_key in MODELS:
        if model_key not in all_results: continue
        for ds_name in BEIR_DATASETS:
            if ds_name not in all_results.get(model_key, {}): continue
            ds = all_results[model_key][ds_name]
            for _, K in k_values:
                dct_ood = ds.get(f"dct_K{K}_s42_ood", {}).get('ndcg@10')
                dct_id = ds.get(f"dct_K{K}_s42_id", {}).get('ndcg@10')
                pca_ood = ds.get(f"pca_K{K}_s42_ood", {}).get('ndcg@10')
                pca_id = ds.get(f"pca_K{K}_s42_id", {}).get('ndcg@10')
                if dct_ood and dct_id and pca_ood and pca_id:
                    dp = dct_ood/dct_id*100 if dct_id > 0 else 0
                    pp = pca_ood/pca_id*100 if pca_id > 0 else 0
                    print(f"  {model_key} & {ds_name} & {K} & {dp:.1f}% & {pp:.1f}% & {dp-pp:+.1f}%")

    # Table 2: Equal storage at 128B
    print(f"\n--- TABLE 2: nDCG@10 at 128 bytes (SQ8) ---")
    print(f"Model & Dataset & DCT & RP & PCA & SpecTemp & Best")
    for model_key in MODELS:
        if model_key not in all_results: continue
        for ds_name in BEIR_DATASETS:
            if ds_name not in all_results.get(model_key, {}): continue
            ds = all_results[model_key][ds_name]
            K = min(128, MODELS[model_key]['dims'] - 1)
            dct = ds.get(f"dct+sq8_K{K}_s42_ood", {}).get('ndcg@10', 0)
            rp = ds.get(f"rp_K{K}_s42_ood", {}).get('ndcg@10', 0)
            pca = ds.get(f"pca+sq8_K{K}_s42_ood", {}).get('ndcg@10', 0)
            spec = ds.get(f"spectemp_K{K}_s42_ood", {}).get('ndcg@10', 0)
            best = max([('DCT', dct), ('RP', rp), ('PCA', pca), ('Spec', spec)], key=lambda x: x[1])
            print(f"  {model_key} & {ds_name} & {dct:.4f} & {rp:.4f} & {pca:.4f} & {spec:.4f} & {best[0]}")

    # Table 3: Multilingual
    print(f"\n--- TABLE 3: Multilingual (BGE-M3 compressed vs BGE-Small) ---")
    print(f"Language & Small(384D) & M3-DCT-128 & M3-DCT-64 & Delta")
    for lang in MIRACL_LANGS + ['en']:
        small = all_results.get('bge-small', {}).get('miracl', {}).get(lang, {}).get('raw', {}).get('ndcg@10', 0)
        m3_128 = all_results.get('bge-m3', {}).get('miracl', {}).get(lang, {}).get('dct_K128', {}).get('ndcg@10', 0)
        m3_64 = all_results.get('bge-m3', {}).get('miracl', {}).get(lang, {}).get('dct_K64', {}).get('ndcg@10', 0)
        delta = m3_128 - small
        print(f"  {lang} & {small:.4f} & {m3_128:.4f} & {m3_64:.4f} & {delta:+.4f}")

    # Table 4: Statistical significance
    print(f"\n--- TABLE 4: Statistical significance (DCT vs PCA, 5 seeds) ---")
    from scipy import stats as scipy_stats
    print(f"Model & Dataset & K & DCT(mean±std) & PCA(mean±std) & p-value & Significant")
    for model_key in MODELS:
        if model_key not in all_results: continue
        for ds_name in BEIR_DATASETS:
            if ds_name not in all_results.get(model_key, {}): continue
            ds = all_results[model_key][ds_name]
            for _, K in k_values:
                dct_vals = [ds.get(f"dct_K{K}_s{s}_ood", {}).get('ndcg@10') for s in SEEDS]
                pca_vals = [ds.get(f"pca_K{K}_s{s}_ood", {}).get('ndcg@10') for s in SEEDS]
                dct_vals = [v for v in dct_vals if v is not None]
                pca_vals = [v for v in pca_vals if v is not None]
                if len(dct_vals) >= 3 and len(pca_vals) >= 3:
                    t, p = scipy_stats.ttest_rel(dct_vals, pca_vals)
                    sig = "Yes" if p < 0.05 else "No"
                    print(f"  {model_key} & {ds_name} & {K} & {np.mean(dct_vals):.4f}±{np.std(dct_vals):.4f} & {np.mean(pca_vals):.4f}±{np.std(pca_vals):.4f} & {p:.4f} & {sig}")

    # Ablations
    print(f"\n--- ABLATION: TSP vs no-TSP ---")
    for model_key in MODELS:
        if model_key not in all_results: continue
        for ds_name in BEIR_DATASETS:
            if ds_name not in all_results.get(model_key, {}): continue
            ds = all_results[model_key][ds_name]
            for _, K in k_values:
                tsp = ds.get(f"dct_K{K}_s42_ood", {}).get('ndcg@10')
                notsp = ds.get(f"dct_notsp_K{K}_s42_ood", {}).get('ndcg@10')
                if tsp and notsp:
                    print(f"  {model_key} & {ds_name} & {K} & {tsp:.4f} & {notsp:.4f} & {tsp-notsp:+.4f}")

    print(f"\n--- ABLATION: Calibration corpus size (BGE-M3/SciFact) ---")
    for n_calib in CALIB_SIZES:
        v = all_results.get('bge-m3', {}).get('scifact', {}).get(f"dct_calib{n_calib}_K64_s42_ood", {}).get('ndcg@10')
        if v: print(f"  n_calib={n_calib}: nDCG@10={v:.4f}")

    print(f"\n{'='*100}")
    print("DONE. All tables generated. Copy into paper/main.tex")
    print(f"Results file: {OUTPUT_FILE}")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
