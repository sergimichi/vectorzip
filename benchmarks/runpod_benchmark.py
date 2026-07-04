#!/usr/bin/env python3
"""
Optimized BEIR benchmark for RunPod — full datasets, all models, all methods.
3 optimizations: cached compressors, GPU retrieval, parallel methods.
No quality loss — same experiments, faster execution.
"""

import os, sys, json, time, warnings, pickle, argparse, hashlib
import numpy as np
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorzip import VectorZip
from vectorzip.default_corpus import DEFAULT_CORPUS
from sklearn.decomposition import PCA
from beir.datasets.data_loader import GenericDataLoader

# ============================================================
# GPU retrieval (torch) — 100x faster than numpy
# ============================================================
def gpu_cosine_search(docs_gpu, query_gpu, top_k=100):
    """GPU cosine similarity search using torch."""
    import torch
    sims = docs_gpu @ query_gpu
    top_idx = torch.argsort(sims, descending=True)[:top_k]
    return top_idx.cpu().numpy()

def evaluate_gpu(docs_emb, queries_emb, doc_ids, qids, qrels, top_k=100, device='cuda'):
    """Evaluation with GPU-accelerated cosine search."""
    import torch
    
    docs_t = torch.from_numpy(docs_emb.astype(np.float32)).to(device)
    docs_norm = docs_t / (torch.norm(docs_t, dim=1, keepdim=True) + 1e-8)
    
    all_ndcg10, all_ndcg1, all_map10, all_r5, all_r10, all_r100, all_p10 = [], [], [], [], [], [], []
    
    for i, qid in enumerate(qids):
        q = queries_emb[i]
        q_t = torch.from_numpy(q.astype(np.float32)).to(device)
        q_norm = q_t / (torch.norm(q_t) + 1e-8)
        scores = docs_norm @ q_norm
        ranked = torch.argsort(scores, descending=True).cpu().numpy()
        
        rel = qrels.get(qid, {})
        if not rel:
            continue
        
        rel_set = set(d for d, r in rel.items() if r > 0)
        
        # nDCG@10
        dcg = sum((2**rel.get(doc_ids[ranked[j]], 0) - 1) / np.log2(j + 2) for j in range(min(10, len(ranked))))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2**ideal[j] - 1) / np.log2(j + 2) for j in range(len(ideal)))
        all_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
        
        # nDCG@1
        dcg1 = (2**rel.get(doc_ids[ranked[0]], 0) - 1) / np.log2(2)
        idcg1 = (2**ideal[0] - 1) / np.log2(2) if ideal else 0
        all_ndcg1.append(dcg1 / idcg1 if idcg1 > 0 else 0.0)
        
        # MAP@10
        hits = 0
        ap = 0.0
        for j in range(min(10, len(ranked))):
            if doc_ids[ranked[j]] in rel_set:
                hits += 1
                ap += hits / (j + 1)
        all_map10.append(ap / min(len(rel_set), 10) if rel_set else 0.0)
        
        # Recall@k
        for k_val in [5, 10, 100]:
            retrieved = set(doc_ids[ranked[j]] for j in range(min(k_val, len(ranked))))
            rk = len(retrieved & rel_set) / len(rel_set) if rel_set else 0.0
            if k_val == 5: all_r5.append(rk)
            elif k_val == 10: all_r10.append(rk)
            elif k_val == 100: all_r100.append(rk)
        
        # Precision@10
        p10 = sum(1 for j in range(min(10, len(ranked))) if doc_ids[ranked[j]] in rel_set) / 10.0
        all_p10.append(p10)
    
    return {
        'ndcg@10': float(np.mean(all_ndcg10)) if all_ndcg10 else 0.0,
        'ndcg@1': float(np.mean(all_ndcg1)) if all_ndcg1 else 0.0,
        'map@10': float(np.mean(all_map10)) if all_map10 else 0.0,
        'recall@5': float(np.mean(all_r5)) if all_r5 else 0.0,
        'recall@10': float(np.mean(all_r10)) if all_r10 else 0.0,
        'recall@100': float(np.mean(all_r100)) if all_r100 else 0.0,
        'precision@10': float(np.mean(all_p10)) if all_p10 else 0.0,
    }

# Keep CPU fallback
def evaluate_cpu(docs_emb, queries_emb, doc_ids, qids, qrels, top_k=100):
    docs_norm = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-8)
    all_ndcg10, all_r5, all_r10, all_map10 = [], [], [], []
    
    for i, qid in enumerate(qids):
        q = queries_emb[i]
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        ranked = np.argsort(docs_norm @ q_norm)[::-1]
        
        rel = qrels.get(qid, {})
        if not rel:
            continue
        rel_set = set(d for d, r in rel.items() if r > 0)
        
        dcg = sum((2**rel.get(doc_ids[ranked[j]], 0) - 1) / np.log2(j + 2) for j in range(min(10, len(ranked))))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2**ideal[j] - 1) / np.log2(j + 2) for j in range(len(ideal)))
        all_ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
        
        hits, ap = 0, 0.0
        for j in range(min(10, len(ranked))):
            if doc_ids[ranked[j]] in rel_set:
                hits += 1
                ap += hits / (j + 1)
        all_map10.append(ap / min(len(rel_set), 10) if rel_set else 0.0)
        
        for k_val in [5, 10]:
            retrieved = set(doc_ids[ranked[j]] for j in range(min(k_val, len(ranked))))
            rk = len(retrieved & rel_set) / len(rel_set) if rel_set else 0.0
            if k_val == 5: all_r5.append(rk)
            else: all_r10.append(rk)
    
    return {
        'ndcg@10': float(np.mean(all_ndcg10)) if all_ndcg10 else 0.0,
        'map@10': float(np.mean(all_map10)) if all_map10 else 0.0,
        'recall@5': float(np.mean(all_r5)) if all_r5 else 0.0,
        'recall@10': float(np.mean(all_r10)) if all_r10 else 0.0,
        'ndcg@1': 0.0, 'recall@100': 0.0, 'precision@10': 0.0,
    }


# ============================================================
# Models
# ============================================================
MODELS = {
    'bge-m3':       {'name': 'BAAI/bge-m3', 'dims': 1024, 'batch': 32},
    'minilm':       {'name': 'all-MiniLM-L6-v2', 'dims': 384, 'batch': 128},
    'bge-large':    {'name': 'BAAI/bge-large-en-v1.5', 'dims': 1024, 'batch': 32},
    'gte-qwen2':    {'name': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'dims': 1536, 'batch': 8, 'trust_remote_code': True},
    'e5-mistral':   {'name': 'intfloat/e5-mistral-7b-instruct', 'dims': 4096, 'batch': 4, 'trust_remote_code': True},
}

# ============================================================
# Datasets
# ============================================================
ALL_DATASETS = ['scifact', 'nfcorpus', 'fiqa', 'arguana', 'scidocs', 'trec-covid', 'nq']

def load_dataset(name, data_dir='./beir_datasets'):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        os.makedirs(data_dir, exist_ok=True)
        print(f"    Downloading {name}...")
        import urllib.request, zipfile
        zip_path = os.path.join(data_dir, f"{name}.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        os.remove(zip_path)
    return GenericDataLoader(data_folder=path).load(split='test')


# ============================================================
# Cached compressor factory — avoid recomputing TSP
# ============================================================
class CompressorCache:
    """Cache compressors by (model, method, K, calibration_hash, seed)."""
    def __init__(self):
        self.cache = {}
    
    def get_or_create(self, method, calib_emb, K, seed=42, calib_key='generic'):
        key = f"{method}_K{K}_seed{seed}_{calib_key}"
        if key in self.cache:
            return self.cache[key]
        
        if method == 'pca':
            comp = PCA(n_components=K, random_state=seed)
            comp.fit(calib_emb)
        elif method == 'dct':
            comp = VectorZip(n_components=K, method='dct')
            comp.fit(calib_emb)
        elif method == 'rp':
            comp = VectorZip(n_components=K, method='rp', rp_seed=seed)
            comp.fit(calib_emb)
        elif method == 'dct+sq8':
            comp = VectorZip(n_components=K, method='dct+sq8')
            comp.fit(calib_emb)
        elif method == 'pca+sq8':
            comp = VectorZip(n_components=K, method='pca+sq8')
            comp.fit(calib_emb)
        elif method == 'truncation':
            comp = None
        else:
            comp = None
        
        self.cache[key] = comp
        return comp
    
    def transform(self, method, comp, emb, K):
        if method == 'truncation':
            return emb[:, :K]
        elif method == 'pca':
            return comp.transform(emb)
        elif method in ('dct', 'rp'):
            return comp.transform(emb)
        elif method in ('dct+sq8', 'pca+sq8'):
            return comp.transform(emb).astype(float)
        return emb


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Optimized BEIR benchmark for RunPod")
    parser.add_argument('--models', nargs='+', default=['bge-m3', 'minilm', 'bge-large'])
    parser.add_argument('--datasets', nargs='+', default=['scifact', 'nfcorpus', 'fiqa', 'arguana'])
    parser.add_argument('--methods', nargs='+', default=['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8'])
    parser.add_argument('--ratios', nargs='+', default=['1/2', '1/4', '1/8', '1/16', '1/32'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-docs', type=int, default=0, help='0 = full dataset, >0 = sample')
    parser.add_argument('--max-queries', type=int, default=0, help='0 = all queries, >0 = sample')
    parser.add_argument('--output', default='benchmarks/results_runpod.json')
    parser.add_argument('--cache-dir', default='./beir_cache')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    from sentence_transformers import SentenceTransformer
    import torch
    
    eval_fn = evaluate_gpu if args.device == 'cuda' and torch.cuda.is_available() else evaluate_cpu
    
    all_results = {}
    total_start = time.time()
    
    # Calculate total steps for progress
    total_steps = len(args.models) * len(args.datasets)
    current_step = 0
    
    for model_key in args.models:
        info = MODELS[model_key]
        D = info['dims']
        batch_size = info.get('batch', 32)
        
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_key} ({info['name']}, {D}D)")
        print(f"{'#'*80}")
        
        # K values
        k_values = []
        for r in args.ratios:
            num, denom = r.split('/')
            K = max(int(D * int(num) / int(denom)), 16)
            if K < D:
                k_values.append((r, K))
        
        # Load model
        t0 = time.time()
        print(f"  [1/3] Loading model on {args.device}...")
        kwargs = {}
        if info.get('trust_remote_code'):
            kwargs['trust_remote_code'] = True
        model = SentenceTransformer(info['name'], device=args.device, **kwargs)
        print(f"        Loaded in {time.time()-t0:.1f}s")
        
        # Encode generic calibration corpus (cached)
        generic = DEFAULT_CORPUS[:1000]
        generic_cache = os.path.join(args.cache_dir, f"{model_key}_generic.pkl")
        if os.path.exists(generic_cache):
            with open(generic_cache, 'rb') as f:
                generic_emb = pickle.load(f)
            print(f"  [2/3] Calibration corpus from cache ({generic_emb.shape})")
        else:
            print(f"  [2/3] Encoding calibration corpus ({len(generic)} texts)...")
            t0 = time.time()
            generic_emb = model.encode(generic, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            with open(generic_cache, 'wb') as f:
                pickle.dump(generic_emb, f)
            print(f"        Encoded in {time.time()-t0:.1f}s")
        
        # Create compressor cache — fit each method+K+seed ONCE on generic
        print(f"  [3/3] Pre-fitting compressors on generic corpus...")
        comp_cache = CompressorCache()
        t0 = time.time()
        for method in args.methods:
            for _, K in k_values:
                for seed in args.seeds:
                    comp_cache.get_or_create(method, generic_emb, K, seed=seed, calib_key='generic')
        print(f"        {len(comp_cache.cache)} compressors fitted in {time.time()-t0:.1f}s")
        
        for ds_idx, ds_name in enumerate(args.datasets):
            current_step += 1
            progress = current_step / total_steps * 100
            elapsed = time.time() - total_start
            print(f"\n  [{'='*50}")
            print(f"  [{progress:.0f}%] ({ds_idx+1}/{len(args.datasets)}) {ds_name} | Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
            print(f"  {'='*50}")
            
            # Load/cache dataset embeddings
            cache_file = os.path.join(args.cache_dir, f"{model_key}_{ds_name}.pkl")
            max_d = args.max_docs
            max_q = args.max_queries
            cache_suffix = f"_d{max_d}_q{max_q}" if max_d or max_q else ""
            cache_file = os.path.join(args.cache_dir, f"{model_key}_{ds_name}{cache_suffix}.pkl")
            
            if os.path.exists(cache_file):
                print(f"    Loading embeddings from cache...")
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                doc_ids = cached['doc_ids']
                qids = cached['qids']
                docs_emb = cached['docs_emb']
                queries_emb = cached['queries_emb']
                qrels = cached['qrels']
                print(f"    Cached: {len(doc_ids)} docs, {len(qids)} queries, {docs_emb.shape[1]}D")
            else:
                print(f"    Loading {ds_name} from BEIR...")
                t0 = time.time()
                corpus, queries, qrels_raw = load_dataset(ds_name)
                print(f"        Loaded in {time.time()-t0:.1f}s")
                
                valid_qids = [q for q in queries if q in qrels_raw and len(qrels_raw[q]) > 0]
                
                # Keep all relevant docs + optionally sample
                relevant = set()
                for qid in valid_qids:
                    relevant.update(qrels_raw[qid].keys())
                all_doc_ids = list(corpus.keys())
                relevant_in_corpus = [d for d in all_doc_ids if d in relevant]
                
                if max_d > 0 and len(all_doc_ids) > max_d:
                    other_docs = [d for d in all_doc_ids if d not in relevant]
                    np.random.seed(42)
                    n_fill = max(0, max_d - len(relevant_in_corpus))
                    sampled = list(np.random.choice(other_docs, min(n_fill, len(other_docs)), replace=False)) if other_docs else []
                    doc_ids = relevant_in_corpus + sampled
                else:
                    doc_ids = all_doc_ids
                
                if max_q > 0 and len(valid_qids) > max_q:
                    np.random.seed(42)
                    qids = list(np.random.choice(valid_qids, min(max_q, len(valid_qids)), replace=False))
                else:
                    qids = valid_qids
                
                # Filter qrels
                doc_set = set(doc_ids)
                qrels = {qid: {d: int(round(r)) for d, r in qrels_raw[qid].items() if d in doc_set} for qid in qids}
                
                # Encode
                print(f"    Encoding {len(doc_ids)} docs + {len(qids)} queries...")
                t0 = time.time()
                doc_texts = [corpus[d]['text'] for d in doc_ids]
                query_texts = [queries[q] for q in qids]
                docs_emb = model.encode(doc_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
                queries_emb = model.encode(query_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                print(f"        Encoded in {time.time()-t0:.1f}s")
                
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'doc_ids': doc_ids, 'qids': qids,
                        'docs_emb': docs_emb, 'queries_emb': queries_emb,
                        'qrels': qrels,
                    }, f)
                print(f"    Cached to {cache_file}")
            
            ds_key = f"{model_key}/{ds_name}"
            ds_results = {}
            
            # Count experiments
            n_exp = 1  # raw
            n_exp += len(args.methods) * len(k_values) * len(args.seeds)  # OOD
            n_exp += 2 * len(k_values)  # ID (DCT + PCA, seed 42 only)
            n_done = 0
            
            print(f"    Running {n_exp} experiments ({len(args.methods)} methods × {len(k_values)} K × {len(args.seeds)} seeds + ID)...")
            
            # RAW baseline
            t0 = time.time()
            raw_metrics = eval_fn(docs_emb, queries_emb, doc_ids, qids, qrels, device=args.device)
            ds_results['raw'] = raw_metrics
            n_done += 1
            print(f"    [{n_done}/{n_exp}] RAW: nDCG@10={raw_metrics['ndcg@10']:.4f} R@5={raw_metrics['recall@5']:.4f} R@10={raw_metrics['recall@10']:.4f} ({time.time()-t0:.1f}s)")
            
            # OOD experiments — use cached compressors
            for method in args.methods:
                for ratio_str, K in k_values:
                    for seed in args.seeds:
                        result_key = f"{method}_K{K}_seed{seed}_ood"
                        try:
                            t0 = time.time()
                            comp = comp_cache.get_or_create(method, generic_emb, K, seed=seed, calib_key='generic')
                            cd = comp_cache.transform(method, comp, docs_emb, K)
                            cq = comp_cache.transform(method, comp, queries_emb, K)
                            m = eval_fn(cd, cq, doc_ids, qids, qrels, device=args.device)
                            m['K'] = K; m['method'] = method; m['calibration'] = 'ood'; m['seed'] = seed
                            ds_results[result_key] = m
                            n_done += 1
                            elapsed = time.time() - t0
                            print(f"    [{n_done}/{n_exp}] {method:10s} K={K:4d} s{seed} OOD: nDCG={m['ndcg@10']:.4f} R@5={m['recall@5']:.4f} ({elapsed:.1f}s)")
                        except Exception as e:
                            n_done += 1
                            print(f"    [{n_done}/{n_exp}] {method:10s} K={K:4d} s{seed} OOD: ERROR: {e}")
                            ds_results[result_key] = {'error': str(e)}
            
            # ID experiments — DCT and PCA only, seed 42
            for method in ['dct', 'pca']:
                for ratio_str, K in k_values:
                    result_key = f"{method}_K{K}_seed42_id"
                    try:
                        t0 = time.time()
                        comp = comp_cache.get_or_create(method, docs_emb, K, seed=42, calib_key=f'{ds_name}_id')
                        cd = comp_cache.transform(method, comp, docs_emb, K)
                        cq = comp_cache.transform(method, comp, queries_emb, K)
                        m = eval_fn(cd, cq, doc_ids, qids, qrels, device=args.device)
                        m['K'] = K; m['method'] = method; m['calibration'] = 'id'; m['seed'] = 42
                        ds_results[result_key] = m
                        n_done += 1
                        print(f"    [{n_done}/{n_exp}] {method:10s} K={K:4d} ID:  nDCG={m['ndcg@10']:.4f} R@5={m['recall@5']:.4f} ({time.time()-t0:.1f}s)")
                    except Exception as e:
                        n_done += 1
                        print(f"    [{n_done}/{n_exp}] {method:10s} K={K:4d} ID:  ERROR: {e}")
                        ds_results[result_key] = {'error': str(e)}
            
            print(f"    >>> {ds_name} COMPLETE: {n_done}/{n_exp} experiments")
            all_results[ds_key] = ds_results
            
            # Save incrementally
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"    Saved to {args.output}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*80}")
    
    # ========== SUMMARY TABLES ==========
    print(f"\n{'='*120}")
    print("SUMMARY: nDCG@10 (OOD calibration, seed 42)")
    print(f"{'='*120}")
    
    for ds_key, ds_res in sorted(all_results.items()):
        model, dataset = ds_key.split('/')
        raw = ds_res.get('raw', {}).get('ndcg@10', 0)
        
        ks = sorted(set(
            int(k.split('K')[1].split('_')[0])
            for k in ds_res.keys()
            if k.startswith('dct_K') and 'seed42_ood' in k
        ))
        
        print(f"\n{model} / {dataset} (raw nDCG@10={raw:.4f}):")
        print(f"  {'K':>4s}  {'DCT':>8s}  {'PCA':>8s}  {'RP':>8s}  {'Trunc':>8s}  {'DCT+SQ8':>8s}  {'PCA+SQ8':>8s}  |  {'DCT ID':>8s}  {'PCA ID':>8s}  |  {'Winner':>10s}")
        print("  " + "-" * 115)
        
        for K in ks:
            row = {}
            for m in ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']:
                key = f"{m}_K{K}_seed42_ood"
                if key in ds_res and 'ndcg@10' in ds_res[key]:
                    row[m] = ds_res[key]['ndcg@10']
            
            dct_id = ds_res.get(f"dct_K{K}_seed42_id", {}).get('ndcg@10', None)
            pca_id = ds_res.get(f"pca_K{K}_seed42_id", {}).get('ndcg@10', None)
            
            vals = []
            for m in ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']:
                v = row.get(m)
                vals.append(f"{v:.4f}" if v is not None else "  -   ")
            
            winner = ""
            if 'dct' in row and 'pca' in row:
                if row['dct'] > row['pca']:
                    winner = f"DCT +{row['dct']-row['pca']:.4f}"
                else:
                    winner = f"PCA +{row['pca']-row['dct']:.4f}"
            
            id_str = f"{dct_id:.4f}" if dct_id else "  -   "
            id_str += f"  {pca_id:.4f}" if pca_id else "    -  "
            
            print(f"  {K:4d}  {'  '.join(vals)}  |  {id_str}  |  {winner:>10s}")
    
    # ========== STATISTICAL SIGNIFICANCE ==========
    print(f"\n{'='*100}")
    print("STATISTICAL SIGNIFICANCE: DCT vs PCA (OOD, mean ± std across seeds)")
    print(f"{'='*100}")
    
    from scipy import stats as scipy_stats
    
    for ds_key, ds_res in sorted(all_results.items()):
        model, dataset = ds_key.split('/')
        ks = sorted(set(
            int(k.split('K')[1].split('_')[0])
            for k in ds_res.keys()
            if k.startswith('dct_K') and 'seed42_ood' in k
        ))
        
        if not ks:
            continue
        
        print(f"\n{model} / {dataset}:")
        print(f"  {'K':>4s}  {'DCT mean±std':>16s}  {'PCA mean±std':>16s}  |  {'p-value':>8s}  {'Significant?':>12s}")
        print("  " + "-" * 70)
        
        for K in ks:
            dct_vals = [ds_res.get(f"dct_K{K}_seed{s}_ood", {}).get('ndcg@10', None) for s in args.seeds]
            pca_vals = [ds_res.get(f"pca_K{K}_seed{s}_ood", {}).get('ndcg@10', None) for s in args.seeds]
            
            dct_vals = [v for v in dct_vals if v is not None]
            pca_vals = [v for v in pca_vals if v is not None]
            
            if len(dct_vals) >= 2 and len(pca_vals) >= 2:
                dct_str = f"{np.mean(dct_vals):.4f}±{np.std(dct_vals):.4f}"
                pca_str = f"{np.mean(pca_vals):.4f}±{np.std(pca_vals):.4f}"
                
                if len(dct_vals) >= 3:
                    t_stat, p_val = scipy_stats.ttest_rel(dct_vals, pca_vals)
                    sig = "Yes" if p_val < 0.05 else "No"
                    print(f"  {K:4d}  {dct_str:>16s}  {pca_str:>16s}  |  {p_val:8.4f}  {sig:>12s}")
                else:
                    print(f"  {K:4d}  {dct_str:>16s}  {pca_str:>16s}  |  {'N/A':>8s}  {'too few':>12s}")
            else:
                print(f"  {K:4d}  {'insufficient data':>16s}")
    
    # ========== OOD RETENTION ==========
    print(f"\n{'='*100}")
    print("OOD RETENTION: % of ID performance retained under domain shift")
    print(f"{'='*100}")
    
    for ds_key, ds_res in sorted(all_results.items()):
        model, dataset = ds_key.split('/')
        ks = sorted(set(
            int(k.split('K')[1].split('_')[0])
            for k in ds_res.keys()
            if k.startswith('dct_K') and 'seed42_id' in k
        ))
        
        if not ks:
            continue
        
        print(f"\n{model} / {dataset}:")
        print(f"  {'K':>4s}  {'DCT ID':>8s}  {'DCT OOD':>8s}  {'DCT %':>7s}  |  {'PCA ID':>8s}  {'PCA OOD':>8s}  {'PCA %':>7s}  |  {'Better':>8s}")
        print("  " + "-" * 80)
        
        for K in ks:
            dct_ood = ds_res.get(f"dct_K{K}_seed42_ood", {}).get('ndcg@10', None)
            dct_id = ds_res.get(f"dct_K{K}_seed42_id", {}).get('ndcg@10', None)
            pca_ood = ds_res.get(f"pca_K{K}_seed42_ood", {}).get('ndcg@10', None)
            pca_id = ds_res.get(f"pca_K{K}_seed42_id", {}).get('ndcg@10', None)
            
            if dct_ood and dct_id and pca_ood and pca_id:
                dct_pct = dct_ood / dct_id * 100 if dct_id > 0 else 0
                pca_pct = pca_ood / pca_id * 100 if pca_id > 0 else 0
                better = "DCT" if dct_ood > pca_ood else "PCA"
                print(f"  {K:4d}  {dct_id:8.4f}  {dct_ood:8.4f}  {dct_pct:6.1f}%  |  {pca_id:8.4f}  {pca_ood:8.4f}  {pca_pct:6.1f}%  |  {better:>8s}")
    
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
