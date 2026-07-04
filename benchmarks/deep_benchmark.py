#!/usr/bin/env python3
"""
Deep benchmark with embedding caching. Encodes each model once per dataset,
then runs all compression methods on cached embeddings.
"""

import os, sys, json, time, warnings, pickle
import numpy as np
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorzip import VectorZip
from vectorzip.default_corpus import DEFAULT_CORPUS
from sklearn.decomposition import PCA
from beir.datasets.data_loader import GenericDataLoader

# ============================================================
# Models
# ============================================================
MODELS = {
    'bge-m3':    {'name': 'BAAI/bge-m3', 'dims': 1024},
    'minilm':    {'name': 'all-MiniLM-L6-v2', 'dims': 384},
    'bge-large': {'name': 'BAAI/bge-large-en-v1.5', 'dims': 1024},
}

# ============================================================
# Dataset loader
# ============================================================
def load_dataset(name, data_dir='/home/sergio/beir_datasets'):
    from beir.datasets.data_loader import GenericDataLoader
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
# Evaluation using proper BEIR metrics
# ============================================================
def evaluate(docs_emb, queries_emb, doc_ids, qids, qrels, top_k=100):
    """Manual BEIR-style evaluation (no pytrec_eval dependency)."""
    docs_norm = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-8)

    all_ndcg10, all_ndcg1, all_map10, all_r5, all_r10, all_r100, all_p10 = [], [], [], [], [], [], []

    for i, qid in enumerate(qids):
        q = queries_emb[i]
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scores = docs_norm @ q_norm
        ranked = np.argsort(scores)[::-1]

        rel = qrels.get(qid, {})
        if not rel:
            continue

        # Binary relevance
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
        for k in [5, 10, 100]:
            retrieved = set(doc_ids[ranked[j]] for j in range(min(k, len(ranked))))
            rk = len(retrieved & rel_set) / len(rel_set) if rel_set else 0.0
            if k == 5: all_r5.append(rk)
            elif k == 10: all_r10.append(rk)
            elif k == 100: all_r100.append(rk)

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

# ============================================================
# Compression
# ============================================================
def compress_and_eval(method, calib_emb, docs_emb, queries_emb, K, doc_ids, qids, qrels):
    """Compress and evaluate in one step."""
    if method == 'raw':
        return evaluate(docs_emb, queries_emb, doc_ids, qids, qrels)

    elif method == 'truncation':
        cd = docs_emb[:, :K]
        cq = queries_emb[:, :K]
        return evaluate(cd, cq, doc_ids, qids, qrels)

    elif method == 'pca':
        pca = PCA(n_components=K, random_state=42)
        pca.fit(calib_emb)
        cd = pca.transform(docs_emb)
        cq = pca.transform(queries_emb)
        return evaluate(cd, cq, doc_ids, qids, qrels)

    elif method == 'dct':
        vz = VectorZip(n_components=K, method='dct')
        vz.fit(calib_emb)
        cd = vz.transform(docs_emb)
        cq = vz.transform(queries_emb)
        return evaluate(cd, cq, doc_ids, qids, qrels)

    elif method == 'rp':
        vz = VectorZip(n_components=K, method='rp', rp_seed=42)
        vz.fit(calib_emb)
        cd = vz.transform(docs_emb)
        cq = vz.transform(queries_emb)
        return evaluate(cd, cq, doc_ids, qids, qrels)

    elif method == 'dct+sq8':
        vz = VectorZip(n_components=K, method='dct+sq8')
        vz.fit(calib_emb)
        cd = vz.transform(docs_emb).astype(float)
        cq = vz.transform(queries_emb).astype(float)
        return evaluate(cd, cq, doc_ids, qids, qrels)

    elif method == 'pca+sq8':
        vz = VectorZip(n_components=K, method='pca+sq8')
        vz.fit(calib_emb)
        cd = vz.transform(docs_emb).astype(float)
        cq = vz.transform(queries_emb).astype(float)
        return evaluate(cd, cq, doc_ids, qids, qrels)

# ============================================================
# Main
# ============================================================
def main():
    DATASETS = ['scifact', 'nfcorpus', 'fiqa', 'arguana']
    METHODS = ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']
    RATIOS = ['1/2', '1/4', '1/8', '1/16', '1/32']
    DEVICE = 'cuda'
    CACHE_DIR = '/home/sergio/beir_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    all_results = {}
    total_start = time.time()

    # Count total steps
    total_models = len(MODELS)
    total_datasets = len(DATASETS)

    for model_key in MODELS:
        info = MODELS[model_key]
        D = info['dims']

        # K values
        k_values = []
        for r in RATIOS:
            num, denom = r.split('/')
            K = max(int(D * int(num) / int(denom)), 16)
            if K < D:
                k_values.append((r, K))

        print(f"\n{'='*80}")
        print(f"MODEL: {model_key} ({info['name']}, {D}D)")
        print(f"{'='*80}")

        # Load model
        t0 = time.time()
        print(f"  [1/4] Loading model on {DEVICE}...")
        model = SentenceTransformer(info['name'], device=DEVICE)
        print(f"        Loaded in {time.time()-t0:.1f}s")

        # Encode generic calibration corpus
        generic = DEFAULT_CORPUS[:1000]
        generic_cache = os.path.join(CACHE_DIR, f"{model_key}_generic.pkl")
        if os.path.exists(generic_cache):
            with open(generic_cache, 'rb') as f:
                generic_emb = pickle.load(f)
            print(f"  [2/4] Calibration corpus loaded from cache ({generic_emb.shape})")
        else:
            print(f"  [2/4] Encoding calibration corpus ({len(generic)} texts)...")
            t0 = time.time()
            generic_emb = model.encode(generic, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
            with open(generic_cache, 'wb') as f:
                pickle.dump(generic_emb, f)
            print(f"        Encoded in {time.time()-t0:.1f}s, shape={generic_emb.shape}")

        for ds_idx, ds_name in enumerate(DATASETS):
            progress = (list(MODELS).index(model_key) * total_datasets + ds_idx + 1) / (total_models * total_datasets) * 100
            elapsed_so_far = time.time() - total_start
            print(f"\n  [{'='*40}")
            print(f"  [{progress:.0f}%] ({ds_idx+1}/{len(DATASETS)}) Dataset: {ds_name} | Elapsed: {elapsed_so_far:.0f}s")
            print(f"  [{'='*40}")

            # Check cache
            cache_file = os.path.join(CACHE_DIR, f"{model_key}_{ds_name}.pkl")
            if os.path.exists(cache_file):
                print(f"    [3/4] Loading embeddings from cache...")
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                doc_ids = cached['doc_ids']
                qids = cached['qids']
                docs_emb = cached['docs_emb']
                queries_emb = cached['queries_emb']
                qrels = cached['qrels']
                print(f"          Cached: {len(doc_ids)} docs, {len(qids)} queries, {docs_emb.shape[1]}D")
            else:
                print(f"    [3/4] Loading {ds_name} from BEIR...")
                t0 = time.time()
                corpus, queries, qrels_raw = load_dataset(ds_name)
                print(f"          Loaded dataset in {time.time()-t0:.1f}s")

                # Filter queries with qrels
                valid_qids = [q for q in queries if q in qrels_raw and len(qrels_raw[q]) > 0]

                # Get all relevant doc IDs
                relevant = set()
                for qid in valid_qids:
                    relevant.update(qrels_raw[qid].keys())

                # Keep relevant docs + sample up to 2000 total
                all_doc_ids = list(corpus.keys())
                relevant_in_corpus = [d for d in all_doc_ids if d in relevant]
                other_docs = [d for d in all_doc_ids if d not in relevant]
                np.random.seed(42)
                n_fill = max(0, 2000 - len(relevant_in_corpus))
                sampled = list(np.random.choice(other_docs, min(n_fill, len(other_docs)), replace=False)) if other_docs else []
                doc_ids = relevant_in_corpus + sampled

                # Sample queries (up to 200)
                np.random.seed(42)
                qids = list(np.random.choice(valid_qids, min(200, len(valid_qids)), replace=False))

                # Filter qrels to sampled docs (convert to int for pytrec_eval)
                doc_set = set(doc_ids)
                qrels = {qid: {d: int(round(r)) for d, r in qrels_raw[qid].items() if d in doc_set} for qid in qids}

                # Encode
                print(f"          Encoding {len(doc_ids)} docs + {len(qids)} queries on GPU...")
                t0 = time.time()
                doc_texts = [corpus[d]['text'] for d in doc_ids]
                query_texts = [queries[q] for q in qids]
                docs_emb = model.encode(doc_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
                queries_emb = model.encode(query_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
                print(f"          Encoded in {time.time()-t0:.1f}s")

                # Cache
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'doc_ids': doc_ids, 'qids': qids,
                        'docs_emb': docs_emb, 'queries_emb': queries_emb,
                        'qrels': qrels,
                    }, f)
                print(f"    Cached to {cache_file}")

            print(f"    [4/4] Running {len(METHODS)} methods x {len(k_values)} K values = {len(METHODS)*len(k_values)} experiments (+{2*len(k_values)} ID)...")
            print(f"    {len(doc_ids)} docs, {len(qids)} queries, {D}D")

            ds_key = f"{model_key}/{ds_name}"
            ds_results = {}
            n_total_exp = len(METHODS) * len(k_values) + 2 * len(k_values) + 1
            n_done = 0

            # Raw baseline
            t0 = time.time()
            raw_metrics = evaluate(docs_emb, queries_emb, doc_ids, qids, qrels)
            ds_results['raw'] = raw_metrics
            n_done += 1
            print(f"    [{n_done}/{n_total_exp}] RAW: nDCG@10={raw_metrics['ndcg@10']:.4f} R@5={raw_metrics['recall@5']:.4f} R@10={raw_metrics['recall@10']:.4f} ({time.time()-t0:.1f}s)")

            # All methods x K values, OOD and ID
            for method in METHODS:
                for ratio_str, K in k_values:
                    # OOD (calibrate on generic)
                    t0 = time.time()
                    m = compress_and_eval(method, generic_emb, docs_emb, queries_emb, K, doc_ids, qids, qrels)
                    m['K'] = K
                    m['method'] = method
                    m['calibration'] = 'ood'
                    ds_results[f"{method}_K{K}_ood"] = m
                    n_done += 1
                    print(f"    [{n_done}/{n_total_exp}] {method:10s} K={K:4d} OOD: nDCG={m['ndcg@10']:.4f} R@5={m['recall@5']:.4f} R@10={m['recall@10']:.4f} ({time.time()-t0:.1f}s)")

                    # ID (calibrate on dataset) for DCT and PCA
                    if method in ('dct', 'pca'):
                        t0 = time.time()
                        m_id = compress_and_eval(method, docs_emb, docs_emb, queries_emb, K, doc_ids, qids, qrels)
                        m_id['K'] = K
                        m_id['method'] = method
                        m_id['calibration'] = 'id'
                        ds_results[f"{method}_K{K}_id"] = m_id
                        n_done += 1
                        print(f"    [{n_done}/{n_total_exp}] {method:10s} K={K:4d} ID:  nDCG={m_id['ndcg@10']:.4f} R@5={m_id['recall@5']:.4f} ({time.time()-t0:.1f}s)")

            print(f"    >>> {ds_name} COMPLETE: {n_done} experiments done")

            all_results[ds_key] = ds_results

            # Save incrementally
            with open('benchmarks/results_deep_benchmark.json', 'w') as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")

    # ========== SUMMARY TABLES ==========
    print(f"\n{'='*120}")
    print("SUMMARY: nDCG@10 (OOD vs ID calibration)")
    print(f"{'='*120}")

    for ds_key, ds_res in sorted(all_results.items()):
        model, dataset = ds_key.split('/')
        raw = ds_res.get('raw', {}).get('ndcg@10', 0)

        ks = sorted(set(
            int(k.split('K')[1].split('_')[0])
            for k in ds_res.keys()
            if k.startswith('dct_K') and 'ood' in k
        ))

        print(f"\n{model} / {dataset} (raw nDCG@10={raw:.4f}):")
        print(f"  {'K':>4s}  {'DCT OOD':>8s}  {'PCA OOD':>8s}  {'RP OOD':>8s}  {'Trunc':>8s}  {'DCT+SQ8':>8s}  {'PCA+SQ8':>8s}  |  {'DCT ID':>8s}  {'PCA ID':>8s}  |  {'OOD winner':>10s}")
        print("  " + "-" * 115)

        for K in ks:
            row = {}
            for m in ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']:
                key = f"{m}_K{K}_ood"
                if key in ds_res:
                    row[m] = ds_res[key]['ndcg@10']

            dct_id = ds_res.get(f"dct_K{K}_id", {}).get('ndcg@10', None)
            pca_id = ds_res.get(f"pca_K{K}_id", {}).get('ndcg@10', None)

            vals = []
            for m in ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']:
                v = row.get(m)
                vals.append(f"{v:.4f}" if v is not None else "  -   ")

            # Winner among DCT vs PCA OOD
            winner = ""
            if 'dct' in row and 'pca' in row:
                if row['dct'] > row['pca']:
                    winner = f"DCT +{row['dct']-row['pca']:.4f}"
                else:
                    winner = f"PCA +{row['pca']-row['dct']:.4f}"

            id_str = f"{dct_id:.4f}" if dct_id else "  -   "
            id_str += f"  {pca_id:.4f}" if pca_id else "    -  "

            print(f"  {K:4d}  {'  '.join(vals)}  |  {id_str}  |  {winner:>10s}")

    # ========== OOD RETENTION RATES ==========
    print(f"\n{'='*100}")
    print("OOD RETENTION: % of ID performance retained under domain shift")
    print(f"{'='*100}")

    for ds_key, ds_res in sorted(all_results.items()):
        model, dataset = ds_key.split('/')
        ks = sorted(set(
            int(k.split('K')[1].split('_')[0])
            for k in ds_res.keys()
            if k.startswith('dct_K') and 'id' in k
        ))

        if not ks:
            continue

        print(f"\n{model} / {dataset}:")
        print(f"  {'K':>4s}  {'DCT ID':>8s}  {'DCT OOD':>8s}  {'DCT %':>7s}  |  {'PCA ID':>8s}  {'PCA OOD':>8s}  {'PCA %':>7s}  |  {'Better OOD':>10s}")
        print("  " + "-" * 85)

        for K in ks:
            dct_ood = ds_res.get(f"dct_K{K}_ood", {}).get('ndcg@10', None)
            dct_id = ds_res.get(f"dct_K{K}_id", {}).get('ndcg@10', None)
            pca_ood = ds_res.get(f"pca_K{K}_ood", {}).get('ndcg@10', None)
            pca_id = ds_res.get(f"pca_K{K}_id", {}).get('ndcg@10', None)

            if dct_ood and dct_id and pca_ood and pca_id:
                dct_pct = dct_ood / dct_id * 100 if dct_id > 0 else 0
                pca_pct = pca_ood / pca_id * 100 if pca_id > 0 else 0
                better = "DCT" if dct_ood > pca_ood else "PCA"
                print(f"  {K:4d}  {dct_id:8.4f}  {dct_ood:8.4f}  {dct_pct:6.1f}%  |  {pca_id:8.4f}  {pca_ood:8.4f}  {pca_pct:6.1f}%  |  {better:>10s}")

    print(f"\nResults saved to benchmarks/results_deep_benchmark.json")

if __name__ == '__main__':
    main()
