#!/usr/bin/env python3
"""
Fast BEIR sanity check — small sample per dataset, all models, all methods.
Verifies DCT vs PCA on real BEIR data in minutes, not hours.
"""

import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorzip import VectorZip
from vectorzip.default_corpus import DEFAULT_CORPUS
from sklearn.decomposition import PCA

# ============================================================
# Metrics
# ============================================================
def ndcg_at_10(gt_indices, test_indices):
    rel = {gt: 10 - i for i, gt in enumerate(gt_indices)}
    dcg = sum((2**rel.get(item, 0) - 1) / np.log2(idx + 2) for idx, item in enumerate(test_indices[:10]))
    idcg = sum((2**(10 - i) - 1) / np.log2(i + 2) for i in range(min(10, len(gt_indices))))
    return dcg / idcg if idcg > 0 else 1.0

def recall_at_k(gt_set, test_indices, k):
    return len(set(test_indices[:k]) & gt_set) / len(gt_set) if gt_set else 0.0

def search(docs, query, top_k=100):
    sims = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query) + 1e-8)
    return np.argsort(sims)[::-1][:top_k]

# ============================================================
# Models
# ============================================================
MODELS = {
    'bge-m3':     {'name': 'BAAI/bge-m3', 'dims': 1024},
    'minilm':     {'name': 'all-MiniLM-L6-v2', 'dims': 384},
    'bge-large':  {'name': 'BAAI/bge-large-en-v1.5', 'dims': 1024},
    'nomic':      {'name': 'nomic-ai/nomic-embed-text-v2', 'dims': 768},
    'gte-qwen2':  {'name': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'dims': 1536, 'trust_remote_code': True},
}

# ============================================================
# Datasets (small samples)
# ============================================================
DATASETS = {
    'scifact':   {'url': 'scifact',     'n_docs': 200, 'n_queries': 50},
    'nfcorpus':  {'url': 'nfcorpus',    'n_docs': 200, 'n_queries': 50},
    'fiqa':      {'url': 'fiqa',        'n_docs': 200, 'n_queries': 50},
    'arguana':   {'url': 'arguana',     'n_docs': 200, 'n_queries': 50},
}

# ============================================================
# Load BEIR dataset (download if needed)
# ============================================================
def load_dataset(name, n_docs, n_queries, data_dir='/home/sergio/beir_datasets'):
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

    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split='test')

    # Filter queries with qrels
    valid_qids = [q for q in queries if q in qrels and len(qrels[q]) > 0]

    # Get relevant doc IDs
    relevant = set()
    for qid in valid_qids:
        relevant.update(qrels[qid].keys())

    # Sample: all relevant docs + random fill
    all_doc_ids = list(corpus.keys())
    relevant_in_corpus = [d for d in all_doc_ids if d in relevant]
    other_docs = [d for d in all_doc_ids if d not in relevant]
    np.random.seed(42)
    n_fill = max(0, n_docs - len(relevant_in_corpus))
    sampled = list(np.random.choice(other_docs, min(n_fill, len(other_docs)), replace=False))
    doc_ids = relevant_in_corpus + sampled

    # Sample queries
    np.random.seed(42)
    sampled_qids = list(np.random.choice(valid_qids, min(n_queries, len(valid_qids)), replace=False))

    doc_texts = [corpus[d]['text'] for d in doc_ids]
    query_texts = [queries[q] for q in sampled_qids]

    # Build qrels filtered
    qrels_f = {}
    doc_set = set(doc_ids)
    for qid in sampled_qids:
        qrels_f[qid] = {d: r for d, r in qrels[qid].items() if d in doc_set}

    return doc_ids, doc_texts, sampled_qids, query_texts, qrels_f

# ============================================================
# Compression
# ============================================================
def compress(method, train_emb, target_emb, K, seed=42):
    if method == 'dct':
        vz = VectorZip(n_components=K, method='dct')
        vz.fit(train_emb)
        return vz.transform(target_emb), vz
    elif method == 'pca':
        pca = PCA(n_components=K, random_state=seed)
        pca.fit(train_emb)
        return pca.transform(target_emb), pca
    elif method == 'rp':
        vz = VectorZip(n_components=K, method='rp', rp_seed=seed)
        vz.fit(train_emb)
        return vz.transform(target_emb), vz
    elif method == 'truncation':
        return target_emb[:, :K], None
    elif method == 'dct+sq8':
        vz = VectorZip(n_components=K, method='dct+sq8')
        vz.fit(train_emb)
        return vz.transform(target_emb).astype(float), vz
    elif method == 'pca+sq8':
        vz = VectorZip(n_components=K, method='pca+sq8')
        vz.fit(train_emb)
        return vz.transform(target_emb).astype(float), vz
    elif method == 'raw':
        return target_emb, None

def transform_queries(method, compressor, query_emb, K):
    if method == 'truncation':
        return query_emb[:, :K]
    elif method == 'pca':
        return compressor.transform(query_emb)
    elif method == 'raw':
        return query_emb
    elif method in ('dct', 'rp', 'dct+sq8', 'pca+sq8'):
        return compressor.transform(query_emb).astype(float) if 'sq8' in method else compressor.transform(query_emb)

# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['bge-m3', 'minilm', 'bge-large', 'nomic', 'gte-qwen2'])
    parser.add_argument('--datasets', nargs='+', default=['scifact', 'nfcorpus', 'fiqa', 'arguana'])
    parser.add_argument('--methods', nargs='+', default=['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8'])
    parser.add_argument('--ratios', nargs='+', default=['1/4', '1/8', '1/16', '1/32'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_docs', type=int, default=200)
    parser.add_argument('--n_queries', type=int, default=50)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    results = {}
    total_start = time.time()

    # Compute total steps for progress
    total_steps = len(args.models) * len(args.datasets)
    current_step = 0

    for model_key in args.models:
        info = MODELS[model_key]
        D = info['dims']
        print(f"\n{'='*70}")
        print(f"MODEL: {model_key} ({info['name']}, {D}D)")
        print(f"{'='*70}")

        # Load model
        t0 = time.time()
        kwargs = {}
        if info.get('trust_remote_code'):
            kwargs['trust_remote_code'] = True
        model = SentenceTransformer(info['name'], device=args.device, **kwargs)
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # Encode generic calibration corpus
        generic = DEFAULT_CORPUS[:500]
        print(f"  Encoding calibration corpus ({len(generic)} texts)...")
        generic_emb = model.encode(generic, batch_size=16, show_progress_bar=False, convert_to_numpy=True)

        # K values for this model
        k_values = []
        for r in args.ratios:
            num, denom = r.split('/')
            K = max(int(D * int(num) / int(denom)), 16)
            if K < D:
                k_values.append(K)

        for ds_name in args.datasets:
            current_step += 1
            progress = current_step / total_steps * 100
            print(f"\n  [{progress:.0f}%] Dataset: {ds_name}")

            ds_cfg = DATASETS.get(ds_name, {'url': ds_name, 'n_docs': args.n_docs, 'n_queries': args.n_queries})
            doc_ids, doc_texts, qids, q_texts, qrels = load_dataset(
                ds_name, ds_cfg['n_docs'], ds_cfg['n_queries']
            )
            print(f"    {len(doc_ids)} docs, {len(qids)} queries")

            # Encode
            print(f"    Encoding docs...")
            docs_emb = model.encode(doc_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
            print(f"    Encoding queries...")
            queries_emb = model.encode(q_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)

            # Ground truth rankings from raw
            gt_ranks = [search(docs_emb, queries_emb[i]) for i in range(len(qids))]

            # Raw baseline
            raw_ndcgs = [ndcg_at_10(gt_ranks[i], search(docs_emb, queries_emb[i])) for i in range(len(qids))]
            raw_r5s = [recall_at_k(set(qrels[qids[i]].keys()), search(docs_emb, queries_emb[i]), 5) for i in range(len(qids))]
            raw_ndcg = np.mean(raw_ndcgs)
            raw_r5 = np.mean(raw_r5s)
            print(f"    RAW: nDCG@10={raw_ndcg:.4f} R@5={raw_r5:.4f}")

            ds_key = f"{model_key}/{ds_name}"
            ds_results = {'raw': {'ndcg@10': raw_ndcg, 'recall@5': raw_r5, 'D': D}}

            # Test each method x K
            for method in args.methods:
                for K in k_values:
                    if method == 'pq' and K > 64:
                        continue
                    try:
                        # OOD calibration
                        comp_docs, compressor = compress(method, generic_emb, docs_emb, K)
                        comp_queries = transform_queries(method, compressor, queries_emb, K)

                        # Evaluate
                        ndcgs, r5s = [], []
                        for i in range(len(qids)):
                            r = search(comp_docs, comp_queries[i])
                            ndcgs.append(ndcg_at_10(gt_ranks[i], r))
                            r5s.append(recall_at_k(set(qrels[qids[i]].keys()), r, 5))

                        m_key = f"{method}_K{K}_ood"
                        ds_results[m_key] = {'ndcg@10': float(np.mean(ndcgs)), 'recall@5': float(np.mean(r5s))}
                        print(f"    {method:10s} K={K:4d} OOD: nDCG={np.mean(ndcgs):.4f} R@5={np.mean(r5s):.4f}")

                    except Exception as e:
                        print(f"    {method:10s} K={K:4d} OOD: ERROR ({e})")

                # ID calibration (DCT + PCA only)
                if method in ('dct', 'pca'):
                    for K in k_values:
                        try:
                            comp_docs, compressor = compress(method, docs_emb, docs_emb, K)
                            comp_queries = transform_queries(method, compressor, queries_emb, K)
                            ndcgs = [ndcg_at_10(gt_ranks[i], search(comp_docs, comp_queries[i])) for i in range(len(qids))]
                            m_key = f"{method}_K{K}_id"
                            ds_results[m_key] = {'ndcg@10': float(np.mean(ndcgs))}
                            print(f"    {method:10s} K={K:4d} ID:  nDCG={np.mean(ndcgs):.4f}")
                        except Exception as e:
                            print(f"    {method:10s} K={K:4d} ID:  ERROR ({e})")

            results[ds_key] = ds_results

            # Save incrementally
            with open('benchmarks/results_fast_benchmark.json', 'w') as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY: nDCG@10 (OOD calibration)")
    print(f"{'='*100}")

    for ds_key, ds_res in results.items():
        if 'raw' not in ds_res:
            continue
        model, dataset = ds_key.split('/')
        raw = ds_res['raw']['ndcg@10']
        D = ds_res['raw']['D']

        # Get unique K values
        ks = sorted(set(
            int(k.split('K')[1].split('_')[0])
            for k in ds_res.keys()
            if k.startswith('dct_K') and 'ood' in k
        ))

        print(f"\n{model} / {dataset} (raw={raw:.4f}, D={D}):")
        print(f"  {'K':>4s}  {'DCT':>8s}  {'PCA':>8s}  {'RP':>8s}  {'Trunc':>8s}  {'DCT+SQ8':>8s}  {'PCA+SQ8':>8s}  |  {'DCT vs PCA':>10s}  |  {'DCT ID':>8s}  {'PCA ID':>8s}")
        print("  " + "-" * 110)

        for K in ks:
            row = {}
            for method in ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']:
                key = f"{method}_K{K}_ood"
                if key in ds_res:
                    row[method] = ds_res[key]['ndcg@10']

            dct_id = ds_res.get(f"dct_K{K}_id", {}).get('ndcg@10', None)
            pca_id = ds_res.get(f"pca_K{K}_id", {}).get('ndcg@10', None)

            vals = []
            for m in ['dct', 'pca', 'rp', 'truncation', 'dct+sq8', 'pca+sq8']:
                v = row.get(m)
                vals.append(f"{v:.4f}" if v else "  -   ")

            delta = ""
            if 'dct' in row and 'pca' in row:
                d = row['dct'] - row['pca']
                delta = f"{d:+.4f} {'DCT' if d > 0 else 'PCA'}"

            id_str = f"{dct_id:.4f}" if dct_id else "  -   "
            id_str += f"  {pca_id:.4f}" if pca_id else "  -   "

            print(f"  {K:4d}  {'  '.join(vals)}  |  {delta:>10s}  |  {id_str}")

    print(f"\nResults saved to benchmarks/results_fast_benchmark.json")

if __name__ == '__main__':
    main()
