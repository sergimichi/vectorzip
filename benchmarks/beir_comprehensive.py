#!/usr/bin/env python3
"""
Comprehensive BEIR benchmark for DCT vs PCA vs other compression methods.

Tests retrieval quality (nDCG@10, Recall@5, Recall@100, MRR@10) under:
- Multiple compression methods: DCT, PCA, RP, PQ, naive truncation, raw
- Multiple compression ratios: K/D = 1/2, 1/4, 1/8, 1/16, 1/32
- Both in-distribution (calibrate on dataset) and OOD (calibrate on generic corpus)
- Statistical significance via multiple seeds

Usage:
    HSA_OVERRIDE_GFX=1100 ./.venv/bin/python benchmarks/beir_comprehensive.py --datasets scifact nfcorpus fiqa --models bge-m3 minilm
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorzip import VectorZip
from vectorzip.default_corpus import DEFAULT_CORPUS


# ============================================================
# Metrics
# ============================================================

def cosine_search(corpus, query, top_k=100):
    """Brute-force cosine similarity search."""
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    scores = corpus_norm @ query_norm
    return np.argsort(scores)[::-1][:top_k], np.sort(scores)[::-1][:top_k]


def evaluate_retrieval(corpus_emb, query_emb, qrels, top_k=100):
    """Evaluate retrieval using BEIR-style metrics."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    results = {}
    for i, (qid, q_emb) in enumerate(query_emb.items()):
        indices, scores = cosine_search(corpus_emb, q_emb, top_k)
        results[qid] = {
            str(corpus_doc_ids[idx]): float(scores[j])
            for j, idx in enumerate(indices)
        }

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        results, qrels, [1, 5, 10, 100]
    )
    return {
        'ndcg@10': ndcg.get('NDCG@10', 0.0),
        'ndcg@1': ndcg.get('NDCG@1', 0.0),
        'map@10': _map.get('MAP@10', 0.0),
        'recall@5': recall.get('Recall@5', 0.0),
        'recall@100': recall.get('Recall@100', 0.0),
        'precision@10': precision.get('P@10', 0.0),
    }


# ============================================================
# Compression methods
# ============================================================

def compress_dct(train_emb, target_emb, K, seed=42):
    """DCT compression with TSP reordering."""
    vz = VectorZip(n_components=K, method='dct')
    vz.fit(train_emb)
    return vz.transform(target_emb), vz


def compress_pca(train_emb, target_emb, K, seed=42):
    """PCA compression."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=K, random_state=seed)
    pca.fit(train_emb)
    return pca.transform(target_emb), pca


def compress_rp(train_emb, target_emb, K, seed=42):
    """Random projection compression."""
    vz = VectorZip(n_components=K, method='rp', rp_seed=seed)
    vz.fit(train_emb)
    return vz.transform(target_emb), vz


def compress_truncation(train_emb, target_emb, K, seed=42):
    """Naive truncation (first K dimensions)."""
    return target_emb[:, :K], None


def compress_pq(train_emb, target_emb, K, seed=42):
    """Product quantization compression."""
    n_subvectors = min(K, 64)
    vz = VectorZip(n_components=n_subvectors, method='pq', pq_clusters=256, rp_seed=seed)
    vz.fit(train_emb)
    codes = vz.transform(target_emb)
    return codes.astype(float), vz


def compress_dct_sq8(train_emb, target_emb, K, seed=42):
    """DCT + scalar quantization hybrid."""
    vz = VectorZip(n_components=K, method='dct+sq8')
    vz.fit(train_emb)
    codes = vz.transform(target_emb)
    return codes.astype(float), vz


def compress_pca_sq8(train_emb, target_emb, K, seed=42):
    """PCA + scalar quantization hybrid."""
    vz = VectorZip(n_components=K, method='pca+sq8')
    vz.fit(train_emb)
    codes = vz.transform(target_emb)
    return codes.astype(float), vz


COMPRESSION_METHODS = {
    'dct': compress_dct,
    'pca': compress_pca,
    'rp': compress_rp,
    'truncation': compress_truncation,
    'pq': compress_pq,
    'dct+sq8': compress_dct_sq8,
    'pca+sq8': compress_pca_sq8,
}


# ============================================================
# Model loading
# ============================================================

MODELS = {
    'bge-m3': {
        'name': 'BAAI/bge-m3',
        'dims': 1024,
        'multilingual': True,
    },
    'minilm': {
        'name': 'all-MiniLM-L6-v2',
        'dims': 384,
        'multilingual': False,
    },
    'bge-large': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dims': 1024,
        'multilingual': False,
    },
    'nomic': {
        'name': 'nomic-ai/nomic-embed-text-v2',
        'dims': 768,
        'multilingual': False,
        'mrl': True,
    },
    'gte-qwen2': {
        'name': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        'dims': 1536,
        'multilingual': True,
    },
}


def load_model(model_key, device='cuda'):
    from sentence_transformers import SentenceTransformer
    info = MODELS[model_key]
    kwargs = {}
    if 'gte' in info['name']:
        kwargs['trust_remote_code'] = True
    model = SentenceTransformer(info['name'], device=device, **kwargs)
    return model


# ============================================================
# BEIR dataset loading
# ============================================================

BEIR_DATASETS = {
    'scifact': 'scifact',
    'nfcorpus': 'nfcorpus',
    'fiqa': 'fiqa',
    'arguana': 'arguana',
    'trec-covid': 'trec-covid',
    'scidocs': 'scidocs',
    'nq': 'nq',
}


def load_beir_dataset(dataset_name, data_dir='/home/sergio/beir_datasets'):
    from beir.datasets.data_loader import GenericDataLoader

    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        os.makedirs(data_dir, exist_ok=True)
        print(f"  Downloading {dataset_name} from {url}...")
        import urllib.request
        import zipfile
        zip_path = os.path.join(data_dir, f"{dataset_name}.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split='test')
    return corpus, queries, qrels


# ============================================================
# Main benchmark
# ============================================================

def run_benchmark(
    dataset_names,
    model_keys,
    methods,
    compression_ratios,
    seeds,
    device='cuda',
    ood_calibration=True,
    output_file='benchmarks/results_beir_comprehensive.json',
):
    """Run the full benchmark suite."""

    all_results = {}

    for model_key in model_keys:
        model_info = MODELS[model_key]
        print(f"\n{'='*80}")
        print(f"MODEL: {model_key} ({model_info['name']}, {model_info['dims']}D)")
        print(f"{'='*80}")

        print(f"  Loading model on {device}...")
        model = load_model(model_key, device=device)
        D = model_info['dims']

        # Generic corpus for OOD calibration
        generic_texts = DEFAULT_CORPUS[:500]
        print(f"  Encoding generic calibration corpus ({len(generic_texts)} texts)...")
        generic_emb = model.encode(generic_texts, show_progress_bar=False, convert_to_numpy=True)

        for ds_name in dataset_names:
            print(f"\n  --- Dataset: {ds_name} ---")
            print(f"  Loading {ds_name}...")
            corpus, queries, qrels = load_beir_dataset(ds_name)

            # Filter to queries that have qrels
            valid_qids = [q for q in queries if q in qrels and len(qrels[q]) > 0]
            print(f"  Corpus: {len(corpus)} docs, Queries: {len(valid_qids)} (with qrels)")

            if len(valid_qids) == 0:
                print(f"  WARNING: No valid queries for {ds_name}, skipping.")
                continue

            # Subsample corpus if too large (for speed)
            max_corpus = 5000
            corpus_ids = list(corpus.keys())
            if len(corpus_ids) > max_corpus:
                # Keep all relevant docs + sample
                relevant_docs = set()
                for qid in valid_qids:
                    relevant_docs.update(qrels[qid].keys())
                relevant_in_corpus = [d for d in corpus_ids if d in relevant_docs]
                other_docs = [d for d in corpus_ids if d not in relevant_docs]
                np.random.seed(42)
                sampled = np.random.choice(other_docs, min(max_corpus - len(relevant_in_corpus), len(other_docs)), replace=False)
                corpus_ids = relevant_in_corpus + list(sampled)

            print(f"  Encoding {len(corpus_ids)} corpus docs...")
            corpus_texts = [corpus[d]['text'] for d in corpus_ids]
            batch_size = 16
            corpus_emb_list = []
            for i in range(0, len(corpus_texts), batch_size):
                batch = corpus_texts[i:i+batch_size]
                emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                corpus_emb_list.append(emb)
            corpus_emb = np.vstack(corpus_emb_list)

            print(f"  Encoding {len(valid_qids)} queries...")
            query_texts = [queries[q] for q in valid_qids]
            query_emb = model.encode(query_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

            # Build query embedding dict
            query_emb_dict = {qid: query_emb[i] for i, qid in enumerate(valid_qids)}

            # Filter qrels to sampled corpus
            qrels_filtered = {}
            for qid in valid_qids:
                qrels_filtered[qid] = {did: rel for did, rel in qrels[qid].items() if did in set(corpus_ids)}

            global corpus_doc_ids
            corpus_doc_ids = corpus_ids

            ds_results = {}

            # Raw baseline (no compression)
            print(f"  Evaluating raw (no compression)...")
            raw_metrics = evaluate_retrieval(corpus_emb, query_emb_dict, qrels_filtered, top_k=100)
            ds_results['raw'] = raw_metrics
            print(f"    Raw: nDCG@10={raw_metrics['ndcg@10']:.4f} R@5={raw_metrics['recall@5']:.4f}")

            # Compression methods
            for method in methods:
                for ratio_name, K in compression_ratios.items():
                    if K >= D:
                        continue
                    if method == 'pq' and K > 128:
                        continue

                    for seed in seeds:
                        seed_key = f"seed_{seed}"
                        result_key = f"{method}_K{K}_{seed_key}"

                        # Calibration: OOD (generic corpus) or ID (dataset corpus)
                        if ood_calibration:
                            calib_emb = generic_emb
                            calib_label = 'ood'
                        else:
                            calib_emb = corpus_emb
                            calib_label = 'id'

                        try:
                            t0 = time.time()
                            compressed_corpus, compressor = COMPRESSION_METHODS[method](
                                calib_emb, corpus_emb, K, seed=seed
                            )

                            # Transform queries with the fitted compressor
                            if method == 'truncation':
                                compressed_queries = query_emb[:, :K]
                            elif method == 'pca':
                                compressed_queries = compressor.transform(query_emb)
                            elif method == 'pq':
                                compressed_queries = compressor.transform(query_emb).astype(float)
                            elif method in ('dct+sq8', 'pca+sq8'):
                                compressed_queries = compressor.transform(query_emb).astype(float)
                            else:
                                compressed_queries = compressor.transform(query_emb)

                            fit_time = time.time() - t0

                            compressed_query_dict = {qid: compressed_queries[i] for i, qid in enumerate(valid_qids)}

                            metrics = evaluate_retrieval(compressed_corpus, compressed_query_dict, qrels_filtered, top_k=100)
                            metrics['fit_time_s'] = fit_time
                            metrics['compression_ratio'] = D / K
                            metrics['calibration'] = calib_label
                            metrics['K'] = K
                            metrics['method'] = method
                            metrics['seed'] = seed

                            ds_results[result_key] = metrics
                            print(f"    {method} K={K} seed={seed}: nDCG@10={metrics['ndcg@10']:.4f} R@5={metrics['recall@5']:.4f} ({fit_time:.1f}s)")

                        except Exception as e:
                            print(f"    ERROR {method} K={K} seed={seed}: {e}")
                            ds_results[result_key] = {'error': str(e)}

            # Also run in-distribution for comparison
            if ood_calibration:
                print(f"  Running in-distribution calibration...")
                for method in ['dct', 'pca']:
                    for ratio_name, K in compression_ratios.items():
                        if K >= D:
                            continue
                        for seed in [42]:
                            result_key = f"{method}_K{K}_seed_{seed}_id"
                            try:
                                compressed, compressor = COMPRESSION_METHODS[method](
                                    corpus_emb, corpus_emb, K, seed=seed
                                )
                                if method == 'pca':
                                    compressed_queries = compressor.transform(query_emb)
                                else:
                                    compressed_queries = compressor.transform(query_emb)
                                compressed_query_dict = {qid: compressed_queries[i] for i, qid in enumerate(valid_qids)}
                                metrics = evaluate_retrieval(compressed, compressed_query_dict, qrels_filtered, top_k=100)
                                metrics['calibration'] = 'id'
                                metrics['K'] = K
                                metrics['method'] = method
                                metrics['seed'] = seed
                                ds_results[result_key] = metrics
                                print(f"    {method} K={K} ID: nDCG@10={metrics['ndcg@10']:.4f}")
                            except Exception as e:
                                print(f"    ERROR {method} K={K} ID: {e}")

            # Save incrementally
            model_ds_key = f"{model_key}/{ds_name}"
            all_results[model_ds_key] = ds_results
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Results saved to {output_file}")

    return all_results


# ============================================================
# Summary table generation
# ============================================================

def generate_summary_table(results_file='benchmarks/results_beir_comprehensive.json'):
    """Generate a summary table from results."""
    with open(results_file, 'r') as f:
        all_results = json.load(f)

    print("\n" + "=" * 120)
    print("SUMMARY: nDCG@10 (OOD calibration, seed 42)")
    print("=" * 120)

    for model_ds_key, ds_results in all_results.items():
        if 'raw' not in ds_results:
            continue
        model, dataset = model_ds_key.split('/')
        raw_ndcg = ds_results['raw']['ndcg@10']

        print(f"\n{model} / {dataset} (raw nDCG@10={raw_ndcg:.4f}):")
        print(f"  {'K':>4s}  {'DCT':>8s}  {'PCA':>8s}  {'RP':>8s}  {'PQ':>8s}  {'Trunc':>8s}  {'DCT+SQ8':>8s}  {'PCA+SQ8':>8s}  |  {'DCT vs PCA':>10s}")
        print("  " + "-" * 100)

        # Group by K
        ks = sorted(set(
            v['K'] for v in ds_results.values()
            if isinstance(v, dict) and 'K' in v and v.get('calibration') == 'ood'
        ))

        for K in ks:
            row = {}
            for method in ['dct', 'pca', 'rp', 'pq', 'truncation', 'dct+sq8', 'pca+sq8']:
                key = f"{method}_K{K}_seed_42"
                if key in ds_results and 'ndcg@10' in ds_results[key]:
                    row[method] = ds_results[key]['ndcg@10']
                else:
                    row[method] = None

            dct_val = f"{row.get('dct', 0):.4f}" if row.get('dct') else "  -   "
            pca_val = f"{row.get('pca', 0):.4f}" if row.get('pca') else "  -   "
            rp_val = f"{row.get('rp', 0):.4f}" if row.get('rp') else "  -   "
            pq_val = f"{row.get('pq', 0):.4f}" if row.get('pq') else "  -   "
            trunc_val = f"{row.get('truncation', 0):.4f}" if row.get('truncation') else "  -   "
            dct_sq8 = f"{row.get('dct+sq8', 0):.4f}" if row.get('dct+sq8') else "  -   "
            pca_sq8 = f"{row.get('pca+sq8', 0):.4f}" if row.get('pca+sq8') else "  -   "

            delta = ""
            if row.get('dct') and row.get('pca'):
                d = row['dct'] - row['pca']
                delta = f"{d:+.4f} {'DCT' if d > 0 else 'PCA'}"

            print(f"  {K:4d}  {dct_val:>8s}  {pca_val:>8s}  {rp_val:>8s}  {pq_val:>8s}  {trunc_val:>8s}  {dct_sq8:>8s}  {pca_sq8:>8s}  |  {delta:>10s}")

        # ID vs OOD comparison
        print(f"  ID vs OOD (DCT vs PCA):")
        for K in ks:
            dct_ood = ds_results.get(f"dct_K{K}_seed_42", {}).get('ndcg@10', None)
            pca_ood = ds_results.get(f"pca_K{K}_seed_42", {}).get('ndcg@10', None)
            dct_id = ds_results.get(f"dct_K{K}_seed_42_id", {}).get('ndcg@10', None)
            pca_id = ds_results.get(f"pca_K{K}_seed_42_id", {}).get('ndcg@10', None)
            if dct_ood and pca_ood and dct_id and pca_id:
                print(f"    K={K:3d}  DCT OOD={dct_ood:.4f} ID={dct_id:.4f}  |  PCA OOD={pca_ood:.4f} ID={pca_id:.4f}  |  DCT OOD%={dct_ood/dct_id*100:.1f}% PCA OOD%={pca_ood/pca_id*100:.1f}%")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive BEIR benchmark")
    parser.add_argument('--datasets', nargs='+', default=['scifact', 'nfcorpus', 'fiqa', 'arguana'],
                        help='BEIR datasets to test')
    parser.add_argument('--models', nargs='+', default=['bge-m3', 'minilm'],
                        help='Models to test')
    parser.add_argument('--methods', nargs='+', default=['dct', 'pca', 'rp', 'truncation', 'pq', 'dct+sq8', 'pca+sq8'],
                        help='Compression methods to test')
    parser.add_argument('--ratios', nargs='+', default=['1/4', '1/8', '1/16', '1/32'],
                        help='Compression ratios (as fractions of original D)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                        help='Random seeds for statistical significance')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', default='benchmarks/results_beir_comprehensive.json',
                        help='Output file')
    parser.add_argument('--summary-only', action='store_true', help='Just print summary table')

    args = parser.parse_args()

    if args.summary_only:
        generate_summary_table(args.output)
        sys.exit(0)

    # Convert ratio strings to K values
    compression_ratios = {}
    for ratio_str in args.ratios:
        num, denom = ratio_str.split('/')
        compression_ratios[ratio_str] = None  # will be computed per-model

    # We need to compute K per model based on D
    # So we pass ratio fractions and compute K = D * num / denom

    all_results = {}

    for model_key in args.models:
        model_info = MODELS[model_key]
        D = model_info['dims']

        # Compute K values for this model
        ratios = {}
        for ratio_str in args.ratios:
            num, denom = ratio_str.split('/')
            K = max(int(D * int(num) / int(denom)), 16)
            if K < D:
                ratios[ratio_str] = K

        # Re-run with this model's K values
        # We need to modify the function to accept pre-computed K values
        # For simplicity, let's just call run_benchmark per model

    # Actually, let's just run it with explicit K values per model
    # The run_benchmark function handles K directly

    # Convert to K-based dict
    # We'll run per model with model-specific K values
    for model_key in args.models:
        model_info = MODELS[model_key]
        D = model_info['dims']

        ratio_dict = {}
        for ratio_str in args.ratios:
            num, denom = ratio_str.split('/')
            K = max(int(D * int(num) / int(denom)), 16)
            if K < D:
                ratio_dict[f"{ratio_str} (K={K})"] = K

        print(f"\n{'#'*80}")
        print(f"# Model: {model_key} ({D}D)")
        print(f"# K values: {[(k, v) for k, v in ratio_dict.items()]}")
        print(f"{'#'*80}")

        results = run_benchmark(
            dataset_names=args.datasets,
            model_keys=[model_key],
            methods=args.methods,
            compression_ratios=ratio_dict,
            seeds=args.seeds,
            device=args.device,
            ood_calibration=True,
            output_file=args.output,
        )
        all_results.update(results)

    # Print summary
    generate_summary_table(args.output)
