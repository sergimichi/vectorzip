import numpy as np
import json
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip

def calculate_ndcg_at_10(gt_indices, test_indices):
    """Calculates NDCG@10 of a test retrieval list compared to ground truth retrieval list.
    Relevance is graded: 10 down to 1 for rank position in gt list.
    """
    relevance_map = {gt_indices[i]: 10 - i for i in range(len(gt_indices))}
    
    # Calculate DCG@10
    dcg = 0.0
    for idx, item in enumerate(test_indices[:10]):
        rel = relevance_map.get(item, 0.0)
        dcg += (2**rel - 1) / np.log2(idx + 2)
        
    # Calculate IDCG@10 (Ideal DCG)
    idcg = 0.0
    for idx in range(len(gt_indices[:10])):
        rel = 10 - idx
        idcg += (2**rel - 1) / np.log2(idx + 2)
        
    if idcg == 0.0:
        return 1.0
    return float(dcg / idcg)

def calculate_recall_at_k(gt_indices, test_indices, k):
    """Calculates Recall@k: fraction of gt_indices[:k] retrieved in test_indices[:k]."""
    gt_set = set(gt_indices[:k])
    test_set = set(test_indices[:k])
    intersection = gt_set.intersection(test_set)
    return float(len(intersection) / k)

def main():
    print("=== Experiment 8: Search Quality Preservation (Recall & NDCG@10) ===")

    # 1. Load model and corpora
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    from vectorzip.default_corpus import DEFAULT_CORPUS
    # Generate 2000 documents by taking wiki sentences and combining/reordering them
    doc_sentences = DEFAULT_CORPUS[:2000]
    
    # Generate 50 queries
    query_sentences = [
        "What is the impact of artificial intelligence on software engineering?",
        "Explain how the discrete cosine transform compresses signals.",
        "How does the traveling salesperson problem optimize dimensions?",
        "What are the clinical findings of metastatic lung cancer?",
        "How is a Delaware corporation structured under the law?",
        "What are the benefits of neural autoencoders vs linear methods?",
        "How do vector databases perform fast similarity search?",
        "Explain the QuickSort algorithm in Python.",
        "What is the difference between PD-1 and PD-L1 in cancer therapy?",
        "How does multiple sclerosis affect the nervous system?"
    ] * 5
    
    print(f"Encoding {len(doc_sentences)} documents...")
    docs_raw = model.encode(doc_sentences, show_progress_bar=False)
    
    print(f"Encoding {len(query_sentences)} queries...")
    queries_raw = model.encode(query_sentences, show_progress_bar=False)
    
    K = 96
    print(f"Fitting PCA and VectorZip on documents (N=384 -> K={K})...")
    
    pca = PCA(n_components=K)
    docs_pca = pca.fit_transform(docs_raw)
    queries_pca = pca.transform(queries_raw)
    
    vz = VectorZip(n_components=K, tsp_optimize=True)
    docs_vz = vz.fit_transform(docs_raw)
    queries_vz = vz.transform(queries_raw)
    
    # 2. Run retrievals
    print("Running search quality evaluations...")
    pca_recall_5, pca_recall_10, pca_ndcg = [], [], []
    vz_recall_5, vz_recall_10, vz_ndcg = [], [], []
    
    for i in range(len(queries_raw)):
        # Raw Ground Truth search
        q_raw = queries_raw[i]
        sims_raw = np.dot(docs_raw, q_raw) / (np.linalg.norm(docs_raw, axis=1) * np.linalg.norm(q_raw) + 1e-8)
        gt_ranks = np.argsort(sims_raw)[::-1][:10]
        
        # PCA search
        q_pca = queries_pca[i]
        sims_pca = np.dot(docs_pca, q_pca) / (np.linalg.norm(docs_pca, axis=1) * np.linalg.norm(q_pca) + 1e-8)
        pca_ranks = np.argsort(sims_pca)[::-1][:10]
        
        # VectorZip search
        q_vz = queries_vz[i]
        sims_vz = np.dot(docs_vz, q_vz) / (np.linalg.norm(docs_vz, axis=1) * np.linalg.norm(q_vz) + 1e-8)
        vz_ranks = np.argsort(sims_vz)[::-1][:10]
        
        # Calculate metrics for PCA
        pca_recall_5.append(calculate_recall_at_k(gt_ranks, pca_ranks, 5))
        pca_recall_10.append(calculate_recall_at_k(gt_ranks, pca_ranks, 10))
        pca_ndcg.append(calculate_ndcg_at_10(gt_ranks, pca_ranks))
        
        # Calculate metrics for VectorZip
        vz_recall_5.append(calculate_recall_at_k(gt_ranks, vz_ranks, 5))
        vz_recall_10.append(calculate_recall_at_k(gt_ranks, vz_ranks, 10))
        vz_ndcg.append(calculate_ndcg_at_10(gt_ranks, vz_ranks))
        
    results = {
        "n_components": K,
        "pca": {
            "recall_5": float(np.mean(pca_recall_5)),
            "recall_10": float(np.mean(pca_recall_10)),
            "ndcg_10": float(np.mean(pca_ndcg))
        },
        "vectorzip": {
            "recall_5": float(np.mean(vz_recall_5)),
            "recall_10": float(np.mean(vz_recall_10)),
            "ndcg_10": float(np.mean(vz_ndcg))
        }
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(f"Search Metrics at 96 dimensions:")
    print(f" -> PCA:       Recall@5: {results['pca']['recall_5']:.4f} | Recall@10: {results['pca']['recall_10']:.4f} | NDCG@10: {results['pca']['ndcg_10']:.4f}")
    print(f" -> VectorZip: Recall@5: {results['vectorzip']['recall_5']:.4f} | Recall@10: {results['vectorzip']['recall_10']:.4f} | NDCG@10: {results['vectorzip']['ndcg_10']:.4f}")

if __name__ == "__main__":
    main()
