import numpy as np
import json
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip

def main():
    print("=== Experiment 9: Downstream RAG Hit Rate ===")

    # 1. Load model and corpora
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    from vectorzip.default_corpus import DEFAULT_CORPUS
    # Take 500 documents
    doc_sentences = DEFAULT_CORPUS[:500]
    
    # Let's create 50 queries where each query's gold document is a specific index
    # We choose gold document indices: 0, 10, 20, 30, ... up to 490
    gold_indices = [i * 10 for i in range(50)]
    
    # We form queries that are slightly rephrased versions of the gold documents
    query_sentences = []
    for idx in gold_indices:
        gold_text = doc_sentences[idx]
        # Simply use the gold text or slightly truncate/modify it
        # Let's take the first 40 characters to represent a query/summary of it
        query_sentences.append(gold_text[:min(60, len(gold_text))])
        
    print(f"Encoding {len(doc_sentences)} documents...")
    docs_raw = model.encode(doc_sentences, show_progress_bar=False)
    
    print(f"Encoding {len(query_sentences)} queries...")
    queries_raw = model.encode(query_sentences, show_progress_bar=False)
    
    K = 96
    print(f"Fitting PCA and VectorZip (N=384 -> K={K})...")
    
    pca = PCA(n_components=K)
    docs_pca = pca.fit_transform(docs_raw)
    queries_pca = pca.transform(queries_raw)
    
    vz = VectorZip(n_components=K, tsp_optimize=True)
    docs_vz = vz.fit_transform(docs_raw)
    queries_vz = vz.transform(queries_raw)
    
    # Evaluate Hit Rate@3
    print("Evaluating RAG Hit Rate@3...")
    raw_hits = []
    pca_hits = []
    vz_hits = []
    
    for i in range(len(queries_raw)):
        gold_idx = gold_indices[i]
        
        # Raw Top-3
        q_raw = queries_raw[i]
        sims_raw = np.dot(docs_raw, q_raw) / (np.linalg.norm(docs_raw, axis=1) * np.linalg.norm(q_raw) + 1e-8)
        raw_top3 = np.argsort(sims_raw)[::-1][:3]
        raw_hits.append(1.0 if gold_idx in raw_top3 else 0.0)
        
        # PCA Top-3
        q_pca = queries_pca[i]
        sims_pca = np.dot(docs_pca, q_pca) / (np.linalg.norm(docs_pca, axis=1) * np.linalg.norm(q_pca) + 1e-8)
        pca_top3 = np.argsort(sims_pca)[::-1][:3]
        pca_hits.append(1.0 if gold_idx in pca_top3 else 0.0)
        
        # VectorZip Top-3
        q_vz = queries_vz[i]
        sims_vz = np.dot(docs_vz, q_vz) / (np.linalg.norm(docs_vz, axis=1) * np.linalg.norm(q_vz) + 1e-8)
        vz_top3 = np.argsort(sims_vz)[::-1][:3]
        vz_hits.append(1.0 if gold_idx in vz_top3 else 0.0)
        
    results = {
        "n_components": K,
        "raw_hit_rate_at_3": float(np.mean(raw_hits)),
        "pca_hit_rate_at_3": float(np.mean(pca_hits)),
        "vectorzip_hit_rate_at_3": float(np.mean(vz_hits))
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(f"RAG Hit Rate@3 Results:")
    print(f" -> Raw embeddings: {results['raw_hit_rate_at_3']:.4f} ({results['raw_hit_rate_at_3']*100:.1f}%)")
    print(f" -> PCA search:     {results['pca_hit_rate_at_3']:.4f} ({results['pca_hit_rate_at_3']*100:.1f}%)")
    print(f" -> VectorZip search: {results['vectorzip_hit_rate_at_3']:.4f} ({results['vectorzip_hit_rate_at_3']*100:.1f}%)")

if __name__ == "__main__":
    main()
