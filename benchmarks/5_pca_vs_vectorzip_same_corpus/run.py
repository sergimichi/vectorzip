import numpy as np
import json
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip

def calculate_similarity_retention(raw, reconstructed):
    """Calculates the average cosine similarity between raw and reconstructed embeddings."""
    similarities = []
    for r, rec in zip(raw, reconstructed):
        denom = (np.linalg.norm(r) * np.linalg.norm(rec))
        if denom < 1e-8:
            sim = 0.0
        else:
            sim = np.dot(r, rec) / denom
        similarities.append(sim)
    return float(np.mean(similarities))

def main():
    print("=== Experiment 5: PCA vs VectorZip on Same Corpus (In-Distribution) ===")

    # 1. Load model and specialized medical sentences from HF
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    print("Loading unique medical flashcards dataset from Hugging Face...")
    try:
        dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        medical_sentences = [dataset[i]['input'] for i in range(500)]
        print(f"Successfully loaded {len(medical_sentences)} unique medical sentences.")
    except Exception as e:
        print(f"Hugging Face load failed ({e}). Using synthetic medical sentences...")
        # Fallback to generating 500 unique sentences
        medical_sentences = [
            f"Patient presents with symptom code {i} indicating potential localized inflammation or tissue response."
            for i in range(500)
        ]
    
    print("Encoding specialized medical sentences...")
    embeddings = model.encode(medical_sentences, show_progress_bar=False)
    M, N = embeddings.shape
    print(f"Embeddings shape: {embeddings.shape} (Full Rank: {np.linalg.matrix_rank(embeddings)})")
    
    target_components = [256, 128, 96, 64, 32, 16]
    
    pca_results = []
    vectorzip_results = []
    deltas = []
    
    for K in target_components:
        print(f"Evaluating target dimension K={K}...")
        
        # PCA
        pca = PCA(n_components=K)
        pca_transformed = pca.fit_transform(embeddings)
        pca_rec = pca.inverse_transform(pca_transformed)
        csr_pca = calculate_similarity_retention(embeddings, pca_rec)
        pca_results.append(csr_pca)
        
        # VectorZip
        vz = VectorZip(n_components=K, tsp_optimize=True)
        vz_transformed = vz.fit_transform(embeddings)
        vz_rec = vz.inverse_transform(vz_transformed)
        csr_vz = calculate_similarity_retention(embeddings, vz_rec)
        vectorzip_results.append(csr_vz)
        
        delta = csr_pca - csr_vz
        deltas.append(delta)
        
        print(f"  -> PCA CSR:       {csr_pca:.4f}")
        print(f"  -> VectorZip CSR: {csr_vz:.4f}")
        print(f"  -> Delta (PCA-VZ): {delta:.4f}")
        
    results = {
        "target_components": target_components,
        "raw_dimension": N,
        "pca_csr": pca_results,
        "vectorzip_csr": vectorzip_results,
        "pca_vz_delta": deltas
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
