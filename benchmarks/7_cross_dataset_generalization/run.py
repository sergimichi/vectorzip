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
    print("=== Experiment 7: Cross-Dataset Generalization (Wiki vs Medical, Code, Legal) ===")

    # 1. Load model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    # 2. Define specialized domain corpora
    from vectorzip.default_corpus import DEFAULT_CORPUS
    generic_sentences = DEFAULT_CORPUS[:500]
    
    print("Loading unique medical flashcards dataset from Hugging Face...")
    try:
        dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        medical_sentences = [dataset[i]['input'] for i in range(500)]
    except Exception as e:
        print(f"Hugging Face load failed ({e}). Using synthetic medical sentences...")
        medical_sentences = [
            f"Patient presents with medical clinical condition code {i} indicating symptoms of specialized pathology."
            for i in range(500)
        ]
        
    code_sentences = [
        f"def quicksort_routine_{i}(arr): return [x for x in arr if x < {i}] + [{i}] + [x for x in arr if x >= {i}]"
        for i in range(500)
    ]
    
    legal_sentences = [
        f"Under section {i} of Delaware code, the party executing the agreement under clause {i} shall maintain absolute confidentiality."
        for i in range(500)
    ]

    print("Encoding corpora...")
    generic_raw = model.encode(generic_sentences, show_progress_bar=False)
    medical_raw = model.encode(medical_sentences, show_progress_bar=False)
    code_raw = model.encode(code_sentences, show_progress_bar=False)
    legal_raw = model.encode(legal_sentences, show_progress_bar=False)
    
    target_components = [96, 64]
    results = {}
    
    for K in target_components:
        print(f"Evaluating target dimension K={K}...")
        
        # Fit models on Generic
        pca = PCA(n_components=K)
        pca.fit(generic_raw)
        
        vz = VectorZip(n_components=K, tsp_optimize=True)
        vz.fit(generic_raw)
        
        # Test A: Medical OOD
        pca_med_csr = calculate_similarity_retention(medical_raw, pca.inverse_transform(pca.transform(medical_raw)))
        vz_med_csr = calculate_similarity_retention(medical_raw, vz.inverse_transform(vz.transform(medical_raw)))
        
        # Test B: Code OOD
        pca_code_csr = calculate_similarity_retention(code_raw, pca.inverse_transform(pca.transform(code_raw)))
        vz_code_csr = calculate_similarity_retention(code_raw, vz.inverse_transform(vz.transform(code_raw)))
        
        # Test C: Legal OOD
        pca_leg_csr = calculate_similarity_retention(legal_raw, pca.inverse_transform(pca.transform(legal_raw)))
        vz_leg_csr = calculate_similarity_retention(legal_raw, vz.inverse_transform(vz.transform(legal_raw)))
        
        results[str(K)] = {
            "pca": {
                "medical_csr": pca_med_csr,
                "code_csr": pca_code_csr,
                "legal_csr": pca_leg_csr
            },
            "vectorzip": {
                "medical_csr": vz_med_csr,
                "code_csr": vz_code_csr,
                "legal_csr": vz_leg_csr
            }
        }
        
        print(f"  -> Medical: PCA CSR: {pca_med_csr:.4f} | VZ CSR: {vz_med_csr:.4f}")
        print(f"  -> Code:    PCA CSR: {pca_code_csr:.4f} | VZ CSR: {vz_code_csr:.4f}")
        print(f"  -> Legal:   PCA CSR: {pca_leg_csr:.4f} | VZ CSR: {vz_leg_csr:.4f}")

    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
