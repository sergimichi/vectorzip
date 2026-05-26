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
    print("=== Experiment 3: Overfitting & Generalization (ID vs OOD) ===")
    
    # 1. Load model and sentences
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    from vectorzip.default_corpus import DEFAULT_CORPUS
    # Load 500 unique general Wikipedia sentences for generic corpus
    generic_sentences = DEFAULT_CORPUS[:500]
    
    # Load 500 unique specialized medical sentences
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
    
    print("Encoding generic (Wiki) sentences...")
    generic_raw = model.encode(generic_sentences, show_progress_bar=False)
    
    print("Encoding specialized (Medical) sentences...")
    medical_raw = model.encode(medical_sentences, show_progress_bar=False)
    
    # Compress 384 dimensions -> 64 dimensions
    K = 64
    print(f"Calibrating estimators on Generic Wiki corpus (K={K})...")
    
    # A. Fit PCA
    pca = PCA(n_components=K)
    pca.fit(generic_raw)
    
    # B. Fit VectorZip
    vz = VectorZip(n_components=K, tsp_optimize=True)
    vz.fit(generic_raw)
    
    # 2. Evaluate In-Distribution (ID) Performance
    print("Evaluating In-Distribution (ID) performance on Generic corpus...")
    # PCA ID
    pca_id_rec = pca.inverse_transform(pca.transform(generic_raw))
    pca_id_csr = calculate_similarity_retention(generic_raw, pca_id_rec)
    
    # VectorZip ID
    vz_id_rec = vz.inverse_transform(vz.transform(generic_raw))
    vz_id_csr = calculate_similarity_retention(generic_raw, vz_id_rec)
    
    # 3. Evaluate Out-of-Distribution (OOD) Performance
    print("Evaluating Out-of-Distribution (OOD) performance on Medical corpus...")
    # PCA OOD
    pca_ood_rec = pca.inverse_transform(pca.transform(medical_raw))
    pca_ood_csr = calculate_similarity_retention(medical_raw, pca_ood_rec)
    
    # VectorZip OOD
    vz_ood_rec = vz.inverse_transform(vz.transform(medical_raw))
    vz_ood_csr = calculate_similarity_retention(medical_raw, vz_ood_rec)
    
    # 4. Measure Generalization Gap (Overfitting Penalty)
    pca_drop = pca_id_csr - pca_ood_csr
    vz_drop = vz_id_csr - vz_ood_csr
    
    # Also evaluate a model fitted SPECIFICALLY on Medical to show domain shift
    vz_adapted = VectorZip(n_components=K, tsp_optimize=True)
    vz_adapted.fit(medical_raw)
    vz_adapted_rec = vz_adapted.inverse_transform(vz_adapted.transform(medical_raw))
    vz_adapted_csr = calculate_similarity_retention(medical_raw, vz_adapted_rec)
    
    results = {
        "target_dimension": K,
        "pca_id_csr": pca_id_csr,
        "pca_ood_csr": pca_ood_csr,
        "pca_overfitting_drop": pca_drop,
        "vectorzip_id_csr": vz_id_csr,
        "vectorzip_ood_csr": vz_ood_csr,
        "vectorzip_overfitting_drop": vz_drop,
        "vectorzip_adapted_medical_csr": vz_adapted_csr
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(f"PCA ID Cosine Similarity:        {pca_id_csr:.4f}")
    print(f"PCA OOD Cosine Similarity:       {pca_ood_csr:.4f}")
    print(f" -> PCA Overfitting Gap:         {pca_drop:.4f} ({pca_drop*100:.1f}%)")
    print(f"VectorZip ID Cosine Similarity:  {vz_id_csr:.4f}")
    print(f"VectorZip OOD Cosine Similarity: {vz_ood_csr:.4f}")
    print(f" -> VectorZip Overfitting Gap:   {vz_drop:.4f} ({vz_drop*100:.1f}%)")
    print(f"VectorZip Calibrated Medical CSR:{vz_adapted_csr:.4f}")

if __name__ == "__main__":
    main()
