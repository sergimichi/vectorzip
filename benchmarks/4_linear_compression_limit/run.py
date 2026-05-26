import numpy as np
import json
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip

def calculate_smoothness_msd(vectors):
    """Calculates the average Mean Squared Difference (MSD) of adjacent components in a set of vectors.
    A lower MSD indicates a smoother, more continuous waveform (signal).
    """
    diffs = np.diff(vectors, axis=1)
    msd = np.mean(diffs ** 2, axis=1)
    return float(np.mean(msd))

def main():
    print("=== Experiment 4: Explainability & Reconstructed Signal Smoothness ===")

    # 1. Load model and sentences
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    from vectorzip.default_corpus import DEFAULT_CORPUS
    sentences = DEFAULT_CORPUS[:500]
    
    print(f"Encoding {len(sentences)} sentences...")
    raw_embeddings = model.encode(sentences, show_progress_bar=False)
    M, N = raw_embeddings.shape
    print(f"Raw embeddings shape: {raw_embeddings.shape}")
    
    # Let's compress from 384 dimensions to 128 dimensions (3x compression)
    K = 128
    
    # 2. Calibrate estimators
    print("Fitting PCA...")
    pca = PCA(n_components=K)
    pca.fit(raw_embeddings)
    
    print("Fitting VectorZip...")
    vz = VectorZip(n_components=K, tsp_optimize=True)
    vz.fit(raw_embeddings)
    
    # 3. Perform reconstructions
    # PCA Reconstructed
    pca_transformed = pca.transform(raw_embeddings)
    pca_rec = pca.inverse_transform(pca_transformed)
    
    # VectorZip Reconstructed (restored to original spatial layout)
    vz_transformed = vz.transform(raw_embeddings)
    vz_rec = vz.inverse_transform(vz_transformed)
    
    # Permuted vectors (the TSP ordering of the original raw vectors)
    raw_permuted = raw_embeddings[:, vz.tsp_indices_]
    
    # Permuted reconstructed (the TSP ordered reconstructed vectors BEFORE un-permuting)
    # This represents the low-pass filtered waveform itself
    # We can reconstruct it in the permuted space by applying idct on the truncated coefficients
    from scipy.fftpack import idct
    C_padded = np.zeros((M, N))
    C_padded[:, :K] = vz_transformed
    vz_rec_permuted = idct(C_padded, type=2, axis=1, norm='ortho')
    
    # 4. Measure Smoothness (MSD)
    msd_raw = calculate_smoothness_msd(raw_embeddings)
    msd_raw_permuted = calculate_smoothness_msd(raw_permuted)
    msd_pca_rec = calculate_smoothness_msd(pca_rec)
    msd_vz_rec = calculate_smoothness_msd(vz_rec)
    msd_vz_rec_permuted = calculate_smoothness_msd(vz_rec_permuted)
    
    results = {
        "original_dimension": N,
        "target_dimension": K,
        "msd_raw_original_order": msd_raw,
        "msd_raw_tsp_order": msd_raw_permuted,
        "msd_pca_reconstructed": msd_pca_rec,
        "msd_vectorzip_reconstructed_original_order": msd_vz_rec,
        "msd_vectorzip_reconstructed_tsp_order": msd_vz_rec_permuted
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(f"Smoothness (MSD - lower is smoother):")
    print(f" -> Raw original order:           {msd_raw:.6f}")
    print(f" -> Raw TSP order (Permuted):     {msd_raw_permuted:.6f}")
    print(f" -> PCA Reconstructed:            {msd_pca_rec:.6f}")
    print(f" -> VectorZip Reconstructed:       {msd_vz_rec:.6f}")
    print(f" -> VectorZip Waveform (TSP space): {msd_vz_rec_permuted:.6f}")
    print(f"TSP reordering made raw vectors {msd_raw/msd_raw_permuted:.2f}x smoother!")
    print(f"VectorZip waveform is {msd_raw/msd_vz_rec_permuted:.2f}x smoother than original vectors!")

if __name__ == "__main__":
    main()
