import numpy as np
import json
import time
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datasets import load_dataset

# Add parent directory to path to import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorzip import VectorZip

class RandomProjectionCompressor:
    """
    Gaussian Random Projections baseline using orthonormal random matrix.
    """
    def __init__(self, n_components=384, seed=42):
        self.n_components = n_components
        self.seed = seed
        self.R = None
        
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[1]
        rng = np.random.default_rng(self.seed)
        G = rng.normal(0.0, 1.0, (self.n_components, N))
        # Gram-Schmidt orthonormalization of rows
        q, r = np.linalg.qr(G.T)
        self.R = q.T  # Shape: (n_components, N)
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
        C = np.dot(X, self.R.T)
        return C[0] if is_1d else C
        
    def inverse_transform(self, C):
        C = np.asarray(C, dtype=float)
        is_1d = C.ndim == 1
        if is_1d:
            C = C[np.newaxis, :]
        X_rec = np.dot(C, self.R)
        return X_rec[0] if is_1d else X_rec

class ProductQuantizer:
    """
    Product Quantization (PQ) baseline using subvector k-means.
    """
    def __init__(self, n_subvectors=8, n_clusters=256, seed=42):
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids = []
        
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        M, N = X.shape
        d = N // self.n_subvectors
        self.centroids = []
        for i in range(self.n_subvectors):
            sub_X = X[:, i*d : (i+1)*d]
            # Use mini-batch k-means or simple k-means with fast parameters for benchmark speed
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=1, max_iter=20)
            kmeans.fit(sub_X)
            self.centroids.append(kmeans.cluster_centers_)
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
        M, N = X.shape
        d = N // self.n_subvectors
        codes = np.zeros((M, self.n_subvectors), dtype=np.uint8)
        for i in range(self.n_subvectors):
            sub_X = X[:, i*d : (i+1)*d]
            cents = self.centroids[i]  # shape (n_clusters, d)
            dists = np.sum((sub_X[:, np.newaxis, :] - cents[np.newaxis, :, :]) ** 2, axis=2)
            codes[:, i] = np.argmin(dists, axis=1)
        return codes[0] if is_1d else codes
        
    def inverse_transform(self, codes):
        is_1d = codes.ndim == 1
        if is_1d:
            codes = codes[np.newaxis, :]
        M, m = codes.shape
        d = self.centroids[0].shape[1]
        N = m * d
        X_rec = np.zeros((M, N))
        for i in range(m):
            cents = self.centroids[i]
            X_rec[:, i*d : (i+1)*d] = cents[codes[:, i]]
        return X_rec[0] if is_1d else X_rec

class ScalarQuantizer:
    """
    Scalar Quantization (SQ8) baseline.
    """
    def __init__(self):
        self.min_val = None
        self.max_val = None
        
    def fit(self, X):
        self.min_val = float(np.min(X))
        self.max_val = float(np.max(X))
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val + 1e-8)
        X_scaled = np.clip(X_scaled, 0.0, 1.0)
        codes = (X_scaled * 255).astype(np.uint8)
        return codes[0] if is_1d else codes
        
    def inverse_transform(self, codes):
        is_1d = codes.ndim == 1
        if is_1d:
            codes = codes[np.newaxis, :]
        X_rec = self.min_val + (codes.astype(float) / 255.0) * (self.max_val - self.min_val)
        return X_rec[0] if is_1d else X_rec

def calculate_csr(raw, reconstructed):
    """Calculates Cosine Similarity Retention (CSR)."""
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
    print("=========================================================")
    print("   RUNNING ALL COMPREHENSIVE COMPRESSION BASELINES      ")
    print("=========================================================")
    
    # 1. Load model and datasets
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    
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
    
    print("Encoding all corpora to generate dense 384D embeddings...")
    generic_raw = model.encode(generic_sentences, show_progress_bar=False)
    medical_raw = model.encode(medical_sentences, show_progress_bar=False)
    code_raw = model.encode(code_sentences, show_progress_bar=False)
    legal_raw = model.encode(legal_sentences, show_progress_bar=False)
    
    # Target dimension: K = 64
    # Note: For Product Quantization, 64 bytes is 64 subvectors (which is m=64, each d=6, clusters=256)
    # This gives an identical byte-width on disk (64 bytes per compressed vector) as K=64 float16 or 32 float32!
    # Wait, K=64 float32 is 256 bytes, K=64 float16 is 128 bytes, K=64 int8 is 64 bytes.
    # To be extremely fair, we evaluate dimensionality reduction to K=64 components.
    K = 64
    
    # 2. Fit and calibrate all baselines on Generic Wiki corpus
    print("\n[Phase 1] Fitting all baselines on Generic Wikipedia Corpus...")
    
    # PCA
    pca = PCA(n_components=K)
    pca.fit(generic_raw)
    
    # VectorZip
    vz = VectorZip(n_components=K, tsp_optimize=True)
    vz.fit(generic_raw)
    
    # Gaussian Random Projections
    rp = RandomProjectionCompressor(n_components=K)
    rp.fit(generic_raw)
    
    # Product Quantization (m=16 subvectors of size 24, k=256 centroids)
    # 16 subvectors takes 16 bytes, which represents extremely high compression
    pq = ProductQuantizer(n_subvectors=16, n_clusters=256)
    pq.fit(generic_raw)
    
    # Scalar Quantization (SQ8)
    sq = ScalarQuantizer()
    sq.fit(generic_raw)
    
    # 3. Evaluate Similarity Retention (CSR) In-Distribution & Out-of-Distribution
    print("\n[Phase 2] Evaluating CSR (Cosine Similarity Retention) on domains:")
    
    def evaluate_model(name, compressor, dataset_raw):
        reconstructed = compressor.inverse_transform(compressor.transform(dataset_raw))
        return calculate_csr(dataset_raw, reconstructed)
        
    print("--- 1. In-Distribution (Wiki) ---")
    csr_id_pca = evaluate_model("PCA", pca, generic_raw)
    csr_id_vz = evaluate_model("VectorZip", vz, generic_raw)
    csr_id_rp = evaluate_model("Random Projections", rp, generic_raw)
    csr_id_pq = evaluate_model("Product Quantization", pq, generic_raw)
    csr_id_sq = evaluate_model("Scalar Quantization", sq, generic_raw)
    
    print(f" PCA CSR:                  {csr_id_pca:.4f}")
    print(f" VectorZip CSR:            {csr_id_vz:.4f}")
    print(f" Random Projections CSR:   {csr_id_rp:.4f}")
    print(f" Product Quantization CSR: {csr_id_pq:.4f}")
    print(f" Scalar Quantization CSR:  {csr_id_sq:.4f}")
    
    print("--- 2. Out-of-Distribution (Medical) ---")
    csr_med_pca = evaluate_model("PCA", pca, medical_raw)
    csr_med_vz = evaluate_model("VectorZip", vz, medical_raw)
    csr_med_rp = evaluate_model("Random Projections", rp, medical_raw)
    csr_med_pq = evaluate_model("Product Quantization", pq, medical_raw)
    csr_med_sq = evaluate_model("Scalar Quantization", sq, medical_raw)
    
    print(f" PCA CSR:                  {csr_med_pca:.4f}")
    print(f" VectorZip CSR:            {csr_med_vz:.4f}")
    print(f" Random Projections CSR:   {csr_med_rp:.4f}")
    print(f" Product Quantization CSR: {csr_med_pq:.4f}")
    print(f" Scalar Quantization CSR:  {csr_med_sq:.4f}")
    
    print("--- 3. Coding Generalization ---")
    csr_code_pca = evaluate_model("PCA", pca, code_raw)
    csr_code_vz = evaluate_model("VectorZip", vz, code_raw)
    csr_code_rp = evaluate_model("Random Projections", rp, code_raw)
    csr_code_pq = evaluate_model("Product Quantization", pq, code_raw)
    csr_code_sq = evaluate_model("Scalar Quantization", sq, code_raw)
    
    print(f" PCA CSR:                  {csr_code_pca:.4f}")
    print(f" VectorZip CSR:            {csr_code_vz:.4f}")
    print(f" Random Projections CSR:   {csr_code_rp:.4f}")
    print(f" Product Quantization CSR: {csr_code_pq:.4f}")
    print(f" Scalar Quantization CSR:  {csr_code_sq:.4f}")
    
    print("--- 4. Legal Generalization ---")
    csr_leg_pca = evaluate_model("PCA", pca, legal_raw)
    csr_leg_vz = evaluate_model("VectorZip", vz, legal_raw)
    csr_leg_rp = evaluate_model("Random Projections", rp, legal_raw)
    csr_leg_pq = evaluate_model("Product Quantization", pq, legal_raw)
    csr_leg_sq = evaluate_model("Scalar Quantization", sq, legal_raw)
    
    print(f" PCA CSR:                  {csr_leg_pca:.4f}")
    print(f" VectorZip CSR:            {csr_leg_vz:.4f}")
    print(f" Random Projections CSR:   {csr_leg_rp:.4f}")
    print(f" Product Quantization CSR: {csr_leg_pq:.4f}")
    print(f" Scalar Quantization CSR:  {csr_leg_sq:.4f}")
    
    # Save baseline results to json
    baseline_results = {
        "K": K,
        "wikipedia_id": {
            "pca": csr_id_pca,
            "vectorzip": csr_id_vz,
            "random_projections": csr_id_rp,
            "product_quantization": csr_id_pq,
            "scalar_quantization": csr_id_sq
        },
        "medical_ood": {
            "pca": csr_med_pca,
            "vectorzip": csr_med_vz,
            "random_projections": csr_med_rp,
            "product_quantization": csr_med_pq,
            "scalar_quantization": csr_med_sq
        },
        "code_ood": {
            "pca": csr_code_pca,
            "vectorzip": csr_code_vz,
            "random_projections": csr_code_rp,
            "product_quantization": csr_code_pq,
            "scalar_quantization": csr_code_sq
        },
        "legal_ood": {
            "pca": csr_leg_pca,
            "vectorzip": csr_leg_vz,
            "random_projections": csr_leg_rp,
            "product_quantization": csr_leg_pq,
            "scalar_quantization": csr_leg_sq
        }
    }
    
    output_file = os.path.join(os.path.dirname(__file__), "baseline_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(baseline_results, f, indent=4)
        
    print(f"\nBaseline comparison metrics successfully written to {output_file}!")
    print("=========================================================")

if __name__ == "__main__":
    main()
