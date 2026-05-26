import time
import numpy as np
import json
import sys
import os
from sklearn.decomposition import PCA

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip

def main():
    print("=== Experiment 10: Large-scale Training Scaling ===")

    N = 1024 # Target high-dimensional embedding size
    K = 384  # Compressed dimension
    
    dataset_sizes = [1000, 2000, 5000, 10000]
    
    pca_fit_times = []
    vectorzip_fit_times = []
    
    for M in dataset_sizes:
        print(f"Benchmarking with dataset size M={M}...")
        
        # Generate synthetic embeddings of shape (M, N)
        X = np.random.randn(M, N).astype(np.float32)
        
        # 1. PCA Fitting
        pca = PCA(n_components=K)
        t_start = time.perf_counter()
        pca.fit(X)
        t_pca = time.perf_counter() - t_start
        pca_fit_times.append(t_pca)
        
        # 2. VectorZip Fitting
        vz = VectorZip(n_components=K, tsp_optimize=True)
        t_start = time.perf_counter()
        vz.fit(X)
        t_vz = time.perf_counter() - t_start
        vectorzip_fit_times.append(t_vz)
        
        print(f"  -> M={M} | PCA Fit: {t_pca:.4f}s | VectorZip Fit: {t_vz:.4f}s")
        
    results = {
        "dataset_sizes": dataset_sizes,
        "dimension": N,
        "n_components": K,
        "pca_fit_times_sec": pca_fit_times,
        "vectorzip_fit_times_sec": vectorzip_fit_times
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
