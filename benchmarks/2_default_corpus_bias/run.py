import numpy as np
import json
import sys
import os
import pickle
from sklearn.decomposition import PCA

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip

def main():
    print("=== Experiment 2: Memory Footprint & Storage Weight ===")
    
    N = 1024 # Original dimension (e.g. BGE-M3, Cohere)
    K = 384  # Target dimension
    
    # 1. Mathematical Footprint Calculations
    # PCA: Needs dense matrix of size N x K (float32 is 4 bytes, float64 is 8 bytes)
    # Plus a mean vector of size N
    pca_matrix_bytes_f32 = (N * K * 4) + (N * 4)
    pca_matrix_bytes_f64 = (N * K * 8) + (N * 8)
    
    # VectorZip: Only needs 1D permutation array of size N (int64 is 8 bytes, int32 is 4 bytes, int16 is 2 bytes)
    vz_indices_bytes_i16 = N * 2
    vz_indices_bytes_i32 = N * 4
    vz_indices_bytes_i64 = N * 8
    
    # 2. Empirical Serialization Size
    print(f"Calibrating a mock model (N={N} -> K={K})...")
    X = np.random.randn(500, N).astype(np.float32)
    
    # PCA Model
    pca = PCA(n_components=K)
    pca.fit(X)
    
    # VectorZip Model
    vz = VectorZip(n_components=K, tsp_optimize=True)
    vz.fit(X)
    
    # Serialize PCA
    pca_pickle = pickle.dumps(pca)
    pca_np_save = io_bytes = pickle.dumps(pca.components_)
    
    # Serialize VectorZip
    # VectorZip indices can be saved as standard JSON (list of ints)
    vz_indices_list = vz.tsp_indices_.tolist()
    vz_json_serialized = json.dumps(vz_indices_list)
    vz_pickle = pickle.dumps(vz.tsp_indices_)
    
    pca_pickle_kb = len(pca_pickle) / 1024.0
    vz_json_kb = len(vz_json_serialized.encode('utf-8')) / 1024.0
    vz_pickle_kb = len(vz_pickle) / 1024.0
    
    # Size ratios
    size_ratio_pickle = len(pca_pickle) / len(vz_pickle)
    size_ratio_json = len(pca_pickle) / len(vz_json_serialized.encode('utf-8'))
    
    results = {
        "original_dimension": N,
        "target_dimension": K,
        "math_pca_bytes_float32": pca_matrix_bytes_f32,
        "math_pca_bytes_float64": pca_matrix_bytes_f64,
        "math_vectorzip_bytes_int16": vz_indices_bytes_i16,
        "math_vectorzip_bytes_int32": vz_indices_bytes_i32,
        "math_vectorzip_bytes_int64": vz_indices_bytes_i64,
        "empirical_pca_pickle_bytes": len(pca_pickle),
        "empirical_pca_pickle_kb": pca_pickle_kb,
        "empirical_vectorzip_json_bytes": len(vz_json_serialized.encode('utf-8')),
        "empirical_vectorzip_json_kb": vz_json_kb,
        "empirical_vectorzip_pickle_bytes": len(vz_pickle),
        "empirical_vectorzip_pickle_kb": vz_pickle_kb,
        "memory_saving_ratio_pickle": size_ratio_pickle,
        "memory_saving_ratio_json": size_ratio_json
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(f"PCA Saved size (Pickle):     {pca_pickle_kb:.2f} KB ({len(pca_pickle)} bytes)")
    print(f"VectorZip Saved size (JSON): {vz_json_kb:.2f} KB ({len(vz_json_serialized.encode('utf-8'))} bytes)")
    print(f"VectorZip Saved size (Pickle):{vz_pickle_kb:.2f} KB ({len(vz_pickle)} bytes)")
    print(f"VectorZip is {size_ratio_json:.1f}x smaller than PCA in production storage!")

if __name__ == "__main__":
    main()
