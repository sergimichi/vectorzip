import time
import numpy as np
import json
import sys
import os
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Add parent directory to path so we can import vectorzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vectorzip import VectorZip, VectorZipModel

def main():
    print("=== Experiment 1: Latency & Transform Phase Benchmark ===")
    
    # 1. Warm up and measure standard embedding model overhead (miniLM)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    sentences = ["Artificial intelligence is transforming modern software engineering."] * 500
    
    print("Running Raw Model encoding warm-up...")
    _ = model.encode(sentences[:10])
    start = time.perf_counter()
    _ = model.encode(sentences, show_progress_bar=False)
    raw_encoding_time = time.perf_counter() - start
    
    vz_model = VectorZipModel(model_name, n_components=128)
    vz_model.fit()
    
    start = time.perf_counter()
    _ = vz_model.encode(sentences, decompress=False, show_progress_bar=False)
    vz_comp_time = time.perf_counter() - start
    
    encoding_overhead = vz_comp_time - raw_encoding_time
    print(f"Raw Model Time (500 sentences):  {raw_encoding_time:.4f}s")
    print(f"VectorZip Time (500 sentences):  {vz_comp_time:.4f}s")
    print(f"Isolate encoding overhead:        {encoding_overhead*1000:.2f}ms")

    # 2. Pure Transform-only Benchmark at scale (1,000 vectors)
    print("\n--- Pure Transform-only Phase Benchmark (1,000 vectors, 1024D -> 384D) ---")
    M, N, K = 1000, 1024, 384
    print(f"Generating synthetic batch: {M} vectors of {N} dimensions...")
    X_large = np.random.randn(M, N).astype(np.float32)
    X_train = np.random.randn(1000, N).astype(np.float32) # calibration subset
    
    print("Fitting PCA estimator...")
    pca = PCA(n_components=K)
    pca.fit(X_train)
    
    print("Fitting VectorZip estimator...")
    vz = VectorZip(n_components=K, tsp_optimize=True)
    vz.fit(X_train)
    
    # Warm up transform
    _ = pca.transform(X_large[:100])
    _ = vz.transform(X_large[:100])
    
    # Benchmark PCA
    print("Benchmarking PCA transform...")
    pca_times = []
    for _ in range(5):
        t_start = time.perf_counter()
        _ = pca.transform(X_large)
        pca_times.append(time.perf_counter() - t_start)
    pca_avg = np.mean(pca_times)
    
    # Benchmark VectorZip
    print("Benchmarking VectorZip transform...")
    vz_times = []
    for _ in range(5):
        t_start = time.perf_counter()
        _ = vz.transform(X_large)
        vz_times.append(time.perf_counter() - t_start)
    vz_avg = np.mean(vz_times)
    
    speedup = pca_avg / vz_avg
    throughput_pca = M / pca_avg
    throughput_vz = M / vz_avg
    
    results = {
        "batch_size": M,
        "original_dimension": N,
        "target_dimension": K,
        "pca_transform_avg_sec": pca_avg,
        "vectorzip_transform_avg_sec": vz_avg,
        "pca_throughput_vectors_per_sec": throughput_pca,
        "vectorzip_throughput_vectors_per_sec": throughput_vz,
        "vectorzip_speedup_factor": speedup,
        "mini_lm_sentences_count": len(sentences),
        "mini_lm_raw_encoding_sec": raw_encoding_time,
        "mini_lm_vectorzip_encoding_sec": vz_comp_time,
        "encoding_overhead_sec": encoding_overhead
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(f"PCA transform time:       {pca_avg:.4f}s ({throughput_pca:.1f} vectors/sec)")
    print(f"VectorZip transform time: {vz_avg:.4f}s ({throughput_vz:.1f} vectors/sec)")
    print(f"VectorZip speedup factor: {speedup:.2f}x faster than PCA!")

if __name__ == "__main__":
    main()
