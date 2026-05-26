import numpy as np
import time

def benchmark_search_latency():
    print("=========================================================")
    print("       SEARCH LATENCY BENCHMARK: 384D vs 64D             ")
    print("=========================================================")
    
    np.random.seed(42)
    
    db_size = 100000
    n_queries = 1000
    
    print(f"Generating synthetic database of size {db_size}...")
    db_384 = np.random.normal(0, 1.0, (db_size, 384)).astype(np.float32)
    db_64 = np.random.normal(0, 1.0, (db_size, 64)).astype(np.float32)
    
    # L2 normalize rows for fast cosine search
    db_384 /= np.linalg.norm(db_384, axis=1, keepdims=True) + 1e-8
    db_64 /= np.linalg.norm(db_64, axis=1, keepdims=True) + 1e-8
    
    print(f"Generating {n_queries} query vectors...")
    queries_384 = np.random.normal(0, 1.0, (n_queries, 384)).astype(np.float32)
    queries_64 = np.random.normal(0, 1.0, (n_queries, 64)).astype(np.float32)
    
    queries_384 /= np.linalg.norm(queries_384, axis=1, keepdims=True) + 1e-8
    queries_64 /= np.linalg.norm(queries_64, axis=1, keepdims=True) + 1e-8
    
    # Warmup
    _ = np.dot(db_384[:1000], queries_384[0])
    _ = np.dot(db_64[:1000], queries_64[0])
    
    # 1. Benchmark 384D (SQ8 / Raw)
    print("Benchmarking 384-dimensional searches...")
    t0 = time.perf_counter()
    for i in range(n_queries):
        q = queries_384[i]
        scores = np.dot(db_384, q)
        top_k = np.argpartition(scores, -10)[-10:]
    t1 = time.perf_counter()
    time_384 = t1 - t0
    qps_384 = n_queries / time_384
    avg_latency_384 = (time_384 / n_queries) * 1000  # in ms
    
    # 2. Benchmark 64D (VectorZip / PCA)
    print("Benchmarking 64-dimensional searches...")
    t0 = time.perf_counter()
    for i in range(n_queries):
        q = queries_64[i]
        scores = np.dot(db_64, q)
        top_k = np.argpartition(scores, -10)[-10:]
    t1 = time.perf_counter()
    time_64 = t1 - t0
    qps_64 = n_queries / time_64
    avg_latency_64 = (time_64 / n_queries) * 1000  # in ms
    
    speedup = time_384 / time_64
    
    print("\n---------------- RESULTS ----------------")
    print(f"384D (SQ8 / Raw) Search:")
    print(f"  -> Total time for {n_queries} queries: {time_384:.4f} seconds")
    print(f"  -> Average latency per query:       {avg_latency_384:.4f} ms")
    print(f"  -> Throughput (QPS):                {qps_384:.1f} queries/sec")
    
    print(f"\n64D (VectorZip / PCA) Search:")
    print(f"  -> Total time for {n_queries} queries: {time_64:.4f} seconds")
    print(f"  -> Average latency per query:       {avg_latency_64:.4f} ms")
    print(f"  -> Throughput (QPS):                {qps_64:.1f} queries/sec")
    
    print(f"\nSearch Speedup of 64D over 384D: {speedup:.2f}x faster!")
    print("=========================================================")
    
    # Save results to json
    results = {
        "db_size": db_size,
        "n_queries": n_queries,
        "d_original": 384,
        "d_compressed": 64,
        "time_384": time_384,
        "avg_latency_384_ms": avg_latency_384,
        "qps_384": qps_384,
        "time_64": time_64,
        "avg_latency_64_ms": avg_latency_64,
        "qps_64": qps_64,
        "speedup": speedup
    }
    
    import json
    with open("search_latency_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    benchmark_search_latency()
