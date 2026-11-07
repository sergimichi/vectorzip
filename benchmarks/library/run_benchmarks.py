import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vectorzip import VectorZip


def generate_correlated_embeddings(n_vectors, orig_dim, seed=42):
    """Generates synthetic embeddings with smooth latent spectral structure for throughput benchmarking."""
    rng = np.random.default_rng(seed)
    c = np.arange(orig_dim)[np.newaxis, :]
    phases = np.arange(n_vectors)[:, np.newaxis] * 0.01
    X = (
        np.sin(c * 0.05 + phases)
        + np.cos(c * 0.1) * 0.2
        + rng.normal(0.0, 0.02, (n_vectors, orig_dim))
    )
    return X


def run_benchmarks():
    print("=== RUNNING VECTORZIP PERFORMANCE & LATENCY BENCHMARKS ===")

    configs = [
        (384, 96, "384D -> 96D (e.g., MiniLM / BGE-Small)"),
        (768, 192, "768D -> 192D (e.g., BGE-Base / Nomic)"),
        (1024, 256, "1024D -> 256D (e.g., BGE-Large / BGE-M3)"),
        (1536, 384, "1536D -> 384D (e.g., GTE-Qwen2 / OpenAI-Small)"),
        (3072, 768, "3072D -> 768D (e.g., OpenAI-Large)"),
    ]

    results_table = [
        "| Dimension Scale | Target Dims | Compression Ratio | MSE Error | Cosine Retention | Transform Latency (1,000 Vectors) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    n_vectors = 1000

    for orig_dim, target_dim, name in configs:
        print(f"\nBenchmarking {name}...")
        X = generate_correlated_embeddings(n_vectors, orig_dim)

        vz = VectorZip(n_components=target_dim, method="dct", tsp_optimize=True, dct_taper="none")

        t_start_fit = time.perf_counter()
        vz.fit(X)
        t_fit = (time.perf_counter() - t_start_fit) * 1000.0

        # Warmup transform
        _ = vz.transform(X[:10])

        t_start_compress = time.perf_counter()
        C = vz.transform(X)
        t_compress = (time.perf_counter() - t_start_compress) * 1000.0

        X_rec = vz.inverse_transform(C)

        mse = float(np.mean((X - X_rec) ** 2))

        norms_o = np.linalg.norm(X, axis=1)
        norms_r = np.linalg.norm(X_rec, axis=1)
        dots = np.sum(X * X_rec, axis=1)
        avg_cosine = float(np.mean(dots / (norms_o * norms_r + 1e-9)))

        ratio = f"{orig_dim / target_dim:.1f}x"

        print(f"  - Fit Time:          {t_fit:.2f} ms")
        print(f"  - Transform Latency: {t_compress:.2f} ms (1,000 vectors)")
        print(f"  - Reconstruction MSE:{mse:.6f}")
        print(f"  - Cosine Retention:  {avg_cosine:.2%}")

        results_table.append(
            f"| {name} | **{target_dim}** | {ratio} | `{mse:.6f}` | **{avg_cosine:.2%}** | `{t_compress:.2f} ms` |"
        )

    print("\n--- Summary Markdown Table ---")
    print("\n".join(results_table))


if __name__ == "__main__":
    run_benchmarks()
