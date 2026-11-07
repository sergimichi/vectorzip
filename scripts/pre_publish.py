import os
import sys
import subprocess
import time


def run_step(command, description, cwd=None):
    print(f"\n--- [STEP] {description} ---")
    print(f"Running: {command}")
    t_start = time.perf_counter()

    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    t_elapsed = time.perf_counter() - t_start

    if result.returncode != 0:
        print(f"FAILED: {description} (Exit code: {result.returncode})")
        print("\n--- STDERR ---")
        print(result.stderr)
        print("\n--- STDOUT ---")
        print(result.stdout)
        sys.exit(1)

    print(f"Success ({t_elapsed:.2f}s).")
    return result.stdout


def main():
    print("=====================================================================")
    print("VECTORZIP PRE-PUBLISH VERIFICATION & BUILD PIPELINE")
    print("=====================================================================")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    python_bin = "./venv/bin/python" if os.path.exists("./venv/bin/python") else sys.executable

    # 1. Run unit & integration tests
    run_step(
        f"{python_bin} tests/test_vectorzip.py",
        "Running public API integration tests",
    )

    # 2. Run extreme edge-case & concurrency tests
    run_step(
        f"{python_bin} tests/test_extreme_edge_cases.py",
        "Running numerical edge-case and concurrency test suite",
    )

    # 3. Run throughput and latency benchmarks
    run_step(
        f"{python_bin} benchmarks/library/run_benchmarks.py",
        "Running projection throughput and fidelity benchmarks",
    )

    # 4. Build distribution packages (.whl and .tar.gz)
    run_step(
        f"{python_bin} -m build",
        "Building distribution packages (.whl and .tar.gz)",
    )

    print("\n" + "=" * 69)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 69)
    print("- Unit, integration, and edge-case test suites passed.")
    print("- Performance benchmarks verified.")
    print("- Distribution artifacts built in 'dist/'.")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
