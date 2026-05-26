import numpy as np
from vectorzip import VectorZip, VectorZipModel

def test_class_api():
    print("\n--- TEST 1: VectorZip Class (Scikit-Learn Style) ---")
    
    # 1. Create synthetic embeddings (10 vectors of 768 dimensions)
    np.random.seed(42)
    original_embeddings = np.random.normal(0, 1.0, (10, 768))
    
    # 2. Initialize compressor to 192 components (4x compression)
    vz = VectorZip(n_components=192)
    
    # 3. Calibrate (Learn TSP permutation based on covariance)
    vz.fit(original_embeddings)
    print("✓ Calibration (Fit) completed.")
    
    # 4. Compress
    compressed = vz.transform(original_embeddings)
    print(f"✓ Compression successful. Original dimension: {original_embeddings.shape} -> Compressed: {compressed.shape}")
    
    # 5. Decompress
    decompressed = vz.inverse_transform(compressed)
    print(f"✓ Decompression successful. Reconstructed to: {decompressed.shape}")
    
    # Evaluate reconstruction error
    mse = np.mean((original_embeddings - decompressed) ** 2)
    print(f"✓ Reconstruction Mean Squared Error (MSE): {mse:.6f}")


def test_wrapper_api():
    print("\n--- TEST 2: VectorZipModel Wrapper (Drop-in Replacement) ---")
    
    # 1. Load the wrapper enveloping the ultra-light all-MiniLM-L6-v2 model (originally 384 dims)
    # We configure it to compress down to 96 dimensions (4x savings)
    print("Loading VectorZipModel ('all-MiniLM-L6-v2' wrapped to 96 dims)...")
    model = VectorZipModel("all-MiniLM-L6-v2", n_components=96)
    
    # 2. Test sentences
    sentences = [
        "Spectral compression saves seventy-five percent of RAM memory.",
        "Vector databases scale incredibly well with short vectors.",
        "Quantum physics studies the behavior of elementary particles."
    ]
    
    # 3. Fit the model to the sentences
    model.fit(sentences)
    print("✓ Model fitted successfully on corpus.")
    
    # 4. Encode text directly. Embeddings are automatically compressed to 96 dims!
    compressed_embeddings = model.encode(sentences)
    print(f"✓ Texts encoded successfully.")
    print(f"✓ Resulting embeddings dimension: {compressed_embeddings.shape}")
    
    # 5. Get decompressed embeddings directly using the decompress=True parameter
    decompressed_embeddings = model.encode(sentences, decompress=True)
    print(f"✓ Decompressed embeddings obtained directly: {decompressed_embeddings.shape}")
    
    # 6. Manually decompress a saved vector
    reconstructed_manually = model.decompress(compressed_embeddings)
    print(f"✓ Manual decompression successful. Dimension: {reconstructed_manually.shape}")

    # 7. Test Serialization
    import shutil
    import os
    save_dir = "/tmp/vectorzip_test_model"
    model.save_pretrained(save_dir)
    print("✓ Model saved successfully.")
    
    loaded_model = VectorZipModel.from_pretrained(save_dir)
    print("✓ Model loaded successfully.")
    loaded_compressed = loaded_model.encode(sentences)
    np.testing.assert_allclose(compressed_embeddings, loaded_compressed, rtol=1e-5, atol=1e-5)
    print("✓ Loaded model produces identical compressed embeddings.")
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

if __name__ == "__main__":
    print("=== STARTING VECTORZIP PYTHON LIBRARY TESTS ===")
    
    # Execute Test 1
    test_class_api()
    
    # Execute Test 2
    test_wrapper_api()
    
    print("\n=======================================================")
    # Visual satisfaction verification
    print("  PYTHON LIBRARY TESTS COMPLETED SUCCESSFULLY!  ")
    print("=======================================================")
