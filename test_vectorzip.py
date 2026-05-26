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

def test_pca_backend():
    print("\n--- TEST 3: VectorZip with Selectable PCA Backend ---")
    
    np.random.seed(42)
    original_embeddings = np.random.normal(0, 1.0, (200, 768))
    
    # 1. Test class-level PCA
    vz = VectorZip(n_components=128, method="pca")
    vz.fit(original_embeddings)
    print("✓ PCA Calibration (Fit) completed.")
    
    compressed = vz.transform(original_embeddings)
    print(f"✓ PCA Compression successful. Original: {original_embeddings.shape} -> Compressed: {compressed.shape}")
    
    decompressed = vz.inverse_transform(compressed)
    print(f"✓ PCA Decompression successful. Reconstructed to: {decompressed.shape}")
    
    mse = np.mean((original_embeddings - decompressed) ** 2)
    print(f"✓ PCA Reconstruction MSE: {mse:.6f}")
    
    # 2. Test model-level PCA wrapper and serialization
    print("Loading PCA-backed VectorZipModel...")
    model = VectorZipModel("all-MiniLM-L6-v2", n_components=32, method="pca")
    
    fit_sentences = [f"This is representative sentence number {i} to calibrate our PCA backend." for i in range(100)]
    sentences = [
        "Principal Component Analysis is a statistically learned projection.",
        "It achieves optimal low-rank matrix approximations on local manifolds.",
        "Our unified toolbox enables selecting either PCA or DCT backends seamlessly."
    ]
    
    model.fit(fit_sentences)
    print("✓ PCA Model fitted successfully on representative corpus.")
    
    compressed_embeddings = model.encode(sentences)
    print(f"✓ PCA Encoded embeddings dimension: {compressed_embeddings.shape}")
    
    import shutil
    import os
    save_dir = "/tmp/vectorzip_test_pca_model"
    model.save_pretrained(save_dir)
    print("✓ PCA Model saved (serialized to JSON config) successfully.")
    
    loaded_model = VectorZipModel.from_pretrained(save_dir)
    print("✓ PCA Model loaded successfully from config.")
    loaded_compressed = loaded_model.encode(sentences)
    np.testing.assert_allclose(compressed_embeddings, loaded_compressed, rtol=1e-5, atol=1e-5)
    print("✓ Loaded PCA model produces identical compressed embeddings.")
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

def test_unified_ecosystem():
    print("\n--- TEST 4: VectorZip Comprehensive Unified Ecosystem ---")
    
    np.random.seed(42)
    original_embeddings = np.random.normal(0, 1.0, (200, 768))
    
    # 1. Test hybrid "dct+sq8"
    vz_hybrid = VectorZip(n_components=64, method="dct+sq8")
    vz_hybrid.fit(original_embeddings)
    codes_hybrid = vz_hybrid.transform(original_embeddings)
    print(f"✓ dct+sq8 compression successful. Shape: {codes_hybrid.shape}, Data type: {codes_hybrid.dtype}")
    assert codes_hybrid.dtype == np.uint8
    rec_hybrid = vz_hybrid.inverse_transform(codes_hybrid)
    print(f"✓ dct+sq8 decompression successful. Shape: {rec_hybrid.shape}")
    
    # 2. Test hybrid "pca+sq8"
    vz_pca_sq8 = VectorZip(n_components=64, method="pca+sq8")
    vz_pca_sq8.fit(original_embeddings)
    codes_pca_sq8 = vz_pca_sq8.transform(original_embeddings)
    print(f"✓ pca+sq8 compression successful. Shape: {codes_pca_sq8.shape}, Data type: {codes_pca_sq8.dtype}")
    assert codes_pca_sq8.dtype == np.uint8
    rec_pca_sq8 = vz_pca_sq8.inverse_transform(codes_pca_sq8)
    print(f"✓ pca+sq8 decompression successful. Shape: {rec_pca_sq8.shape}")
    
    # 3. Test Gaussian Random Projections "rp"
    vz_rp = VectorZip(n_components=64, method="rp")
    vz_rp.fit(original_embeddings)
    codes_rp = vz_rp.transform(original_embeddings)
    print(f"✓ rp compression successful. Shape: {codes_rp.shape}")
    rec_rp = vz_rp.inverse_transform(codes_rp)
    print(f"✓ rp decompression successful. Shape: {rec_rp.shape}")
    
    # 4. Test Product Quantization "pq"
    vz_pq = VectorZip(n_components=16, method="pq", pq_clusters=64)
    vz_pq.fit(original_embeddings)
    codes_pq = vz_pq.transform(original_embeddings)
    print(f"✓ pq compression successful. Shape: {codes_pq.shape}, Data type: {codes_pq.dtype}")
    assert codes_pq.dtype == np.uint8
    rec_pq = vz_pq.inverse_transform(codes_pq)
    print(f"✓ pq decompression successful. Shape: {rec_pq.shape}")
    
    # 5. Test pure Scalar Quantization "sq8"
    vz_sq8 = VectorZip(method="sq8")
    vz_sq8.fit(original_embeddings)
    codes_sq8 = vz_sq8.transform(original_embeddings)
    print(f"✓ sq8 compression successful. Shape: {codes_sq8.shape}, Data type: {codes_sq8.dtype}")
    assert codes_sq8.dtype == np.uint8
    rec_sq8 = vz_sq8.inverse_transform(codes_sq8)
    print(f"✓ sq8 decompression successful. Shape: {rec_sq8.shape}")

    # 6. Test Matryoshka Slicing "matryoshka"
    vz_mat = VectorZip(n_components=64, method="matryoshka")
    vz_mat.fit(original_embeddings)
    codes_mat = vz_mat.transform(original_embeddings)
    print(f"✓ matryoshka compression successful. Shape: {codes_mat.shape}")
    rec_mat = vz_mat.inverse_transform(codes_mat)
    print(f"✓ matryoshka decompression successful. Shape: {rec_mat.shape}")
    
    # 7. Test hybrid "matryoshka+sq8"
    vz_mat_sq8 = VectorZip(n_components=64, method="matryoshka+sq8")
    vz_mat_sq8.fit(original_embeddings)
    codes_mat_sq8 = vz_mat_sq8.transform(original_embeddings)
    print(f"✓ matryoshka+sq8 compression successful. Shape: {codes_mat_sq8.shape}, Data type: {codes_mat_sq8.dtype}")
    assert codes_mat_sq8.dtype == np.uint8
    rec_mat_sq8 = vz_mat_sq8.inverse_transform(codes_mat_sq8)
    print(f"✓ matryoshka+sq8 decompression successful. Shape: {rec_mat_sq8.shape}")
    
    # 8. Test Matryoshka Model-level Wrapper with serialization
    print("Loading Matryoshka-backed VectorZipModel...")
    model_mat = VectorZipModel("all-MiniLM-L6-v2", n_components=64, method="matryoshka+sq8")
    sentences = ["Matryoshka Representation Learning is supported out of the box."]
    compressed_mat = model_mat.encode(sentences)
    print(f"✓ Matryoshka Model encoded on the fly successful: {compressed_mat.shape}")
    
    import shutil
    import os
    save_dir = "/tmp/vectorzip_test_mat_model"
    model_mat.save_pretrained(save_dir)
    print("✓ Matryoshka Model saved successfully.")
    
    loaded_model = VectorZipModel.from_pretrained(save_dir)
    print("✓ Matryoshka Model loaded successfully.")
    loaded_compressed = loaded_model.encode(sentences)
    np.testing.assert_allclose(compressed_mat, loaded_compressed, rtol=1e-5, atol=1e-5)
    print("✓ Loaded Matryoshka model produces identical compressed embeddings.")
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


def test_universal_model_wrapping():
    print("\n--- TEST 5: Universal Framework Wrapping in VectorZipModel ---")
    
    # 1. Custom LangChain style model wrapper
    class DummyLangChainModel:
        def embed_documents(self, texts):
            # Returns standard float arrays
            return [[0.1 * (i + 1)] * 384 for i in range(len(texts))]
            
    base_lc = DummyLangChainModel()
    model_lc = VectorZipModel(base_lc, n_components=64, method="dct+sq8")
    
    # Fit & encode
    texts = ["LangChain document 1", "LangChain document 2"]
    compressed_lc = model_lc.encode(texts)
    print(f"✓ LangChain Model wrapped natively. Shape: {compressed_lc.shape}, Dtype: {compressed_lc.dtype}")
    assert compressed_lc.shape == (2, 64)
    assert compressed_lc.dtype == np.uint8
    
    # 2. Custom LlamaIndex style model wrapper
    class DummyLlamaIndexModel:
        def get_text_embeddings(self, texts):
            return [[0.2 * (i + 1)] * 384 for i in range(len(texts))]
            
    base_li = DummyLlamaIndexModel()
    model_li = VectorZipModel(base_li, n_components=64, method="pca+sq8")
    
    compressed_li = model_li.encode(texts)
    print(f"✓ LlamaIndex Model wrapped natively. Shape: {compressed_li.shape}, Dtype: {compressed_li.dtype}")
    assert compressed_li.shape == (2, 64)
    assert compressed_li.dtype == np.uint8
    
    # 3. Custom callable (lambda / function) wrapper
    def dummy_callable(texts):
        return [[0.3] * 384 for _ in texts]
        
    model_call = VectorZipModel(dummy_callable, n_components=32, method="sq8")
    compressed_call = model_call.encode(texts)
    print(f"✓ Raw Callable wrapped natively. Shape: {compressed_call.shape}, Dtype: {compressed_call.dtype}")
    assert compressed_call.shape == (2, 384)
    assert compressed_call.dtype == np.uint8


if __name__ == "__main__":
    print("=== STARTING VECTORZIP PYTHON LIBRARY TESTS ===")
    
    # Execute Test 1
    test_class_api()
    
    # Execute Test 2
    test_wrapper_api()
    
    # Execute Test 3 (PCA Backend)
    test_pca_backend()
    
    # Execute Test 4 (Unified Ecosystem)
    test_unified_ecosystem()
    
    # Execute Test 5 (Universal Framework Wrapping)
    test_universal_model_wrapping()
    
    print("\n=======================================================")
    # Visual satisfaction verification
    print("  PYTHON LIBRARY TESTS COMPLETED SUCCESSFULLY!  ")
    print("=======================================================")


