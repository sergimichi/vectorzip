import numpy as np
import os
import unittest
import warnings
import threading
import shutil
from vectorzip import VectorZip, VectorZipModel

class TestVectorZipExtremeEdgeCases(unittest.TestCase):
    def setUp(self):
        # Base representative data for quick fitting
        np.random.seed(42)
        self.base_data = np.random.randn(150, 128)
        self.large_data = np.random.randn(50, 384)

    def test_1_dimension_bounds(self):
        print("\n[EXTREME TEST 1] Dimension Bounds and Shapes")
        
        # A. Empty datasets (0 samples or 0 dimensions)
        vz = VectorZip(n_components=16, method="dct")
        with self.assertRaises(ValueError):
            vz.fit(np.zeros((0, 128)))
            
        with self.assertRaises(ValueError):
            vz.fit(np.zeros((10, 0)))

        # B. 1D single-vector inputs (should work on transform/inverse_transform by promoting)
        vz.fit(self.base_data)
        single_vector = np.random.randn(128)
        comp = vz.transform(single_vector)
        self.assertEqual(comp.ndim, 1)
        self.assertEqual(comp.shape[0], 16)
        
        rec = vz.inverse_transform(comp)
        self.assertEqual(rec.ndim, 1)
        self.assertEqual(rec.shape[0], 128)
        print("✓ Promoted 1D arrays successfully.")

        # C. Too high n_components (exceeding physical dimensions)
        vz_too_high = VectorZip(n_components=256, method="dct")
        with self.assertRaises(ValueError):
            vz_too_high.fit(self.base_data)
            
        vz_pca_high = VectorZip(n_components=256, method="pca")
        with self.assertRaises(ValueError):
            vz_pca_high.fit(self.base_data)
            
        vz_rp_high = VectorZip(n_components=256, method="rp")
        with self.assertRaises(ValueError):
            vz_rp_high.fit(self.base_data)
            
        vz_mat_high = VectorZip(n_components=256, method="matryoshka")
        with self.assertRaises(ValueError):
            vz_mat_high.fit(self.base_data)
        print("✓ Prevented target dimensions exceeding original dimensions.")

        # D. Zero or negative n_components
        with self.assertRaises(ValueError):
            vz_invalid = VectorZip(n_components=0, method="dct")
            vz_invalid.fit(self.base_data)
            
        with self.assertRaises(ValueError):
            vz_invalid = VectorZip(n_components=-5, method="dct")
            vz_invalid.fit(self.base_data)
        print("✓ Caught zero/negative n_components successfully.")

    def test_2_numeric_degeneracy(self):
        print("\n[EXTREME TEST 2] Numeric Degeneracy & Edge Data")
        
        # A. Constant zeros matrix (zero-variance)
        zero_matrix = np.zeros((20, 128))
        for method in ["dct", "pca", "rp", "pq", "sq8", "matryoshka"]:
            vz = VectorZip(n_components=16, method=method)
            vz.fit(zero_matrix)
            comp = vz.transform(zero_matrix)
            rec = vz.inverse_transform(comp)
            self.assertEqual(comp.shape[0], 20)
            self.assertEqual(rec.shape, zero_matrix.shape)
        print("✓ Zero-variance matrices processed successfully across all backends.")

        # B. Identical duplicate rows
        identical_row = np.random.randn(128)
        dup_matrix = np.tile(identical_row, (20, 1)).astype(float)
        vz = VectorZip(n_components=16, method="pca")
        vz.fit(dup_matrix)
        comp = vz.transform(dup_matrix)
        rec = vz.inverse_transform(comp)
        self.assertEqual(comp.shape[0], 20)
        self.assertEqual(rec.shape, dup_matrix.shape)
        print("✓ Duplicated matrices (rank-1 degeneracy) processed successfully.")

        # C. Extremely large and small floats
        extreme_large = self.base_data * 1e10
        extreme_small = self.base_data * 1e-10
        
        vz_large = VectorZip(n_components=32, method="dct+sq8")
        vz_large.fit(extreme_large)
        comp_large = vz_large.transform(extreme_large)
        self.assertEqual(comp_large.dtype, np.uint8)
        
        vz_small = VectorZip(n_components=32, method="dct+sq8")
        vz_small.fit(extreme_small)
        comp_small = vz_small.transform(extreme_small)
        self.assertEqual(comp_small.dtype, np.uint8)
        print("✓ Extreme float scales processed successfully without underflow/overflow.")

        # D. Invalid numerical inputs (NaNs, Infs)
        bad_matrix = self.base_data.copy()
        bad_matrix[2, 5] = np.nan
        bad_matrix[4, 8] = np.inf
        
        vz_nan = VectorZip(n_components=16, method="dct")
        # Should raise error due to NaN/Inf in input
        with self.assertRaises(ValueError):
            vz_nan.fit(bad_matrix)
        print("✓ NaN/Inf input detected and prevented from corrupting calibration.")

    def test_3_cross_configuration_combinatorial_matrix(self):
        print("\n[EXTREME TEST 3] Combinatorial Config Matrix")
        methods = [
            "dct", "pca", "rp", "pq", "sq8", "matryoshka", "mrl",
            "dct+sq8", "pca+sq8", "rp+sq8", "matryoshka+sq8", "mrl+sq8"
        ]
        
        for m in methods:
            n_comp = 8 if "pq" not in m else 4
            vz = VectorZip(n_components=n_comp, method=m)
            vz.fit(self.base_data)
            
            # Compress
            comp = vz.transform(self.base_data)
            self.assertEqual(comp.shape[0], self.base_data.shape[0])
            if "sq8" in m or m == "pq":
                self.assertEqual(comp.dtype, np.uint8)
                
            # Decompress
            rec = vz.inverse_transform(comp)
            self.assertEqual(rec.shape, self.base_data.shape)
            print(f"  - Config '{m}' verified successfully.")
        print("✓ All 12 configuration combos passed verification.")

    def test_4_error_handling_mismatches(self):
        print("\n[EXTREME TEST 4] Error Handling and Dimension Mismatches")
        
        # A. Invalid method name
        with self.assertRaises(ValueError):
            VectorZip(method="dct+invalid_quant")
            
        with self.assertRaises(ValueError):
            VectorZip(method="magic_compression")

        # B. Dimension mismatch on transform
        vz = VectorZip(n_components=16, method="dct")
        vz.fit(self.base_data) # fit with 128-dimensional data
        
        different_dim_data = np.random.randn(10, 256)
        with self.assertRaises(ValueError):
            # Transform should raise error because dimension is 256, not 128
            vz.transform(different_dim_data)
            
        # C. Mismatch during inverse_transform
        invalid_codes = np.random.randn(10, 32) # fitted components is 16
        with self.assertRaises(ValueError):
            vz.inverse_transform(invalid_codes)
        print("✓ All dimension mismatches on transform/inverse_transform caught cleanly.")

    def test_5_numerical_lossless_limits(self):
        print("\n[EXTREME TEST 5] Lossless Limits & Exact Reconstruction")
        
        # When components match original dimension exactly, DCT should be lossless
        vz = VectorZip(n_components=128, method="dct", tsp_optimize=False)
        vz.fit(self.base_data)
        comp = vz.transform(self.base_data)
        rec = vz.inverse_transform(comp)
        
        mse = np.mean((self.base_data - rec) ** 2)
        print(f"✓ Lossless DCT Reconstruction MSE: {mse:.2e}")
        self.assertLess(mse, 1e-12)
        
        # Same for PCA when n_components equals original dimension
        vz_pca = VectorZip(n_components=128, method="pca")
        vz_pca.fit(self.base_data)
        comp_pca = vz_pca.transform(self.base_data)
        rec_pca = vz_pca.inverse_transform(comp_pca)
        
        mse_pca = np.mean((self.base_data - rec_pca) ** 2)
        print(f"✓ Lossless PCA Reconstruction MSE: {mse_pca:.2e}")
        self.assertLess(mse_pca, 1e-12)

    def test_6_high_concurrency_thread_safety(self):
        print("\n[EXTREME TEST 6] High Concurrency Thread-Safety")
        
        vz = VectorZip(n_components=32, method="dct+sq8")
        vz.fit(self.large_data)
        
        results = []
        errors = []
        
        def worker(thread_idx):
            try:
                # Concurrent compression & decompression
                chunk = np.random.randn(20, 384)
                comp = vz.transform(chunk)
                rec = vz.inverse_transform(comp)
                self.assertEqual(comp.shape, (20, 32))
                self.assertEqual(rec.shape, (20, 384))
                results.append(thread_idx)
            except Exception as e:
                errors.append(e)
                
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        self.assertEqual(len(errors), 0, f"Concurrent execution errors: {errors}")
        self.assertEqual(len(results), 25)
        print("✓ Completed 25 concurrent threads without race conditions or memory faults.")

    def test_7_wrapper_extreme_inputs(self):
        print("\n[EXTREME TEST 7] Wrapper Extreme Text Inputs")
        
        # Dummy callable embedder representing complex unicode or empty structures
        def dummy_embedder(texts):
            return np.random.randn(len(texts), 384)
            
        model = VectorZipModel(dummy_embedder, n_components=64, method="dct+sq8")
        
        # A. Empty list of sentences
        with self.assertRaises(Exception):
            model.encode([])
            
        # B. Unicode, emojis, weird encodings
        weird_texts = [
            "Hola Mundo! 🇪🇸",
            "こんにちは 🇯🇵",
            "مرحبا بالعالم 🇸🇦",
            "Character set: \x00\x01\x02\n\t",
            "A" * 10000 # Very long sentence
        ]
        
        comp = model.encode(weird_texts)
        self.assertEqual(comp.shape, (5, 64))
        self.assertEqual(comp.dtype, np.uint8)
        print("✓ Processed complex unicode, long sentences, and special characters successfully.")

    def test_8_serialization_resiliency(self):
        print("\n[EXTREME TEST 8] Serialization and Tampering Resiliency")
        
        save_dir = "/tmp/vectorzip_extreme_serialization"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            
        # 1. Fit and Save a PCA-backed VectorZipModel
        vz = VectorZip(n_components=32, method="pca")
        vz.fit(self.large_data)
        
        # Wrapper around lambda returning fitted components
        model = VectorZipModel(lambda texts: self.large_data[:len(texts)], n_components=32, method="pca")
        model.compressor = vz
        model.save_pretrained(save_dir)
        
        # Check config exists
        self.assertTrue(os.path.exists(os.path.join(save_dir, "vectorzip_config.json")))
        
        # 2. Tamper with the config file to write garbage values
        config_file = os.path.join(save_dir, "vectorzip_config.json")
        with open(config_file, "r") as f:
            content = f.read()
            
        # Replace method "pca" with a fake invalid method
        tampered_content = content.replace('"method": "pca"', '"method": "invalid_tampered_method"')
        with open(config_file, "w") as f:
            f.write(tampered_content)
            
        # Try loading tampered model - should catch invalid method cleanly
        with self.assertRaises(ValueError):
            VectorZipModel.from_pretrained(save_dir)
            
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        print("✓ Serialization tampering caught cleanly with no undefined states.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔥 STARTING EXTREME EDGE CASE & ROBUSTNESS TESTING SUITE 🔥")
    print("="*60)
    unittest.main()
