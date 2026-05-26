import numpy as np
import json
import os
import warnings
from scipy.fftpack import dct, idct

class RandomProjectionCompressor:
    """
    Gaussian Random Projections compressor using orthonormal random matrix.
    """
    def __init__(self, n_components=384, seed=42):
        self.n_components = n_components
        self.seed = seed
        self.R_ = None
        
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[1]
        rng = np.random.default_rng(self.seed)
        G = rng.normal(0.0, 1.0, (self.n_components, N))
        # Gram-Schmidt orthonormalization
        q, r = np.linalg.qr(G.T)
        self.R_ = q.T  # Shape: (n_components, N)
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
        C = np.dot(X, self.R_.T)
        return C[0] if is_1d else C
        
    def inverse_transform(self, C):
        C = np.asarray(C, dtype=float)
        is_1d = C.ndim == 1
        if is_1d:
            C = C[np.newaxis, :]
        X_rec = np.dot(C, self.R_)
        return X_rec[0] if is_1d else X_rec


class ProductQuantizer:
    """
    Product Quantization (PQ) compressor using subvector k-means.
    """
    def __init__(self, n_subvectors=8, n_clusters=256, seed=42):
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids_ = None
        
    def fit(self, X):
        from sklearn.cluster import KMeans
        X = np.asarray(X, dtype=float)
        M, N = X.shape
        d = N // self.n_subvectors
        self.centroids_ = []
        n_clus = min(self.n_clusters, M)
        n_clus = max(1, n_clus)
        for i in range(self.n_subvectors):
            sub_X = X[:, i*d : (i+1)*d]
            kmeans = KMeans(n_clusters=n_clus, random_state=self.seed, n_init=1, max_iter=20)
            kmeans.fit(sub_X)
            self.centroids_.append(kmeans.cluster_centers_)
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
            cents = self.centroids_[i]
            dists = np.sum((sub_X[:, np.newaxis, :] - cents[np.newaxis, :, :]) ** 2, axis=2)
            codes[:, i] = np.argmin(dists, axis=1)
        return codes[0] if is_1d else codes
        
    def inverse_transform(self, codes):
        codes = np.asarray(codes, dtype=int)
        is_1d = codes.ndim == 1
        if is_1d:
            codes = codes[np.newaxis, :]
        M, m = codes.shape
        d = self.centroids_0_shape = self.centroids_[0].shape[1]
        N = m * d
        X_rec = np.zeros((M, N))
        for i in range(m):
            cents = self.centroids_[i]
            X_rec[:, i*d : (i+1)*d] = cents[codes[:, i]]
        return X_rec[0] if is_1d else X_rec


class ScalarQuantizer:
    """
    Column-wise Scalar Quantization (SQ8) to 8-bit integers.
    """
    def __init__(self):
        self.min_val_ = None
        self.max_val_ = None
        
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_val_ = np.min(X, axis=0)
        self.max_val_ = np.max(X, axis=0)
        self.max_val_ = np.where(self.max_val_ == self.min_val_, self.max_val_ + 1e-8, self.max_val_)
        return self
        
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
        X_scaled = (X - self.min_val_) / (self.max_val_ - self.min_val_)
        X_scaled = np.clip(X_scaled, 0.0, 1.0)
        codes = (X_scaled * 255.0).astype(np.uint8)
        return codes[0] if is_1d else codes
        
    def inverse_transform(self, codes):
        codes = np.asarray(codes, dtype=float)
        is_1d = codes.ndim == 1
        if is_1d:
            codes = codes[np.newaxis, :]
        X_rec = self.min_val_ + (codes / 255.0) * (self.max_val_ - self.min_val_)
        return X_rec[0] if is_1d else X_rec


class VectorZip:
    """
    VectorZip: Unified vector compression ecosystem for RAG embeddings.
    
    Acts as a comprehensive general-purpose vector compression toolbox supporting:
        - "dct": Travelling Salesperson (TSP) 2-opt reordering + Discrete Cosine Transform (DCT-II).
        - "pca": Traditional Principal Component Analysis (optimal local manifolds).
        - "rp": Gaussian Random Projections.
        - "pq": Product Quantization.
        - "sq8": Scalar Quantization to int8.
        - "matryoshka" (or "mrl"): Matryoshka Representation Learning static coordinate truncation.
        - Hybrid Multiplicative Modes: "dct+sq8", "pca+sq8", "rp+sq8", "matryoshka+sq8"
          (reducing/slicing dimensionality *then* quantizing floats to bytes).
          
    API mimics scikit-learn estimators:
        - fit(X)
        - transform(X)
        - fit_transform(X)
        - inverse_transform(C)
    """
    def __init__(self, n_components=384, method="dct", tsp_optimize=True, pq_subvectors=8, pq_clusters=256, rp_seed=42):
        self.n_components = n_components
        self.method = method.lower()
        self.tsp_optimize = tsp_optimize
        self.pq_subvectors = pq_subvectors
        self.pq_clusters = pq_clusters
        self.rp_seed = rp_seed
        
        parts = self.method.split("+")
        self.reduction_method = parts[0]
        self.quantization_method = parts[1] if len(parts) > 1 else None
        
        valid_reductions = ["dct", "pca", "rp", "pq", "sq8", "matryoshka", "mrl"]
        if self.reduction_method not in valid_reductions:
            raise ValueError(f"Invalid compression method: '{self.method}'")
        if self.quantization_method and self.quantization_method != "sq8":
            raise ValueError(f"Invalid quantization method: '{self.quantization_method}'")
            
        self.tsp_indices_ = None
        self.pca_ = None
        self.rp_ = None
        self.pq_ = None
        self.sq_ = None
        self.original_dim_ = None

    def fit(self, X):
        """
        Fits the selected compression backend to a corpus of vectors.
        """
        X = np.asarray(X, dtype=float)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input X contains NaNs or Infs which are invalid for vector compression.")
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
            
        M, N = X.shape
        if M == 0 or N == 0:
            raise ValueError("Input corpus cannot be empty (n_samples > 0 and n_features > 0 required).")
            
        if self.n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
            
        self.original_dim_ = N
        
        # Fit dimensionality reduction backend
        if self.reduction_method == "dct":
            if self.n_components > N:
                raise ValueError(f"Target n_components ({self.n_components}) cannot exceed dimension {N}.")
            if self.tsp_optimize:
                self.tsp_indices_ = self._solve_tsp(X)
            else:
                self.tsp_indices_ = np.arange(N)
        elif self.reduction_method == "pca":
            if self.n_components > N:
                raise ValueError(f"Target n_components ({self.n_components}) cannot exceed dimension {N}.")
            from sklearn.decomposition import PCA
            self.pca_ = PCA(n_components=self.n_components)
            self.pca_.fit(X)
        elif self.reduction_method == "rp":
            if self.n_components > N:
                raise ValueError(f"Target n_components ({self.n_components}) cannot exceed dimension {N}.")
            self.rp_ = RandomProjectionCompressor(n_components=self.n_components, seed=self.rp_seed)
            self.rp_.fit(X)
        elif self.reduction_method == "pq":
            # For PQ, we use n_components as the subvector partition count
            self.pq_ = ProductQuantizer(n_subvectors=self.n_components, n_clusters=self.pq_clusters, seed=self.rp_seed)
            self.pq_.fit(X)
        elif self.reduction_method in ["matryoshka", "mrl"]:
            if self.n_components > N:
                raise ValueError(f"Target n_components ({self.n_components}) cannot exceed dimension {N}.")
            # Static slicing does not require learning coordinates
            pass
        elif self.reduction_method == "sq8":
            self.sq_ = ScalarQuantizer()
            self.sq_.fit(X)
            
        # Fit secondary quantization layer if hybrid requested
        if self.quantization_method == "sq8":
            C_float = self._transform_reduction(X)
            self.sq_ = ScalarQuantizer()
            self.sq_.fit(C_float)
            
        return self

    def _transform_reduction(self, X):
        if self.reduction_method == "dct":
            X_perm = X[:, self.tsp_indices_]
            C_full = dct(X_perm, type=2, axis=1, norm='ortho')
            return C_full[:, :self.n_components]
        elif self.reduction_method == "pca":
            return self.pca_.transform(X)
        elif self.reduction_method == "rp":
            return self.rp_.transform(X)
        elif self.reduction_method == "pq":
            return self.pq_.transform(X)
        elif self.reduction_method in ["matryoshka", "mrl"]:
            return X[:, :self.n_components]
        elif self.reduction_method == "sq8":
            return X

    def transform(self, X):
        """
        Compresses the vectors using the unified pipeline.
        """
        X = np.asarray(X, dtype=float)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input X contains NaNs or Infs which are invalid for vector compression.")
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
            
        if self.original_dim_ is not None and X.shape[1] != self.original_dim_:
            raise ValueError(f"Dimension mismatch: Input has {X.shape[1]} features, but VectorZip was fitted on {self.original_dim_} features.")
            
        C_float = self._transform_reduction(X)
        
        # Product Quantization returns byte codes directly
        if self.reduction_method == "pq":
            return C_float[0] if is_1d else C_float
            
        # Apply secondary scalar quantization layer
        if self.quantization_method == "sq8":
            C_quant = self.sq_.transform(C_float)
            return C_quant[0] if is_1d else C_quant
        elif self.reduction_method == "sq8":
            C_quant = self.sq_.transform(X)
            return C_quant[0] if is_1d else C_quant
            
        return C_float[0] if is_1d else C_float

    def fit_transform(self, X):
        """
        Fits the compressor and transforms the input in a single step.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, C):
        """
        Reconstructs the compressed representations back to the original vector space.
        """
        C = np.asarray(C)
        is_1d = C.ndim == 1
        if is_1d:
            C = C[np.newaxis, :]
            
        expected_components = self.n_components if self.reduction_method != "sq8" else self.original_dim_
        if C.shape[1] != expected_components:
            raise ValueError(f"Dimension mismatch: Input has {C.shape[1]} components, but expected {expected_components}.")
            
        # Dequantize back to float first if quantized
        if self.quantization_method == "sq8":
            C_float = self.sq_.inverse_transform(C)
        elif self.reduction_method == "sq8":
            C_float = self.sq_.inverse_transform(C)
            return C_float[0] if is_1d else C_float
        else:
            C_float = C.astype(float)
            
        # Decompress back to original dimensions
        if self.reduction_method == "dct":
            M, K = C_float.shape
            N = len(self.tsp_indices_)
            C_padded = np.zeros((M, N))
            C_padded[:, :K] = C_float
            X_rec_perm = idct(C_padded, type=2, axis=1, norm='ortho')
            inv_tsp_indices = np.argsort(self.tsp_indices_)
            X_rec = X_rec_perm[:, inv_tsp_indices]
        elif self.reduction_method == "pca":
            X_rec = self.pca_.inverse_transform(C_float)
        elif self.reduction_method == "rp":
            X_rec = self.rp_.inverse_transform(C_float)
        elif self.reduction_method == "pq":
            X_rec = self.pq_.inverse_transform(C_float)
        elif self.reduction_method in ["matryoshka", "mrl"]:
            M, K = C_float.shape
            N = self.original_dim_ if self.original_dim_ is not None else K
            X_rec = np.zeros((M, N))
            X_rec[:, :K] = C_float
            
        return X_rec[0] if is_1d else X_rec

    def _solve_tsp(self, X):
        M, N = X.shape
        if N <= 1:
            return np.arange(N)
            
        mean_sq = np.mean(X ** 2, axis=0)
        dot_product = np.dot(X.T, X) / M
        dist_matrix = mean_sq[:, np.newaxis] + mean_sq[np.newaxis, :] - 2 * dot_product
        dist_matrix = np.clip(dist_matrix, 0, None)
        
        unvisited = set(range(N))
        current = 0
        path = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda node: dist_matrix[current, node])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        path = self._two_opt(path, dist_matrix)
        return np.array(path)

    def _two_opt(self, path, dist_matrix):
        n = len(path)
        best_path = path[:]
        improved = True
        max_iters = 100
        iters = 0
        while improved and iters < max_iters:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n + 1):
                    if j - i == 1:
                        continue
                    
                    node_i_prev = best_path[i - 1]
                    node_i = best_path[i]
                    node_j_prev = best_path[j - 1]
                    
                    if j < n:
                        node_j = best_path[j]
                        current_edges_cost = dist_matrix[node_i_prev, node_i] + dist_matrix[node_j_prev, node_j]
                        new_edges_cost = dist_matrix[node_i_prev, node_j_prev] + dist_matrix[node_i, node_j]
                    else:
                        current_edges_cost = dist_matrix[node_i_prev, node_i]
                        new_edges_cost = dist_matrix[node_i_prev, node_j_prev]
                    
                    if new_edges_cost < current_edges_cost:
                        best_path[i:j] = best_path[i:j][::-1]
                        improved = True
            iters += 1
        return best_path

    def _path_cost(self, path, dist_matrix):
        return sum(dist_matrix[path[i], path[i+1]] for i in range(len(path) - 1))


class VectorZipModel:
    """
    VectorZipModel: Seamless drop-in wrapper substitute for embedding libraries.
    
    Encapsulates any SentenceTransformer model, automatically training the 
    VectorZip compressor during its first encoding run, and seamlessly yielding
    highly compressed embeddings.
    """
    def __init__(self, model_name_or_path, n_components=384, method="dct", **kwargs):
        if isinstance(model_name_or_path, str):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name_or_path, **kwargs)
        else:
            self.model = model_name_or_path
        self.n_components = n_components
        self.method = method.lower()
        self.compressor = None

    def _get_embeddings(self, sentences, **kwargs):
        if hasattr(self.model, "encode"):
            return self.model.encode(sentences, **kwargs)
        elif hasattr(self.model, "embed_documents"):
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.asarray(self.model.embed_documents(sentences))
        elif hasattr(self.model, "get_text_embeddings"):
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.asarray(self.model.get_text_embeddings(sentences))
        elif hasattr(self.model, "embed"):
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.asarray(self.model.embed(sentences))
        elif callable(self.model):
            return np.asarray(self.model(sentences))
        else:
            raise AttributeError("The wrapped model does not have a recognized encoding method (encode, embed_documents, get_text_embeddings, embed) and is not callable.")

    def fit(self, sentences=None, **kwargs):
        """
        Fits the compressor to a representative corpus of sentences.
        If sentences is None, uses an internal generic diverse corpus.
        """
        if sentences is None:
            from vectorzip.default_corpus import DEFAULT_CORPUS
            sentences = DEFAULT_CORPUS
            
        embeddings = self._get_embeddings(sentences, **kwargs)
        self.compressor = VectorZip(n_components=self.n_components, method=self.method)
        self.compressor.fit(embeddings)
        return self

    def encode(self, sentences, decompress=False, **kwargs):
        """
        Encodes sentences and automatically yields compressed (or decompressed) embeddings.
        """
        if not sentences or len(sentences) == 0:
            raise ValueError("Input sentences list cannot be empty.")
            
        if self.compressor is None:
            # Fast static fitting bypass for MRL or SQ8 (no generic corpus required)
            if self.method in ["matryoshka", "mrl", "sq8", "matryoshka+sq8", "mrl+sq8"]:
                self.compressor = VectorZip(n_components=self.n_components, method=self.method)
                # Quick dimensions setup
                first_emb = self._get_embeddings(sentences[:2], **kwargs)
                self.compressor.fit(first_emb)
            else:
                warnings.warn("VectorZipModel not explicitly fitted. Automatically calibrating using the internal generic corpus. For optimal domain-specific performance, call `.fit(your_corpus)` first.")
                self.fit()

        embeddings = self._get_embeddings(sentences, **kwargs)
        compressed = self.compressor.transform(embeddings)
        
        if decompress:
            return self.compressor.inverse_transform(compressed)
            
        return compressed

    def save_pretrained(self, path):
        """
        Saves the underlying model and VectorZip configuration to disk in pure JSON.
        """
        os.makedirs(path, exist_ok=True)
        if hasattr(self.model, "save") and callable(self.model.save):
            try:
                self.model.save(path)
            except Exception as e:
                warnings.warn(f"Could not save base model weights: {e}. Only saving VectorZip config.")
        elif hasattr(self.model, "save_pretrained") and callable(self.model.save_pretrained):
            try:
                self.model.save_pretrained(path)
            except Exception as e:
                warnings.warn(f"Could not save base model weights: {e}. Only saving VectorZip config.")
        
        config = {
            "n_components": self.n_components,
            "method": self.method,
        }
        if self.compressor is not None:
            config["tsp_optimize"] = self.compressor.tsp_optimize
            config["pq_subvectors"] = self.compressor.pq_subvectors
            config["pq_clusters"] = self.compressor.pq_clusters
            config["rp_seed"] = self.compressor.rp_seed
            
            if self.compressor.original_dim_ is not None:
                config["original_dim_"] = self.compressor.original_dim_
            
            # Serialize DCT
            if self.compressor.tsp_indices_ is not None:
                config["tsp_indices_"] = self.compressor.tsp_indices_.tolist()
                
            # Serialize PCA
            if self.compressor.pca_ is not None:
                config["pca_components_"] = self.compressor.pca_.components_.tolist()
                config["pca_mean_"] = self.compressor.pca_.mean_.tolist()
                config["pca_explained_variance_"] = self.compressor.pca_.explained_variance_.tolist()
                
            # Serialize RP
            if self.compressor.rp_ is not None and self.compressor.rp_.R_ is not None:
                config["rp_matrix_"] = self.compressor.rp_.R_.tolist()
                
            # Serialize PQ
            if self.compressor.pq_ is not None and self.compressor.pq_.centroids_ is not None:
                config["pq_centroids_"] = [c.tolist() for c in self.compressor.pq_.centroids_]
                
            # Serialize SQ
            if self.compressor.sq_ is not None:
                config["sq_min_val_"] = self.compressor.sq_.min_val_.tolist()
                config["sq_max_val_"] = self.compressor.sq_.max_val_.tolist()
                
        with open(os.path.join(path, "vectorzip_config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Loads a pre-trained VectorZipModel from disk.
        """
        config_path = os.path.join(model_name_or_path, "vectorzip_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            n_components = config.get("n_components", 384)
            method = config.get("method", "dct")
            model = cls(model_name_or_path, n_components=n_components, method=method, **kwargs)
            
            model.compressor = VectorZip(
                n_components=n_components, 
                method=method,
                tsp_optimize=config.get("tsp_optimize", True),
                pq_subvectors=config.get("pq_subvectors", 8),
                pq_clusters=config.get("pq_clusters", 256),
                rp_seed=config.get("rp_seed", 42)
            )
            
            # Restore state based on config fields
            if "original_dim_" in config:
                model.compressor.original_dim_ = config["original_dim_"]
                
            if "tsp_indices_" in config:
                model.compressor.tsp_indices_ = np.array(config["tsp_indices_"])
                
            if "pca_components_" in config:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                pca.components_ = np.array(config["pca_components_"])
                pca.mean_ = np.array(config["pca_mean_"])
                pca.explained_variance_ = np.array(config["pca_explained_variance_"])
                pca.n_components_ = n_components
                pca.n_features_in_ = pca.components_.shape[1]
                pca.singular_values_ = np.ones(n_components)
                pca.noise_variance_ = 0.0
                model.compressor.pca_ = pca
                
            if "rp_matrix_" in config:
                model.compressor.rp_ = RandomProjectionCompressor(n_components=n_components, seed=config.get("rp_seed", 42))
                model.compressor.rp_.R_ = np.array(config["rp_matrix_"])
                
            if "pq_centroids_" in config:
                model.compressor.pq_ = ProductQuantizer(n_subvectors=n_components, n_clusters=config.get("pq_clusters", 256), seed=config.get("rp_seed", 42))
                model.compressor.pq_.centroids_ = [np.array(c) for c in config["pq_centroids_"]]
                
            if "sq_min_val_" in config:
                model.compressor.sq_ = ScalarQuantizer()
                model.compressor.sq_.min_val_ = np.array(config["sq_min_val_"])
                model.compressor.sq_.max_val_ = np.array(config["sq_max_val_"])
                
            return model
        else:
            return cls(model_name_or_path, **kwargs)

    def decompress(self, C):
        """
        Exposed high-fidelity decompression function.
        """
        if self.compressor is None:
            raise RuntimeError("Model must encode some sentences first to calibrate dimensions before decompressing.")
        return self.compressor.inverse_transform(C)


# One-line Functional APIs
def compress(X, n_components=384, method="dct", tsp_optimize=True):
    """
    One-line functional interface to fit and compress embeddings.
    """
    vz = VectorZip(n_components=n_components, method=method, tsp_optimize=tsp_optimize)
    return vz.fit_transform(X)

def decompress(C):
    """
    One-line functional interface to reconstruct compressed embeddings.
    """
    raise NotImplementedError("Direct global decompress requires static mappings. Use VectorZip or VectorZipModel class instead.")
