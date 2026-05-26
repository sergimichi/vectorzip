import numpy as np
from scipy.fftpack import dct, idct

class VectorZip:
    """
    VectorZip: High-performance spectral vector compression for RAG systems.
    
    Transforms high-dimensional dense vector embeddings into low-entropy, 
    smoother discrete waveforms using Traveling Salesperson Problem (TSP) 2-opt reordering 
    and Discrete Cosine Transform (DCT-II) projections.
    
    API mimics scikit-learn estimators:
        - fit(X)
        - transform(X)
        - fit_transform(X)
        - inverse_transform(C)
    """
    def __init__(self, n_components=384, tsp_optimize=True):
        self.n_components = n_components
        self.tsp_optimize = tsp_optimize
        self.tsp_indices_ = None

    def fit(self, X):
        """
        Fits the compressor to a corpus of vectors.
        Computes the covariance matrix and solves the TSP dimension permutation.
        """
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
            
        M, N = X.shape
        
        if self.n_components > N:
            raise ValueError(f"Target n_components ({self.n_components}) cannot be greater than original dimension size ({N}).")
        
        if self.tsp_optimize:
            self.tsp_indices_ = self._solve_tsp(X)
        else:
            self.tsp_indices_ = np.arange(N)
            
        return self

    def transform(self, X):
        """
        Compresses the vectors into low-dimensional representations.
        """
        if self.tsp_indices_ is None:
            raise RuntimeError("VectorZip must be fitted before calling transform.")
            
        X = np.asarray(X, dtype=float)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[np.newaxis, :]
            
        M, N = X.shape
        if N != len(self.tsp_indices_):
            raise ValueError(f"Dimension mismatch. Expected {len(self.tsp_indices_)} dimensions, got {N}.")
            
        X_perm = X[:, self.tsp_indices_]
        
        C_full = dct(X_perm, type=2, axis=1, norm='ortho')
        C_trunc = C_full[:, :self.n_components]
        
        if is_1d:
            C_trunc = C_trunc[0]
            
        return C_trunc

    def fit_transform(self, X):
        """
        Fits the compressor and transforms the input in a single step.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, C):
        """
        Reconstructs the compressed spectral vectors back to the original space.
        """
        if self.tsp_indices_ is None:
            raise RuntimeError("VectorZip must be fitted before calling inverse_transform.")
            
        C = np.asarray(C, dtype=float)
        is_1d = C.ndim == 1
        if is_1d:
            C = C[np.newaxis, :]
            
        M, K = C.shape
        N = len(self.tsp_indices_)
        
        if K != self.n_components:
            raise ValueError(f"Component size mismatch. Expected {self.n_components} elements, got {K}.")
            
        C_padded = np.zeros((M, N))
        C_padded[:, :K] = C
        X_rec_perm = idct(C_padded, type=2, axis=1, norm='ortho')
            
        inv_tsp_indices = np.argsort(self.tsp_indices_)
        X_rec = X_rec_perm[:, inv_tsp_indices]
        
        if is_1d:
            X_rec = X_rec[0]
            
        return X_rec

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
    def __init__(self, model_name_or_path, n_components=384, **kwargs):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name_or_path, **kwargs)
        self.n_components = n_components
        self.compressor = None

    def fit(self, sentences=None, **kwargs):
        """
        Fits the compressor to a representative corpus of sentences.
        If sentences is None, uses an internal generic diverse corpus.
        """
        if sentences is None:
            from vectorzip.default_corpus import DEFAULT_CORPUS
            sentences = DEFAULT_CORPUS
            
        embeddings = self.model.encode(sentences, **kwargs)
        self.compressor = VectorZip(n_components=self.n_components)
        self.compressor.fit(embeddings)
        return self

    def encode(self, sentences, decompress=False, **kwargs):
        """
        Encodes sentences and automatically yields compressed (or decompressed) embeddings.
        """
        if self.compressor is None:
            import warnings
            warnings.warn("VectorZipModel not explicitly fitted. Automatically calibrating using the internal generic corpus. For optimal domain-specific performance, call `.fit(your_corpus)` first.")
            self.fit()

        # 1. Generate high-dimensional embeddings using native model
        embeddings = self.model.encode(sentences, **kwargs)
        
        # 2. Compress vectors
        compressed = self.compressor.transform(embeddings)
        
        # 3. Decompress if requested
        if decompress:
            return self.compressor.inverse_transform(compressed)
            
        return compressed

    def save_pretrained(self, path):
        """
        Saves the underlying model and VectorZip configuration to disk.
        """
        import os
        import json
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
        
        config = {
            "n_components": self.n_components,
        }
        if self.compressor is not None:
            config["tsp_optimize"] = self.compressor.tsp_optimize
            if self.compressor.tsp_indices_ is not None:
                config["tsp_indices_"] = self.compressor.tsp_indices_.tolist()
                
        with open(os.path.join(path, "vectorzip_config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Loads a pre-trained VectorZipModel from disk.
        """
        import os
        import json
        config_path = os.path.join(model_name_or_path, "vectorzip_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            n_components = config.get("n_components", 384)
            model = cls(model_name_or_path, n_components=n_components, **kwargs)
            
            if "tsp_indices_" in config:
                model.compressor = VectorZip(n_components=n_components, tsp_optimize=config.get("tsp_optimize", True))
                model.compressor.tsp_indices_ = np.array(config["tsp_indices_"])
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
def compress(X, n_components=384, tsp_optimize=True):
    """
    One-line functional interface to fit and compress embeddings.
    """
    vz = VectorZip(n_components=n_components, tsp_optimize=tsp_optimize)
    return vz.fit_transform(X)

def decompress(C):
    """
    One-line functional interface to reconstruct compressed embeddings.
    Not recommended for production RAG (use VectorZip class for static mappings),
    but highly useful for quick analytical scripts.
    """
    # Simply reconstructs assuming identity mapping for quick testing
    # Note: Requires fitted static compressor for true mapping consistency.
    raise NotImplementedError("Direct global decompress requires static mappings. Use VectorZip or VectorZipModel class instead.")
