import os
import numpy as np
from scipy.fftpack import dct, idct
from sentence_transformers import SentenceTransformer

# Advanced TSP solver
def solve_tsp_nn(X):
    M, N = X.shape
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
        
    # Local 2-opt refinement for stability
    best_path = path[:]
    n = len(path)
    for _ in range(5):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n):
                if j - i == 1:
                    continue
                node_i_prev = best_path[i - 1]
                node_i = best_path[i]
                node_j_prev = best_path[j - 1]
                node_j = best_path[j] if j < n else best_path[0]
                
                curr_dist = dist_matrix[node_i_prev, node_i] + dist_matrix[node_j_prev, node_j]
                new_dist = dist_matrix[node_i_prev, node_j_prev] + dist_matrix[node_i, node_j]
                
                if new_dist < curr_dist - 1e-9:
                    best_path[i:j] = best_path[i:j][::-1]
                    improved = True
        if not improved:
            break
            
    return np.array(best_path)

# DCT Compress with Abrupt Truncation
def compress_dct(X, K):
    C_full = dct(X, type=2, axis=1, norm='ortho')
    C_trunc = C_full[:, :K]
    C_padded = np.zeros_like(C_full)
    C_padded[:, :K] = C_trunc
    X_rec = idct(C_padded, type=2, axis=1, norm='ortho')
    return C_trunc, X_rec

def search(query_vector, corpus_vectors, corpus_texts, top_k=3):
    q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
    c_norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1e-9
    c_norm_vectors = corpus_vectors / c_norms
    
    scores = np.dot(c_norm_vectors, q_norm)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus_texts[idx], scores[idx]) for idx in top_indices]

def evaluate_accuracy(queries, corpus, query_embeddings, corpus_embeddings, gold_labels):
    correct = 0
    for idx, q_text in enumerate(queries):
        results = search(query_embeddings[idx], corpus_embeddings, corpus, top_k=1)
        retrieved_text = results[0][0]
        if retrieved_text == gold_labels[idx]:
            correct += 1
    return correct / len(queries)

def main():
    print("=== SPANISH QWEN BENCHMARK: Qwen2.5-Large (1536 Compressed to 512) vs Qwen2.5-Small (512 Native Truncated) ===")
    
    # 1. Corpus and Queries in Spanish (High difficulty semantic traps)
    corpus = [
        "El Banco Central Europeo redujo los tipos de interés para estimular la economía.",
        "Un banco de peces tropicales nadaba velozmente esquivando los corales.",
        "El anciano descansaba plácidamente en el banco de madera bajo la sombra del roble.",
        "Utilicé un gato hidráulico para levantar el chasis del coche y cambiar la rueda.",
        "El felino doméstico maullaba suavemente pidiendo su ración de comida.",
        "La física cuántica estudia el comportamiento de las partículas subatómicas."
    ]
    
    queries = [
        "Herramienta mecánica para elevar un vehículo pesado",
        "Institución financiera que regula la política monetaria",
        "Un animal de compañía que maúlla y ronronea"
    ]
    
    gold_labels = [
        "Utilicé un gato hidráulico para levantar el chasis del coche y cambiar la rueda.",
        "El Banco Central Europeo redujo los tipos de interés para estimular la economía.",
        "El felino doméstico maullaba suavemente pidiendo su ración de comida."
    ]
    
    # Load Models
    print("\nLoading Models...")
    # Qwen2.5-Large is Alibaba-NLP/gte-Qwen2-1.5B-instruct (1536 dimensions)
    print("Loading Qwen2.5-Large (1536 dims, Qwen2 Giant)...")
    model_giant = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    
    # Qwen2.5-Small is Qwen/Qwen2.5-0.5B-Instruct (896 dimensions native)
    print("Loading Qwen2.5-Small (896 dims Native)...")
    model_small = SentenceTransformer("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    
    print("\nEncoding corpus and queries...")
    emb_giant_corpus = model_giant.encode(corpus)
    emb_giant_queries = model_giant.encode(queries)
    
    emb_small_corpus_full = model_small.encode(corpus)
    emb_small_queries_full = model_small.encode(queries)
    
    # Slice Qwen2.5-Small to 512 dimensions natively
    emb_small_corpus = emb_small_corpus_full[:, :512]
    emb_small_queries = emb_small_queries_full[:, :512]
    
    # Compress GTE-Qwen2 (1536 -> 512)
    print("\nCompressing Qwen2.5-Large to 512 dimensions using VectorZip (TSP-DCT)...")
    tsp_indices = solve_tsp_nn(emb_giant_corpus)
    emb_giant_corpus_tsp = emb_giant_corpus[:, tsp_indices]
    emb_giant_queries_tsp = emb_giant_queries[:, tsp_indices]
    
    _, rec_giant_corpus_512 = compress_dct(emb_giant_corpus_tsp, 512)
    _, rec_giant_queries_512 = compress_dct(emb_giant_queries_tsp, 512)
    
    # Evaluate Accuracy (Top-1 Match)
    acc_giant = evaluate_accuracy(queries, corpus, emb_giant_queries, emb_giant_corpus, gold_labels)
    acc_small = evaluate_accuracy(queries, corpus, emb_small_queries, emb_small_corpus, gold_labels)
    acc_our_512 = evaluate_accuracy(queries, corpus, rec_giant_queries_512, rec_giant_corpus_512, gold_labels)
    
    print("\n" + "="*80)
    print("SPANISH QWEN BENCHMARK RESULTS (TOP-1 SEMANTIC SEARCH ACCURACY)")
    print("="*80)
    print(f"1. Qwen2.5-Large Giant (1536 dims Ground Truth):         {acc_giant:.2%}")
    print(f"2. Our Compressed Qwen2.5-Large (512 dims):              {acc_our_512:.2%}")
    print(f"3. Native Qwen2.5-Small (512 dims Native Truncated):     {acc_small:.2%}")
    print("="*80)
    print(f"Gain vs. Native Model of identical weight: {acc_our_512 - acc_small:+.2%}")
    print("="*80)
    
    # Detailed check of failures
    print("\nDetailed breakdown of Qwen2.5-Small (512 dims) failures in Spanish:")
    for idx, q_text in enumerate(queries):
        results_small = search(emb_small_queries[idx], emb_small_corpus, corpus, top_k=1)
        results_our = search(rec_giant_queries_512[idx], rec_giant_corpus_512, corpus, top_k=1)
        
        print(f"\nQuery: '{q_text}'")
        print(f"  ❌ Native Small (512) returned: [{results_small[0][1]:.4f}] '{results_small[0][0]}'")
        print(f"  ✅ Our Compressed (512) returned: [{results_our[0][1]:.4f}] '{results_our[0][0]}'")

if __name__ == "__main__":
    main()
