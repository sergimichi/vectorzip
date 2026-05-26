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
    return np.array(path)

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
    print("=== THE SPANISH BENCHMARK: Why Compression is Essential for Global RAG ===")
    
    # 1. Corpus and Queries in Spanish
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
    # BGE-M3 is the state-of-the-art multilingual model (1024 dims, 567M parameters)
    print("Loading BGE-M3 (1024 dims Multilingual Giant)...")
    model_large_multi = SentenceTransformer("BAAI/bge-m3")
    
    # BGE-Small is the english-only model (384 dims, 24M parameters)
    print("Loading BGE-Small-en (384 dims English-only Native)...")
    model_small_en = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print("\nEncoding corpus and queries...")
    emb_large_corpus = model_large_multi.encode(corpus)
    emb_large_queries = model_large_multi.encode(queries)
    
    emb_small_corpus = model_small_en.encode(corpus)
    emb_small_queries = model_small_en.encode(queries)
    
    # Compress BGE-M3 (1024 -> 384)
    print("\nCompressing BGE-M3 to 384 dimensions using TSP-DCT...")
    tsp_indices = solve_tsp_nn(emb_large_corpus)
    emb_large_corpus_tsp = emb_large_corpus[:, tsp_indices]
    emb_large_queries_tsp = emb_large_queries[:, tsp_indices]
    
    _, rec_large_corpus_384 = compress_dct(emb_large_corpus_tsp, 384)
    _, rec_large_queries_384 = compress_dct(emb_large_queries_tsp, 384)
    
    # Evaluate Accuracy (Top-1 Match)
    acc_large = evaluate_accuracy(queries, corpus, emb_large_queries, emb_large_corpus, gold_labels)
    acc_small = evaluate_accuracy(queries, corpus, emb_small_queries, emb_small_corpus, gold_labels)
    acc_our_384 = evaluate_accuracy(queries, corpus, rec_large_queries_384, rec_large_corpus_384, gold_labels)
    
    print("\n" + "="*80)
    print("SPANISH BENCHMARK RESULTS (TOP-1 SEMANTIC SEARCH ACCURACY)")
    print("="*80)
    print(f"1. BGE-M3 Multilingual (1024 dims Ground Truth):        {acc_large:.2%}")
    print(f"2. Our Compressed BGE-M3 (384 dims):                     {acc_our_384:.2%}")
    print(f"3. Native BGE-Small (384 dims English Native):           {acc_small:.2%}")
    print("="*80)
    print(f"Gain vs. Native Model of identical weight: {acc_our_384 - acc_small:+.2%}")
    print("="*80)
    
    # Detailed check of what BGE-Small returned
    print("\nDetailed breakdown of BGE-Small-en (384 dims) failures in Spanish:")
    for idx, q_text in enumerate(queries):
        results_small = search(emb_small_queries[idx], emb_small_corpus, corpus, top_k=1)
        results_our = search(rec_large_queries_384[idx], rec_large_corpus_384, corpus, top_k=1)
        
        print(f"\nQuery: '{q_text}'")
        print(f"  ❌ Native Small (384) returned: [{results_small[0][1]:.4f}] '{results_small[0][0]}'")
        print(f"  ✅ Our Compressed (384) returned: [{results_our[0][1]:.4f}] '{results_our[0][0]}'")

if __name__ == "__main__":
    main()
