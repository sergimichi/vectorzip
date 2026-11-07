import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vectorzip import VectorZip


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
    for idx in range(len(queries)):
        results = search(query_embeddings[idx], corpus_embeddings, corpus, top_k=1)
        retrieved_text = results[0][0]
        if retrieved_text == gold_labels[idx]:
            correct += 1
    return correct / len(queries)


def main():
    print("=== QWEN MULTILINGUAL DEMO: GTE-Qwen2-1.5B (1536D -> 512D) vs. Qwen2.5-0.5B (512D) ===")

    corpus = [
        "El Banco Central Europeo redujo los tipos de interés para estimular la economía.",
        "Un banco de peces tropicales nadaba velozmente esquivando los corales.",
        "El anciano descansaba plácidamente en el banco de madera bajo la sombra del roble.",
        "Utilicé un gato hidráulico para levantar el chasis del coche y cambiar la rueda.",
        "El felino doméstico maullaba suavemente pidiendo su ración de comida.",
        "La física cuántica estudia el comportamiento de las partículas subatómicas.",
    ]

    queries = [
        "Herramienta mecánica para elevar un vehículo pesado",
        "Institución financiera que regula la política monetaria",
        "Un animal de compañía que maúlla y ronronea",
    ]

    gold_labels = [
        "Utilicé un gato hidráulico para levantar el chasis del coche y cambiar la rueda.",
        "El Banco Central Europeo redujo los tipos de interés para estimular la economía.",
        "El felino doméstico maullaba suavemente pidiendo su ración de comida.",
    ]

    print("\nLoading models...")
    model_giant = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    model_small = SentenceTransformer("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    print("Encoding corpus and queries...")
    emb_giant_corpus = model_giant.encode(corpus)
    emb_giant_queries = model_giant.encode(queries)

    emb_small_corpus = model_small.encode(corpus)[:, :512]
    emb_small_queries = model_small.encode(queries)[:, :512]

    print("Compressing GTE-Qwen2-1.5B (1536D -> 512D) using VectorZip (TSP-DCT)...")
    vz = VectorZip(n_components=512, method="dct", tsp_optimize=True)
    comp_giant_corpus_512 = vz.fit_transform(emb_giant_corpus)
    comp_giant_queries_512 = vz.transform(emb_giant_queries)

    acc_giant = evaluate_accuracy(queries, corpus, emb_giant_queries, emb_giant_corpus, gold_labels)
    acc_small = evaluate_accuracy(queries, corpus, emb_small_queries, emb_small_corpus, gold_labels)
    acc_vz_512 = evaluate_accuracy(queries, corpus, comp_giant_queries_512, comp_giant_corpus_512, gold_labels)

    print("\n" + "=" * 80)
    print("QWEN MULTILINGUAL DEMO RESULTS (TOP-1 RETRIEVAL ACCURACY)")
    print("=" * 80)
    print(f"1. GTE-Qwen2-1.5B Uncompressed (1536D):      {acc_giant:.2%}")
    print(f"2. VectorZip Compressed GTE-Qwen2 (512D):    {acc_vz_512:.2%}")
    print(f"3. Native Qwen2.5-0.5B Sliced (512D):        {acc_small:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
