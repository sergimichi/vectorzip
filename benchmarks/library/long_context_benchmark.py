import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vectorzip import VectorZip


def search_top_1(query_vector, corpus_vectors):
    q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
    c_norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1e-9
    c_norm_vectors = corpus_vectors / c_norms

    scores = np.dot(c_norm_vectors, q_norm)
    return int(np.argmax(scores)), float(np.max(scores))


def main():
    print("=== LONG-CONTEXT RETRIEVAL DEMO: Compressed BGE-M3 (384D) vs. Native BGE-Small (384D) ===")

    filler = (
        "The culinary history of rice cultivation dates back thousands of years to ancient Asia, "
        "where early farmers domesticated wild grains in flooded paddies. Over centuries, rice "
        "spread along trade routes to the Mediterranean, becoming a staple ingredient in various "
        "cultures. In Spain, the region of Valencia perfected the art of rice cooking, creating "
        "the world-famous paella. Traditional Valencian paella is cooked over an open wood fire "
        "using local ingredients like chicken, rabbit, green beans, and saffron. The wood smoke "
        "infuses the rice with a subtle, aromatic flavor that cannot be replicated on a standard stove. "
    )

    secret_fact = "The secret activation code of the satellite is 7741-XYZ."

    doc_beginning = f"{secret_fact} " + filler * 12
    doc_middle = (filler * 6) + f" {secret_fact} " + (filler * 6)
    doc_end = (filler * 12) + f" {secret_fact}"

    control_docs = [
        "The history of agriculture shows that crop rotation and soil enrichment were key "
        "innovations that allowed civilizations to thrive. Rice, wheat, and maize formed the "
        "foundation of human nutrition across three continents, each adapted to local climates. "
        + filler * 11,
        "Traditional gastronomy relies on fresh, seasonal ingredients sourced from local farms. "
        "Valencian chefs argue that the choice of firewood, such as orange wood, is critical to "
        "achieving the perfect socarrat, the caramelized layer of rice at the bottom of the pan. "
        + filler * 11,
    ]

    query = "Find the document containing the secret activation code of the satellite."
    positions = ["BEGINNING (Word ~10)", "MIDDLE (Word ~550)", "END (Word ~1100)"]
    test_docs = [doc_beginning, doc_middle, doc_end]

    print("\nLoading models...")
    model_m3 = SentenceTransformer("BAAI/bge-m3")
    model_small = SentenceTransformer("BAAI/bge-small-en-v1.5")

    for pos_idx, position in enumerate(positions):
        print("\n" + "=" * 80)
        print(f"TESTING NEEDLE POSITION: {position}")
        print("=" * 80)

        corpus = [test_docs[pos_idx]] + control_docs

        emb_small_corpus = model_small.encode(corpus)
        emb_small_query = model_small.encode([query])[0]
        idx_small, score_small = search_top_1(emb_small_query, emb_small_corpus)

        emb_m3_corpus = model_m3.encode(corpus)
        emb_m3_query = model_m3.encode([query])[0]

        vz = VectorZip(n_components=384, method="dct", tsp_optimize=True)
        comp_m3_corpus = vz.fit_transform(emb_m3_corpus)
        comp_m3_query = vz.transform(emb_m3_query)

        idx_comp, score_comp = search_top_1(comp_m3_query, comp_m3_corpus)

        print(f"  Native BGE-Small (384D, 512-token limit):      {'PASS' if idx_small == 0 else 'FAIL'} (Top-1 idx: {idx_small}, score: {score_small:.4f})")
        print(f"  VectorZip Compressed BGE-M3 (384D, 8K limit):  {'PASS' if idx_comp == 0 else 'FAIL'} (Top-1 idx: {idx_comp}, score: {score_comp:.4f})")


if __name__ == "__main__":
    main()
