"""Script de validation de l'indexation Qdrant — Module 1."""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    COLLECTION = "FAQ_Multilingue"

    # 1. Compte total des points
    info = client.get_collection(COLLECTION)
    total = info.points_count
    print(f"\n=== Collection '{COLLECTION}' ===")
    print(f"Total points indexes : {total}")

    # 2. Répartition par type (faq vs pdf)
    faq_count = client.count(
        collection_name=COLLECTION,
        count_filter={"must": [{"key": "type", "match": {"value": "faq"}}]},
    )
    pdf_count = client.count(
        collection_name=COLLECTION,
        count_filter={"must": [{"key": "type", "match": {"value": "pdf"}}]},
    )
    print(f"  - FAQ       : {faq_count.count} points")
    print(f"  - PDF/corpus: {pdf_count.count} points")

    # 3. Test de recherche sémantique
    questions_test = [
        "comment obtenir mon visa étudiant ?",
        "documents nécessaires pour l'inscription",
        "logement sur le campus TUSUR",
    ]

    print("\n=== Tests de recherche sémantique ===")
    for question in questions_test:
        vector = model.encode(question).tolist()
        results = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=2,
        )
        print(f"\nQ: {question!r}")
        for i, r in enumerate(results.points, 1):
            content_preview = r.payload.get("content", "")[:100].replace("\n", " ")
            src = r.payload.get("source_file") or r.payload.get("source", "pdf")
            print(f"  {i}. [score={r.score:.3f}] [{src}] {content_preview}...")
