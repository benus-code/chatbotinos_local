from indexation import client, collection_name, model, search_faq

questions = [
    "comment obtenir mon visa etudiant",
    "Je veux savoir comment avoir l'invitation",
    "Quels documents pour visa",
]

if __name__ == "__main__":
    for q in questions:
        print(f"\nQuestion: {q}")
        results = search_faq(client, collection_name, model, q, limit=3)
        for r in results:
            content = r.payload.get("content", "")[:80]
            print(f"  score={r.score:.4f} | {content}")
