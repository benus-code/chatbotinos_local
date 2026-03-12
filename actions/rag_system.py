"""
**Tester des Modèles d'Embedding**

Ce notebook accompagne la vidéo tutoriel sur le test et l'utilisation de modèles d'embedding avec Hugging Face et Mistral Al dans le cadre du projet RAG (Retrieval-Augmented Generation) pour la mairie de Trifouillis-sur-Loire.


Objectif: Explorer comment générer des représentations vectorielles 

*   Élément de liste
*   Élément de liste

(embeddings) de textes pour améliorer la recherche d'informations pertinentes.
Outils utilisés:

Hugging Face: Pour accéder à des modèles d'embedding open source via la bibliothèque sentence-transformers.

Mistral Al: Pour utiliser des modèles d'embedding performants via leur API serverless.

# **Objectifs:**
En comparant la similarité entre vecteur_requête et les vecteur_document disponibles, 
le système RAG peut identifier le document le plus pertinent pour répondre à la question, même si les mots exacts ne sont pas identiques.
"""

# Installation des bibliothèques (à exécuter dans le terminal Conda)
# Dans votre terminal Conda, exécutez :
# pip install sentence-transformers numpy

# Chargement du modèle et vectorisation de textes
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # Éviter la division par zéro
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    # Charger le modèle depuis Hugging Face Hub
    print("Chargement du modèle Sentence Transformer...")
    try:
        model_hf = SentenceTransformer("C:/Users/Ivana Leba/.cache\huggingface/modules/transformers_modules/jinaai/xlm-roberta-flash-implementation", trust_remote_code=True, local_files_only=True)
        print("Modèle chargé !")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Textes d'exemple (simulant une question et un extrait de document)
    textes_hf = [
        "Quelles sont les heures d'ouverture de la mairie ?",  # Simulation question citoyen
        "La mairie de Trifouillis-sur-Loire est ouverte du lundi au vendredi de 8h30 à 17h00.",  # Simulation info document
        "Le marché hebdomadaire a lieu tous les samedis matin sur la place centrale."  # Info non pertinente
    ]
    
    # Générer les embeddings
    print("\nGénération des embeddings...")
    embeddings_hf = model_hf.encode(textes_hf)
    print("Embeddings générés !")
    print(f"Shape du premier embedding: {embeddings_hf[0].shape}")  # Montre la dimension du vecteur
    
    # Embedding de la question
    embedding_question = embeddings_hf[0]
    # Embeddings des documents potentiels
    embedding_doc1 = embeddings_hf[1]
    embedding_doc2 = embeddings_hf[2]
    
    # Calcul des similarités
    similarity_q_doc1 = cosine_similarity(embedding_question, embedding_doc1)
    similarity_q_doc2 = cosine_similarity(embedding_question, embedding_doc2)
    
    print(f"Question: '{textes_hf[0]}'")
    print("-" * 50)
    print(f"Document 1: '{textes_hf[1]}'")
    print(f"Similarité avec la question: {similarity_q_doc1:.4f}")
    print("-" * 50)
    print(f"Document 2: '{textes_hf[2]}'")
    print(f"Similarité avec la question: {similarity_q_doc2:.4f}")
    print("-" * 50)
    
    if similarity_q_doc1 > similarity_q_doc2:
        print("\n✅ Conclusion: Le document 1 est sémantiquement plus proche de la question.")
    else:
        print("\n❌ Conclusion: Le document 2 est sémantiquement plus proche de la question.")

if __name__ == "__main__":
    main()