import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
client = QdrantClient(host="localhost", port=6333, timeout=60)
# Utilisation de ton modèle Qwen
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
collection_name = "FAQ_Multilingue"

# --- 2. FONCTIONS DE TRAITEMENT ---

def charger_donnees_json(chemin_du_fichier):
    """Charge le fichier JSON généré précédemment."""
    with open(chemin_du_fichier, 'r', encoding='utf-8') as f:
        return json.load(f)

def generer_points_pdf(liste_extraits, modele_embedding):
    """Transforme chaque bloc JSON en point vectoriel pour Qdrant."""
    points = []
    for item in liste_extraits:
        id_unique = str(uuid.uuid4())
        # On vectorise le texte français (la langue de recherche)
        vecteur = modele_embedding.encode(item["texte_fr"]).tolist()
        
        points.append(PointStruct(
            id=id_unique,
            vector=vecteur,
           payload={
    "content": item["texte_fr"],
    "texte_ru": item["texte_ru"],
    "source": item["source"],
    "page": item["page"],
    "type": "pdf"  # Pour distinguer la source dans actions.py
}
        ))
    return points

def indexer_dans_qdrant(client, nom_collection, points):
    """Envoie les points vers la base de données."""
    client.upsert(collection_name=nom_collection, points=points)

# --- 3. EXÉCUTION ---

if __name__ == "__main__":
    # 1. Vérifier si la collection existe au lieu de la recréer
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )

    # 2. Charger le JSON
    fichier_source = "corpus_pdf_traduit.json"
    donnees = charger_donnees_json(fichier_source)

    # 3. Préparer les points
    mes_points = generer_points_pdf(donnees, model)

    # 4. Envoyer à Qdrant
    indexer_dans_qdrant(client, collection_name, mes_points)

    print(f"✅ Terminé ! {len(mes_points)} blocs du PDF sont maintenant dans Qdrant.")