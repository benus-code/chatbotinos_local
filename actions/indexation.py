from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
import uuid
from qdrant_client.http.models import PointStruct
import re

# --- 1. CONFIGURATION ET MODÈLES (Accessibles par l'import) ---
client = QdrantClient(host="localhost", port=6333, timeout=60)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
collection_name = "FAQ_Multilingue"

# --- 2. DÉFINITIONS DES FONCTIONS DE TRAITEMENT ---

def charger_et_decouper_faq(chemin_du_fichier):
    with open(chemin_du_fichier, 'r', encoding='utf-8') as f:
        texte_a_vectoriser = f.read()
    blocs = re.split(r'\n(?=Q-)', texte_a_vectoriser)
    return [bloc.strip() for bloc in blocs if bloc.strip()]

def structurer_bloc(bloc):
    match_q = re.search(r'Q-"(.*?)"', bloc)
    match_r = re.search(r'R-(.*)', bloc, re.DOTALL)
    return {
        "texte_a_vectoriser": bloc,
        "question": match_q.group(1) if match_q else "",
        "reponse": match_r.group(1).strip() if match_r else ""
    }


def generer_points_avec_uuid(liste_faq, modele_embedding, nom_du_fichier_source):
    points = []
    for item in liste_faq:
        id_unique = str(uuid.uuid4())
         # On combine tout pour que l'IA "comprenne" le bloc entier
        texte_a_vectoriser = f"Question: {item['question']} Réponse: {item['reponse']}"
        vecteur = modele_embedding.encode(texte_a_vectoriser).tolist()
        points.append(PointStruct(
            id=id_unique,
            vector=vecteur,
            payload={
    "content": f"Question: {item['question']}\nRéponse: {item['reponse']}",
    "source_file": nom_du_fichier_source,
    "type": "faq" # Petite astuce pour que le bot sache d'où ça vient !
}
        ))
    return points

def nettoyer_et_remplacer(client, nom_collection, nom_fichier, nouveaux_points):
    client.delete(
        collection_name=nom_collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=nom_fichier))]
        )
    )
    client.upsert(collection_name=nom_collection, points=nouveaux_points)

def chercher_faq(client, nom_collection, modele, question_utilisateur, limite=3):
    vecteur_question = modele.encode(question_utilisateur).tolist()
    resultat = client.query_points(
        collection_name=nom_collection,
        query=vecteur_question,
        limit=limite
    )
    return resultat.points

# --- 3. BLOC D'EXÉCUTION (Uniquement quand on lance ce fichier directement) ---

if __name__ == "__main__":
    # 1. Vérifier si la collection existe au lieu de la recréer
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )

    # Chargement et structuration
    nom_du_fichier_source = "FAQ.txt"
    mon_contenu = charger_et_decouper_faq(nom_du_fichier_source)
    faq_structuree = [structurer_bloc(bloc) for bloc in mon_contenu]

    # Génération et envoi des points
    mes_points_test = generer_points_avec_uuid(faq_structuree, model, nom_du_fichier_source)
    
    nettoyer_et_remplacer(
        client=client, 
        nom_collection=collection_name, 
        nom_fichier=nom_du_fichier_source, 
        nouveaux_points=mes_points_test
    )

    print(f"✅ Données indexées avec succès dans '{collection_name}' !")

    # Test rapide de recherche
    ma_question = "Je veux savoir comment avoir l'invitation"
    content = chercher_faq(client, collection_name, model, ma_question)

    print(f"\n🔍 Test de recherche pour : '{ma_question}'")
    for i, res in enumerate(content):
        print(f"Rang {i+1} (Score: {res.score:.4f}): {res.payload['content']}")