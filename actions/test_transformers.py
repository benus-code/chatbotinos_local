from sentence_transformers import SentenceTransformer
import torch

# Le nom exact du modèle sur Hugging Face
model_id = "Qwen/Qwen3-Embedding-0.6B"

print("--- Nettoyage et préparation ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du matériel : {device.upper()}")

print(f"Chargement de {model_id} (environ 1.2 GB)...")

# trust_remote_code est requis pour l'architecture Qwen3
model = SentenceTransformer(model_id, trust_remote_code=True, device=device)

print("\n✅ SUCCÈS : Le modèle Qwen3 est chargé sur l'ordinateur d'IVANA !")

# Petit test rapide
sentences = ["Bonjour tout le monde", "Привет мир", "Hello world"]
embeddings = model.encode(sentences)
print(f"Test réussi : {len(sentences)} phrases transformées en vecteurs.")
# Ajoute cette ligne à la fin de ton script de test
print(f"La dimension de vos vecteurs est : {embeddings.shape[1]}")