from qdrant_client import QdrantClient
from qdrant_client.http import models

collection_name = "FAQ_Multilingue"

if __name__ == "__main__":
    # Connexion à votre instance Qdrant locale
    client = QdrantClient(host="localhost", port=6333)

    # Création de la collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024, # La dimension que nous avons trouvée
            distance=models.Distance.COSINE
        )
    )

    print(f"La collection '{collection_name}' est prête sur votre dashboard !")