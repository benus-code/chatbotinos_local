# chatbotinos_local — Guide d'utilisation

Chatbot universitaire basé sur Rasa + RAG (Qdrant) avec support multilingue français/russe.

---

## Prérequis

- Git
- Docker et Docker Compose (v2+)
- Python 3.10+ (uniquement pour lancer les tests en local sans Docker)

---

## 1. Cloner le projet (nouvelle machine)

```bash
git clone <url-du-repo>
cd chatbotinos_local
```

---

## 2. Travailler sur une branche spécifique

```bash
# Voir toutes les branches disponibles (locales + distantes)
git branch -a

# Se mettre sur une branche spécifique
git checkout claude/add-level-one-tests-ppwl8

# Récupérer les dernières mises à jour de cette branche
git pull origin claude/add-level-one-tests-ppwl8
```

> Si la branche n'existe pas encore en local :
> ```bash
> git fetch origin
> git checkout -b claude/add-level-one-tests-ppwl8 origin/claude/add-level-one-tests-ppwl8
> ```

---

## 3. Après un `git pull` — que faire avec Docker ?

### Cas A — Nouvelle machine (premier lancement)

```bash
docker compose build --no-cache
docker compose up -d
```

### Cas B — Tu as reçu des modifications dans `actions/` ou les `Dockerfile`

Le cas le plus fréquent. Il faut rebuild l'action server :

```bash
docker compose build action_server
docker compose up -d action_server
```

Si `docker-compose.yml` lui-même a changé, fais un restart complet :

```bash
docker compose down
docker compose build
docker compose up -d
```

### Cas C — Modifications uniquement dans les fichiers Rasa (`data/`, `config.yml`, `domain.yml`)

Pas besoin de rebuild. Un simple restart suffit :

```bash
docker compose restart rasa
```

---

## 4. Vérifier que tout fonctionne

```bash
# Status de tous les services (qdrant, action_server, rasa)
docker compose ps

# Logs en direct (Ctrl+C pour quitter)
docker compose logs -f action_server
docker compose logs -f rasa

# Health checks
curl http://localhost:5055/health   # action server
curl http://localhost:5005/         # rasa
```

Réponse attendue de l'action server : un JSON `{"status": "ok"}`.

---

## 5. Lancer les tests unitaires (sans Docker)

```bash
# Depuis la racine du projet
python -m pytest actions/tests/ -v
```

Les tests n'ont pas besoin de Docker ni de GPU — tout est mocké (Qdrant, SentenceTransformer).

Pour un seul fichier :

```bash
python -m pytest actions/tests/test_faq_indexation.py -v
python -m pytest actions/tests/test_pdf_chunker.py -v
```

---

## 6. Arrêter les services

```bash
docker compose down
```

Pour tout supprimer (volumes inclus) :

```bash
docker compose down -v
```

---

## Architecture des services

| Service        | Port  | Description                        |
|----------------|-------|------------------------------------|
| `qdrant`       | 6333  | Base vectorielle (embeddings)      |
| `action_server`| 5055  | Serveur d'actions Rasa custom      |
| `rasa`         | 5005  | API principale Rasa                |

---

## Variables d'environnement utiles

| Variable             | Défaut | Description                              |
|----------------------|--------|------------------------------------------|
| `INTENT_HIGH_CONF`   | 0.80   | Seuil confiance haute (intent)           |
| `INTENT_LOW_CONF`    | 0.45   | Seuil confiance basse (intent)           |
| `MIN_RELEVANCE_SCORE`| 0.50   | Score minimum de pertinence RAG          |
| `RAG_MIN_SCORE`      | 0.55   | Score minimum Qdrant                     |
| `RAG_COLLECTION`     | documents | Nom de la collection Qdrant           |

Ces variables peuvent être surchargées dans `docker-compose.yml` ou via `-e` dans `docker run`.

---

## Voir aussi

- [`docs/docker-test.md`](docs/docker-test.md) — tester l'action server en isolation
- [`docs/plan_implementation_v1_rasa_rag.md`](docs/plan_implementation_v1_rasa_rag.md) — plan d'implémentation V1
