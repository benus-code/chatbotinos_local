# Tester l'action server avec Docker

## 1) Build de l'image
Depuis la racine du projet:

```bash
docker build --no-cache -t chatbotinos-action-server:local .
```

## 2) Lancer le conteneur action server

```bash
docker run --rm -it \
  --name action_server_test \
  -p 5055:5055 \
  -e INTENT_HIGH_CONF=0.80 \
  -e INTENT_LOW_CONF=0.45 \
  -e MIN_RELEVANCE_SCORE=0.50 \
  chatbotinos-action-server:local
```

## 3) Vérifier que le serveur répond
Dans un autre terminal:

```bash
curl http://localhost:5055/health
```

Réponse attendue: un JSON indiquant que le serveur d'actions est `ok`.

## 4) Si vous utilisez docker compose
Si votre `docker-compose.yml` contient un service `action_server`:

```bash
docker compose build --no-cache action_server
docker compose up -d action_server
docker compose logs -f action_server
```

## 5) Dépannage rapide
- Si `docker logs chatbotinos_local-action_server` renvoie *No such container*, le conteneur n'a pas été créé (échec de build ou nom différent).
- Listez les conteneurs:

```bash
docker ps -a
```

- Vérifiez le nom exact puis relancez les logs:

```bash
docker logs -f <nom_du_conteneur>
```

### Important pour votre erreur actuelle
Si `docker compose build action_server` continue d'exécuter la ligne:
`pip install --no-cache-dir qdrant-client sentence-transformers requests pypdf langchain`
alors Compose n'utilise pas le bon Dockerfile.

Exemple attendu dans `docker-compose.yml`:

```yaml
services:
  action_server:
    build:
      context: ./actions
      dockerfile: Dockerfile
    ports:
      - "5055:5055"
```

Avec cette config, le build utilisera `actions/Dockerfile` (dépendances épinglées + torch CPU).

