# Plan d'implémentation V1 (4 semaines) — Rasa + RAG + LLM local (open source)

## Objectif global
Construire un assistant robuste qui suit ce flux:
1. **Rasa NLU** détecte l’intention et son niveau de confiance.
2. Si l’intention est sûre et couverte par des règles/réponses: réponse déterministe Rasa.
3. Sinon: interrogation **RAG** (Qdrant + embeddings).
4. Si la récupération est insuffisante: fallback configuré.
5. Option V1.1: génération finale contrôlée par un **LLM local** avec citations.

---

## Semaine 1 — Stabiliser l’architecture et le routage

### Objectifs
- Mettre en place un flux clair « intention -> RAG -> fallback ».
- Rendre les seuils configurables et observables.
- Eviter les comportements implicites non traçables.

### Tâches
- Ajouter des variables d’environnement:
  - `INTENT_HIGH_CONF` (ex: 0.75)
  - `INTENT_LOW_CONF` (ex: 0.45)
  - `RAG_MIN_SCORE` (ex: 0.55)
- Implémenter une logique de décision explicite dans l’action custom:
  - **Cas A**: intention haute confiance -> réponse métier/règle.
  - **Cas B**: confiance moyenne ou ambiguë -> interrogation RAG.
  - **Cas C**: faible confiance + score RAG faible -> fallback.
- Unifier les messages de fallback (`utter_default`) et aide à reformulation.
- Ajouter des logs structurés par requête:
  - intention détectée
  - confiance intention
  - top score RAG
  - route choisie

### Livrables
- Routage stable + seuils paramétrables.
- Logs exploitables pour calibration.

### Risques à surveiller
- Seuils trop stricts => trop de fallback.
- Seuils trop permissifs => réponses hors sujet.

---

## Semaine 2 — Qualité RAG (ingestion + retrieval)

### Objectifs
- Améliorer la pertinence des passages retournés.
- Réduire la perte de contexte liée au chunking.

### Tâches
- Standardiser l’ingestion documents:
  - extraction texte propre (PDF/OCR)
  - normalisation UTF-8, suppression bruit
  - langue détectée (`fr`, `ru`, etc.)
- Mettre un chunking glissant:
  - taille cible: 400–800 tokens
  - overlap: 10–20%
- Enrichir les métadonnées Qdrant:
  - `document_id`, `source`, `page`, `section`, `language`, `chunk_index`
- Ajouter des filtres de recherche:
  - langue utilisateur prioritaire
  - source/document si contexte connu
- Tester `top_k` (3/5/8) et comparer précision perçue.

### Livrables
- Pipeline d’ingestion reproductible.
- Retrieval plus stable sur questions paraphrasées.

### Risques à surveiller
- OCR bruité dégrade fortement les embeddings.
- Métadonnées incomplètes => filtrage inefficace.

---

## Semaine 3 — Intégrer un LLM local contrôlé (open source)

### Objectifs
- Ajouter une génération de réponse plus naturelle.
- Limiter les hallucinations via contrainte stricte de contexte.

### Tâches
- Choisir un runtime local open source:
  - **Ollama** (simple à démarrer) ou **vLLM** (perf plus avancée).
- Choisir un modèle adapté hardware:
  - 7B instruct quantisé pour machines modestes.
  - 8B/13B selon RAM/VRAM disponible.
- Prompt de sécurité (RAG-grounded):
  - "Réponds uniquement à partir des extraits fournis."
  - "Si info absente, dis-le explicitement."
  - "Cite les sources (document/page)."
- Ajouter un mode `NO_ANSWER` quand contexte insuffisant.
- Mettre timeout + gestion d’erreurs (LLM indisponible -> fallback propre).

### Livrables
- Réponses plus cohérentes en conversation.
- Diminution des hallucinations grâce à la contrainte de contexte.

### Risques à surveiller
- Latence trop élevée sur CPU.
- Variabilité des réponses sans contrôle du prompt.

---

## Semaine 4 — Évaluation, calibration et durcissement

### Objectifs
- Mesurer la qualité de façon fiable.
- Stabiliser avant passage en usage réel.

### Tâches
- Créer un petit jeu d’évaluation (100 questions):
  - 40 FAQ exactes
  - 30 paraphrases
  - 20 hors périmètre
  - 10 ambiguës/multilingues
- Définir des métriques simples:
  - Intention accuracy (top-1)
  - Taux de fallback utile
  - Recall@k retrieval
  - Taux de réponses avec source correcte
- Faire 2–3 itérations de seuils:
  - `INTENT_HIGH_CONF`, `INTENT_LOW_CONF`, `RAG_MIN_SCORE`
- Mettre des tests de non-régression (smoke tests) sur:
  - intents critiques
  - questions de référence
- Documenter un guide exploitation:
  - indexation doc
  - redémarrage services
  - check santé Qdrant/Action server/LLM

### Livrables
- Rapport court d’évaluation + seuils retenus.
- Checklist d’exploitation V1.

---

## Pile technologique 100% open source (recommandée)
- **Orchestration conversationnelle**: Rasa OSS
- **Vector DB**: Qdrant OSS
- **Embeddings**: sentence-transformers / modèles open source HF
- **LLM local**: Mistral/Llama/Qwen instruct via Ollama ou vLLM
- **API**: FastAPI (facultatif pour découplage embedding/LLM)
- **Observabilité**: logs structurés + dashboard minimal

---

## Gestion des difficultés techniques (pragmatique)

### 1) Ressources machine (RAM/VRAM)
- Commencer avec un modèle embedding compact.
- Utiliser un LLM quantisé (4-bit/8-bit) pour CPU/GPU limité.
- Déporter ingestion lourde hors du runtime Rasa (batch offline).

### 2) Calibration des seuils
- Ne pas fixer une fois pour toutes: calibrer à partir de logs réels.
- Travailler avec un set d’évaluation fixe pour comparer objectivement.
- Ajuster un seul seuil à la fois pour isoler l’effet.

### 3) Qualité des documents
- Prioriser extraction propre avant toute optimisation modèle.
- Nettoyer entêtes/pieds de page répétitifs.
- Conserver les pages et sections dans les métadonnées.

### 4) Évaluation fiable
- Mesurer systématiquement au lieu de juger “à l’impression”.
- Garder un historique des scores après chaque changement.
- Bloquer les régressions avec un test minimal automatisé.

---

## Critères de sortie V1
- L’utilisateur obtient une réponse pertinente ou un fallback clair.
- Les réponses générées citent la source quand disponible.
- Les cas non couverts sont explicitement reconnus.
- Les performances sont acceptables pour ton matériel cible.
