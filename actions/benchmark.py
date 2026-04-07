"""Benchmark script — mesure la précision du RAG à chaque niveau d'amélioration.

Usage (dans le container) :
    python benchmark.py --level 0          # Baseline avant corrections
    python benchmark.py --level 1          # Après corrections Niveau 1
    python benchmark.py --level 2          # Après NLLB-200
    python benchmark.py --level 3          # Après LLM Ollama
    python benchmark.py --output rapport.json

Chaque run produit un JSON horodaté dans benchmark_results/ pour comparaison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from indexation import client, collection_name, model, search_by_type

logging.basicConfig(level=logging.WARNING)

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

# ---------------------------------------------------------------------------
# Jeu de test fixe — NE PAS MODIFIER entre les niveaux pour comparaison valide
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    # --- Questions FAQ (source attendue : FAQ.txt) ---
    {
        "id": "faq_01",
        "question": "Comment obtenir un visa étudiant pour la Russie ?",
        "source_attendue": "FAQ.txt",
        "mots_cles_reponse": ["visa", "département", "international"],
    },
    {
        "id": "faq_02",
        "question": "Combien de temps faut-il pour obtenir le visa ?",
        "source_attendue": "FAQ.txt",
        "mots_cles_reponse": ["30", "40", "jours", "invitation"],
    },
    {
        "id": "faq_03",
        "question": "Quels documents faut-il pour s'inscrire à l'université ?",
        "source_attendue": "FAQ.txt",
        "mots_cles_reponse": ["passeport", "baccalauréat", "diplôme"],
    },
    {
        "id": "faq_04",
        "question": "Comment obtenir l'invitation de l'université ?",
        "source_attendue": "FAQ.txt",
        "mots_cles_reponse": ["10%", "frais", "scolarité"],
    },
    {
        "id": "faq_05",
        "question": "Puis-je travailler avec un visa étudiant ?",
        "source_attendue": "FAQ.txt",
        "mots_cles_reponse": [],  # réponse vide dans la FAQ — cas limite
    },
    # --- Questions PDF résidence (source attendue : PDF) ---
    {
        "id": "pdf_01",
        "question": "Quelles sont les règles à respecter en résidence universitaire ?",
        "source_attendue": "Polozhenie_o_studencheskom_obschezhitii.pdf",
        "mots_cles_reponse": ["règles", "résidence", "logement"],
    },
    {
        "id": "pdf_02",
        "question": "Combien coûte la résidence universitaire ?",
        "source_attendue": "Polozhenie_o_studencheskom_obschezhitii.pdf",
        "mots_cles_reponse": ["frais", "hébergement", "recteur"],
    },
    {
        "id": "pdf_03",
        "question": "Comment payer les frais de résidence ?",
        "source_attendue": "Polozhenie_o_studencheskom_obschezhitii.pdf",
        "mots_cles_reponse": ["paiement", "bourse", "caisse"],
    },
    {
        "id": "pdf_04",
        "question": "Quels sont les droits des étudiants en résidence ?",
        "source_attendue": "Polozhenie_o_studencheskom_obschezhitii.pdf",
        "mots_cles_reponse": ["droit", "chambre", "résider"],
    },
    {
        "id": "pdf_05",
        "question": "Est-ce qu'on peut recevoir des visiteurs en résidence ?",
        "source_attendue": "Polozhenie_o_studencheskom_obschezhitii.pdf",
        "mots_cles_reponse": ["visiteur", "invité", "autorisation"],
    },
    # --- Questions hors-sujet (aucune réponse attendue) ---
    {
        "id": "oos_01",
        "question": "Quel est le meilleur restaurant de Tomsk ?",
        "source_attendue": None,
        "mots_cles_reponse": [],
    },
    {
        "id": "oos_02",
        "question": "Comment faire une pizza ?",
        "source_attendue": None,
        "mots_cles_reponse": [],
    },
]


# ---------------------------------------------------------------------------
# Évaluation d'une question
# ---------------------------------------------------------------------------

def _source_from_payload(payload: Dict[str, Any]) -> str:
    """Retourne le nom de la source depuis le payload Qdrant."""
    return payload.get("source_file") or payload.get("source") or "inconnu"


def _check_mots_cles(content: str, mots_cles: List[str]) -> float:
    """Ratio de mots-clés attendus trouvés dans le contenu (0.0 → 1.0)."""
    if not mots_cles:
        return 1.0  # pas de mots-clés définis = non évalué
    content_lower = content.lower()
    trouvés = sum(1 for m in mots_cles if m.lower() in content_lower)
    return round(trouvés / len(mots_cles), 2)


def _is_useful(r: Any) -> bool:
    """Même filtre que actions.py — élimine les FAQ avec réponse vide."""
    content = r.payload.get("content", "").strip()
    if len(content) < 30:
        return False
    if r.payload.get("type") == "faq":
        answer = content.split("Réponse:", 1)[-1].strip() if "Réponse:" in content else ""
        return len(answer) > 5 and not answer.startswith("??")
    return True


def evaluer_question(q: Dict) -> Dict:
    """Interroge Qdrant et évalue la qualité de la réponse.
    Même logique que actions.py : recherche séparée FAQ et PDF, puis sélection du meilleur.
    Та же логика, что в actions.py: раздельный поиск FAQ и PDF, затем выбор лучшего.
    """
    # Recherche uniquement dans le PDF — même comportement que actions.py en mode test.
    # Поиск только по PDF — то же поведение, что и actions.py в тестовом режиме.
    pdf_results = search_by_type(client, collection_name, model, q["question"], "pdf", limit=5)

    best_pdf = next((r for r in pdf_results if r.score >= RAG_MIN_SCORE and _is_useful(r)), None)

    if not best_pdf:
        return {
            "id": q["id"],
            "question": q["question"],
            "repondu": False,
            "score_top": round(pdf_results[0].score, 4) if pdf_results else 0.0,
            "source_retournee": None,
            "source_correcte": q["source_attendue"] is None,
            "longueur_reponse": 0,
            "mots_cles_ratio": 0.0,
            "contenu_tronque": "",
        }

    top = best_pdf

    score = round(top.score, 4)
    source = _source_from_payload(top.payload)
    content = top.payload.get("content", "")
    repondu = True

    source_correcte: bool
    if q["source_attendue"] is None:
        # Question hors-sujet : bonne réponse = ne rien retourner (score < seuil)
        source_correcte = not repondu
    else:
        source_correcte = q["source_attendue"] in source and repondu

    return {
        "id": q["id"],
        "question": q["question"],
        "repondu": repondu,
        "score_top": score,
        "source_retournee": source,
        "source_correcte": source_correcte,
        "longueur_reponse": len(content),
        "mots_cles_ratio": _check_mots_cles(content, q["mots_cles_reponse"]) if repondu else 0.0,
        "contenu_tronque": content[:150] + ("…" if len(content) > 150 else ""),
    }


# ---------------------------------------------------------------------------
# Calcul des métriques agrégées
# ---------------------------------------------------------------------------

def calculer_metriques(resultats: List[Dict]) -> Dict:
    """Calcule précision, rappel, F1 et statistiques générales."""
    total = len(resultats)
    repondus = [r for r in resultats if r["repondu"]]
    sources_correctes = [r for r in resultats if r["source_correcte"]]

    # Precision = parmi les réponses données, combien sont correctes
    precision = round(
        len([r for r in repondus if r["source_correcte"]]) / len(repondus), 3
    ) if repondus else 0.0

    # Rappel = parmi les questions attendant une réponse, combien ont été répondues correctement
    questions_avec_reponse = [q for q in TEST_QUESTIONS if q["source_attendue"] is not None]
    rappel = round(
        len([r for r in resultats if r["source_correcte"] and r["repondu"]
             and any(q["id"] == r["id"] for q in questions_avec_reponse)]) /
        len(questions_avec_reponse), 3
    ) if questions_avec_reponse else 0.0

    f1 = round(
        2 * precision * rappel / (precision + rappel), 3
    ) if (precision + rappel) > 0 else 0.0

    longueurs = [r["longueur_reponse"] for r in repondus]
    mots_cles_ratios = [r["mots_cles_ratio"] for r in repondus if r["mots_cles_ratio"] > 0]

    return {
        "total_questions": total,
        "questions_repondues": len(repondus),
        "sources_correctes": len(sources_correctes),
        "precision": precision,
        "rappel": rappel,
        "f1_score": f1,
        "longueur_moy_reponse": round(sum(longueurs) / len(longueurs)) if longueurs else 0,
        "longueur_max_reponse": max(longueurs) if longueurs else 0,
        "mots_cles_ratio_moy": round(sum(mots_cles_ratios) / len(mots_cles_ratios), 2) if mots_cles_ratios else 0.0,
        "score_moy_top": round(sum(r["score_top"] for r in resultats) / total, 4),
    }


# ---------------------------------------------------------------------------
# Génération du rapport
# ---------------------------------------------------------------------------

def generer_rapport(level: int, resultats: List[Dict], metriques: Dict) -> Dict:
    niveaux = {
        0: "Baseline (avant corrections)",
        1: "Niveau 1 — Troncature + Seuil RAG + Split articles",
        2: "Niveau 2 — Traduction NLLB-200",
        3: "Niveau 3 — Génération LLM (Ollama)",
        4: "Niveau 4 — Hybrid search BM25+vecteur",
    }
    return {
        "niveau": level,
        "description": niveaux.get(level, f"Niveau {level}"),
        "date": datetime.now().isoformat(),
        "rag_min_score": RAG_MIN_SCORE,
        "metriques": metriques,
        "details": resultats,
    }


def afficher_rapport(rapport: Dict) -> None:
    m = rapport["metriques"]
    print("\n" + "=" * 60)
    print(f"  BENCHMARK RAG — {rapport['description']}")
    print(f"  Date : {rapport['date'][:19]}")
    print(f"  Seuil RAG_MIN_SCORE : {rapport['rag_min_score']}")
    print("=" * 60)
    print(f"  Questions testées    : {m['total_questions']}")
    print(f"  Réponses données     : {m['questions_repondues']}")
    print(f"  Sources correctes    : {m['sources_correctes']}")
    print(f"  Précision            : {m['precision']:.1%}")
    print(f"  Rappel               : {m['rappel']:.1%}")
    print(f"  F1-score             : {m['f1_score']:.1%}")
    print(f"  Score moyen top-1    : {m['score_moy_top']:.4f}")
    print(f"  Longueur moy réponse : {m['longueur_moy_reponse']} chars")
    print(f"  Longueur max réponse : {m['longueur_max_reponse']} chars")
    print(f"  Mots-clés trouvés    : {m['mots_cles_ratio_moy']:.0%}")
    print("-" * 60)
    print("  Détail par question :")
    for r in rapport["details"]:
        statut = "✓" if r["source_correcte"] else "✗"
        print(f"  [{statut}] {r['id']:8s} score={r['score_top']:.3f} | {r['question'][:45]}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG chatbot TUSUR")
    parser.add_argument("--level", type=int, default=0, help="Numéro du niveau (0=baseline)")
    parser.add_argument("--output", type=str, default=None, help="Fichier JSON de sortie")
    args = parser.parse_args()

    print(f"Évaluation de {len(TEST_QUESTIONS)} questions...")
    resultats = [evaluer_question(q) for q in TEST_QUESTIONS]
    metriques = calculer_metriques(resultats)
    rapport = generer_rapport(args.level, resultats, metriques)

    afficher_rapport(rapport)

    # Sauvegarde automatique dans benchmark_results/
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"niveau_{args.level}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    print(f"Rapport sauvegardé → {filename}")

    # Sauvegarde optionnelle à l'emplacement spécifié
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)
        print(f"Rapport sauvegardé → {args.output}")


if __name__ == "__main__":
    main()
