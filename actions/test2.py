import re


def charger_et_decouper_faq(chemin_du_fichier):
    with open(chemin_du_fichier, 'r', encoding='utf-8') as f:
        texte_complet = f.read()

    blocs = re.split(r'\n(?=Q-)', texte_complet)
    chunks_faq = [bloc.strip() for bloc in blocs if bloc.strip()]

    for i, bloc in enumerate(chunks_faq):
        print(f"--- Bloc {i+1} ---")
        print(bloc)

    return chunks_faq


def structurer_bloc(bloc):
    match_q = re.search(r'Q-"(.*?)"', bloc)
    match_r = re.search(r'R-(.*)', bloc, re.DOTALL)

    question = match_q.group(1) if match_q else ""
    reponse = match_r.group(1).strip() if match_r else ""

    return {
        "texte_complet": bloc,
        "question": question,
        "reponse": reponse
    }


if __name__ == "__main__":
    mon_contenu = charger_et_decouper_faq("FAQ.txt")
    faq_structuree = [structurer_bloc(bloc) for bloc in mon_contenu]

    if faq_structuree:
        print("Exemple de dictionnaire structuré :")
        print(faq_structuree[2])
