import re

# On utilise un nom de variable, par exemple 'chemin_du_fichier'
def charger_et_decouper_faq(chemin_du_fichier):
    # On utilise cette variable pour ouvrir le fichier
    with open(chemin_du_fichier, 'r', encoding='utf-8') as f:
        texte_complet = f.read()
    
    blocs = re.split(r'\n(?=Q-)', texte_complet)
    chunks_faq = [bloc.strip() for bloc in blocs if bloc.strip()]
    
    for i, bloc in enumerate(chunks_faq):
        print(f"--- Bloc {i+1} ---")
        print(bloc)
        
    return chunks_faq

# C'est ici, lors de l'appel, que l'on donne le nom du fichier
mon_contenu = charger_et_decouper_faq("FAQ.txt")

def structurer_bloc(bloc):
    # On cherche la partie après Q- et avant R-
    match_q = re.search(r'Q-"(.*?)"', bloc)
    # On cherche la partie après R-
    match_r = re.search(r'R-(.*)', bloc, re.DOTALL)
    
    question = match_q.group(1) if match_q else ""
    reponse = match_r.group(1).strip() if match_r else ""
    
    return {
        "texte_complet": bloc, # pour la vectorisation
        "question": question,  # Métadonnée
        "reponse": reponse     # Métadonnée
    }

# On transforme chaque texte brut en dictionnaire
faq_structuree = [structurer_bloc(bloc) for bloc in mon_contenu]

# Petit test pour vérifier le premier élément
if faq_structuree:
    print("Exemple de dictionnaire structuré :")
    print(faq_structuree[2])



    