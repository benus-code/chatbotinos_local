import fitz
import re
import json
import time
from deep_translator import GoogleTranslator

# --- FONCTIONS DE BASE ---

def extraire_texte_propre(doc):
    """Extrait tout le texte d'une page, trié du haut vers le bas."""
    texte_complet = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        # Tri spatial indispensable pour garder l'ordre logique
        blocks.sort(key=lambda b: b[1]) 
        
        for b in blocks:
            texte = b[4].strip()
            if texte:
                texte_complet.append(texte)
    return "\n".join(texte_complet)

def parser_le_document_ameliore(texte_brut):
    """Découpe le texte en articles."""
    pattern_section = re.compile(r'^(\d+)\.\s+(.+)$', re.MULTILINE)
    pattern_article = re.compile(r'^(\d+\.\d+)\.\s+(.+)$', re.MULTILINE)
    pattern_doc_type = re.compile(r'^(ПРИКАЗ|ПОЛОЖЕНИЕ)', re.MULTILINE)
    
    lignes = texte_brut.split('\n')
    chunks = []
    current_article = None
    current_section = None
    document_info = {"type": "unknown", "title": ""}
    
    # Détection en-tête
    for ligne in lignes[:20]:
        doc_match = pattern_doc_type.search(ligne)
        if doc_match:
            document_info["type"] = doc_match.group(1)
            document_info["title"] = ligne.strip()
            break
    
    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne: continue
        
        sec_match = pattern_section.match(ligne)
        if sec_match and not pattern_article.match(ligne):
            current_section = {"num": sec_match.group(1), "titre": sec_match.group(2)}
            continue
            
        art_match = pattern_article.match(ligne)
        if art_match:
            if current_article:
                current_article["texte"] = re.sub(r'\s+', ' ', current_article["texte"]).strip()
                chunks.append(current_article)
            
            current_article = {
                "metadata": {
                    "article_num": art_match.group(1),
                    "article_titre": art_match.group(2),
                    "section_num": current_section["num"] if current_section else "",
                    "section_titre": current_section["titre"] if current_section else "",
                    "document_type": document_info["type"]
                },
                "texte": ligne
            }
        elif current_article:
            current_article["texte"] += " " + ligne
            
    if current_article:
        current_article["texte"] = re.sub(r'\s+', ' ', current_article["texte"]).strip()
        chunks.append(current_article)
    
    return chunks, document_info

def detecter_type_article(texte, titre):
    texte_lower = texte.lower()
    titre_lower = titre.lower()
    if any(word in titre_lower for word in ["право", "имеют право"]): return "rights"
    elif any(word in titre_lower for word in ["обязан"]): return "obligations"
    elif any(word in titre_lower for word in ["запреща"]): return "prohibitions"
    return "general"

# --- TRADUCTION ---

def traduire_chunk(text, translator):
    """Traduit un texte avec gestion d'erreur."""
    try:
        # La pause est importante pour ne pas être banni par Google
        time.sleep(0.5) 
        return translator.translate(text)
    except Exception as e:
        return f"[Erreur traduction: {e}]"

# --- UTILISATION PRINCIPALE ---

if __name__ == "__main__":
    fichier_pdf = "Polozhenie_o_studencheskom_obschezhitii.pdf"
    
    print(f"📄 Traitement de {fichier_pdf}...")
    doc = fitz.open(fichier_pdf)
    texte_brut = extraire_texte_propre(doc)
    resultats, doc_info = parser_le_document_ameliore(texte_brut)
    
    # CORRECTION ICI : Instanciation correcte
    translator = GoogleTranslator(source='ru', target='fr')
    
    print(f"🌍 Traduction en cours ({len(resultats)} articles)...")
    
    for article in resultats:
        # 1. Classification
        article["metadata"]["type"] = detecter_type_article(article["texte"], article["metadata"]["article_titre"])
        
        # 2. Traduction
        print(f" -> Traduction article {article['metadata']['article_num']}...")
        article["texte_fr"] = traduire_chunk(article["texte"], translator)
    
    # Sauvegarde
    output = {"document_info": doc_info, "total_articles": len(resultats), "chunks": resultats}
    with open("chunks_tusur_final_fr.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Terminé ! Sauvegardé dans chunks_tusur_final_fr.json")