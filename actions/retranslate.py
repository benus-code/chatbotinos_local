"""
Retraduit les chunks dont texte_fr contient une erreur de traduction,
en utilisant MarianMT local (pas de connexion internet requise).
"""
import html
import json

from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-ru-fr"
PATH = "/app/actions/chunks_tusur_final_fr.json"
MAX_TOKENS = 490


def translate(text: str, tokenizer, model) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
    translated = model.generate(**inputs)
    return html.unescape(tokenizer.decode(translated[0], skip_special_tokens=True))


def main():
    print("Chargement du modèle MarianMT...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data["chunks"])
    fixed = 0
    errors = 0

    for i, chunk in enumerate(data["chunks"]):
        texte_fr = chunk.get("texte_fr", "")
        if "[Erreur traduction" in texte_fr or not texte_fr.strip():
            texte_ru = chunk.get("texte", "")
            if not texte_ru.strip():
                continue
            try:
                chunk["texte_fr"] = translate(texte_ru, tokenizer, model)
                fixed += 1
                if fixed % 10 == 0:
                    print(f"  {fixed} chunks retraduits ({i+1}/{total})...")
            except Exception as e:
                errors += 1
                print(f"  Erreur chunk {i}: {e}")

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nTerminé : {fixed} retraduits, {errors} erreurs sur {total} total")
    print("Exemple:", data["chunks"][0]["texte_fr"][:200])


if __name__ == "__main__":
    main()
