import json
import html

path = '/app/actions/chunks_tusur_final_fr.json'
with open(path) as f:
    data = json.load(f)

count = 0
for chunk in data['chunks']:
    if 'texte_fr' in chunk:
        fixed = html.unescape(chunk['texte_fr'])
        if fixed != chunk['texte_fr']:
            chunk['texte_fr'] = fixed
            count += 1

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f'Corrige {count} chunks')
print('Exemple:', data['chunks'][0]['texte_fr'][:200])
