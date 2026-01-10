import json

with open(r'E:\Antigravity\Black_Yak\PDF_Translator\garment_dict.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

empty_full = []
for lang, terms in d.items():
    for k, v in terms.items():
        if isinstance(v, dict) and not v.get('full'):
            empty_full.append((lang, k, v))

print(f"Found {len(empty_full)} entries with empty 'full' value:")
for item in empty_full[:30]:
    print(item)
