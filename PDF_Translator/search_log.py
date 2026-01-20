import re

log_path = r"E:\Antigravity\Black_Yak\PDF_Translator\overlap_debug.log"

# Search for Korean text patterns
search_terms = ['콘실', '지퍼', 'Concealed', 'Zipper']

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

for term in search_terms:
    matches = [(i, line.strip()) for i, line in enumerate(lines) if term in line]
    print(f"\n=== '{term}' found {len(matches)} times ===")
    for idx, line in matches[-10:]:  # Show last 10 matches
        print(f"Line {idx}: {line[:150]}...")
