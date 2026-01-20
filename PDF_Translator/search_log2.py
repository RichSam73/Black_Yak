import re

log_path = r"E:\Antigravity\Black_Yak\PDF_Translator\overlap_debug.log"

# Search for overlap detection results and abbreviation patterns
search_patterns = [
    'needs_abbreviation',
    'y_overlap=True',
    'ABBREVIATE',
    'TRANSLATED:',
    'SKIP english',
]

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Get last 200 lines for recent activity
recent_lines = lines[-300:]
print("\n=== Last 300 lines summary ===")

for pattern in search_patterns:
    matches = [(i, line.strip()) for i, line in enumerate(recent_lines) if pattern in line]
    print(f"\n--- '{pattern}' found {len(matches)} times ---")
    for idx, line in matches[:5]:  # Show first 5 matches
        print(f"  {line[:200]}")
