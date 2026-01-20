import re

log_path = r"E:\Antigravity\Black_Yak\PDF_Translator\overlap_debug.log"

# Search for various translation-related patterns
search_patterns = [
    'SKIP',
    'English',
    'english',
    'render',
    'draw',
    'paste',
    'Original:',
    'translated_text',
]

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

print(f"Total size: {len(content)} bytes")

for pattern in search_patterns:
    count = content.count(pattern)
    print(f"'{pattern}': {count} occurrences")
    
# Also check if there's a separate app log
import os
app_log = r"E:\Antigravity\Black_Yak\PDF_Translator\app.log"
if os.path.exists(app_log):
    print(f"\n=== app.log exists, size: {os.path.getsize(app_log)} bytes ===")
else:
    print("\n=== app.log not found ===")
    
# Check debug_images folder
debug_folder = r"E:\Antigravity\Black_Yak\PDF_Translator\debug_images"
if os.path.exists(debug_folder):
    files = os.listdir(debug_folder)
    print(f"\n=== debug_images folder: {len(files)} files ===")
    for f in files[-10:]:
        print(f"  {f}")
