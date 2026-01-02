# -*- coding: utf-8 -*-
"""
img2table PoC Test
- colspan/rowspan HTML output check
- PaddleOCR integration test
"""

import os
import sys

print("=" * 60)
print("img2table PoC Test")
print("=" * 60)

# 1. img2table import
print("\n1. img2table import...")
from img2table.document import Image
from img2table.ocr import PaddleOCR
print("   [OK] import done")

# 2. PaddleOCR engine init
print("\n2. PaddleOCR engine init...")
ocr = PaddleOCR(lang="en")
print("   [OK] OCR engine ready")

# 3. Test images
test_images = [
    r"e:\Antigravity\Black_Yak\Reference\BY_Original_Table.png",
    r"e:\Antigravity\Black_Yak\Reference\005M_Table.png",
    r"e:\Antigravity\Black_Yak\Reference\Submaterial_information.png"
]

# 4. Test each image
for img_path in test_images:
    print("\n" + "=" * 60)
    print(f"Testing: {os.path.basename(img_path)}")
    print("=" * 60)

    if not os.path.exists(img_path):
        print(f"   [ERROR] File not found: {img_path}")
        continue

    try:
        # Load image
        doc = Image(src=img_path)

        # Extract tables
        print("\n3. Extracting tables...")
        tables = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,       # implicit row detection
            borderless_tables=False,  # bordered tables
            min_confidence=50         # min confidence
        )

        print(f"   [OK] Extracted tables: {len(tables)}")

        # Analyze each table
        for idx, table in enumerate(tables):
            print(f"\n--- Table {idx + 1} ---")

            # HTML output
            html = table.html
            print(f"\n[HTML length]: {len(html)} chars")

            # colspan/rowspan check
            has_colspan = 'colspan' in html
            has_rowspan = 'rowspan' in html
            print(f"[has colspan]: {'YES' if has_colspan else 'NO'}")
            print(f"[has rowspan]: {'YES' if has_rowspan else 'NO'}")

            # HTML preview (first 1500 chars)
            print(f"\n[HTML preview]:")
            print(html[:1500])
            if len(html) > 1500:
                print("...")

            # DataFrame
            df = table.df
            print(f"\n[DataFrame shape]: {df.shape}")
            print(f"[Columns]: {len(df.columns)}")
            print(f"[Rows]: {len(df)}")

            # DataFrame preview
            print(f"\n[DataFrame preview]:")
            print(df.head(10).to_string())

            # Save HTML file
            output_name = os.path.basename(img_path).replace('.png', f'_img2table_{idx+1}.html')
            output_path = os.path.join(r"e:\Antigravity\Black_Yak\Reference\CometApp", output_name)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{os.path.basename(img_path)} - Table {idx+1}</title>
    <style>
        table {{ border-collapse: collapse; margin: 20px; }}
        td, th {{ border: 1px solid black; padding: 8px; text-align: center; }}
        th {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <h2>{os.path.basename(img_path)} - Table {idx+1}</h2>
    <p>has colspan: {has_colspan} | has rowspan: {has_rowspan}</p>
    {html}
</body>
</html>""")
            print(f"\n[Saved]: {output_path}")

    except Exception as e:
        print(f"   [ERROR]: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
