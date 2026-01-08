from img2table.document import Image as Img2TableImage
from img2table.ocr import PaddleOCR
import os

# Create a dummy image or use existing one if possible
img_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"

if not os.path.exists(img_path):
    print("Image not found, skipping test")
    exit()

ocr = PaddleOCR(lang="korean")
doc = Img2TableImage(src=img_path)

# Extract tables
tables = doc.extract_tables(ocr=ocr, implicit_rows=True, borderless_tables=False, min_confidence=50)

if tables:
    print(f"Tables found: {len(tables)}")
    t = tables[0]
    # Inspect if we can access confidence
    # Usually img2table returns Table objects. 
    # Let's check if the dataframe or other properties have confidence.
    print("DF columns:", t.df.columns)
    print("First cell value:", t.df.iloc[0,0])
    
    # Check for private attributes or extra metadata
    if hasattr(t, 'df_confidence'): # Hypothetical
        print("Found df_confidence!")
    else:
        print("No df_confidence attribute found directly.")
        
    dir_t = dir(t)
    print("Table attributes:", [d for d in dir_t if not d.startswith('_')])
else:
    print("No tables found")
