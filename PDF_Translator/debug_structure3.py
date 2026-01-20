# PaddleOCR OCRResult 키와 값 확인
import sys
try:
    import fitz
except ImportError:
    import pymupdf as fitz
from paddleocr import PaddleOCR

pdf_path = r"E:\Antigravity\Black_Yak\Reference\Translator\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"

# PDF를 이미지로 변환
doc = fitz.open(pdf_path)
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img_path = "temp_debug_page.png"
pix.save(img_path)

# OCR 초기화 및 실행
ocr_engine = PaddleOCR(lang="korean", device="gpu")
result = ocr_engine.predict(img_path)

# OCRResult 키와 값 확인
ocr_result = result[0]
print(f"Keys: {list(ocr_result.keys())}")

for key in ocr_result.keys():
    value = ocr_result[key]
    print(f"\n--- {key} ---")
    print(f"Type: {type(value)}")
    if hasattr(value, '__len__'):
        print(f"Length: {len(value)}")
    if isinstance(value, list) and len(value) > 0:
        print(f"First item type: {type(value[0])}")
        print(f"First item: {value[0]}")

# 정리
import os
os.remove(img_path)
doc.close()
