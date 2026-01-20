# PaddleOCR OCRResult 구조 확인
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

# OCRResult 구조 출력
ocr_result = result[0]
print(f"Type: {type(ocr_result)}")
print(f"\nAttributes (non-private):")
for attr in dir(ocr_result):
    if not attr.startswith('_'):
        print(f"  {attr}")

# 주요 속성 확인
print(f"\n--- Checking common attributes ---")
if hasattr(ocr_result, 'rec_texts'):
    print(f"rec_texts: {ocr_result.rec_texts[:5]}")
if hasattr(ocr_result, 'rec_scores'):
    print(f"rec_scores: {ocr_result.rec_scores[:5]}")
if hasattr(ocr_result, 'dt_polys'):
    print(f"dt_polys (first): {ocr_result.dt_polys[0] if ocr_result.dt_polys else None}")
if hasattr(ocr_result, 'boxes'):
    print(f"boxes (first): {ocr_result.boxes[0] if hasattr(ocr_result.boxes, '__getitem__') else ocr_result.boxes}")
if hasattr(ocr_result, 'texts'):
    print(f"texts: {list(ocr_result.texts)[:5] if hasattr(ocr_result.texts, '__iter__') else ocr_result.texts}")

# 정리
import os
os.remove(img_path)
doc.close()
