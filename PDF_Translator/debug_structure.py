# PaddleOCR 새 API 결과 구조 확인
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
print(f"Image: {pix.width} x {pix.height}")

# OCR 초기화 및 실행
ocr_engine = PaddleOCR(lang="korean", device="gpu")
result = ocr_engine.predict(img_path)

# 결과 구조 출력
print(f"\nResult type: {type(result)}")
print(f"Result length: {len(result) if result else 0}")

if result:
    print(f"\nresult[0] type: {type(result[0])}")
    if hasattr(result[0], '__len__'):
        print(f"result[0] length: {len(result[0])}")
    
    # 첫 번째 항목 상세 출력
    if result[0]:
        first_item = result[0] if not hasattr(result[0], '__getitem__') else result[0][0] if len(result[0]) > 0 else result[0]
        print(f"\nFirst item type: {type(first_item)}")
        print(f"First item: {first_item}")
        
        # dict인 경우 키 출력
        if isinstance(first_item, dict):
            print(f"Keys: {first_item.keys()}")
        
        # 리스트인 경우 요소 출력
        if isinstance(first_item, list):
            for idx, elem in enumerate(first_item[:3]):
                print(f"  [{idx}] type={type(elem)}, value={elem}")

# 정리
import os
os.remove(img_path)
doc.close()
