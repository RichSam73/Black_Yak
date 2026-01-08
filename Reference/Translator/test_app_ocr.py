# -*- coding: utf-8 -*-
"""
app.py의 OCR 함수를 직접 테스트
"""

import os
import sys
import io
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 테스트 설정
TEST_PDF = r"E:\Antigravity\Black_Yak\Reference\Translator\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"
OUTPUT_DIR = r"E:\Antigravity\Black_Yak\Reference\Translator\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pdf_to_images_app_style(pdf_path, zoom=2.0):
    """app.py와 동일한 방식으로 PDF 변환"""
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img_path = os.path.join(OUTPUT_DIR, f"app_page_{page_num + 1}.png")
        pix.save(img_path)
        images.append(img_path)
        print(f"[PDF] Page {page_num+1}: {pix.width}x{pix.height}")

    doc.close()
    return images


def get_ocr_results_app_style(image_path):
    """app.py와 동일한 OCR 함수"""
    ocr = PaddleOCR(use_textline_orientation=True, lang="korean")

    # ★ 핵심: BGR→RGB 변환
    img_bgr = cv2.imread(image_path)
    print(f"[OCR] Image shape (BGR): {img_bgr.shape}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = ocr.predict(img_rgb)

    texts = []
    if result:
        for item in result:
            rec_texts = []
            rec_scores = []
            dt_polys = []

            # OCRResult 객체 처리 (새 PaddleOCR API)
            if hasattr(item, 'rec_texts'):
                rec_texts = item.rec_texts or []
                rec_scores = item.rec_scores or []
                dt_polys = item.dt_polys if hasattr(item, 'dt_polys') and item.dt_polys is not None else []
                print(f"[OCR] rec_texts: {len(rec_texts)}, dt_polys: {len(dt_polys) if dt_polys is not None else 0}")

                # 첫 번째 좌표 출력
                if dt_polys is not None and len(dt_polys) > 0:
                    first = dt_polys[0]
                    if hasattr(first, 'tolist'):
                        first = first.tolist()
                    print(f"[OCR] First dt_poly: {first}")

            elif isinstance(item, dict):
                rec_texts = item.get('rec_text', item.get('rec_texts', []))
                rec_scores = item.get('rec_score', item.get('rec_scores', []))
                dt_polys = item.get('dt_polys', [])

            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
                rec_scores = [rec_scores]
                dt_polys = [dt_polys]

            for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                text_str = str(text)
                # 한글이 포함된 텍스트만 추출
                if any('\uac00' <= c <= '\ud7a3' for c in text_str):
                    bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
                    texts.append({
                        "bbox": bbox,
                        "text": text_str,
                        "confidence": float(score) if score else 1.0
                    })

    return texts


def test_ocr():
    """OCR 테스트"""
    print("=" * 60)
    print("app.py OCR 함수 테스트")
    print("=" * 60)

    # 1. PDF → 이미지 (app.py 방식)
    print("\n[1] PDF → 이미지 변환 (zoom=2.0)...")
    image_paths = pdf_to_images_app_style(TEST_PDF, zoom=2.0)
    img_path = image_paths[0]

    # 2. OCR 실행
    print("\n[2] OCR 실행...")
    texts = get_ocr_results_app_style(img_path)
    print(f"\n총 {len(texts)}개 한글 텍스트 감지")

    # 처음 5개 bbox 출력
    print("\n[3] 처음 5개 bbox 좌표:")
    for i, t in enumerate(texts[:5]):
        print(f"  {i+1}. '{t['text']}': {t['bbox']}")

    # 3. bbox 시각화
    print("\n[4] bbox 시각화...")
    img = cv2.imread(img_path)
    for t in texts:
        bbox = t["bbox"]
        pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    output_path = os.path.join(OUTPUT_DIR, "app_style_detected.png")
    cv2.imwrite(output_path, img)
    print(f"  저장: {output_path}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_ocr()
