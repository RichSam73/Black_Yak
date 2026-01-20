# 겹침 디버그 테스트 스크립트
# 사용법: python debug_overlap.py "경로/파일.pdf" 1

import sys
sys.path.insert(0, r"E:\Antigravity\Black_Yak\PDF_Translator")

# app.py의 extract_texts_single 함수 결과를 분석
from app import extract_texts_single, get_ocr_engine
import fitz

if len(sys.argv) < 3:
    print("Usage: python debug_overlap.py <pdf_path> <page_number>")
    sys.exit(1)

pdf_path = sys.argv[1]
page_num = int(sys.argv[2]) - 1  # 0-based

# PDF를 이미지로 변환
doc = fitz.open(pdf_path)
page = doc.load_page(page_num)
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img_path = "temp_debug_page.png"
pix.save(img_path)

# OCR 실행
print(f"\n{'='*60}")
print(f"[DEBUG] PDF: {pdf_path}, Page: {page_num + 1}")
print(f"{'='*60}\n")

texts = extract_texts_single(img_path)

# 한글 포함 텍스트 분석
korean_texts = [(i, t) for i, t in enumerate(texts) if t.get('has_korean', True)]
english_texts = [(i, t) for i, t in enumerate(texts) if not t.get('has_korean', True)]

print(f"Total texts: {len(texts)}")
print(f"Korean texts (to translate): {len(korean_texts)}")
print(f"English texts (skip render): {len(english_texts)}")

print(f"\n--- Korean texts (번역 대상) ---")
for i, t in korean_texts[:20]:
    bbox = t['bbox']
    x = min(p[0] for p in bbox)
    print(f"  [{i:3d}] x={x:4.0f} | '{t['text'][:50]}'")

print(f"\n--- English texts (렌더링 건너뜀) ---")
for i, t in english_texts[:20]:
    bbox = t['bbox']
    x = min(p[0] for p in bbox)
    print(f"  [{i:3d}] x={x:4.0f} | '{t['text'][:50]}'")

# 겹침 가능성 체크: 같은 Y 위치에 있는 텍스트들
print(f"\n--- 겹침 가능성 분석 (같은 Y 범위) ---")
for i, t1 in enumerate(texts):
    bbox1 = t1['bbox']
    y1_min = min(p[1] for p in bbox1)
    y1_max = max(p[1] for p in bbox1)
    
    for j, t2 in enumerate(texts):
        if i >= j:
            continue
        bbox2 = t2['bbox']
        y2_min = min(p[1] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)
        
        # Y 겹침 체크
        if not (y1_max < y2_min or y2_max < y1_min):
            x1 = min(p[0] for p in bbox1)
            x2 = min(p[0] for p in bbox2)
            if abs(x1 - x2) < 100:  # X 위치가 가까운 경우
                print(f"  [{i}] '{t1['text'][:30]}' ↔ [{j}] '{t2['text'][:30]}'")
                print(f"       Y1: {y1_min:.0f}~{y1_max:.0f}, Y2: {y2_min:.0f}~{y2_max:.0f}")

import os
os.remove(img_path)
doc.close()
print("\n[DONE]")
