# 겹침 디버그 테스트 스크립트 (PaddleOCR 새 API)
import sys
try:
    import fitz
except ImportError:
    import pymupdf as fitz
import numpy as np
from paddleocr import PaddleOCR

pdf_path = r"E:\Antigravity\Black_Yak\Reference\Translator\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"
page_num = 3  # 0-based (4페이지)

print(f"\n{'='*60}")
print(f"[DEBUG] PDF: {pdf_path}")
print(f"[DEBUG] Page: {page_num + 1}")
print(f"{'='*60}\n")

# PDF를 이미지로 변환
print("[1/3] PDF를 이미지로 변환 중...")
doc = fitz.open(pdf_path)
page = doc.load_page(page_num)
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img_path = "temp_debug_page.png"
pix.save(img_path)
print(f"  → 이미지 크기: {pix.width} x {pix.height}")

# OCR 초기화
print("\n[2/3] PaddleOCR 초기화 중...")
ocr_engine = PaddleOCR(lang="korean", device="gpu")

# OCR 실행
print("\n[3/3] OCR 실행 중...")
result = ocr_engine.predict(img_path)
ocr_result = result[0]

# 새 API 형식으로 데이터 추출
texts = []
dt_polys = ocr_result['dt_polys']
rec_texts = ocr_result['rec_texts']
rec_scores = ocr_result['rec_scores']

for i in range(len(rec_texts)):
    poly = dt_polys[i].tolist() if hasattr(dt_polys[i], 'tolist') else dt_polys[i]
    text_str = rec_texts[i]
    score = rec_scores[i]
    
    # 한글 포함 여부
    has_korean = any('\uac00' <= c <= '\ud7a3' for c in text_str)
    
    texts.append({
        "bbox": poly,
        "text": text_str,
        "confidence": float(score),
        "has_korean": has_korean
    })

print(f"\n{'='*60}")
print(f"[결과] 총 {len(texts)}개 텍스트 인식")
print(f"{'='*60}")

# 한글 vs 영어 분류
korean_texts = [(i, t) for i, t in enumerate(texts) if t.get('has_korean', True)]
english_texts = [(i, t) for i, t in enumerate(texts) if not t.get('has_korean', True)]

print(f"\n한글 텍스트 (번역 대상): {len(korean_texts)}개")
print(f"영어 텍스트 (렌더링 건너뜀): {len(english_texts)}개")

# "콘실" 또는 "지퍼" 포함 텍스트 찾기
print(f"\n{'='*60}")
print("[검색] '콘실' 또는 '지퍼' 포함 텍스트")
print(f"{'='*60}")

found = False
for i, t in enumerate(texts):
    if '콘실' in t['text'] or '지퍼' in t['text'] or 'Zipper' in t['text'] or 'zipper' in t['text']:
        found = True
        bbox = t['bbox']
        x = min(p[0] for p in bbox)
        y = min(p[1] for p in bbox)
        w = max(p[0] for p in bbox) - x
        h = max(p[1] for p in bbox) - y
        print(f"  [{i:3d}] has_korean={t['has_korean']} | x={x:4.0f}, y={y:4.0f}, w={w:3.0f}, h={h:3.0f}")
        print(f"        text: '{t['text']}'")

if not found:
    print("  → 해당 텍스트 없음")

# 겹침 가능성 분석
print(f"\n{'='*60}")
print("[분석] 겹침 가능성 (같은 Y 범위에서 X가 가까운 텍스트)")
print(f"{'='*60}")

overlap_candidates = []
for i, t1 in enumerate(texts):
    bbox1 = t1['bbox']
    x1 = min(p[0] for p in bbox1)
    y1_min = min(p[1] for p in bbox1)
    y1_max = max(p[1] for p in bbox1)
    w1 = max(p[0] for p in bbox1) - x1
    
    for j, t2 in enumerate(texts):
        if i >= j:
            continue
        bbox2 = t2['bbox']
        x2 = min(p[0] for p in bbox2)
        y2_min = min(p[1] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)
        
        # Y 겹침 체크
        if not (y1_max < y2_min or y2_max < y1_min):
            # X 위치가 매우 가까운 경우 (30px 이내)
            if abs(x1 - x2) < 30:
                overlap_candidates.append((i, j, t1, t2, abs(x1-x2)))

print(f"  X 거리 30px 이내 + Y 겹침: {len(overlap_candidates)}쌍")
for i, j, t1, t2, dist in overlap_candidates[:20]:
    print(f"\n  [{i}] '{t1['text'][:40]}' (has_korean={t1['has_korean']})")
    print(f"  [{j}] '{t2['text'][:40]}' (has_korean={t2['has_korean']})")
    print(f"       X 거리: {dist:.0f}px")

# 한글+영어 혼합 텍스트 확인
print(f"\n{'='*60}")
print("[분석] 한글+영어 혼합 텍스트 (has_korean=True)")
print(f"{'='*60}")

mixed_texts = []
for i, t in enumerate(texts):
    if t['has_korean']:
        has_english = any('a' <= c.lower() <= 'z' for c in t['text'])
        if has_english:
            mixed_texts.append((i, t))

print(f"  한글+영어 혼합: {len(mixed_texts)}개")
for i, t in mixed_texts[:15]:
    bbox = t['bbox']
    x = min(p[0] for p in bbox)
    print(f"  [{i:3d}] x={x:4.0f} | '{t['text'][:50]}'")

# 정리
import os
os.remove(img_path)
doc.close()
print(f"\n{'='*60}")
print("[완료]")
print(f"{'='*60}")
