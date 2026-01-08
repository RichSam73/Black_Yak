# -*- coding: utf-8 -*-
"""
텍스트 지우기 테스트 프로그램
- 한글 감지 → 지우기 → 번역 텍스트 삽입 단계별 테스트
"""

import os
import sys
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import re

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 테스트 설정
TEST_PDF = r"E:\Antigravity\Black_Yak\Reference\Translator\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"
BASE_OUTPUT_DIR = r"E:\Antigravity\Black_Yak\Reference\Translator\output"

# 테스트 실행 시 날짜/시간 폴더 생성
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 폰트 설정
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"


def contains_korean(text):
    """텍스트에 한글이 포함되어 있는지 확인"""
    return bool(re.search(r'[가-힣]', text))


def pdf_to_image(pdf_path, page_num=0, zoom=2.0):
    """PDF 페이지를 이미지로 변환 (app.py와 동일한 방식)"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # ★ app.py와 동일하게 zoom=2.0 사용
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # ★ app.py와 동일: 임시 파일로 저장 후 cv2로 읽기
    temp_path = os.path.join(OUTPUT_DIR, "temp_page.png")
    pix.save(temp_path)
    doc.close()

    # cv2로 읽기 (BGR 형식)
    img = cv2.imread(temp_path)
    return img


def get_background_color_from_edges(img, bbox, margin=10):
    """bbox 주변 가장자리에서 배경색 샘플링"""
    height, width = img.shape[:2]
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    edge_pixels = []

    # 상단 가장자리 (margin 픽셀 위)
    sample_y = max(0, y_min - margin)
    for x in range(max(0, x_min), min(width, x_max)):
        pixel = tuple(img[sample_y, x])
        edge_pixels.append(pixel)

    # 하단 가장자리 (margin 픽셀 아래)
    sample_y = min(height - 1, y_max + margin)
    for x in range(max(0, x_min), min(width, x_max)):
        pixel = tuple(img[sample_y, x])
        edge_pixels.append(pixel)

    # 좌측 가장자리 (margin 픽셀 왼쪽)
    sample_x = max(0, x_min - margin)
    for y in range(max(0, y_min), min(height, y_max)):
        pixel = tuple(img[y, sample_x])
        edge_pixels.append(pixel)

    # 우측 가장자리 (margin 픽셀 오른쪽)
    sample_x = min(width - 1, x_max + margin)
    for y in range(max(0, y_min), min(height, y_max)):
        pixel = tuple(img[y, sample_x])
        edge_pixels.append(pixel)

    if edge_pixels:
        # 가장 많이 등장하는 색상 선택
        most_common = Counter(edge_pixels).most_common(1)[0][0]
        return most_common

    return (255, 255, 255)  # 기본값: 흰색


def erase_text_region_v1(img, bbox):
    """방법 1: 배경색 샘플링 + 사각형 채우기"""
    height, width = img.shape[:2]

    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    # 글자 높이에 비례한 마진
    text_height = y_max - y_min
    margin = max(3, int(text_height * 0.15))

    # 마진 적용한 확장 영역
    x_min_ext = max(0, x_min - margin)
    y_min_ext = max(0, y_min - margin)
    x_max_ext = min(width, x_max + margin)
    y_max_ext = min(height, y_max + margin)

    # 배경색 샘플링
    bg_color = get_background_color_from_edges(img, bbox, margin=margin + 5)
    bg_color_int = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))

    # 배경색으로 사각형 채우기
    cv2.rectangle(img, (x_min_ext, y_min_ext), (x_max_ext, y_max_ext), bg_color_int, -1)

    return img


def erase_text_region_v2(img, bbox):
    """방법 2: 흰색으로 덮기 (단순)"""
    height, width = img.shape[:2]

    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    text_height = y_max - y_min
    margin = max(3, int(text_height * 0.15))

    x_min_ext = max(0, x_min - margin)
    y_min_ext = max(0, y_min - margin)
    x_max_ext = min(width, x_max + margin)
    y_max_ext = min(height, y_max + margin)

    # 흰색으로 덮기
    cv2.rectangle(img, (x_min_ext, y_min_ext), (x_max_ext, y_max_ext), (255, 255, 255), -1)

    return img


def erase_text_region_v3(img, bbox):
    """방법 3: 폴리곤 마스크 + 배경색"""
    height, width = img.shape[:2]

    # 배경색 샘플링
    bg_color = get_background_color_from_edges(img, bbox, margin=10)
    bg_color_int = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))

    # bbox를 폴리곤으로 변환 (약간 확장)
    pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)

    # 폴리곤 중심 계산
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    # 폴리곤 확장 (중심에서 바깥으로 5% 확장)
    expanded_pts = []
    for pt in pts:
        dx = pt[0] - center_x
        dy = pt[1] - center_y
        new_x = int(center_x + dx * 1.1)
        new_y = int(center_y + dy * 1.1)
        expanded_pts.append([new_x, new_y])
    expanded_pts = np.array(expanded_pts, dtype=np.int32)

    # 폴리곤 채우기
    cv2.fillPoly(img, [expanded_pts], bg_color_int)

    return img


def draw_text_in_bbox(draw, bbox, text, font_path=FONT_PATH):
    """bbox 영역에 텍스트 삽입"""
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    box_width = x_max - x_min
    box_height = y_max - y_min

    # 폰트 크기 자동 조절
    font_size = max(10, int(box_height * 0.8))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 텍스트 크기 측정
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 폰트 크기 재조절 (너비에 맞춤)
    if text_width > box_width:
        font_size = int(font_size * box_width / text_width * 0.9)
        font_size = max(8, font_size)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

    # 텍스트 위치 (좌측 정렬, 수직 중앙)
    text_x = x_min
    text_y = y_min + (box_height - font_size) // 2

    # 검은색으로 텍스트 그리기
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))


def test_erase_methods():
    """여러 지우기 방법 테스트"""
    print("=" * 60)
    print("텍스트 지우기 테스트 프로그램 - OCR 좌표 정렬 디버깅")
    print("=" * 60)

    # 1. PDF를 이미지로 변환 (app.py와 동일: zoom=2.0)
    print("\n[1단계] PDF를 이미지로 변환 (app.py와 동일: zoom=2.0)...")
    img_original = pdf_to_image(TEST_PDF, page_num=0, zoom=2.0)
    print(f"  - 이미지 크기: {img_original.shape} (height, width, channels)")

    # 원본 저장
    cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original.png"), img_original)
    print(f"  - 저장: 0_original.png")

    # 2. OCR로 텍스트 감지
    print("\n[2단계] PaddleOCR로 텍스트 감지...")

    # OCR은 RGB 이미지를 기대함 - BGR에서 RGB로 변환
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # ★ 방법 1: 전처리 비활성화 - 원본 이미지 좌표로 bbox 반환
    ocr = PaddleOCR(
        lang='korean',
        use_doc_orientation_classify=False,  # 문서 방향 분류 끄기
        use_doc_unwarping=False,             # 문서 왜곡 보정 끄기
        use_textline_orientation=False       # 텍스트라인 방향 분류 끄기
    )
    result = ocr.predict(img_rgb)  # RGB 이미지 전달

    detections = []
    preprocessed_img = None  # ★ 전처리된 이미지 저장용

    # 새 PaddleOCR API 결과 파싱 - 디버깅
    print(f"  - 결과 타입: {type(result)}")
    if result:
        print(f"  - 결과 길이: {len(result)}")
        if len(result) > 0:
            print(f"  - 첫번째 항목 타입: {type(result[0])}")
            # dict인 경우 키 출력
            if hasattr(result[0], 'keys'):
                print(f"  - 키 목록: {list(result[0].keys())}")

        # 결과가 OCRResult 또는 dict 리스트인 경우 (새 API)
        for item in result:
            if hasattr(item, 'keys') or hasattr(item, 'rec_texts'):
                # ★ doc_preprocessor_res에서 전처리된 이미지 추출
                # OCRResult는 [] 접근으로 키에 접근해야 함
                try:
                    doc_res = item['doc_preprocessor_res']
                    print(f"  - doc_preprocessor_res 타입: {type(doc_res)}")
                    if doc_res is not None:
                        # DocPreprocessorResult도 [] 접근 사용
                        preprocessed_img = doc_res['output_img']
                        print(f"  - ★ 전처리된 이미지 추출 성공!")
                        print(f"  - 전처리 이미지 shape: {preprocessed_img.shape}")
                        print(f"  - 원본 이미지 shape: {img_original.shape}")
                except Exception as e:
                    print(f"  - doc_preprocessor_res 접근 오류: {e}")

                # PaddleX OCRResult 객체 또는 dict 형태
                # 키: rec_texts, rec_scores, rec_polys (또는 dt_polys)
                polys = getattr(item, 'rec_polys', None) or item.get('rec_polys', [])
                if not polys:
                    polys = getattr(item, 'dt_polys', None) or item.get('dt_polys', [])
                texts = getattr(item, 'rec_texts', None) or item.get('rec_texts', [])
                scores = getattr(item, 'rec_scores', None) or item.get('rec_scores', [])

                print(f"  - polys 개수: {len(polys) if polys else 0}")
                print(f"  - texts 개수: {len(texts) if texts else 0}")

                # 첫 번째 bbox 좌표 확인 (디버깅)
                if polys is not None and len(polys) > 0:
                    first_poly = polys[0]
                    if hasattr(first_poly, 'tolist'):
                        first_poly = first_poly.tolist()
                    print(f"  - 첫번째 bbox 예시: {first_poly}")
                    print(f"  - bbox 타입: {type(polys[0])}")

                # dt_polys도 확인 (원본 detection polygons)
                dt_polys = getattr(item, 'dt_polys', None) or item.get('dt_polys', [])
                if dt_polys is not None and len(dt_polys) > 0:
                    first_dt = dt_polys[0]
                    if hasattr(first_dt, 'tolist'):
                        first_dt = first_dt.tolist()
                    print(f"  - dt_polys 첫번째 예시: {first_dt}")
                    print(f"  - dt_polys 개수: {len(dt_polys)}")

                # rec_boxes도 확인 ([x_min, y_min, x_max, y_max] 형식)
                rec_boxes = getattr(item, 'rec_boxes', None) or item.get('rec_boxes', [])
                if rec_boxes is not None and len(rec_boxes) > 0:
                    first_box = rec_boxes[0]
                    if hasattr(first_box, 'tolist'):
                        first_box = first_box.tolist()
                    print(f"  - rec_boxes 첫번째 예시: {first_box} (x_min, y_min, x_max, y_max)")
                    print(f"  - rec_boxes 개수: {len(rec_boxes)}")

                # ★ dt_polys를 기본으로 사용 (원본 detection 좌표)
                if dt_polys is not None and len(dt_polys) > 0:
                    print(f"  → dt_polys 사용 (원본 detection 좌표)")
                    polys = dt_polys

                if polys and texts:
                    for i, (b, t) in enumerate(zip(polys, texts)):
                        score = scores[i] if scores and i < len(scores) else 0.0
                        # numpy array를 리스트로 변환
                        if hasattr(b, 'tolist'):
                            b = b.tolist()
                        detections.append({
                            "bbox": b,
                            "text": t,
                            "confidence": float(score)
                        })
                        print(f"  - 감지: '{t}' (신뢰도: {score:.2f})")
            elif isinstance(item, (list, tuple)):
                # 리스트 형태 (이전 API)
                if len(item) >= 2:
                    bbox = item[0]
                    if isinstance(item[1], (list, tuple)):
                        text = item[1][0]
                        confidence = item[1][1] if len(item[1]) > 1 else 0.0
                    else:
                        text = str(item[1])
                        confidence = item[2] if len(item) > 2 else 0.0

                    if bbox and text:
                        detections.append({
                            "bbox": bbox,
                            "text": text,
                            "confidence": float(confidence)
                        })
                        print(f"  - 감지: '{text}' (신뢰도: {confidence:.2f})")

    # ★ 전처리된 이미지가 있으면 사용, 없으면 원본 사용
    if preprocessed_img is not None:
        print(f"\n  ★ 전처리된 이미지 사용 (bbox 좌표와 일치)")
        img_for_bbox = preprocessed_img
        if len(img_for_bbox.shape) == 3 and img_for_bbox.shape[2] == 3:
            # RGB → BGR 변환 (OpenCV용)
            img_for_bbox = cv2.cvtColor(img_for_bbox, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "0_preprocessed.png"), img_for_bbox)
        print(f"  - 전처리 이미지 크기: {img_for_bbox.shape}")
    else:
        print(f"\n  ⚠ 전처리된 이미지 없음, 원본 사용")
        img_for_bbox = img_original

    print(f"\n  총 {len(detections)}개 텍스트 감지됨")

    # 한글이 포함된 텍스트만 필터링
    korean_detections = [d for d in detections if contains_korean(d["text"])]
    english_only_detections = [d for d in detections if not contains_korean(d["text"])]

    print(f"  - 한글 포함: {len(korean_detections)}개")
    print(f"  - 영어만: {len(english_only_detections)}개")

    # 한글 포함 텍스트 출력
    print("\n  [한글 포함 텍스트]")
    for det in korean_detections[:20]:  # 처음 20개만
        print(f"    - '{det['text']}'")
    if len(korean_detections) > 20:
        print(f"    ... 외 {len(korean_detections) - 20}개")

    # 3. 감지 영역 시각화 (한글만 빨간색, 영어는 파란색)
    print("\n[3단계] 감지 영역 시각화...")
    img_bbox = img_original.copy()
    for det in korean_detections:
        pts = np.array([[int(p[0]), int(p[1])] for p in det["bbox"]], dtype=np.int32)
        cv2.polylines(img_bbox, [pts], True, (0, 0, 255), 2)  # 한글: 빨간색
    for det in english_only_detections:
        pts = np.array([[int(p[0]), int(p[1])] for p in det["bbox"]], dtype=np.int32)
        cv2.polylines(img_bbox, [pts], True, (255, 0, 0), 1)  # 영어: 파란색 (얇게)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_detected_bbox.png"), img_bbox)
    print(f"  - 저장: 1_detected_bbox.png (빨간색=한글, 파란색=영어)")

    # 4. 방법별 지우기 테스트 (한글만 지우기!)
    print("\n[4단계] 지우기 방법 테스트 (한글만 지우기)...")

    # 방법 1: 배경색 샘플링
    img_v1 = img_original.copy()
    for det in korean_detections:  # 한글만!
        img_v1 = erase_text_region_v1(img_v1, det["bbox"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_erased_v1_bgcolor.png"), img_v1)
    print(f"  - 방법1 (배경색 샘플링): 2_erased_v1_bgcolor.png")

    # 방법 2: 흰색 덮기
    img_v2 = img_original.copy()
    for det in korean_detections:  # 한글만!
        img_v2 = erase_text_region_v2(img_v2, det["bbox"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_erased_v2_white.png"), img_v2)
    print(f"  - 방법2 (흰색): 2_erased_v2_white.png")

    # 방법 3: 폴리곤 마스크
    img_v3 = img_original.copy()
    for det in korean_detections:  # 한글만!
        img_v3 = erase_text_region_v3(img_v3, det["bbox"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_erased_v3_polygon.png"), img_v3)
    print(f"  - 방법3 (폴리곤): 2_erased_v3_polygon.png")

    # 5. 하드코딩된 번역으로 텍스트 삽입 테스트 (방법1 기준)
    print("\n[5단계] 번역 텍스트 삽입 (하드코딩)...")

    # 간단한 번역 매핑 (일부만)
    translation_map = {
        "남성": "Men's",
        "다운자켓": "Down Jacket",
        "봉제": "Sewing",
        "작업": "Work",
        "지시서": "Instructions",
        "원단": "Fabric",
        "안감": "Lining",
        "겉감": "Shell",
        "소매": "Sleeve",
        "후드": "Hood",
        "에리": "Collar",
        "지퍼": "Zipper",
        "밑단": "Hem",
        "앞판": "Front",
        "뒷판": "Back",
        "주머니": "Pocket",
        "로고": "LOGO",
    }

    # PIL 이미지로 변환
    img_result = Image.fromarray(cv2.cvtColor(img_v1, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    for det in korean_detections:  # 한글만!
        original_text = det["text"]
        # 번역 찾기
        translated = original_text
        for kr, en in translation_map.items():
            if kr in original_text:
                translated = original_text.replace(kr, en)
                break

        # 텍스트 삽입
        draw_text_in_bbox(draw, det["bbox"], translated)

    # 저장
    img_result.save(os.path.join(OUTPUT_DIR, "3_translated.png"))
    print(f"  - 저장: 3_translated.png")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print(f"결과 폴더: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    test_erase_methods()
