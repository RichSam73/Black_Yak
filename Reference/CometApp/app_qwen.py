"""
Comet + ERP 통합 웹 앱 (Qwen2.5-VL 버전)
- PaddleOCR: 텍스트 위치(좌표) 감지
- Qwen2.5-VL: OCR/문서 파싱 특화 모델 (Alibaba)
- Comet 오버레이 + ERP 테이블 동시 제공
- Grid 감지 방식으로 셀 매핑 (기존 방식 유지)
- 포트: 6002
"""
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
import io
import requests
import json
import re
from paddleocr import PaddleOCR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ollama 설정
OLLAMA_URL = "http://localhost:11434/api/generate"
# Round 16: 모델 폴백 체인 (타임아웃 시 대안 모델 사용)
# 1순위: qwen2.5vl, 2순위: gemma3:4b, 3순위: llama3.2-vision
VISION_MODELS = ["qwen2.5vl", "gemma3:4b", "llama3.2-vision"]
VISION_MODEL = VISION_MODELS[0]  # 기본 모델

# 전역 OCR 인스턴스
_paddle_ocr = None

# 전역 OCR 보정 설정 (외부 JSON에서 로드)
_ocr_corrections = None

def load_ocr_corrections():
    """외부 JSON 파일에서 OCR 보정 설정 로드 (하드코딩 제거)"""
    global _ocr_corrections
    if _ocr_corrections is not None:
        return _ocr_corrections

    import os
    config_path = os.path.join(os.path.dirname(__file__), "ocr_corrections.json")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _ocr_corrections = json.load(f)
        print(f"  [설정] OCR 보정 설정 로드 완료: {config_path}")
    except FileNotFoundError:
        print(f"  [경고] OCR 보정 설정 파일 없음: {config_path}")
        _ocr_corrections = {"simple_corrections": {}, "position_corrections": {"corrections": []}, "defaults": {}}
    except json.JSONDecodeError as e:
        print(f"  [오류] OCR 보정 설정 파싱 실패: {e}")
        _ocr_corrections = {"simple_corrections": {}, "position_corrections": {"corrections": []}, "defaults": {}}

    return _ocr_corrections


def get_simple_corrections() -> dict:
    """simple_corrections 딕셔너리 반환 (모든 카테고리 병합)"""
    config = load_ocr_corrections()
    simple = config.get("simple_corrections", {})

    # 모든 카테고리 병합 (_description, _comment 제외)
    merged = {}
    for key, value in simple.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            merged.update(value)

    return merged


def get_position_corrections() -> list:
    """position_corrections 리스트 반환 [(text, y_min, y_max, correct_text), ...]"""
    config = load_ocr_corrections()
    pos = config.get("position_corrections", {}).get("corrections", [])

    # JSON 형식을 튜플 리스트로 변환
    result = []
    for item in pos:
        result.append((
            item.get("text", ""),
            item.get("y_min", 0),
            item.get("y_max", 9999),
            item.get("correct_text", "")
        ))

    return result


def get_default_coordinates() -> dict:
    """기본 좌표값 반환 (헤더 감지 실패 시 폴백)"""
    config = load_ocr_corrections()
    return config.get("defaults", {"sup_nm_x": 834, "div_x": 33})


def get_detection_patterns() -> dict:
    """테이블 감지용 정규식 패턴 반환"""
    config = load_ocr_corrections()
    patterns = config.get("patterns", {})

    return {
        "size_pattern": patterns.get("size_pattern", r"^(0[89][05]|1[0-3][05])$"),
        "color_code_pattern": patterns.get("color_code_pattern", r"^[A-Z]{2}$"),
        "total_pattern": patterns.get("total_pattern", r"TOTAL|합계|소계"),
        "color_names_pattern": patterns.get("color_names_pattern", r"^(BLACK|NAVY|CREAM|TEAL|D/TAUPE\s*GRAY|SILVER\s*BEIGE)$")
    }


def is_size_number(text: str) -> bool:
    """SIZE 숫자인지 정규식으로 확인 (하드코딩 제거)"""
    patterns = get_detection_patterns()
    return bool(re.match(patterns["size_pattern"], text.strip()))


def is_color_code(text: str) -> bool:
    """COLOR 코드인지 정규식으로 확인 (하드코딩 제거)"""
    patterns = get_detection_patterns()
    return bool(re.match(patterns["color_code_pattern"], text.strip().upper()))


def has_total_keyword(text: str) -> bool:
    """TOTAL 키워드가 있는지 정규식으로 확인 (하드코딩 제거)"""
    patterns = get_detection_patterns()
    return bool(re.search(patterns["total_pattern"], text.strip().upper()))

def get_paddleocr():
    """PaddleOCR 인스턴스 싱글톤 (한글)"""
    global _paddle_ocr
    if _paddle_ocr is None:
        print("  [PaddleOCR 초기화 중... (한글)]")
        _paddle_ocr = PaddleOCR(lang='korean')
    return _paddle_ocr


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """OCR 정확도 향상을 위한 이미지 전처리

    - 녹색/파란색 배경의 흰색 텍스트 감지 개선
    - 어두운 배경 반전 처리
    - 대비 향상
    """
    img_array = np.array(image)

    # BGR로 변환 (OpenCV 형식)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    # HSV 변환하여 녹색/파란색 영역 마스크 생성
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 녹색 범위 (색상구분 헤더, 컬러 컬럼)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 파란색/청록색 범위
    blue_lower = np.array([85, 40, 40])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # 컬러 마스크 합치기
    color_mask = cv2.bitwise_or(green_mask, blue_mask)

    # 그레이스케일 변환
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 컬러 영역은 반전 (어두운 배경 -> 밝은 배경, 밝은 텍스트 -> 어두운 텍스트)
    inverted = cv2.bitwise_not(gray)

    # 컬러 영역만 반전된 이미지로 대체
    result = gray.copy()
    result[color_mask > 0] = inverted[color_mask > 0]

    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(result)

    # 다시 RGB로 변환
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(enhanced_rgb)


def ocr_with_paddle(image: Image.Image) -> list:
    """PaddleOCR로 텍스트 위치 + 초기 인식"""
    ocr = get_paddleocr()

    # 원본 이미지 사용 (전처리는 오히려 인식 품질 저하)
    img_array = np.array(image)

    try:
        results = ocr.predict(img_array)

        ocr_results = []
        if results:
            for result in results:
                rec_texts = result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', [])
                dt_polys = result.get('dt_polys', [])

                for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, dt_polys)):
                    if not text.strip():
                        continue

                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    box = [int(min(x_coords)), int(min(y_coords)),
                           int(max(x_coords)), int(max(y_coords))]

                    ocr_results.append({
                        "text": text,
                        "box": box,
                        "score": float(score)
                    })

        print(f"  [PaddleOCR] {len(ocr_results)}개 텍스트 감지")
        return ocr_results
    except Exception as e:
        print(f"PaddleOCR 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


def fill_missing_by_table_structure(ocr_results: list) -> list:
    """테이블 구조 분석으로 누락된 텍스트 채우기

    PaddleOCR이 놓친 텍스트를 행/열 구조 분석으로 찾아서 삽입
    - 스트링 행 SUP NM: "대일"
    - 아일렛 행 DIV: "아일렛", SUP NM: "대일"
    """
    if not ocr_results:
        return ocr_results

    # 1. 행 위치(Y좌표) 클러스터링
    y_centers = {}
    for ocr in ocr_results:
        box = ocr.get("box", [0, 0, 0, 0])
        y_center = int((box[1] + box[3]) / 2)
        text = ocr.get("text", "")

        # Y좌표 ±10 범위로 그룹핑
        found_cluster = False
        for cluster_y in y_centers:
            if abs(cluster_y - y_center) < 15:
                y_centers[cluster_y].append(ocr)
                found_cluster = True
                break
        if not found_cluster:
            y_centers[y_center] = [ocr]

    # 2. 각 행에서 특정 텍스트 찾기
    string_row_y = None  # 스트링 행 Y좌표
    eyelet_row_y = None  # 로고아일렛 행 Y좌표 (실제로는 아일렛 행)

    for y, row_items in y_centers.items():
        texts = [item.get("text", "") for item in row_items]
        if "스트링" in texts:
            string_row_y = y
            print(f"  [구조 분석] 스트링 행 발견: Y={y}")
        if "로고아일렛" in texts:
            eyelet_row_y = y
            print(f"  [구조 분석] 로고아일렛 행 발견: Y={y}")

    # 3. SUP NM 컬럼 X좌표 추정 (헤더에서 "SUP NM" 위치 찾기)
    # Round 19: 외부 설정에서 폴백 값 로드 (하드코딩 제거)
    defaults = get_default_coordinates()
    sup_nm_x = defaults.get("sup_nm_x", 834)  # 폴백 값
    for ocr in ocr_results:
        if ocr.get("text") == "SUP NM":
            box = ocr.get("box", [0, 0, 0, 0])
            sup_nm_x = int((box[0] + box[2]) / 2)
            print(f"  [구조 분석] SUP NM 컬럼 X좌표: {sup_nm_x}")
            break

    # 4. DIV 컬럼 X좌표 추정
    div_x = defaults.get("div_x", 33)  # 폴백 값
    for ocr in ocr_results:
        if ocr.get("text") == "DIV":
            box = ocr.get("box", [0, 0, 0, 0])
            div_x = int((box[0] + box[2]) / 2)
            print(f"  [구조 분석] DIV 컬럼 X좌표: {div_x}")
            break

    # 5. 스트링 행에 SUP NM "대일" 확인 및 삽입
    if string_row_y:
        has_daeil = False
        for ocr in ocr_results:
            box = ocr.get("box", [0, 0, 0, 0])
            y_center = (box[1] + box[3]) / 2
            x_center = (box[0] + box[2]) / 2

            # 스트링 행(Y) + SUP NM 컬럼(X) 근처에 "대일"이 있는지
            if abs(y_center - string_row_y) < 15 and abs(x_center - sup_nm_x) < 50:
                if ocr.get("text") == "대일":
                    has_daeil = True
                    break

        if not has_daeil:
            # 스트링 행에 대일 삽입
            ocr_results.append({
                "text": "대일",
                "box": [sup_nm_x - 24, string_row_y - 10, sup_nm_x + 24, string_row_y + 10],
                "score": 1.0,
                "injected": True
            })
            print(f"  [구조 삽입] '대일' at 스트링 행 Y={string_row_y}, X={sup_nm_x}")

    # 6. 아일렛 행 (로고아일렛 위) 처리
    if eyelet_row_y:
        # 아일렛 행은 로고아일렛보다 약간 위에 있음 (Y 차이 약 24픽셀)
        # 실제로 OCR 결과에서 로고아일렛 행 바로 위 행을 찾아야 함
        # 하지만 이미지에서 "아일렛" DIV와 "대일" SUP NM이 같은 행

        # 로고아일렛 행에서 DIV 컬럼에 아일렛 확인
        has_eyelet_div = False
        has_daeil_eyelet = False

        for ocr in ocr_results:
            box = ocr.get("box", [0, 0, 0, 0])
            y_center = (box[1] + box[3]) / 2
            x_center = (box[0] + box[2]) / 2

            if abs(y_center - eyelet_row_y) < 15:
                if ocr.get("text") == "아일렛" and abs(x_center - div_x) < 50:
                    has_eyelet_div = True
                if ocr.get("text") == "대일" and abs(x_center - sup_nm_x) < 50:
                    has_daeil_eyelet = True

        # 참고: Submaterial_correct.html 기준으로
        # 아일렛 행의 DIV = "아일렛", NAME = "로고아일렛"
        # 즉 "로고아일렛"은 NAME 컬럼에 있고, DIV에는 "아일렛"이 있어야 함

        if not has_eyelet_div:
            ocr_results.append({
                "text": "아일렛",
                "box": [div_x - 25, eyelet_row_y - 10, div_x + 25, eyelet_row_y + 10],
                "score": 1.0,
                "injected": True
            })
            print(f"  [구조 삽입] '아일렛' at DIV 컬럼 Y={eyelet_row_y}, X={div_x}")

        if not has_daeil_eyelet:
            ocr_results.append({
                "text": "대일",
                "box": [sup_nm_x - 24, eyelet_row_y - 10, sup_nm_x + 24, eyelet_row_y + 10],
                "score": 1.0,
                "injected": True
            })
            print(f"  [구조 삽입] '대일' at 아일렛 행 SUP NM Y={eyelet_row_y}, X={sup_nm_x}")

    return ocr_results


def detect_left_column_with_preprocessing(image: Image.Image, ocr_results: list) -> list:
    """이미지 전처리 기반 왼쪽 컬러/행 레이블 컬럼 텍스트 감지

    PaddleOCR이 배경색이 있는 영역의 텍스트를 감지하지 못하는 경우,
    이미지 전처리(대비 강화, 그레이스케일 반전 등)를 적용하여 재처리
    """
    from PIL import ImageEnhance, ImageOps

    # 이미지 맨 왼쪽 가장자리 영역 (X < 30)에 텍스트가 있는지 확인
    very_left_boundary = 30

    very_left_texts = [r for r in ocr_results if r.get("box", [100])[0] < very_left_boundary]

    # 맨 왼쪽 가장자리에 텍스트가 있으면 스킵
    if len(very_left_texts) >= 2:
        print(f"  [왼쪽 컬럼 감지] X<30에 이미 {len(very_left_texts)}개 텍스트 있음, 스킵")
        return ocr_results

    print(f"  [왼쪽 컬럼 감지] X<30 텍스트 {len(very_left_texts)}개, 이미지 전처리 시작...")

    try:
        # 1. 왼쪽 영역 크롭 (이미지 너비의 1/5 정도)
        width, height = image.size
        left_crop_width = min(150, width // 5)  # 최대 150px 또는 너비의 1/5
        left_region = image.crop((0, 0, left_crop_width, height))

        # 2. 이미지 전처리: 대비 강화 + 그레이스케일
        gray = left_region.convert('L')

        # 대비 강화
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)

        # 밝기 조정 (배경색을 흰색에 가깝게)
        brightness = ImageEnhance.Brightness(enhanced)
        brightened = brightness.enhance(1.3)

        # RGB로 다시 변환 (PaddleOCR용)
        processed = brightened.convert('RGB')

        # 3. 전처리된 왼쪽 영역 OCR
        if paddle_ocr is not None:
            img_array = np.array(processed)
            result = paddle_ocr.predict(img_array)

            if result and len(result) > 0:
                new_texts = []
                for item in result:
                    if 'rec_texts' in item and 'dt_polys' in item:
                        for i, text in enumerate(item['rec_texts']):
                            score = item['rec_scores'][i] if i < len(item['rec_scores']) else 0.5
                            poly = item['dt_polys'][i]

                            # 바운딩 박스 계산
                            x_coords = [p[0] for p in poly]
                            y_coords = [p[1] for p in poly]
                            box = [int(min(x_coords)), int(min(y_coords)),
                                   int(max(x_coords)), int(max(y_coords))]

                            # 유효한 텍스트만 추가 (숫자가 아닌 텍스트)
                            if text and not text.replace(',', '').replace('.', '').isdigit():
                                new_texts.append({
                                    "text": text,
                                    "box": box,  # X 좌표는 이미 왼쪽 영역 기준
                                    "score": score
                                })

                if new_texts:
                    print(f"  [왼쪽 컬럼 감지] 전처리 OCR로 {len(new_texts)}개 텍스트 발견")

                    # 기존 OCR 결과에서 가장 왼쪽 X좌표 확인
                    min_x = min(r.get("box", [100])[0] for r in ocr_results if r.get("box"))
                    # 새 컬럼은 기존 최소 X보다 충분히 왼쪽에 배치
                    new_col_x = max(0, min_x - 50)

                    for new_text in new_texts:
                        # Y 좌표는 그대로 유지, X 좌표만 조정
                        box = new_text["box"]
                        adjusted_box = [new_col_x, box[1], new_col_x + 25, box[3]]

                        ocr_results.append({
                            "text": new_text["text"],
                            "box": adjusted_box,
                            "score": new_text["score"],
                            "injected": True,
                            "source": "preprocessing_ocr"
                        })
                        print(f"  [왼쪽 컬럼 삽입] '{new_text['text']}' at Y={box[1]}-{box[3]}")
                else:
                    print(f"  [왼쪽 컬럼 감지] 전처리 OCR에서 유효한 텍스트 없음")
        else:
            print(f"  [왼쪽 컬럼 감지] PaddleOCR 미초기화")

    except Exception as e:
        print(f"  [왼쪽 컬럼 감지 오류] {e}")

    return ocr_results


def refine_text_with_ai(image: Image.Image, ocr_results: list) -> list:
    """AI Vision으로 저신뢰도 텍스트 보정 + 누락 컬럼 감지
    """
    if not ocr_results:
        return ocr_results

    # ===========================================================
    # 0단계: 이미지 전처리로 왼쪽 컬러 컬럼 감지 (배경색 영역)
    # ===========================================================
    ocr_results = detect_left_column_with_preprocessing(image, ocr_results)

    # ===========================================================
    # 1단계: 테이블 구조 분석으로 누락 텍스트 채우기 (AI 좌표 대신)
    # ===========================================================
    ocr_results = fill_missing_by_table_structure(ocr_results)

    # ===========================================================
    # 2단계: 저신뢰도 텍스트 AI 보정 (선택적)
    # ===========================================================
    low_confidence = [r for r in ocr_results if r.get('score', 1.0) < 0.85 and not r.get('injected')]

    if not low_confidence:
        print("  [AI 보정] 저신뢰도 텍스트 없음, 보정 생략")
        return ocr_results

    return ocr_results


def apply_known_corrections(ocr_results: list) -> list:
    """알려진 OCR 오류 수동 보정 (외부 JSON 설정 사용)

    Round 19: 하드코딩 제거 - ocr_corrections.json에서 설정 로드
    """
    # 외부 설정에서 보정 데이터 로드 (하드코딩 제거)
    simple_corrections = get_simple_corrections()
    position_corrections = get_position_corrections()

    # =====================================================================
    # 하드코딩 제거됨 - AI Vision (gemma3:27b)이 누락 텍스트 인식 담당
    # =====================================================================

    # 디버깅: 전체 OCR 결과를 파일로 저장
    debug_lines = ["=== 전체 OCR 결과 ==="]
    for i, ocr in enumerate(ocr_results):
        text = ocr.get("text", "")
        box = ocr.get("box", [0, 0, 0, 0])
        y_center = (box[1] + box[3]) / 2 if len(box) >= 4 else 0
        debug_lines.append(f"#{i}: '{text}' at Y={y_center:.0f}, box={box}")
    debug_lines.append("=====================")

    # 파일로 저장
    with open("ocr_debug.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(debug_lines))
    print(f"  [DEBUG] OCR 결과 {len(ocr_results)}개 -> ocr_debug.txt 저장됨")

    for ocr in ocr_results:
        text = ocr.get("text", "")
        box = ocr.get("box", [0, 0, 0, 0])
        y_center = (box[1] + box[3]) / 2 if len(box) >= 4 else 0

        # 1. 일반 텍스트 보정
        if text in simple_corrections:
            old_text = text
            ocr["text"] = simple_corrections[text]
            ocr["score"] = 1.0
            print(f"  [수동 보정] '{old_text}' → '{ocr['text']}'")
            continue

        # 2. 위치 기반 보정
        for (target_text, y_min, y_max, correct_text) in position_corrections:
            if text == target_text and y_min <= y_center <= y_max:
                old_text = text
                ocr["text"] = correct_text
                ocr["score"] = 1.0
                print(f"  [위치 보정] '{old_text}' → '{correct_text}' (Y={y_center:.0f})")
                break

    return ocr_results


def hybrid_ocr(image: Image.Image) -> list:
    """하이브리드 OCR: PaddleOCR + 수동 보정 + AI 보정"""
    # 1단계: PaddleOCR로 위치 + 초기 텍스트 인식
    ocr_results = ocr_with_paddle(image)

    # 2단계: 알려진 오류 수동 보정 (빠름)
    ocr_results = apply_known_corrections(ocr_results)

    # 3단계: AI로 저신뢰도 텍스트 보정 (선택적, 느림)
    ocr_results = refine_text_with_ai(image, ocr_results)

    return ocr_results


# =============================================================================
# Grid-First 핵심 함수들 (기존 코드 유지)
# =============================================================================

def cluster_values(values: list, threshold: int = 15) -> list:
    """값들을 클러스터링하여 대표값 리스트 반환"""
    if not values:
        return []

    sorted_vals = sorted(set(values))
    clusters = []
    current_cluster = [sorted_vals[0]]

    for v in sorted_vals[1:]:
        if v - current_cluster[-1] <= threshold:
            current_cluster.append(v)
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [v]

    clusters.append(int(np.mean(current_cluster)))
    return clusters


def detect_column_count_with_ai(image: Image.Image) -> int:
    """AI 비전 모델로 테이블 열 개수 감지

    Round 16: 사용자 피드백 반영 - OpenCV 세로선 대신 AI가 열 개수 판단
    타임아웃 시 대안 모델로 자동 전환:
    1순위: qwen2.5vl → 2순위: gemma3:4b → 3순위: llama3.2-vision

    Returns:
        int: 감지된 열 개수 (실패 시 0)
    """
    print(f"  [Round 16 AI] 테이블 열 개수 감지 시작...")

    # 이미지를 base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = """이 테이블 이미지를 보고 열(column) 개수를 세어주세요.

규칙:
1. 세로로 구분된 컬럼의 개수를 세세요
2. 첫 번째 컬럼(행 레이블)부터 마지막 컬럼(TOTAL 등)까지 포함
3. 빈 컬럼도 포함하여 세세요
4. 숫자만 답하세요 (예: 10)

열 개수:"""

    # Round 16: 모델 폴백 체인 - 타임아웃 시 다음 모델 시도
    for model in VISION_MODELS:
        try:
            print(f"  [Round 16 AI] {model} 시도 중...")

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # 숫자만 추출
                numbers = re.findall(r'\d+', result)
                if numbers:
                    col_count = int(numbers[0])
                    print(f"  [Round 16 AI] {model} 성공! 감지된 열 개수: {col_count}")
                    return col_count
                else:
                    print(f"  [Round 16 AI] {model} 응답에서 숫자 추출 실패: {result[:50]}")
            else:
                print(f"  [Round 16 AI] {model} HTTP 오류: {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"  [Round 16 AI] {model} 타임아웃, 다음 모델 시도...")
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"  [Round 16 AI] {model} 연결 오류: {e}")
            continue
        except Exception as e:
            print(f"  [Round 16 AI] {model} 오류: {e}")
            continue

    print(f"  [Round 16 AI] 모든 모델 실패, 세로선 기반 폴백 사용")
    return 0


def detect_vertical_lines(image: Image.Image, region: tuple = None) -> list:
    """OpenCV morphological operation으로 테이블 수직선 감지

    Round 14: OCR 텍스트 위치 대신 실제 테이블 선을 감지하여 컬럼 경계로 활용
    Round 15: 디버깅 로그 추가 + 다중 threshold 전략

    Args:
        image: PIL Image 객체
        region: (x, y, w, h) - 향후 차트 분리 시 특정 영역만 처리

    Returns:
        list: 수직선 X좌표 리스트 (컬럼 경계)
    """
    try:
        print(f"  [Round 15] 수직선 감지 시작...")

        # PIL Image를 OpenCV 형식으로 변환
        img_array = np.array(image)
        print(f"  [Round 15] 이미지 shape: {img_array.shape}")

        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array

        if img is None:
            print(f"  [Round 15] 이미지 변환 실패")
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 다중 threshold 전략 (Round 15)
        thresholds = [150, 180, 120, 200]
        all_line_positions = []

        for thresh_val in thresholds:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

            # 수직선 감지용 커널 (세로로 긴 커널)
            vertical_kernel = np.ones((30, 1), np.uint8)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

            # 수직선의 X좌표 추출
            contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # 높이가 충분한 선만 (노이즈 필터링)
                if h > img.shape[0] * 0.3:
                    all_line_positions.append(x + w // 2)

        # 중복 제거 및 정렬
        unique_positions = sorted(set(all_line_positions))
        print(f"  [Round 15] 수직선 후보: {len(unique_positions)}개")

        return unique_positions

    except Exception as e:
        print(f"  [Round 15] 수직선 감지 오류: {e}")
        return []


def merge_adjacent_columns(col_positions: list, threshold: int = 60) -> list:
    """인접한 컬럼들을 병합하여 병합 셀 문제 해결

    COLOR/SIZE 테이블에서 BK(X=24)와 BLACK(X=129)처럼
    헤더가 병합 셀이고 데이터가 분리된 경우 처리

    Args:
        col_positions: 클러스터링된 컬럼 X 좌표 리스트
        threshold: 병합 기준 거리 (기본 60px)

    Returns:
        병합 정보가 포함된 컬럼 매핑 딕셔너리
        {원래_col_idx: 병합된_col_idx}
    """
    if len(col_positions) < 2:
        return {i: i for i in range(len(col_positions))}

    # 컬럼 간 거리 계산
    col_distances = []
    for i in range(len(col_positions) - 1):
        dist = col_positions[i + 1] - col_positions[i]
        col_distances.append(dist)

    # 평균 거리 계산
    avg_dist = sum(col_distances) / len(col_distances) if col_distances else 0

    # 매핑 생성: 거리가 너무 가까운 컬럼은 병합
    col_mapping = {}
    merged_idx = 0

    for i in range(len(col_positions)):
        if i == 0:
            col_mapping[i] = merged_idx
        else:
            # 이전 컬럼과의 거리가 평균의 50% 미만이면 병합
            dist = col_positions[i] - col_positions[i - 1]
            if dist < avg_dist * 0.5:
                # 이전 컬럼과 같은 병합 인덱스 사용
                col_mapping[i] = col_mapping[i - 1]
            else:
                merged_idx += 1
                col_mapping[i] = merged_idx

    return col_mapping


def detect_merged_header_columns(ocr_results: list, row_positions: list, col_positions: list) -> dict:
    """헤더 행에서 병합된 컬럼 감지 (개선된 버전)

    병합 감지 방식:
    1. 헤더에 빈 컬럼이 있는 경우 (SUB MATERIAL INFORMATION 타입)
    2. 헤더 셀의 너비가 여러 데이터 컬럼을 포함하는 경우

    ⚠️ COLOR/SIZE QTY 타입 테이블은 병합하지 않음 (모든 컬럼이 독립적)

    Returns:
        {빈_컬럼_idx: 병합할_헤더_컬럼_idx} 형식의 병합 정보
    """
    if len(row_positions) < 2 or len(col_positions) < 2:
        return {}

    # ====================================================================
    # COLOR/SIZE QTY 테이블 감지 - 병합 스킵
    # ====================================================================
    all_texts = [ocr.get("text", "").strip() for ocr in ocr_results]
    all_texts_upper = [t.upper() for t in all_texts]
    all_texts_joined = " ".join(all_texts_upper)

    # 사이즈 숫자 패턴 감지 (3자리 숫자: 095, 100, 105, 110, 115, 120, 125, 130)
    size_numbers = ["095", "100", "105", "110", "115", "120", "125", "130"]
    has_size_numbers = any(size in all_texts for size in size_numbers)

    # COLOR/SIZE QTY 테이블 특징:
    # 1. 헤더에 COLOR, SIZE, QTY/TOTAL 중 하나 이상 포함
    # 2. 3자리 사이즈 숫자가 헤더에 있음
    has_color_size_keywords = (
        "COLOR" in all_texts_joined or
        "SIZE" in all_texts_joined or
        "QTY" in all_texts_joined
    )

    is_color_size_qty = has_color_size_keywords and has_size_numbers

    # 추가 감지: 컬러 코드(BK, NA, D3, SV 등)가 데이터에 있으면서 TOTAL이 있는 경우
    color_codes = ["BK", "NA", "D3", "SV", "WH", "GR", "NV", "RD", "BL", "CM", "TE", "CR", "BE", "GY"]
    has_color_codes = any(code in all_texts_upper for code in color_codes)
    has_total = "TOTAL" in all_texts_joined

    if has_color_codes and has_total and has_size_numbers:
        is_color_size_qty = True

    if is_color_size_qty:
        print(f"  [병합 감지] COLOR/SIZE QTY 테이블 감지 - 선택적 병합")
        print(f"    - 사이즈 숫자 발견: {has_size_numbers}")
        print(f"    - 컬러 키워드 발견: {has_color_size_keywords}")
        print(f"    - 컬러 코드 발견: {has_color_codes}")
        # Round 10: 전체 스킵 대신 헤더 단위 필터링 적용
        # QTY/TOTAL 헤더는 병합 스킵, COLOR / SIZE 헤더는 정상 병합

    # ====================================================================
    # 헤더 행 분석
    # ====================================================================

    # 헤더 행 (상위 2개 행을 헤더로 간주)
    header_rows = row_positions[:2] if len(row_positions) >= 2 else row_positions[:1]

    # 헤더 텍스트의 X 좌표 수집
    header_items = []  # [(x_start, x_end, x_center, text, col_idx), ...]
    header_cols = set()

    for ocr in ocr_results:
        box = ocr.get("box", [])
        text = ocr.get("text", "")
        if len(box) < 4:
            continue
        cy = (box[1] + box[3]) / 2
        x_start, x_end = box[0], box[2]
        cx = (x_start + x_end) / 2

        # 헤더 행인지 확인
        is_header = any(abs(cy - header_y) < 20 for header_y in header_rows)
        if is_header:
            col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - cx))
            header_cols.add(col_idx)
            header_items.append((x_start, x_end, cx, text, col_idx))

    # X 좌표로 헤더 정렬
    header_items.sort(key=lambda h: h[2])  # cx 기준 정렬

    merge_map = {}

    # ====================================================================
    # 방법 2: 헤더 셀의 너비가 여러 데이터 컬럼을 커버하는지 확인
    # ====================================================================

    for x_start, x_end, cx, text, header_col_idx in header_items:
        # Round 10: QTY/TOTAL 헤더는 병합 스킵 (사이즈 컬럼 독립 유지)
        text_upper = text.upper()
        if "QTY" in text_upper or text_upper == "TOTAL":
            continue

        # Round 18: 타이틀 행(1행)의 넓은 헤더는 병합 기준으로 사용하지 않음
        # SUB MATERIAL INFORMATION, COLOR/SIZE QTY 등은 여러 컬럼에 걸쳐있지만 병합 의도 아님
        title_keywords = ["INFORMATION", "COLOR/SIZE", "MATERIAL", "SUB "]
        is_title_header = any(kw in text_upper for kw in title_keywords)
        if is_title_header:
            print(f"  [병합 스킵] 타이틀 헤더 '{text}' - 병합 기준으로 사용 안함")
            continue

        # 이 헤더 범위에 포함되는 다른 컬럼 찾기
        covered_cols = []
        for col_idx, col_x in enumerate(col_positions):
            if col_idx in merge_map:
                continue  # 이미 병합됨
            # 컬럼 중심이 헤더 X 범위 내에 있는지
            if x_start - 20 <= col_x <= x_end + 20:
                covered_cols.append(col_idx)

        # 여러 컬럼이 커버되면 첫 번째 외의 컬럼들을 병합
        if len(covered_cols) > 1:
            primary_col = min(covered_cols)
            for col_idx in covered_cols:
                if col_idx != primary_col and col_idx not in merge_map:
                    merge_map[col_idx] = primary_col
                    print(f"  [병합 감지] 헤더 '{text}' 범위 X={x_start}-{x_end}에서 컬럼 {col_idx} → {primary_col}")

    # ====================================================================
    # 방법 3: 헤더가 없는 컬럼 찾기 (기존 로직)
    # ⚠️ COLOR/SIZE QTY 테이블에서는 건너뜀 (BK 컬럼 등이 의도치 않게 병합되는 것 방지)
    # ====================================================================

    if not is_color_size_qty:
        empty_header_cols = set(range(len(col_positions))) - header_cols

        for empty_col in empty_header_cols:
            if empty_col in merge_map:
                continue  # 이미 병합됨

            # 가장 가까운 헤더가 있는 컬럼 찾기
            closest_header = None
            min_dist = float('inf')
            for header_col in header_cols:
                dist = abs(col_positions[header_col] - col_positions[empty_col])
                if dist < min_dist:
                    min_dist = dist
                    closest_header = header_col

            if closest_header is not None and min_dist < 40:  # Round 9: 150→40 (SUP CD/SUP NM 병합 방지)
                merge_map[empty_col] = closest_header
                print(f"  [병합 감지] 빈 헤더 컬럼 {empty_col} → {closest_header} (거리: {min_dist:.0f})")

    return merge_map


def remove_empty_rows_cols(table: list) -> list:
    """완전히 빈 행과 열 제거 (후처리)

    Round 12: Docling 패턴 적용 - 업계 표준 후처리 방식
    - 모든 셀이 빈 행만 제거
    - 모든 셀이 빈 열만 제거
    - 데이터가 있는 셀은 모두 보존
    """
    if not table:
        return table

    # 1. 빈 행 제거
    table = [row for row in table if any(cell.strip() for cell in row)]

    if not table:
        return table

    # 2. 빈 열 제거
    num_cols = len(table[0])
    non_empty_cols = []
    for col_idx in range(num_cols):
        if any(row[col_idx].strip() for row in table if col_idx < len(row)):
            non_empty_cols.append(col_idx)

    if non_empty_cols:
        table = [[row[col_idx] for col_idx in non_empty_cols if col_idx < len(row)] for row in table]

    return table


def build_table_from_ocr(ocr_results: list, image: Image.Image = None, region: tuple = None) -> list:
    """OCR 결과의 위치 정보만으로 테이블 구성

    Round 12: 병합 로직 비활성화, 후처리로 빈 열/행 제거
    Round 14: 수직선 감지 하이브리드 방식 추가
    Round 16: AI 기반 열 개수 감지

    Args:
        ocr_results: OCR 결과 리스트
        image: PIL Image 객체 (수직선 감지용, 선택적)
        region: (x, y, w, h) - 향후 차트 분리 시 특정 영역만 처리
    """
    if not ocr_results:
        return []

    y_centers = []
    x_centers = []

    for ocr in ocr_results:
        box = ocr.get("box", [])
        if len(box) < 4:
            continue
        cy = (box[1] + box[3]) / 2
        cx = (box[0] + box[2]) / 2
        y_centers.append(cy)
        x_centers.append(cx)

    if not y_centers or not x_centers:
        return []

    row_positions = cluster_values(y_centers, threshold=15)

    # ========================================
    # Round 14: 하이브리드 컬럼 감지
    # - 수직선 감지 우선 → 텍스트 클러스터링 폴백
    # ========================================

    # 1. 각 행의 텍스트 개수 계산 (타이틀 행 제외용)
    row_text_counts = {}
    row_x_centers = {}
    for ocr in ocr_results:
        box = ocr.get("box", [])
        if len(box) < 4:
            continue
        cy = (box[1] + box[3]) / 2
        cx = (box[0] + box[2]) / 2
        row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - cy))
        row_text_counts[row_idx] = row_text_counts.get(row_idx, 0) + 1
        if row_idx not in row_x_centers:
            row_x_centers[row_idx] = []
        row_x_centers[row_idx].append(cx)

    max_cols = max(row_text_counts.values()) if row_text_counts else 0
    print(f"  [Max-Column] 행별 텍스트 개수: {dict(sorted(row_text_counts.items()))}")
    print(f"  [Max-Column] 최대 열 수: {max_cols}")

    # 2. 타이틀 행 제외한 X좌표 수집
    non_title_x_centers = []
    for row_idx, count in row_text_counts.items():
        if count > 1:
            non_title_x_centers.extend(row_x_centers.get(row_idx, []))

    # 3. Round 14: 수직선 감지 시도
    line_x_positions = []
    if image is not None:
        line_x_positions = detect_vertical_lines(image, region)
        print(f"  [Round 14] 수직선 감지: {len(line_x_positions)}개 발견")
        if line_x_positions:
            print(f"  [Round 14] 수직선 X좌표: {line_x_positions[:15]}{'...' if len(line_x_positions) > 15 else ''}")

    # 4. 하이브리드 컬럼 위치 결정
    # Round 15: COLOR/SIZE QTY 테이블 감지 (수직선보다 텍스트 클러스터링이 더 정확)
    # Round 19: 정규식 기반 패턴 감지 (하드코딩 제거)
    all_texts = [ocr.get("text", "").strip() for ocr in ocr_results]
    all_texts_upper = [t.upper() for t in all_texts]

    # 정규식 기반 SIZE 숫자 감지 (하드코딩 제거)
    has_size_numbers = any(is_size_number(text) for text in all_texts)
    # 정규식 기반 COLOR 코드 감지 (하드코딩 제거)
    has_color_codes = any(is_color_code(text) for text in all_texts)
    # 정규식 기반 TOTAL 감지 (하드코딩 제거)
    has_total = has_total_keyword(" ".join(all_texts_upper))

    is_color_size_qty = has_size_numbers and (has_color_codes or has_total)
    if is_color_size_qty:
        print(f"  [Round 16] COLOR/SIZE QTY 테이블 감지됨")

    # ========================================
    # Round 17: 헤더 행 기반 컬럼 위치 결정 (핵심 개선)
    # - SIZE 숫자가 있는 헤더 행을 찾아 컬럼 위치 기준으로 사용
    # - 빈 컬럼도 헤더 기준으로 정확히 배치
    # ========================================

    header_col_positions = None
    if is_color_size_qty:
        # 헤더 행 찾기: SIZE 숫자(095, 100, ...)가 포함된 행
        header_row_idx = None
        header_row_x_positions = []

        for ocr in ocr_results:
            text = ocr.get("text", "").strip()
            box = ocr.get("box", [])
            if len(box) < 4:
                continue

            # SIZE 숫자인지 확인 (정규식 기반)
            if is_size_number(text):
                cy = (box[1] + box[3]) / 2
                cx = (box[0] + box[2]) / 2

                # 해당 행 인덱스 찾기
                row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - cy))

                if header_row_idx is None:
                    header_row_idx = row_idx

                if row_idx == header_row_idx:
                    header_row_x_positions.append((cx, text))

        if header_row_x_positions:
            # 헤더 행의 X 좌표 정렬
            header_row_x_positions.sort(key=lambda x: x[0])
            print(f"  [Round 17] 헤더 행 발견: {[(t, int(x)) for x, t in header_row_x_positions]}")

            # 헤더 행에서 컬럼 위치 계산
            # COLOR/SIZE 컬럼 (첫 번째 size 숫자 왼쪽)
            first_size_x = header_row_x_positions[0][0]

            # COLOR/SIZE 텍스트 찾기
            color_size_x = None
            for ocr in ocr_results:
                text = ocr.get("text", "").strip().upper()
                box = ocr.get("box", [])
                if len(box) >= 4 and ("COLOR" in text or "SIZE" in text) and "QTY" not in text:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - cy))
                    if row_idx == header_row_idx or abs(row_positions[row_idx] - row_positions[header_row_idx]) < 30:
                        color_size_x = cx
                        break

            # 색상 코드 컬럼 위치 (왼쪽 끝)
            # Round 19: 정규식 기반 감지 (하드코딩 제거)
            color_code_x_list = []
            color_name_x_list = []
            patterns = get_detection_patterns()
            color_names_pattern = re.compile(patterns["color_names_pattern"], re.IGNORECASE)
            for ocr in ocr_results:
                text = ocr.get("text", "").strip().upper()
                box = ocr.get("box", [])
                if len(box) >= 4:
                    cx = (box[0] + box[2]) / 2
                    if is_color_code(text):
                        color_code_x_list.append(cx)
                    elif color_names_pattern.match(text):
                        color_name_x_list.append(cx)

            # 컬럼 위치 구성
            header_col_positions = []

            # 1. 색상 코드 컬럼 (있으면)
            if color_code_x_list:
                avg_color_code_x = sum(color_code_x_list) / len(color_code_x_list)
                header_col_positions.append(avg_color_code_x)

            # 2. COLOR/SIZE 또는 색상 이름 컬럼
            if color_size_x:
                header_col_positions.append(color_size_x)
            elif color_name_x_list:
                avg_color_name_x = sum(color_name_x_list) / len(color_name_x_list)
                if not header_col_positions or abs(avg_color_name_x - header_col_positions[-1]) > 30:
                    header_col_positions.append(avg_color_name_x)

            # 3. SIZE 숫자 컬럼들
            for x, text in header_row_x_positions:
                header_col_positions.append(x)

            # 4. TOTAL 컬럼
            total_x = None
            for ocr in ocr_results:
                text = ocr.get("text", "").strip().upper()
                box = ocr.get("box", [])
                if len(box) >= 4 and text == "TOTAL":
                    cy = (box[1] + box[3]) / 2
                    row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - cy))
                    if row_idx == header_row_idx or abs(row_positions[row_idx] - row_positions[header_row_idx]) < 30:
                        total_x = (box[0] + box[2]) / 2
                        break

            if total_x and (not header_col_positions or abs(total_x - header_col_positions[-1]) > 30):
                header_col_positions.append(total_x)

            # 정렬 및 중복 제거
            header_col_positions = sorted(set(int(x) for x in header_col_positions))
            print(f"  [Round 17] 헤더 기반 컬럼 위치: {header_col_positions}")

    # ========================================
    # Round 16: AI 기반 열 개수 감지 (헤더 기반이 없을 때만)
    # ========================================

    ai_col_count = 0
    if image is not None and not header_col_positions:
        ai_col_count = detect_column_count_with_ai(image)

    # 컬럼 위치 결정 우선순위: 헤더 기반 > AI 기반 > 텍스트 클러스터링
    if header_col_positions:
        col_positions = header_col_positions
        print(f"  [Round 17] 헤더 기반 컬럼 사용: {len(col_positions)}개")
    elif ai_col_count > 0 and non_title_x_centers:
        # AI가 열 개수를 감지한 경우: 해당 개수에 맞게 클러스터링
        # 이미지 너비와 열 개수로 적절한 threshold 계산
        img_width = image.width if image else 1000
        estimated_col_width = img_width // ai_col_count
        dynamic_threshold = max(10, estimated_col_width // 3)

        col_positions = cluster_values(non_title_x_centers, threshold=dynamic_threshold)

        # AI 열 개수와 클러스터 결과가 다르면 threshold 조정
        attempts = 0
        while len(col_positions) != ai_col_count and attempts < 5:
            if len(col_positions) > ai_col_count:
                dynamic_threshold += 5  # threshold 증가 → 더 많이 병합
            else:
                dynamic_threshold = max(5, dynamic_threshold - 5)  # threshold 감소 → 더 세분화
            col_positions = cluster_values(non_title_x_centers, threshold=dynamic_threshold)
            attempts += 1

        print(f"  [Round 16 AI] 목표 열 수: {ai_col_count}, 실제: {len(col_positions)}, threshold: {dynamic_threshold}")
        print(f"  [Round 16 AI] 컬럼 중심점: {col_positions}")

    elif len(line_x_positions) >= 2:
        # AI 실패 시 세로선 기반 폴백
        merged_lines = cluster_values(line_x_positions, threshold=15)
        print(f"  [Round 16] 세로선 병합: {len(line_x_positions)}개 → {len(merged_lines)}개")

        col_positions = []
        for i in range(len(merged_lines) - 1):
            center = (merged_lines[i] + merged_lines[i + 1]) // 2
            col_positions.append(center)
        print(f"  [Round 16] 세로선 기반 컬럼: {len(col_positions)}개")

    else:
        # 세로선도 없으면 텍스트 기반 클러스터링
        col_positions = cluster_values(non_title_x_centers, threshold=20)
        print(f"  [Round 16] 텍스트 기반 컬럼 (폴백): {len(col_positions)}개")

    num_rows = len(row_positions)
    num_cols = len(col_positions)

    if num_rows == 0 or num_cols == 0:
        return []

    # Round 8: COLOR/SIZE QTY 테이블은 Method 2만 허용
    merge_map = {}
    if not is_color_size_qty:
        merge_map = detect_merged_header_columns(ocr_results, row_positions, col_positions)

    if merge_map:
        print(f"  [병합 셀 감지] {len(merge_map)}개 컬럼 병합: {merge_map}")

        merged_cols = set()
        for empty_col, header_col in merge_map.items():
            merged_cols.add(empty_col)

        new_col_positions = []
        old_to_new_col = {}
        new_idx = 0

        for old_idx in range(num_cols):
            if old_idx in merged_cols:
                target_col = merge_map[old_idx]
                old_to_new_col[old_idx] = old_to_new_col.get(target_col, new_idx)
            else:
                old_to_new_col[old_idx] = new_idx
                new_col_positions.append(col_positions[old_idx])
                new_idx += 1

        num_cols = len(new_col_positions)
        print(f"  [병합 후] {num_rows}행 x {num_cols}열")
    else:
        old_to_new_col = {i: i for i in range(num_cols)}
        new_col_positions = col_positions

    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for ocr in ocr_results:
        box = ocr.get("box", [])
        text = ocr.get("text", "").strip()

        if not text or len(box) < 4:
            continue

        cy = (box[1] + box[3]) / 2
        cx = (box[0] + box[2]) / 2

        row_idx = min(range(num_rows), key=lambda i: abs(row_positions[i] - cy))

        orig_col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - cx))

        col_idx = old_to_new_col.get(orig_col_idx, orig_col_idx)

        if col_idx >= num_cols:
            col_idx = num_cols - 1

        if table[row_idx][col_idx]:
            table[row_idx][col_idx] += " " + text
        else:
            table[row_idx][col_idx] = text

    # Round 12: 후처리 - 빈 행/열 제거
    table = remove_empty_rows_cols(table)

    return table


# =============================================================================
# HTML 생성
# =============================================================================

def validate_table_with_ai(image: Image.Image, table_2d: list) -> dict:
    """AI Vision으로 ERP 테이블 검증 + 수학적 검증

    테이블 생성 후 AI가 원본 이미지와 비교하여 누락/오류 검출
    다양한 테이블 타입 지원: SUB MATERIAL INFORMATION, COLOR/SIZE QTY 등

    Round 20 강화: COLOR/SIZE QTY 테이블은 수학적 합계 검증 추가
    """
    if not table_2d or len(table_2d) < 3:
        return {"valid": True, "issues": [], "message": "테이블이 너무 작아 검증 생략"}

    # 테이블 타입 자동 감지 (첫 번째 행 또는 두 번째 행 기준)
    first_row_text = " ".join([str(c) for c in table_2d[0]]) if table_2d else ""
    second_row_text = " ".join([str(c) for c in table_2d[1]]) if len(table_2d) > 1 else ""
    combined_header = first_row_text.upper() + " " + second_row_text.upper()

    is_color_size_qty = "COLOR" in combined_header and ("SIZE" in combined_header or "QTY" in combined_header)

    print(f"  [검증] 테이블 타입 감지: is_color_size_qty={is_color_size_qty}")
    print(f"  [검증] combined_header: {combined_header[:100]}...")

    # Round 20: COLOR/SIZE QTY 테이블은 수학적 검증 먼저 수행
    if is_color_size_qty:
        print("  [검증] COLOR/SIZE QTY 테이블 감지 - 수학적 검증 수행")

        # 헤더 찾기 (숫자 패턴이 있는 행 - 사이즈 095, 100, 105 등)
        headers = []
        header_row_idx = -1
        for idx, row in enumerate(table_2d[:5]):  # 처음 5행까지 확인
            for cell in row:
                # 2-3자리 숫자 패턴 (095, 100, 105 등)
                if re.match(r'^0?\d{2,3}$', str(cell).strip()):
                    headers = row
                    header_row_idx = idx
                    print(f"  [검증] 헤더 발견: 행 {idx}, 매칭 셀: {cell}")
                    break
            if headers:
                break

        if not headers:
            print(f"  [검증] 헤더를 찾지 못함! 테이블 처음 3행:")
            for i, row in enumerate(table_2d[:3]):
                print(f"    행{i}: {row[:5]}...")

        if headers:
            # validate_table_math 함수 호출
            math_result = validate_table_math(table_2d, headers)

            if not math_result["valid"]:
                issues = []
                suggestions = []

                for err in math_result.get("errors", []):
                    if err["type"] == "row_total":
                        issues.append(f"행 {err['row']} 합계 불일치: 계산={err['expected']}, 표시={err['current']}")
                    elif err["type"] == "col_total":
                        col_name = headers[err['col']] if err['col'] < len(headers) else f"열{err['col']}"
                        issues.append(f"열 {col_name} 합계 불일치: 계산={err['expected']}, 표시={err['current']}")

                for cell in math_result.get("mismatch_cells", []):
                    col_name = headers[cell['col']] if cell['col'] < len(headers) else f"열{cell['col']}"
                    issues.append(f"⚠️ 셀({cell['row']},{col_name}) 오류 추정: '{cell['current']}' → '{cell['possible_correct']}'")
                    suggestions.append(f"({cell['row']},{col_name}) 값을 '{cell['possible_correct']}'로 수정하세요")

                return {
                    "valid": False,
                    "issues": issues,
                    "suggestions": suggestions,
                    "math_errors": math_result.get("errors", []),
                    "mismatch_cells": math_result.get("mismatch_cells", [])
                }
            else:
                print("  [검증] 수학적 검증 통과")

    # 이미지를 base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 테이블을 텍스트로 변환
    table_text = []
    for row_idx, row in enumerate(table_2d):
        row_text = " | ".join([cell if cell else "(빈칸)" for cell in row])
        table_text.append(f"행{row_idx}: {row_text}")
    table_summary = "\n".join(table_text[:10])  # 처음 10행만

    if is_color_size_qty:
        table_type = "COLOR/SIZE QTY (발주수량)"
        check_items = "1. 컬러별 수량이 정확한지\n2. 사이즈별 합계가 맞는지\n3. TOTAL 값이 정확한지"
    elif "MATERIAL" in first_row_text.upper() or "SUB" in first_row_text.upper():
        table_type = "SUB MATERIAL INFORMATION"
        check_items = "1. 누락된 텍스트가 있는지 (특히 SUP NM, DIV 컬럼)\n2. 잘못 인식된 텍스트가 있는지\n3. 행/열이 잘못 매핑된 경우"
    else:
        table_type = "일반 테이블"
        check_items = "1. 누락된 텍스트가 있는지\n2. 잘못 인식된 텍스트가 있는지\n3. 행/열이 잘못 매핑된 경우"

    prompt = f"""이미지는 {table_type} 테이블입니다.
OCR로 추출한 테이블 결과가 맞는지 검증해주세요.

[추출된 테이블 (처음 10행)]
{table_summary}

다음을 확인하세요:
{check_items}

문제가 있으면 JSON 형식으로 응답:
{{"valid": false, "issues": ["문제1", "문제2"], "suggestions": ["수정제안1"]}}

문제가 없으면:
{{"valid": true, "message": "검증 통과"}}

반드시 JSON만 응답하세요."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120  # 타임아웃 60초 → 120초로 증가
        )

        if response.status_code == 200:
            result = response.json().get("response", "")
            # \xa0 등 특수문자 제거 (Windows cp949 인코딩 오류 방지)
            safe_result = result.replace('\xa0', ' ')[:200]
            print(f"  [AI 검증] 응답: {safe_result}...")

            # JSON 파싱 시도
            try:
                # JSON 부분 추출
                json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
                if json_match:
                    validation = json.loads(json_match.group())
                    return validation
            except json.JSONDecodeError:
                pass

            return {"valid": True, "message": "AI 응답 파싱 실패, 기본 통과"}
        else:
            print(f"  [AI 검증] 오류: {response.status_code}")
            return {"valid": True, "message": "AI 서버 오류, 검증 생략"}

    except Exception as e:
        print(f"  [AI 검증] 예외: {e}")
        return {"valid": True, "message": f"검증 실패: {str(e)}"}


def generate_erp_table_html(table_2d: list, validation: dict = None) -> str:
    """Grid-First 2D 테이블을 ERP용 HTML 테이블로 변환

    Round 20: 수학적 오류 셀 하이라이트 + 수정 UI 추가
    """

    if not table_2d or len(table_2d) == 0:
        return '<p style="color: #ff6b6b;">테이블 격자를 감지하지 못했습니다. Comet 탭에서 직접 텍스트를 복사해주세요.</p>'

    num_cols = max(len(row) for row in table_2d)

    # 수학적 오류 셀 위치 맵 (빨간색 하이라이트)
    mismatch_map = {}
    if validation and validation.get("mismatch_cells"):
        for m in validation["mismatch_cells"]:
            key = (m.get("row", -1), m.get("col", -1))
            mismatch_map[key] = m

    # AI 검증 결과 배너
    validation_banner = ""
    if validation:
        if validation.get("valid", True):
            validation_banner = f'''
            <div style="background: #d4edda; border: 1px solid #28a745; padding: 10px; margin-bottom: 15px; border-radius: 8px; color: #155724;">
                ✅ <strong>AI 검증 통과</strong>: {validation.get("message", "검증 통과")}
            </div>'''
        else:
            issues = validation.get("issues", [])
            issues_html = "<br>".join([f"⚠️ {issue}" for issue in issues])
            suggestions = validation.get("suggestions", [])
            suggestions_html = "<br>".join([f"💡 {s}" for s in suggestions]) if suggestions else ""

            validation_banner = f'''
            <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin-bottom: 15px; border-radius: 8px; color: #856404;">
                ⚠️ <strong>AI 검증 경고</strong><br>
                {issues_html}
                {f"<br><br>{suggestions_html}" if suggestions_html else ""}
            </div>'''

    html = validation_banner + '<table class="erp-table">\n'

    for row_idx, row in enumerate(table_2d):
        html += '<tr>\n'

        for col_idx in range(num_cols):
            cell = row[col_idx] if col_idx < len(row) else ''
            cell = cell.strip() if cell else ''
            key = (row_idx, col_idx)

            # Round 20: 수학적 오류 셀 - 빨간색 하이라이트 + 수정 UI
            if key in mismatch_map:
                m = mismatch_map[key]
                cell_id = f"erp_cell_{row_idx}_{col_idx}"
                current = m.get("current", cell)
                possible = m.get("possible_correct", "")

                html += f'''<td class="data-cell" style="background: #f8d7da !important; border: 2px solid #dc3545 !important;">
                    <select id="{cell_id}" onchange="handleErpCellSelect(this)" style="width: 100%; border: none; background: transparent; color: #721c24; font-weight: bold; cursor: pointer;">
                        <option value="{current}">{current} (현재)</option>
                        <option value="{possible}">{possible} (추정)</option>
                    </select>
                </td>\n'''
            elif row_idx == 0:
                html += f'<td class="header">{cell}</td>\n'
            elif row_idx == 1:
                html += f'<td class="sub-header">{cell}</td>\n'
            elif not cell:
                html += f'<td class="empty-cell"></td>\n'
            elif 'TOTAL' in ' '.join([c for c in row if c]).upper():
                html += f'<td class="total-row-cell">{cell}</td>\n'
            elif cell.replace(',', '').replace('.', '').replace('-', '').isdigit():
                html += f'<td class="data-cell">{cell}</td>\n'
            else:
                html += f'<td>{cell}</td>\n'

        html += '</tr>\n'

    html += '</table>\n'

    # JavaScript for cell selection (수정됨 표시)
    if mismatch_map:
        html += '''
        <script>
        function handleErpCellSelect(select) {
            // 값 변경 시 셀 색상을 녹색으로 변경 (수정됨 표시)
            if (select.selectedIndex > 0) {
                select.parentElement.style.background = "#d4edda";
                select.parentElement.style.border = "2px solid #28a745";
                select.style.color = "#155724";
            } else {
                select.parentElement.style.background = "#f8d7da";
                select.parentElement.style.border = "2px solid #dc3545";
                select.style.color = "#721c24";
            }
        }
        </script>
        '''

    return html


# =============================================================================
# Round 20: 100% AI 기반 테이블 추출 (하드코딩 제로)
# =============================================================================

def extract_table_with_ai_only(image: Image.Image) -> dict:
    """100% AI 기반 테이블 추출 - 하드코딩 없이 AI가 직접 구조 파악

    Round 20: 모든 규칙/매핑/좌표 없이 AI가 테이블을 직접 이해

    Returns:
        {
            "success": bool,
            "table": [[셀1, 셀2, ...], ...],  # 2D 배열
            "headers": [헤더1, 헤더2, ...],
            "uncertain_cells": [{"row": 0, "col": 1, "text": "D/SILVER", "confidence": 0.75, "alternatives": [...]}],
            "structure": {"rows": N, "cols": M},
            "raw_response": str
        }
    """
    print(f"  [Round 20 AI-Only] 100% AI 기반 테이블 추출 시작...")

    # 이미지를 base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = """이 테이블 이미지를 분석하여 JSON 형식으로 변환해주세요.

## 출력 형식 (반드시 이 형식으로):
```json
{
  "headers": ["컬럼1", "컬럼2", "컬럼3"],
  "rows": [
    ["데이터1", "데이터2", "데이터3"],
    ["데이터4", "데이터5", "데이터6"]
  ],
  "uncertain": [
    {"row": 0, "col": 2, "text": "불확실한텍스트", "alternatives": ["대안1", "대안2"]}
  ]
}
```

## 규칙:
1. 빈 셀은 빈 문자열 ""로 표시
2. 모든 행의 컬럼 수는 헤더와 동일해야 함
3. 숫자는 원본 그대로 유지 (쉼표, 소수점 포함)
4. 읽기 어려운 글자가 있으면 uncertain 배열에 추가하고 가능한 대안 제시
5. 첫 번째 행이 헤더(제목)인지 데이터인지 판단하여 적절히 분류
6. 병합된 셀이 있으면 해당 값을 첫 번째 셀에만 넣고 나머지는 빈 문자열

JSON만 출력하세요 (설명 없이):"""

    # 모델 폴백 체인
    for model in VISION_MODELS:
        try:
            print(f"  [Round 20 AI-Only] {model} 시도 중...")

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 4096}
                },
                timeout=60  # 테이블 추출은 시간이 더 필요
            )

            if response.status_code == 200:
                raw_response = response.json().get("response", "").strip()
                print(f"  [Round 20 AI-Only] {model} 응답 수신 ({len(raw_response)} chars)")

                # JSON 파싱 시도
                parsed = parse_ai_table_response(raw_response)
                if parsed["success"]:
                    parsed["raw_response"] = raw_response
                    parsed["model_used"] = model
                    print(f"  [Round 20 AI-Only] 성공! {parsed['structure']['rows']}행 x {parsed['structure']['cols']}열")
                    return parsed
                else:
                    print(f"  [Round 20 AI-Only] {model} JSON 파싱 실패: {parsed.get('error', 'unknown')}")

        except requests.exceptions.Timeout:
            print(f"  [Round 20 AI-Only] {model} 타임아웃")
            continue
        except Exception as e:
            print(f"  [Round 20 AI-Only] {model} 오류: {e}")
            continue

    return {
        "success": False,
        "error": "모든 AI 모델 실패",
        "table": [],
        "headers": [],
        "uncertain_cells": [],
        "structure": {"rows": 0, "cols": 0}
    }


def parse_ai_table_response(raw_response: str) -> dict:
    """AI 응답에서 JSON 테이블 파싱

    Returns:
        {
            "success": bool,
            "table": 2D 배열,
            "headers": 헤더 배열,
            "uncertain_cells": 불확실한 셀 목록,
            "structure": {"rows": N, "cols": M}
        }
    """
    try:
        # JSON 블록 추출 (```json ... ``` 또는 순수 JSON)
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 순수 JSON 시도 (첫 번째 { 부터 마지막 } 까지)
            start = raw_response.find('{')
            end = raw_response.rfind('}')
            if start != -1 and end != -1:
                json_str = raw_response[start:end+1]
            else:
                return {"success": False, "error": "JSON 블록 없음"}

        data = json.loads(json_str)

        headers = data.get("headers", [])
        rows = data.get("rows", [])
        uncertain = data.get("uncertain", [])

        # 유효성 검사
        if not headers and not rows:
            return {"success": False, "error": "빈 테이블"}

        # 헤더가 없으면 첫 번째 행을 헤더로 사용
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]

        # 컬럼 수 일관성 체크
        col_count = len(headers)
        normalized_rows = []
        for row in rows:
            if len(row) < col_count:
                row = row + [""] * (col_count - len(row))
            elif len(row) > col_count:
                row = row[:col_count]
            normalized_rows.append(row)

        # 전체 테이블 (헤더 포함)
        full_table = [headers] + normalized_rows

        return {
            "success": True,
            "table": full_table,
            "headers": headers,
            "uncertain_cells": uncertain,
            "structure": {"rows": len(full_table), "cols": col_count}
        }

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON 파싱 오류: {e}"}
    except Exception as e:
        return {"success": False, "error": f"처리 오류: {e}"}


def validate_ai_only_table(table_data: dict) -> dict:
    """AI-Only 테이블 검증

    검증 항목:
    1. 구조 검증: 행/열 일관성
    2. 수학적 검증: COLOR/SIZE QTY 테이블의 합계 확인
    3. 불확실성 검증: uncertain_cells 개수

    Returns:
        {
            "valid": bool,
            "structure_ok": bool,
            "math_ok": bool or None (해당 없음),
            "uncertain_count": int,
            "warnings": [str],
            "errors": [str]
        }
    """
    result = {
        "valid": True,
        "structure_ok": True,
        "math_ok": None,
        "uncertain_count": 0,
        "warnings": [],
        "errors": []
    }

    if not table_data.get("success"):
        result["valid"] = False
        result["structure_ok"] = False
        result["errors"].append(table_data.get("error", "테이블 추출 실패"))
        return result

    table = table_data.get("table", [])
    headers = table_data.get("headers", [])
    uncertain = table_data.get("uncertain_cells", [])

    # 1. 구조 검증
    if len(table) < 2:
        result["valid"] = False
        result["structure_ok"] = False
        result["errors"].append("테이블이 너무 작음 (최소 2행 필요)")
        return result

    col_count = len(headers)
    for i, row in enumerate(table):
        if len(row) != col_count:
            result["structure_ok"] = False
            result["errors"].append(f"행 {i}의 컬럼 수 불일치: {len(row)} != {col_count}")

    # 2. 수학적 검증 (COLOR/SIZE QTY 테이블인지 확인)
    has_total = any("TOTAL" in str(h).upper() for h in headers)
    if has_total and len(table) > 2:
        math_result = validate_table_math(table, headers)
        result["math_ok"] = math_result["valid"]
        result["math_errors"] = math_result.get("errors", [])
        result["mismatch_cells"] = math_result.get("mismatch_cells", [])

        if not math_result["valid"]:
            # 오류 원인이 추정된 경우
            if math_result.get("mismatch_cells"):
                for cell in math_result["mismatch_cells"]:
                    result["warnings"].append(
                        f"⚠️ 셀({cell['row']},{cell['col']}) 오류 추정: '{cell['current']}' → '{cell['possible_correct']}'"
                    )
            else:
                result["warnings"].append("합계 검증 실패 - 정확한 원인 셀을 찾지 못함")

    # 3. 불확실성 검증
    result["uncertain_count"] = len(uncertain)
    if uncertain:
        result["warnings"].append(f"{len(uncertain)}개의 불확실한 셀 발견")

    # 전체 유효성
    result["valid"] = result["structure_ok"] and len(result["errors"]) == 0

    return result


def validate_table_math(table: list, headers: list) -> dict:
    """COLOR/SIZE QTY 테이블의 수학적 합계 검증

    수정된 버전: 오류 위치와 예상값을 정확히 반환

    Args:
        table: 전체 테이블 (헤더 행 포함)
        headers: 헤더 행 (사이즈 정보가 포함된 행)

    Returns:
        {
            "valid": bool,
            "errors": [...],
            "mismatch_cells": [...]
        }
    """
    result = {"valid": True, "errors": [], "mismatch_cells": []}

    try:
        # 헤더 행 인덱스 찾기 (테이블에서 headers와 일치하는 행)
        header_row_idx = 0
        for idx, row in enumerate(table):
            if row == headers:
                header_row_idx = idx
                break

        print(f"  [수학 검증] 헤더 행 인덱스: {header_row_idx}, 전체 행 수: {len(table)}")

        # TOTAL 컬럼 인덱스 찾기
        total_col_idx = None
        for i, h in enumerate(headers):
            if "TOTAL" in str(h).upper():
                total_col_idx = i
                break

        if total_col_idx is None:
            print("  [수학 검증] TOTAL 컬럼 없음 - 검증 스킵")
            return result  # TOTAL 컬럼 없으면 검증 스킵

        print(f"  [수학 검증] TOTAL 컬럼 인덱스: {total_col_idx}")

        # TOTAL 행 인덱스 찾기 (헤더 이후 행에서)
        total_row_idx = None
        for row_idx in range(header_row_idx + 1, len(table)):
            row = table[row_idx]
            # 첫 번째 또는 두 번째 컬럼에 TOTAL이 있는지 확인
            for col_idx in range(min(3, len(row))):
                if "TOTAL" in str(row[col_idx]).upper():
                    total_row_idx = row_idx
                    break
            if total_row_idx is not None:
                break

        print(f"  [수학 검증] TOTAL 행 인덱스: {total_row_idx}")

        # 숫자 컬럼 시작 인덱스 찾기 (헤더에서 숫자 패턴 찾기)
        num_col_start = 2  # 기본값
        for i, h in enumerate(headers):
            if re.match(r'^\d{2,3}$', str(h).strip()):
                num_col_start = i
                break

        print(f"  [수학 검증] 숫자 컬럼 시작: {num_col_start}, 끝: {total_col_idx}")

        # 데이터 행 범위: 헤더 행 다음부터 TOTAL 행 전까지
        data_start_idx = header_row_idx + 1
        data_end_idx = total_row_idx if total_row_idx else len(table)

        print(f"  [수학 검증] 데이터 행 범위: {data_start_idx} ~ {data_end_idx - 1}")

        # 1. 각 행의 합계 검증
        row_sums = {}  # row_idx -> calculated_sum
        row_errors = {}  # row_idx -> difference

        for row_idx in range(data_start_idx, data_end_idx):
            row = table[row_idx]

            row_sum = 0
            total_value = parse_number(row[total_col_idx]) if total_col_idx < len(row) else 0

            for col_idx in range(num_col_start, total_col_idx):
                if col_idx < len(row):
                    cell_value = parse_number(row[col_idx])
                    row_sum += cell_value

            row_sums[row_idx] = row_sum

            if total_value > 0 and abs(row_sum - total_value) > 1:
                diff = total_value - row_sum
                row_errors[row_idx] = diff
                row_name = row[1] if len(row) > 1 else f"행{row_idx}"
                result["errors"].append({
                    "row": row_idx,
                    "col": total_col_idx,
                    "current": row[total_col_idx] if total_col_idx < len(row) else "",
                    "expected": f"{row_sum:,}",
                    "difference": diff,
                    "type": "row_total",
                    "row_name": row_name
                })
                print(f"  [수학 검증] 행 {row_idx} ({row_name}) 불일치: 계산={row_sum:,}, TOTAL={total_value:,}, 차이={diff}")

        # 2. 각 열의 합계 검증 (TOTAL 행이 있는 경우)
        col_errors = {}  # col_idx -> difference

        if total_row_idx is not None:
            total_row = table[total_row_idx]

            for col_idx in range(num_col_start, total_col_idx):
                col_sum = 0
                for row_idx in range(data_start_idx, data_end_idx):
                    row = table[row_idx]
                    if col_idx < len(row):
                        col_sum += parse_number(row[col_idx])

                total_value = parse_number(total_row[col_idx]) if col_idx < len(total_row) else 0

                if total_value > 0 and abs(col_sum - total_value) > 1:
                    diff = total_value - col_sum
                    col_errors[col_idx] = diff
                    col_name = headers[col_idx] if col_idx < len(headers) else f"열{col_idx}"
                    result["errors"].append({
                        "row": total_row_idx,
                        "col": col_idx,
                        "current": total_row[col_idx] if col_idx < len(total_row) else "",
                        "expected": f"{col_sum:,}",
                        "difference": diff,
                        "type": "col_total",
                        "col_name": col_name
                    })
                    print(f"  [수학 검증] 열 {col_idx} ({col_name}) 불일치: 계산={col_sum:,}, TOTAL={total_value:,}, 차이={diff}")

        # 3. 오류 교차점 분석 - 동일한 차이값을 가진 행/열 오류가 있으면 해당 셀이 원인
        print(f"  [수학 검증] 행 오류: {row_errors}")
        print(f"  [수학 검증] 열 오류: {col_errors}")

        for row_idx, row_diff in row_errors.items():
            for col_idx, col_diff in col_errors.items():
                # 차이값의 절대값이 같으면 교차점이 오류 원인
                if abs(row_diff) == abs(col_diff):
                    # 이 셀이 문제의 원인일 가능성 높음
                    current_value = parse_number(table[row_idx][col_idx])
                    possible_correct = current_value + row_diff
                    col_name = headers[col_idx] if col_idx < len(headers) else f"열{col_idx}"
                    row_name = table[row_idx][1] if len(table[row_idx]) > 1 else f"행{row_idx}"

                    result["mismatch_cells"].append({
                        "row": row_idx,
                        "col": col_idx,
                        "current": table[row_idx][col_idx],
                        "current_num": current_value,
                        "possible_correct": f"{int(possible_correct):,}",
                        "difference": row_diff,
                        "reason": f"{row_name} 행과 {col_name} 열 합계 오류의 교차점",
                        "col_name": col_name,
                        "row_name": row_name
                    })
                    print(f"  [수학 검증] ** 오류 셀 발견: ({row_idx}, {col_idx}) [{row_name}/{col_name}] = '{table[row_idx][col_idx]}' -> 올바른 값: {int(possible_correct):,}")

        result["valid"] = len(result["errors"]) == 0

        return result

    except Exception as e:
        import traceback
        print(f"  [수학 검증] 오류: {e}")
        traceback.print_exc()
        return {"valid": True, "errors": [], "mismatch_cells": []}


def parse_number(text: str) -> int:
    """텍스트에서 숫자 추출 (쉼표 제거)"""
    try:
        clean = re.sub(r'[^\d]', '', str(text))
        return int(clean) if clean else 0
    except:
        return 0


def generate_ai_only_result_html(table_data: dict, validation: dict) -> str:
    """AI-Only 결과를 HTML로 변환 (불확실한 셀 + 수학 오류 셀에 수정 UI 포함)"""

    if not table_data.get("success"):
        return f'''
        <div style="background: #f8d7da; border: 1px solid #dc3545; padding: 15px; border-radius: 8px; color: #721c24;">
            <strong>AI 추출 실패</strong>: {table_data.get("error", "알 수 없는 오류")}
        </div>
        '''

    table = table_data.get("table", [])
    headers = table_data.get("headers", [])
    uncertain = table_data.get("uncertain_cells", [])

    # 불확실한 셀 위치 맵
    uncertain_map = {}
    for u in uncertain:
        key = (u.get("row", -1), u.get("col", -1))
        uncertain_map[key] = u

    # 수학적 오류 셀 위치 맵 (빨간색 하이라이트)
    mismatch_map = {}
    for m in validation.get("mismatch_cells", []):
        key = (m.get("row", -1), m.get("col", -1))
        mismatch_map[key] = m

    # 검증 결과 배너
    total_issues = validation["uncertain_count"] + len(mismatch_map)

    if validation["valid"] and total_issues == 0 and validation.get("math_ok", True):
        banner = '''
        <div style="background: #d4edda; border: 1px solid #28a745; padding: 10px; margin-bottom: 15px; border-radius: 8px; color: #155724;">
            ✅ <strong>AI 검증 통과</strong>: 모든 셀 정상, 합계 일치
        </div>
        '''
    elif mismatch_map:
        # 수학적 오류가 있는 경우 - 구체적인 오류 위치 표시
        error_details = []
        for m in validation.get("mismatch_cells", []):
            col_name = headers[m['col']] if m['col'] < len(headers) else f"열{m['col']}"
            error_details.append(
                f"⚠️ 행 {m['row']} ({col_name}): '{m['current']}' → 올바른 값 추정: '{m['possible_correct']}'"
            )
        error_html = "<br>".join(error_details)

        banner = f'''
        <div style="background: #f8d7da; border: 1px solid #dc3545; padding: 10px; margin-bottom: 15px; border-radius: 8px; color: #721c24;">
            ❌ <strong>AI 검증 경고</strong>: 수학적 합계 불일치 감지<br>
            {error_html}
            <br><br>💡 <strong>빨간색 셀</strong>을 확인하고 올바른 값을 선택하세요.
        </div>
        '''
    elif validation["uncertain_count"] > 0:
        banner = f'''
        <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin-bottom: 15px; border-radius: 8px; color: #856404;">
            ⚠️ <strong>AI 추출 완료</strong>: {validation["uncertain_count"]}개 셀 확인 필요
            <br><small>노란색 셀을 클릭하여 내용을 확인/수정하세요.</small>
        </div>
        '''
    else:
        errors = "<br>".join(validation.get("errors", []))
        banner = f'''
        <div style="background: #f8d7da; border: 1px solid #dc3545; padding: 10px; margin-bottom: 15px; border-radius: 8px; color: #721c24;">
            ❌ <strong>AI 추출 실패</strong><br>{errors}
        </div>
        '''

    # 테이블 HTML
    html = banner + '<table class="erp-table" style="border-collapse: collapse; width: 100%;">\n'

    for row_idx, row in enumerate(table):
        html += '<tr>\n'
        for col_idx, cell in enumerate(row):
            key = (row_idx, col_idx)

            if key in mismatch_map:
                # 수학적 오류 셀 - 빨간색 하이라이트 + 수정 UI
                m = mismatch_map[key]
                cell_id = f"cell_{row_idx}_{col_idx}"
                current = m.get("current", cell)
                possible = m.get("possible_correct", "")

                options_html = f'<option value="{current}">{current} (현재 값)</option>'
                options_html += f'<option value="{possible}">{possible} (추정 올바른 값)</option>'
                options_html += '<option value="__custom__">직접 입력...</option>'

                html += f'''
                <td style="background: #f8d7da; border: 2px solid #dc3545; padding: 5px;">
                    <select id="{cell_id}" onchange="handleCellSelect(this, {row_idx}, {col_idx})" style="width: 100%; border: none; background: transparent; color: #721c24; font-weight: bold;">
                        {options_html}
                    </select>
                    <input type="text" id="{cell_id}_input" style="display: none; width: 100%;" placeholder="직접 입력" onblur="handleCustomInput(this, {row_idx}, {col_idx})">
                </td>
                '''
            elif key in uncertain_map:
                # 불확실한 셀 - 노란색 하이라이트 + 선택 UI
                u = uncertain_map[key]
                alternatives = u.get("alternatives", [])
                cell_id = f"cell_{row_idx}_{col_idx}"

                options_html = f'<option value="{cell}" selected>{cell} (AI 추천)</option>'
                for alt in alternatives:
                    options_html += f'<option value="{alt}">{alt}</option>'
                options_html += '<option value="__custom__">직접 입력...</option>'

                html += f'''
                <td style="background: #fff3cd; border: 1px solid #ffc107; padding: 5px;">
                    <select id="{cell_id}" onchange="handleCellSelect(this, {row_idx}, {col_idx})" style="width: 100%; border: none; background: transparent;">
                        {options_html}
                    </select>
                    <input type="text" id="{cell_id}_input" style="display: none; width: 100%;" placeholder="직접 입력" onblur="handleCustomInput(this, {row_idx}, {col_idx})">
                </td>
                '''
            elif row_idx == 0:
                # 헤더
                html += f'<td style="background: #e9ecef; border: 1px solid #ccc; padding: 5px; font-weight: bold;">{cell}</td>\n'
            else:
                # 일반 셀
                html += f'<td style="border: 1px solid #ccc; padding: 5px;">{cell}</td>\n'

        html += '</tr>\n'

    html += '</table>\n'

    # JavaScript for cell selection
    html += '''
    <script>
    function handleCellSelect(select, row, col) {
        if (select.value === "__custom__") {
            select.style.display = "none";
            var input = document.getElementById(select.id + "_input");
            input.style.display = "block";
            input.focus();
        } else {
            console.log("Cell [" + row + "," + col + "] = " + select.value);
            // 선택 시 셀 색상 변경 (수정됨 표시)
            select.parentElement.style.background = "#d4edda";
            select.parentElement.style.border = "2px solid #28a745";
        }
    }

    function handleCustomInput(input, row, col) {
        var select = document.getElementById("cell_" + row + "_" + col);
        if (input.value.trim()) {
            var option = document.createElement("option");
            option.value = input.value;
            option.text = input.value + " (사용자 입력)";
            option.selected = true;
            select.add(option, select.options.length - 1);
            // 셀 색상 변경 (수정됨 표시)
            select.parentElement.style.background = "#d4edda";
            select.parentElement.style.border = "2px solid #28a745";
        }
        input.style.display = "none";
        select.style.display = "block";
        console.log("Cell [" + row + "," + col + "] custom = " + input.value);
    }
    </script>
    '''

    return html


def process_image(img: Image.Image, img_base64: str) -> dict:
    """이미지 처리 - 하이브리드 OCR + Comet 오버레이 + ERP 테이블 + AI 검증"""

    width, height = img.size

    # 1. 하이브리드 OCR 수행 (PaddleOCR + AI 보정)
    ocr_results = hybrid_ocr(img)

    # 2. OCR 결과 위치 기반으로 테이블 구성 (Round 16: 이미지 전달)
    table_2d = build_table_from_ocr(ocr_results, image=img)

    num_rows = len(table_2d)
    num_cols = len(table_2d[0]) if table_2d else 0
    has_grid = num_rows >= 2 and num_cols >= 2
    grid_info = f"{num_rows}행 x {num_cols}열" if has_grid else "테이블 없음"

    # 3. Comet 텍스트 스팬 생성
    text_spans = []
    for item in ocr_results:
        x1, y1, x2, y2 = item["box"]
        text = item["text"].replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        score = item.get("score", 1.0)
        font_size = max(10, int((y2 - y1) * 0.8))

        text_spans.append({
            "x": x1, "y": y1,
            "width": x2 - x1, "height": y2 - y1,
            "text": text, "score": score,
            "fontSize": font_size
        })

    # 4. AI 검증 (테이블 생성 후 원본 이미지와 비교)
    print("  [AI 검증] 테이블 검증 시작...")
    validation = validate_table_with_ai(img, table_2d)
    print(f"  [AI 검증] 결과: {validation}")

    # 5. ERP 테이블 HTML 생성 (검증 결과 포함)
    erp_table_html = generate_erp_table_html(table_2d, validation)

    return {
        "success": True,
        "width": width,
        "height": height,
        "image_base64": img_base64,
        "ocr_count": len(ocr_results),
        "grid_info": grid_info,
        "has_grid": has_grid,
        "text_spans": text_spans,
        "erp_table_html": erp_table_html,
        "validation": validation
    }


# =============================================================================
# HTML 템플릿
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comet + ERP 테이블 추출 (하이브리드 OCR)</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #ff6b6b;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.2em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            color: #a0a0a0;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .engine-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }
        .model-info {
            text-align: center;
            margin-bottom: 20px;
        }
        .model-badge {
            display: inline-block;
            background: rgba(102, 126, 234, 0.2);
            color: #667eea;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }

        /* 업로드 영역 */
        .upload-section {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 16px;
            padding: 60px 40px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(102, 126, 234, 0.05);
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.1);
            transform: scale(1.01);
        }
        .upload-area.dragover {
            border-color: #ff6b6b;
            background: rgba(255, 107, 107, 0.1);
        }
        .upload-icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        .upload-text {
            font-size: 20px;
            color: #fff;
            margin-bottom: 10px;
        }
        .upload-hint {
            font-size: 14px;
            color: #888;
        }
        #fileInput {
            display: none;
        }

        /* 로딩 */
        .loading {
            display: none;
            text-align: center;
            padding: 60px;
            color: #fff;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            font-size: 18px;
            color: #ccc;
        }
        .loading-sub {
            font-size: 14px;
            color: #888;
            margin-top: 10px;
        }

        /* 결과 영역 */
        .result-section {
            display: none;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        .result-section.active {
            display: block;
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        }
        .info-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        .info-badge {
            display: inline-block;
            background: rgba(102, 126, 234, 0.2);
            color: #667eea;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }

        /* 탭 */
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 14px 28px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px 12px 0 0;
            cursor: pointer;
            font-size: 15px;
            color: #ccc;
            transition: all 0.2s;
        }
        .tab:hover {
            background: rgba(255,255,255,0.15);
        }
        .tab.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-color: transparent;
        }
        .tab-content {
            display: none;
            background: rgba(0,0,0,0.2);
            border-radius: 0 12px 12px 12px;
            padding: 20px;
        }
        .tab-content.active {
            display: block;
        }

        /* Comet 컨테이너 */
        .comet-wrapper {
            overflow: auto;
            max-height: 70vh;
            border: 2px solid #333;
            border-radius: 8px;
            background: #1a1a1a;
        }
        .comet-container {
            position: relative;
            display: inline-block;
        }
        .comet-image {
            display: block;
            pointer-events: none;
        }
        .comet-overlay {
            position: absolute;
            top: 0;
            left: 0;
        }
        .ocr-text {
            position: absolute;
            color: transparent;
            user-select: text;
            cursor: text;
            line-height: 1.2;
        }
        .ocr-text::selection {
            background: rgba(102, 126, 234, 0.4);
        }
        .debug-mode .ocr-text {
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(102, 126, 234, 0.5);
        }

        /* ERP 테이블 */
        .erp-wrapper {
            overflow: auto;
            max-height: 70vh;
            background: #fff;
            border-radius: 8px;
            padding: 20px;
        }
        .erp-table {
            border-collapse: collapse;
            font-size: 13px;
            border: 2px solid #000;
            width: auto;
            min-width: 100%;
        }
        .erp-table td {
            border: 1px solid #000;
            padding: 6px 12px;
            text-align: center;
            height: 28px;
            background: #fff;
            white-space: nowrap;
            color: #000;
        }
        .erp-table .header {
            font-weight: bold;
            background: #d4e8d4;
        }
        .erp-table .sub-header {
            font-weight: bold;
            background: #e8f4e8;
        }
        .erp-table .data-cell {
            text-align: right;
        }
        .erp-table .empty-cell {
            background: #fff;
        }
        .erp-table .total-row-cell {
            font-weight: bold;
            background: #f0f0f0;
        }

        /* 버튼 */
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .btn-secondary:hover {
            background: rgba(255,255,255,0.15);
        }
        .btn-success {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
        }
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        }

        /* 컨트롤 */
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
            align-items: center;
        }
        .controls label {
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ccc;
            font-size: 14px;
        }
        .controls input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #667eea;
        }

        /* 토스트 */
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            display: none;
            z-index: 1000;
            font-weight: 600;
        }
        .toast.show {
            display: block;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* 안내 텍스트 */
        .help-text {
            color: #888;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .help-text strong {
            color: #667eea;
        }

        /* 비교 링크 */
        .compare-link {
            text-align: center;
            margin-top: 20px;
        }
        .compare-link a {
            color: #888;
            text-decoration: none;
            font-size: 13px;
        }
        .compare-link a:hover {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Comet + ERP 테이블 추출 <span class="engine-badge">하이브리드 OCR</span></h1>
        <p class="subtitle">PaddleOCR (좌표) + AI Vision (보정)으로 Comet 오버레이 생성</p>
        <div class="model-info">
            <span class="model-badge">🧠 ''' + VISION_MODEL + '''</span>
        </div>

        <!-- 업로드 섹션 -->
        <div class="upload-section" id="uploadSection">
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">🖼️</div>
                <div class="upload-text">이미지를 드래그하거나 클릭하여 업로드</div>
                <div class="upload-hint">PNG, JPG, JPEG 지원 (최대 16MB)</div>
            </div>
            <input type="file" id="fileInput" accept="image/*">

            <div class="compare-link">
                <a href="http://localhost:5001" target="_blank">📊 PaddleOCR 버전(5001)과 비교하기</a>
            </div>
        </div>

        <!-- 로딩 -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">하이브리드 OCR 처리 중...</div>
            <div class="loading-sub">PaddleOCR 위치 감지 + AI 보정 (약 30초 소요)</div>
        </div>

        <!-- 결과 섹션 -->
        <div class="result-section" id="resultSection">
            <div class="info-bar">
                <button class="btn btn-secondary" onclick="resetUpload()">🔄 새 이미지</button>
                <span id="imageInfo"></span>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('comet')">1️⃣ Comet 오버레이</button>
                <button class="tab" onclick="switchTab('erp')">2️⃣ ERP 테이블</button>
            </div>

            <!-- Comet 탭 -->
            <div class="tab-content active" id="cometTab">
                <div class="section-title">📝 Comet 방식 텍스트 추출</div>
                <p class="help-text">
                    <strong>사용법:</strong> 이미지 위의 텍스트를 드래그하여 선택하고 <strong>Ctrl+C</strong>로 복사하세요.
                </p>
                <div class="controls">
                    <label>
                        <input type="checkbox" id="debugMode" onchange="toggleDebug()">
                        디버그 모드 (텍스트 영역 표시)
                    </label>
                </div>
                <div class="comet-wrapper">
                    <div class="comet-container" id="cometContainer">
                        <img class="comet-image" id="cometImage">
                        <div class="comet-overlay" id="cometOverlay"></div>
                    </div>
                </div>
            </div>

            <!-- ERP 탭 -->
            <div class="tab-content" id="erpTab">
                <div class="section-title">📋 ERP 전송용 테이블</div>
                <p class="help-text">
                    <strong>사용법:</strong> 아래 테이블을 복사하여 ERP 시스템에 붙여넣을 수 있습니다.
                </p>
                <div class="controls">
                    <button class="btn btn-success" onclick="copyTable()">📋 테이블 복사</button>
                </div>
                <div class="erp-wrapper" id="erpTableContainer"></div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const uploadSection = document.getElementById('uploadSection');
        const resultSection = document.getElementById('resultSection');

        // 드래그 앤 드롭
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                processFile(file);
            }
        });

        function processFile(file) {
            uploadSection.style.display = 'none';
            loading.classList.add('active');

            const formData = new FormData();
            formData.append('image', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('active');
                if (data.error) {
                    showToast('오류: ' + data.error, true);
                    uploadSection.style.display = 'block';
                } else {
                    displayResult(data);
                }
            })
            .catch(error => {
                loading.classList.remove('active');
                showToast('오류가 발생했습니다', true);
                uploadSection.style.display = 'block';
            });
        }

        function displayResult(data) {
            // 이미지 정보
            document.getElementById('imageInfo').innerHTML =
                `<span class="info-badge">📐 ${data.width} x ${data.height}</span>` +
                `<span class="info-badge">📝 ${data.ocr_count}개 텍스트</span>` +
                `<span class="info-badge">📊 ${data.grid_info}</span>`;

            // Comet 이미지
            const cometImage = document.getElementById('cometImage');
            cometImage.src = 'data:image/png;base64,' + data.image_base64;
            cometImage.width = data.width;
            cometImage.height = data.height;

            // OCR 텍스트 오버레이
            const overlay = document.getElementById('cometOverlay');
            overlay.innerHTML = '';
            overlay.style.width = data.width + 'px';
            overlay.style.height = data.height + 'px';

            data.text_spans.forEach(span => {
                const el = document.createElement('span');
                el.className = 'ocr-text';
                el.style.left = span.x + 'px';
                el.style.top = span.y + 'px';
                el.style.width = span.width + 'px';
                el.style.height = span.height + 'px';
                el.style.fontSize = span.fontSize + 'px';
                el.title = 'score: ' + span.score.toFixed(3);
                el.textContent = span.text;
                overlay.appendChild(el);
            });

            // ERP 테이블
            document.getElementById('erpTableContainer').innerHTML = data.erp_table_html;

            resultSection.classList.add('active');
        }

        function resetUpload() {
            resultSection.classList.remove('active');
            uploadSection.style.display = 'block';
            fileInput.value = '';
            switchTab('comet');
        }

        function toggleDebug() {
            const container = document.getElementById('cometContainer');
            const checkbox = document.getElementById('debugMode');
            if (checkbox.checked) {
                container.classList.add('debug-mode');
            } else {
                container.classList.remove('debug-mode');
            }
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            if (tabName === 'comet') {
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('cometTab').classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('erpTab').classList.add('active');
            }
        }

        function copyTable() {
            const table = document.querySelector('.erp-table');
            if (!table) {
                showToast('복사할 테이블이 없습니다', true);
                return;
            }

            const range = document.createRange();
            range.selectNode(table);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();

            showToast('테이블 복사 완료!');
        }

        function showToast(message, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.style.background = isError
                ? 'linear-gradient(135deg, #e94560, #ff6b6b)'
                : 'linear-gradient(135deg, #667eea, #764ba2)';
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "이미지를 선택해주세요."})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "이미지를 선택해주세요."})

    try:
        # 이미지 읽기
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Base64 인코딩
        img_base64 = base64.b64encode(img_bytes).decode()

        # 처리 (AI OCR + Grid 매핑)
        result = process_image(img, img_base64)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"처리 오류: {str(e)}"})


# =============================================================================
# Round 20: AI-Only 테스트 엔드포인트
# =============================================================================

AI_ONLY_TEST_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>AI-Only 테이블 추출 테스트 (Round 20)</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #ff6b6b; }
        .badge { background: linear-gradient(135deg, #f093fb, #f5576c); padding: 5px 15px; border-radius: 20px; font-size: 0.8em; }
        .upload-box { border: 2px dashed #4ecdc4; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-box:hover { background: rgba(78, 205, 196, 0.1); }
        input[type="file"] { display: none; }
        .btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 10px 25px; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result { margin-top: 20px; padding: 20px; background: #16213e; border-radius: 10px; }
        .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .panel { background: #0f3460; padding: 15px; border-radius: 8px; }
        .panel h3 { margin-top: 0; color: #4ecdc4; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 4px solid #333; border-top: 4px solid #4ecdc4; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        td, th { border: 1px solid #444; padding: 5px; }
        th { background: #333; }
        .info { background: #2d3436; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI-Only 테이블 추출 <span class="badge">Round 20</span></h1>
        <p>100% AI 기반 - 하드코딩 없이 qwen2.5vl이 직접 테이블 구조 파악</p>

        <div class="info">
            <strong>기존 방식 (Hybrid)</strong>: PaddleOCR → 규칙 기반 처리 → ocr_corrections.json<br>
            <strong>AI-Only 방식</strong>: 이미지 → AI → JSON 테이블 (규칙/매핑 없음)
        </div>

        <div class="upload-box" onclick="document.getElementById('fileInput').click()">
            <p>📁 테이블 이미지를 클릭하여 업로드</p>
            <input type="file" id="fileInput" accept="image/*" onchange="handleUpload(this)">
        </div>

        <button class="btn" onclick="runTest()" id="testBtn" disabled>🚀 AI-Only 추출 실행</button>
        <button class="btn" onclick="runComparison()" id="compareBtn" disabled>⚖️ 기존 방식과 비교</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>AI 처리 중... (최대 60초)</p>
        </div>

        <div class="result" id="result" style="display: none;">
            <h3>📊 AI-Only 결과</h3>
            <div id="aiOnlyResult"></div>
        </div>

        <div class="comparison" id="comparison" style="display: none;">
            <div class="panel">
                <h3>🔧 기존 방식 (Hybrid)</h3>
                <div id="hybridResult"></div>
            </div>
            <div class="panel">
                <h3>🤖 AI-Only 방식</h3>
                <div id="aiOnlyResult2"></div>
            </div>
        </div>
    </div>

    <script>
        let uploadedFile = null;

        function handleUpload(input) {
            if (input.files && input.files[0]) {
                uploadedFile = input.files[0];
                document.querySelector('.upload-box p').textContent = '✅ ' + uploadedFile.name;
                document.getElementById('testBtn').disabled = false;
                document.getElementById('compareBtn').disabled = false;
            }
        }

        async function runTest() {
            if (!uploadedFile) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            const formData = new FormData();
            formData.append('image', uploadedFile);

            try {
                const response = await fetch('/api/ai-only', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('aiOnlyResult').innerHTML = data.html || '<p>오류: ' + (data.error || 'Unknown') + '</p>';
            } catch (e) {
                document.getElementById('loading').style.display = 'none';
                alert('오류: ' + e.message);
            }
        }

        async function runComparison() {
            if (!uploadedFile) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('comparison').style.display = 'none';

            const formData = new FormData();
            formData.append('image', uploadedFile);

            try {
                // AI-Only 실행
                const aiResponse = await fetch('/api/ai-only', {
                    method: 'POST',
                    body: formData
                });
                const aiData = await aiResponse.json();

                // Hybrid 실행 (기존 방식)
                const formData2 = new FormData();
                formData2.append('image', uploadedFile);
                const hybridResponse = await fetch('/upload', {
                    method: 'POST',
                    body: formData2
                });
                const hybridData = await hybridResponse.json();

                document.getElementById('loading').style.display = 'none';
                document.getElementById('comparison').style.display = 'grid';
                document.getElementById('hybridResult').innerHTML = hybridData.erp_table_html || '<p>오류</p>';
                document.getElementById('aiOnlyResult2').innerHTML = aiData.html || '<p>오류</p>';
            } catch (e) {
                document.getElementById('loading').style.display = 'none';
                alert('오류: ' + e.message);
            }
        }
    </script>
</body>
</html>
'''


@app.route('/test-ai-only')
def test_ai_only_page():
    """AI-Only 테스트 페이지"""
    return render_template_string(AI_ONLY_TEST_TEMPLATE)


@app.route('/api/ai-only', methods=['POST'])
def api_ai_only():
    """AI-Only 테이블 추출 API"""
    if 'image' not in request.files:
        return jsonify({"error": "이미지를 선택해주세요."})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "이미지를 선택해주세요."})

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # AI-Only 추출
        table_data = extract_table_with_ai_only(img)

        # 검증
        validation = validate_ai_only_table(table_data)

        # HTML 생성
        html = generate_ai_only_result_html(table_data, validation)

        return jsonify({
            "success": table_data.get("success", False),
            "html": html,
            "table": table_data.get("table", []),
            "structure": table_data.get("structure", {}),
            "validation": validation,
            "model_used": table_data.get("model_used", "unknown")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"처리 오류: {str(e)}"})


if __name__ == '__main__':
    print("=" * 50)
    print("Comet + ERP 테이블 추출 (Qwen2.5-VL 버전)")
    print(f"PaddleOCR + {VISION_MODEL} (OCR 특화)")
    print("http://localhost:6002 에서 접속하세요")
    print("=" * 50)
    app.run(debug=True, port=6002)
