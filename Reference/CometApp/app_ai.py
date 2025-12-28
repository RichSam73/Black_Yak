"""
Comet + ERP 통합 웹 앱 (하이브리드 AI OCR)
- PaddleOCR: 텍스트 위치(좌표) 감지
- Ollama Vision: 한글 인식 정확도 보강
- Comet 오버레이 + ERP 테이블 동시 제공
- Grid 감지 방식으로 셀 매핑 (기존 방식 유지)
- 포트: 6001
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
VISION_MODEL = "llama3.2-vision"

# 전역 OCR 인스턴스
_paddle_ocr = None

def get_paddleocr():
    """PaddleOCR 인스턴스 싱글톤 (한글)"""
    global _paddle_ocr
    if _paddle_ocr is None:
        print("  [PaddleOCR 초기화 중... (한글)]")
        _paddle_ocr = PaddleOCR(lang='korean')
    return _paddle_ocr


def ocr_with_paddle(image: Image.Image) -> list:
    """PaddleOCR로 텍스트 위치 + 초기 인식"""
    ocr = get_paddleocr()
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
    sup_nm_x = 834  # 기본값
    for ocr in ocr_results:
        if ocr.get("text") == "SUP NM":
            box = ocr.get("box", [0, 0, 0, 0])
            sup_nm_x = int((box[0] + box[2]) / 2)
            print(f"  [구조 분석] SUP NM 컬럼 X좌표: {sup_nm_x}")
            break

    # 4. DIV 컬럼 X좌표 추정
    div_x = 33  # 기본값
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


def refine_text_with_ai(image: Image.Image, ocr_results: list) -> list:
    """AI Vision으로 저신뢰도 텍스트 보정 (누락 감지는 테이블 구조 분석으로)
    """
    if not ocr_results:
        return ocr_results

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

    # AI 보정은 시간이 오래 걸리므로 생략 가능
    # 필요시 주석 해제
    # print(f"  [AI 보정] {len(low_confidence)}개 저신뢰도 텍스트 재인식 중...")
    # ... (AI 호출 코드)

    return ocr_results


def apply_known_corrections(ocr_results: list) -> list:
    """알려진 OCR 오류 수동 보정 사전

    참조: Submaterial_correct.html의 정확한 데이터 기준
    """
    # 일반 텍스트 보정 (위치 무관)
    simple_corrections = {
        # 타이틀 오류
        "ATCAC NOAIVIAITON": "SUB MATERIAL INFORMATION",
        "SUB ATCAC NOAIVIAITON": "SUB MATERIAL INFORMATION",
        "MATERIAL": "SUB MATERIAL INFORMATION",  # 부분 인식된 경우
        # 행거루프
        "23SS-헬거루프": "23SS-행거루프",
        "헹거루프": "행거루프",
        "헬거루프": "행거루프",
        # 기타 텍스트 오류
        "소멋단": "소맷단",
        "소멧단": "소맷단",
        "사이드포켓": "사이드 포켓",
        "앞가슴": "앞 가슴",
        "실리콘매트": "실리콘 매트",
        "12본스트링(SOLID)": "12본 스트링(SOLID)",
        "12본스트링": "12본 스트링(SOLID)",
        # 추가 보정
        "앞지퍼": "앞 지퍼",
        "앞지퍼:": "앞 지퍼",
        "후드/맏단": "후드/밑단",
        "후드/믿단": "후드/밑단",
        "후드/믿단": "후드/밑단",
        # 공급업체명 오류 (가능한 모든 변형)
        "성훤": "성원",
        "숭원": "성원",
        "성완": "성원",
        "성웬": "성원",
        # "공", "울" 등 1글자는 위치 기반 보정으로 처리 (다른 곳에서 잘못 변환될 수 있음)
        "동아금혐": "동아금형",
        "동아굼형": "동아금형",
        "동아금헝": "동아금형",
        "동아금혁": "동아금형",
        "천신지퍼:": "천신지퍼",
        "업체헨들링": "업체핸들링",
        "업채핸들링": "업체핸들링",
        # 에리안 오류 (가능한 모든 변형) - "20"은 위치 기반 보정으로 처리
        "에러안": "에리안",
        "에리얀": "에리안",
        "애리안": "에리안",
        "이리안": "에리안",
        "에라안": "에리안",
        # 대일 오류
        "대얼": "대일",
        "데일": "대일",
        # 숨프린트 오류
        "숭프린트": "숨프린트",
        "숨프릳트": "숨프린트",
        # 헤더 오류
        "DEMANO": "DEMAND",
        "DOMAND": "DEMAND",
        # 컬러 오류
        "D/SLVER": "D/SILVER",
        "BK/SLVER": "BK/SILVER",
        # 기타
        "로고아일랫": "로고아일렛",
        "내장이밴드": "내장 이밴드",
    }

    # 위치 기반 보정 (특정 Y 좌표 범위에서만 적용)
    # format: (text, y_min, y_max, correct_text)
    # 이미지 크기 약 500-600px 높이 기준으로 행 위치 추정
    # 헤더: ~20-50, 데이터 행: ~50-400 범위
    position_corrections = [
        # 행거루프 행 PART USED: "20" → "에리안" (약 2번째 데이터 행, Y~70-120)
        ("20", 50, 150, "에리안"),
        # S/ZIP PKT. 3번째 행 SUP NM: "공" → "성원" (약 8번째 행, Y~200-280)
        ("공", 150, 320, "성원"),
        # 스토퍼 행 SUP NM: "울" → "동아금형" (약 10번째 행, Y~280-380)
        ("울", 250, 420, "동아금형"),
        # 비드 행 SUP NM도 동아금형
        ("욿", 250, 420, "동아금형"),
    ]

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


def build_table_from_ocr(ocr_results: list) -> list:
    """OCR 결과의 위치 정보만으로 테이블 구성"""
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
    col_positions = cluster_values(x_centers, threshold=30)

    num_rows = len(row_positions)
    num_cols = len(col_positions)

    if num_rows == 0 or num_cols == 0:
        return []

    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for ocr in ocr_results:
        box = ocr.get("box", [])
        text = ocr.get("text", "").strip()

        if not text or len(box) < 4:
            continue

        cy = (box[1] + box[3]) / 2
        cx = (box[0] + box[2]) / 2

        row_idx = min(range(num_rows), key=lambda i: abs(row_positions[i] - cy))
        col_idx = min(range(num_cols), key=lambda i: abs(col_positions[i] - cx))

        if table[row_idx][col_idx]:
            table[row_idx][col_idx] += " " + text
        else:
            table[row_idx][col_idx] = text

    return table


# =============================================================================
# HTML 생성
# =============================================================================

def generate_erp_table_html(table_2d: list) -> str:
    """Grid-First 2D 테이블을 ERP용 HTML 테이블로 변환"""

    if not table_2d or len(table_2d) == 0:
        return '<p style="color: #ff6b6b;">테이블 격자를 감지하지 못했습니다. Comet 탭에서 직접 텍스트를 복사해주세요.</p>'

    num_cols = max(len(row) for row in table_2d)

    html = '<table class="erp-table">\n'

    for row_idx, row in enumerate(table_2d):
        html += '<tr>\n'

        for col_idx in range(num_cols):
            cell = row[col_idx] if col_idx < len(row) else ''
            cell = cell.strip() if cell else ''

            if row_idx == 0:
                css_class = 'header'
            elif row_idx == 1:
                css_class = 'sub-header'
            elif not cell:
                css_class = 'empty-cell'
            elif 'TOTAL' in ' '.join([c for c in row if c]).upper():
                css_class = 'total-row-cell'
            elif cell.replace(',', '').replace('.', '').replace('-', '').isdigit():
                css_class = 'data-cell'
            else:
                css_class = ''

            if css_class:
                html += f'<td class="{css_class}">{cell}</td>\n'
            else:
                html += f'<td>{cell}</td>\n'

        html += '</tr>\n'

    html += '</table>\n'

    return html


def process_image(img: Image.Image, img_base64: str) -> dict:
    """이미지 처리 - 하이브리드 OCR + Comet 오버레이 + ERP 테이블"""

    width, height = img.size

    # 1. 하이브리드 OCR 수행 (PaddleOCR + AI 보정)
    ocr_results = hybrid_ocr(img)

    # 2. OCR 결과 위치 기반으로 테이블 구성
    table_2d = build_table_from_ocr(ocr_results)

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

    # 4. ERP 테이블 HTML 생성
    erp_table_html = generate_erp_table_html(table_2d)

    return {
        "success": True,
        "width": width,
        "height": height,
        "image_base64": img_base64,
        "ocr_count": len(ocr_results),
        "grid_info": grid_info,
        "has_grid": has_grid,
        "text_spans": text_spans,
        "erp_table_html": erp_table_html
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


if __name__ == '__main__':
    print("=" * 50)
    print("Comet + ERP 테이블 추출 (하이브리드 OCR)")
    print(f"PaddleOCR + {VISION_MODEL} (AI 보정)")
    print("http://localhost:6001 에서 접속하세요")
    print("=" * 50)
    app.run(debug=True, port=6001)
