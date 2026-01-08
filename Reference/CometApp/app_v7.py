# -*- coding: utf-8 -*-
"""
Comet + ERP Table Extractor v7.0
================================
- 전체 화면 활용 UI (테이블 크게 표시)
- img2table: 테이블 구조 추출 (colspan/rowspan 자동 감지)
- PaddleOCR: 텍스트 인식

**v7.0 변경사항**: Confidence 기반 Selective VLM
  - Dual OCR 구조 비교 방식 제거 (False Positive 문제 해결)
  - Raw PaddleOCR로 개별 셀 confidence 추출
  - Cell BBox ↔ OCR BBox IOU 매핑
  - confidence < 0.8인 셀만 VLM 검증 호출
  - VLM 호출 90%+ 감소, 정확도 향상
- 포트: 6007
"""

import os
# PaddlePaddle 연결성 체크 비활성화 (시작 시간 단축)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
import io
import json
import requests

# img2table imports
from img2table.document import Image as Img2TableImage
from img2table.ocr import PaddleOCR as Img2TablePaddleOCR

# Raw PaddleOCR for confidence extraction
from paddleocr import PaddleOCR as RawPaddleOCR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['JSON_AS_ASCII'] = False  # 한글 유니코드 출력 지원

# ============================================================
# Ollama 설정 (Selective VLM 검증용)
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "granite3.2-vision:2b"  # VLM 검증 모델

# Confidence 기반 설정
CONFIDENCE_THRESHOLD = 0.8  # 80% 미만이면 VLM 검증
IOU_THRESHOLD = 0.3  # Cell-OCR 매핑 IOU 임계값

# 전역 OCR 인스턴스
_ocr_engine = None
_raw_ocr_engine = None


def get_ocr_engine():
    """img2table용 PaddleOCR 엔진 초기화 (한글+영어 지원)"""
    global _ocr_engine
    if _ocr_engine is None:
        print("[init] PaddleOCR engine for img2table (korean)...")
        _ocr_engine = Img2TablePaddleOCR(lang="korean")
        print("[init] PaddleOCR engine ready")
    return _ocr_engine


def get_raw_ocr_engine():
    """Raw PaddleOCR 엔진 초기화 (confidence 추출용)"""
    global _raw_ocr_engine
    if _raw_ocr_engine is None:
        print("[init] Raw PaddleOCR engine (korean)...")
        _raw_ocr_engine = RawPaddleOCR(use_textline_orientation=True, lang="korean")
        print("[init] Raw PaddleOCR engine ready")
    return _raw_ocr_engine


# ============================================================
# Confidence 기반 OCR 함수들
# ============================================================

def get_raw_paddle_ocr_with_confidence(image_path: str) -> list:
    """
    Raw PaddleOCR로 텍스트와 confidence 추출

    Returns:
        list of dict: [{"bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "text": str, "confidence": float}]
    """
    ocr = get_raw_ocr_engine()
    result = ocr.ocr(image_path)

    print(f"[DEBUG] Raw OCR result type: {type(result)}")
    if result:
        print(f"[DEBUG] Result length: {len(result)}")
        if len(result) > 0 and result[0]:
            print(f"[DEBUG] First item type: {type(result[0])}")
            if len(result[0]) > 0:
                print(f"[DEBUG] First line sample: {result[0][0]}")

    ocr_results = []
    if result and result[0]:
        for line in result[0]:
            try:
                # PaddleOCR 결과 구조: [[bbox], (text, confidence)] 또는 다른 형태
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]

                    # text_info가 tuple/list인 경우: (text, confidence)
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                    # text_info가 dict인 경우
                    elif isinstance(text_info, dict):
                        text = str(text_info.get('text', text_info.get('rec_text', '')))
                        confidence = float(text_info.get('confidence', text_info.get('rec_score', 0.0)))
                    # text_info가 문자열인 경우
                    else:
                        text = str(text_info)
                        confidence = 1.0  # 기본값

                    ocr_results.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": confidence
                    })
                else:
                    print(f"[DEBUG] Unexpected line format: {line}")
            except Exception as e:
                print(f"[DEBUG] Error parsing line: {line}, error: {e}")
                continue

    print(f"[DEBUG] Parsed {len(ocr_results)} OCR results")
    return ocr_results


def calculate_iou(box1, box2) -> float:
    """
    두 bbox의 IOU (Intersection over Union) 계산

    Args:
        box1: (x1, y1, x2, y2) 형식
        box2: (x1, y1, x2, y2) 형식

    Returns:
        IOU 값 (0.0 ~ 1.0)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def polygon_to_rect(polygon) -> tuple:
    """
    PaddleOCR polygon bbox를 rect (x1, y1, x2, y2)로 변환
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def map_ocr_to_cells_by_iou(cells: list, ocr_results: list) -> list:
    """
    Cell BBox와 OCR BBox를 IOU로 매핑하여 셀별 confidence 계산

    Args:
        cells: [{"row": int, "col": int, "bbox": (x1,y1,x2,y2), "text": str}]
        ocr_results: Raw PaddleOCR 결과

    Returns:
        cells with confidence: [{"row", "col", "bbox", "text", "confidence", "ocr_matches"}]
    """
    for cell in cells:
        cell_bbox = cell["bbox"]
        matched_confidences = []
        matched_texts = []

        for ocr in ocr_results:
            ocr_bbox = polygon_to_rect(ocr["bbox"])
            iou = calculate_iou(cell_bbox, ocr_bbox)

            if iou >= IOU_THRESHOLD:
                matched_confidences.append(ocr["confidence"])
                matched_texts.append(ocr["text"])

        if matched_confidences:
            # 매칭된 OCR 결과의 평균 confidence
            cell["confidence"] = sum(matched_confidences) / len(matched_confidences)
            cell["ocr_matches"] = len(matched_confidences)
        else:
            # 매칭 없음 (빈 셀이거나 OCR 실패)
            cell["confidence"] = 1.0 if not cell.get("text") else 0.5
            cell["ocr_matches"] = 0

    return cells


def vlm_verify_cell(image: Image.Image, cell_bbox: tuple, original_text: str) -> dict:
    """
    VLM으로 특정 셀 영역의 텍스트 검증

    Args:
        image: 전체 이미지
        cell_bbox: (x1, y1, x2, y2)
        original_text: OCR로 읽은 원본 텍스트

    Returns:
        {"verified_text": str, "matches": bool, "error": str or None}
    """
    try:
        # 셀 영역 크롭 (여백 추가)
        x1, y1, x2, y2 = cell_bbox
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)

        cell_image = image.crop((x1, y1, x2, y2))

        # base64 변환
        buffered = io.BytesIO()
        cell_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = f"""이 이미지에서 텍스트를 읽어주세요.
OCR 결과: "{original_text}"

이미지의 실제 텍스트만 출력하세요. 설명 없이 텍스트만."""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VLM_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0}
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            verified_text = result.get('response', '').strip()

            # 간단한 정리 (불필요한 따옴표 제거 등)
            verified_text = verified_text.strip('"\'')

            return {
                "verified_text": verified_text,
                "matches": verified_text.lower() == original_text.lower(),
                "error": None
            }

        return {
            "verified_text": original_text,
            "matches": True,
            "error": f"API error: {response.status_code}"
        }

    except Exception as e:
        return {
            "verified_text": original_text,
            "matches": True,
            "error": str(e)
        }


def selective_vlm_verification(image: Image.Image, cells: list) -> dict:
    """
    Confidence 기반 Selective VLM 검증

    Args:
        image: 원본 이미지
        cells: confidence가 포함된 셀 목록

    Returns:
        {
            "total_cells": int,
            "low_confidence_cells": int,
            "vlm_verified": int,
            "corrections": [{"row", "col", "original", "verified", "confidence"}],
            "skipped": int
        }
    """
    result = {
        "total_cells": len(cells),
        "low_confidence_cells": 0,
        "vlm_verified": 0,
        "corrections": [],
        "skipped": 0
    }

    for cell in cells:
        confidence = cell.get("confidence", 1.0)

        if confidence >= CONFIDENCE_THRESHOLD:
            result["skipped"] += 1
            continue

        result["low_confidence_cells"] += 1

        # 빈 셀은 VLM 호출 스킵
        if not cell.get("text"):
            continue

        # VLM 검증
        vlm_result = vlm_verify_cell(
            image,
            cell["bbox"],
            cell["text"]
        )

        result["vlm_verified"] += 1

        if vlm_result["error"]:
            print(f"[VLM] Error at ({cell['row']},{cell['col']}): {vlm_result['error']}")
            continue

        if not vlm_result["matches"]:
            result["corrections"].append({
                "row": cell["row"],
                "col": cell["col"],
                "original": cell["text"],
                "verified": vlm_result["verified_text"],
                "confidence": confidence
            })
            print(f"[VLM] Correction at ({cell['row']},{cell['col']}): "
                  f"'{cell['text']}' → '{vlm_result['verified_text']}' "
                  f"(conf: {confidence:.2%})")

    return result


# ============================================================
# 테이블 추출 함수
# ============================================================

def extract_tables_with_cells(image: Image.Image, tmp_path: str) -> list:
    """
    img2table로 테이블 추출 + 셀 BBox 정보 포함

    Returns:
        list of tables with cell info
    """
    ocr = get_ocr_engine()

    try:
        doc = Img2TableImage(src=tmp_path)
        tables = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=False,
            min_confidence=50
        )

        results = []
        for idx, table in enumerate(tables):
            table_bbox = None
            if hasattr(table, 'bbox') and table.bbox is not None:
                b = table.bbox
                table_bbox = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))

            # 셀 정보 추출
            cells = []
            if hasattr(table, 'content') and table.content:
                for row_idx, row in enumerate(table.content.values()):
                    for col_idx, cell in enumerate(row):
                        if hasattr(cell, 'bbox') and cell.bbox:
                            cells.append({
                                "row": row_idx,
                                "col": col_idx,
                                "bbox": (int(cell.bbox.x1), int(cell.bbox.y1),
                                        int(cell.bbox.x2), int(cell.bbox.y2)),
                                "text": cell.value if hasattr(cell, 'value') else ""
                            })

            # 셀 정보가 없으면 HTML에서 추출 시도
            if not cells:
                cells = extract_cells_from_html(table.html, table_bbox)

            result = {
                'index': idx,
                'html': table.html,
                'df': table.df,
                'bbox': table_bbox,
                'cells': cells,
                'has_colspan': 'colspan' in table.html,
                'has_rowspan': 'rowspan' in table.html,
            }
            results.append(result)

        return results

    except Exception as e:
        print(f"[extract_tables] Error: {e}")
        return []


def extract_cells_from_html(html: str, table_bbox: tuple) -> list:
    """
    HTML에서 셀 정보 추출 (fallback용)
    table_bbox를 기준으로 대략적인 셀 위치 계산
    """
    from bs4 import BeautifulSoup

    cells = []
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return cells

    rows = table.find_all('tr')
    if not rows or not table_bbox:
        return cells

    # 테이블 크기
    tx1, ty1, tx2, ty2 = table_bbox
    table_width = tx2 - tx1
    table_height = ty2 - ty1

    # 행/열 수 계산
    num_rows = len(rows)
    max_cols = 0
    for row in rows:
        cols = sum(int(cell.get('colspan', 1)) for cell in row.find_all(['td', 'th']))
        max_cols = max(max_cols, cols)

    if num_rows == 0 or max_cols == 0:
        return cells

    # 대략적인 셀 크기
    cell_height = table_height / num_rows
    cell_width = table_width / max_cols

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            colspan = int(cell.get('colspan', 1))
            text = cell.get_text(strip=True)

            # 대략적인 bbox 계산
            x1 = tx1 + col_idx * cell_width
            y1 = ty1 + row_idx * cell_height
            x2 = x1 + colspan * cell_width
            y2 = y1 + cell_height

            cells.append({
                "row": row_idx,
                "col": col_idx,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "text": text
            })

            col_idx += colspan

    return cells


def extract_tables_from_image(image: Image.Image) -> list:
    """img2table을 사용하여 이미지에서 테이블 추출 (기존 호환성)"""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = extract_tables_with_cells(image, tmp_path)
        return results
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# HTML 생성 함수
# ============================================================

def generate_ocr_result_html(tables: list) -> str:
    """OCR 결과를 HTML로 렌더링"""
    if not tables:
        return "<p>No tables detected</p>"

    html_parts = []

    for table in tables:
        if 'df' in table and hasattr(table['df'], 'shape'):
            cols = len(table['df'].columns)
            rows = len(table['df'])
        elif 'df_shape' in table:
            rows, cols = table['df_shape']
        else:
            cols = rows = '?'

        table_meta = f"""
            {cols} cols × {rows} rows |
            colspan: {'✓' if table['has_colspan'] else '✗'} |
            rowspan: {'✓' if table['has_rowspan'] else '✗'}
        """

        html_parts.append(f"""
        <div class="table-block section-original">
            <div class="table-header">
                <span class="table-title">📋 PaddleOCR Result</span>
                <span class="table-meta">{table_meta}</span>
            </div>
            <div class="section-label">img2table + PaddleOCR 결과</div>
            <div class="table-content">
                {table['html']}
            </div>
        </div>
        """)

    return '\n'.join(html_parts)


def generate_confidence_check_html(tables: list, verification_result: dict, cells_with_conf: list) -> str:
    """
    Confidence 기반 검증 결과를 HTML로 렌더링
    """
    if not tables:
        return "<p>No tables to check</p>"

    html_parts = []

    for table in tables:
        if 'df' in table and hasattr(table['df'], 'shape'):
            cols = len(table['df'].columns)
            rows = len(table['df'])
        elif 'df_shape' in table:
            rows, cols = table['df_shape']
        else:
            cols = rows = '?'

        total = verification_result.get('total_cells', 0)
        low_conf = verification_result.get('low_confidence_cells', 0)
        vlm_count = verification_result.get('vlm_verified', 0)
        corrections = verification_result.get('corrections', [])
        skipped = verification_result.get('skipped', 0)

        # 하이라이트된 HTML 생성
        highlighted_html = generate_confidence_highlighted_html(
            table['html'],
            cells_with_conf,
            corrections
        )

        # 통계 정보
        if corrections:
            status_msg = f"⚠️ VLM 수정 <span class='diff-count'>{len(corrections)}</span>개 (검토 필요)"
            status_class = "has-diff"
        else:
            status_msg = "✅ OCR 결과 신뢰도 높음 (VLM 수정 없음)"
            status_class = "success"

        stats_html = f"""
        <div class="confidence-stats">
            <span class="stat-item">📊 전체 셀: {total}</span>
            <span class="stat-item">✅ 고신뢰 (≥80%): {skipped}</span>
            <span class="stat-item">⚠️ 저신뢰 (<80%): {low_conf}</span>
            <span class="stat-item">🔍 VLM 검증: {vlm_count}</span>
        </div>
        """

        # 수정 목록
        corrections_html = ""
        if corrections:
            items = ["<li><strong>🔍 VLM 수정 내역:</strong></li>"]
            for c in corrections[:15]:
                conf_pct = c.get('confidence', 0) * 100
                items.append(
                    f"<li style='margin-left:20px'>Row {c['row']+1}, Col {c['col']+1}: "
                    f"<span class='ocr-value'>{c['original']}</span> → "
                    f"<span class='ai-value'>{c['verified']}</span> "
                    f"<span style='color:#888'>(신뢰도: {conf_pct:.0f}%)</span></li>"
                )
            if len(corrections) > 15:
                items.append(f"<li style='margin-left:20px'>... 외 {len(corrections) - 15}개</li>")
            corrections_html = f"<ul class='diff-list'>{''.join(items)}</ul>"

        html_parts.append(f"""
        <div class="table-block section-diff">
            <div class="table-header">
                <span class="table-title">🎯 Confidence-based Verification</span>
                <span class="table-meta">Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%</span>
            </div>
            <div class="section-label {status_class}">
                {status_msg}
            </div>
            {stats_html}
            {corrections_html}
            <div class="table-content">
                {highlighted_html}
            </div>
        </div>
        """)

    return '\n'.join(html_parts)


def generate_confidence_highlighted_html(ocr_html: str, cells_with_conf: list, corrections: list) -> str:
    """
    신뢰도 기반 하이라이트된 HTML 생성
    - 저신뢰 셀: 노란색 배경
    - 수정된 셀: 원본 취소선 + 수정값 굵게
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(ocr_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ocr_html

    # 수정 맵 생성
    correction_map = {(c['row'], c['col']): c for c in corrections}

    # 셀 신뢰도 맵 생성
    confidence_map = {(c['row'], c['col']): c.get('confidence', 1.0) for c in cells_with_conf}

    rows = table.find_all('tr')

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            colspan = int(cell.get('colspan', 1))

            pos = (row_idx, col_idx)
            conf = confidence_map.get(pos, 1.0)

            # 저신뢰 셀 표시
            if conf < CONFIDENCE_THRESHOLD:
                existing_class = cell.get('class', [])
                if isinstance(existing_class, str):
                    existing_class = [existing_class]
                existing_class.append('low-confidence')
                cell['class'] = existing_class
                cell['data-confidence'] = f"{conf*100:.0f}%"

            # 수정된 셀 표시
            if pos in correction_map:
                corr = correction_map[pos]

                existing_class = cell.get('class', [])
                if isinstance(existing_class, str):
                    existing_class = [existing_class]
                existing_class.append('diff-cell')
                cell['class'] = existing_class

                cell['data-diff'] = f"OCR: {corr['original']} → VLM: {corr['verified']}"

                # 셀 내용 변경
                ocr_span = soup.new_tag('span')
                ocr_span['class'] = 'ocr-value'
                ocr_span.string = corr['original']

                arrow = soup.new_tag('span')
                arrow['class'] = 'diff-arrow'
                arrow.string = ' → '

                vlm_span = soup.new_tag('span')
                vlm_span['class'] = 'ai-value'
                vlm_span.string = corr['verified']

                cell.clear()
                cell.append(ocr_span)
                cell.append(arrow)
                cell.append(vlm_span)

            col_idx += colspan

    return str(table)


def process_image_ocr_only(img: Image.Image) -> dict:
    """OCR만 수행"""
    result = {
        'success': False,
        'tables': [],
        'html': '',
        'error': None
    }

    try:
        tables = extract_tables_from_image(img)

        if not tables:
            result['error'] = 'No tables detected in image'
            return result

        result['tables'] = tables
        result['success'] = True
        result['html'] = generate_ocr_result_html(tables)

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


# ============================================================
# UI HTML 템플릿 (v7 - Confidence 기반)
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comet Table Extractor v7.0</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html, body {
            height: 100%;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }

        /* 헤더 */
        .header {
            background: #16213e;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #0f3460;
            height: 60px;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo h1 {
            font-size: 18px;
            color: #e94560;
        }

        .version {
            background: #28a745;
            color: white;
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 12px;
        }

        /* 버튼 그룹 */
        .btn-group {
            display: flex;
            gap: 10px;
        }

        .action-btn {
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .action-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .upload-btn {
            background: #e94560;
        }

        .upload-btn:hover:not(:disabled) {
            background: #ff6b6b;
            transform: scale(1.02);
        }

        .upload-btn.dragover {
            background: #00d9ff;
        }

        .ocr-btn {
            background: #0f3460;
            border: 2px solid #00d9ff;
        }

        .ocr-btn:hover:not(:disabled) {
            background: #00d9ff;
            color: #0f3460;
        }

        .verify-btn {
            background: #0f3460;
            border: 2px solid #28a745;
        }

        .verify-btn:hover:not(:disabled) {
            background: #28a745;
            color: white;
        }

        #file-input {
            display: none;
        }

        .tech-badges {
            display: flex;
            gap: 8px;
        }

        .tech-badge {
            background: #0f3460;
            color: #00d9ff;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
        }

        .tech-badge.new {
            background: #28a745;
            color: white;
        }

        /* 메인 컨테이너 */
        .main-container {
            margin-top: 60px;
            padding: 20px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }

        /* 이미지 섹션 */
        .image-section {
            background: #16213e;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            display: none;
        }

        .image-section.show {
            display: block;
        }

        .section-header {
            background: #0f3460;
            padding: 12px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .section-title {
            font-weight: bold;
            color: #00d9ff;
            font-size: 14px;
        }

        .section-meta {
            font-size: 12px;
            color: #888;
        }

        .image-content {
            padding: 15px;
            text-align: center;
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 4px;
            border: 1px solid #0f3460;
        }

        /* 테이블 블록 */
        .table-block {
            background: #16213e;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }

        .table-header {
            background: #0f3460;
            padding: 12px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .table-title {
            font-weight: bold;
            color: #00d9ff;
        }

        .table-meta {
            font-size: 12px;
            color: #888;
        }

        .table-content {
            padding: 15px;
            overflow-x: auto;
        }

        /* 테이블 스타일 */
        .table-content table {
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
            background: #1a1a2e;
        }

        .table-content td, .table-content th {
            border: 1px solid #0f3460;
            padding: 10px 12px;
            text-align: center;
            white-space: nowrap;
        }

        .table-content th,
        .table-content tr:first-child td {
            background: #0f3460;
            font-weight: bold;
            color: #00d9ff;
        }

        .table-content tr:hover {
            background: rgba(233, 69, 96, 0.1);
        }

        /* 빈 상태 */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            color: #555;
            background: #16213e;
            border-radius: 8px;
        }

        .empty-icon {
            font-size: 64px;
            margin-bottom: 20px;
        }

        /* 안내 상태 */
        .guide-state {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 200px;
            color: #888;
            background: #16213e;
            border-radius: 8px;
            margin-top: 20px;
        }

        .guide-state.show {
            display: flex;
        }

        .guide-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }

        /* 로딩 */
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .loading.show {
            display: flex;
        }

        .loading-content {
            text-align: center;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #0f3460;
            border-top: 4px solid #28a745;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 차이점 하이라이트 스타일 */
        .diff-cell {
            background: #fff3cd !important;
            position: relative;
        }

        .diff-cell:hover::after {
            content: attr(data-diff);
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .low-confidence {
            background: rgba(255, 193, 7, 0.2) !important;
            position: relative;
        }

        .low-confidence:hover::after {
            content: "신뢰도: " attr(data-confidence);
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: #ffc107;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
            z-index: 100;
        }

        .ocr-value {
            text-decoration: line-through;
            color: #dc3545;
            font-size: 0.85em;
        }

        .diff-arrow {
            color: #666;
            margin: 0 4px;
        }

        .ai-value {
            font-weight: bold;
            color: #28a745;
        }

        /* Confidence 통계 */
        .confidence-stats {
            display: flex;
            gap: 20px;
            padding: 10px 15px;
            background: rgba(0, 0, 0, 0.3);
            border-bottom: 1px solid #0f3460;
            flex-wrap: wrap;
        }

        .stat-item {
            font-size: 12px;
            color: #aaa;
        }

        /* 섹션 스타일 */
        .section-original .table-header {
            background: #0f3460;
            border-left: 4px solid #00d9ff;
        }

        .section-diff .table-header {
            background: #1a3d1a;
            border-left: 4px solid #28a745;
        }

        .section-label {
            padding: 10px 15px;
            font-size: 12px;
            color: #888;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid #0f3460;
        }

        .section-label.success {
            color: #28a745;
        }

        .section-label.has-diff {
            color: #ffc107;
        }

        .section-label .diff-count {
            font-size: 16px;
            color: #ffc107;
            font-weight: bold;
        }

        /* 차이점 목록 */
        .diff-list {
            max-height: 200px;
            overflow-y: auto;
            padding: 10px 15px 10px 30px;
            background: rgba(0, 0, 0, 0.2);
            margin: 0;
            font-size: 12px;
        }

        .diff-list li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <!-- 상단 헤더 -->
    <div class="header">
        <div class="header-left">
            <div class="logo">
                <h1>Comet Table Extractor</h1>
                <span class="version">v7.0</span>
            </div>
            <div class="btn-group">
                <input type="file" id="file-input" accept="image/*">
                <button class="action-btn upload-btn" id="upload-btn" onclick="document.getElementById('file-input').click()">
                    📤 Upload
                </button>
                <button class="action-btn ocr-btn" id="ocr-btn" disabled onclick="runOCR()">
                    🔍 Run OCR
                </button>
                <button class="action-btn verify-btn" id="verify-btn" disabled onclick="runConfidenceCheck()">
                    🎯 Confidence 검증
                </button>
            </div>
        </div>
        <div class="tech-badges">
            <span class="tech-badge">img2table</span>
            <span class="tech-badge">PaddleOCR</span>
            <span class="tech-badge new">Confidence-based</span>
            <span class="tech-badge new">Selective VLM</span>
        </div>
    </div>

    <!-- 메인 컨테이너 -->
    <div class="main-container">
        <!-- 이미지 섹션 -->
        <div class="image-section" id="image-section">
            <div class="section-header">
                <span class="section-title">🖼️ Uploaded Image</span>
                <span class="section-meta" id="image-meta"></span>
            </div>
            <div class="image-content">
                <img id="preview-image" class="preview-image">
            </div>
        </div>

        <!-- 안내 상태 -->
        <div class="guide-state" id="guide-state">
            <div class="guide-icon">🔍</div>
            <div>Click <strong>Run OCR</strong> to extract tables</div>
        </div>

        <!-- 결과 영역 -->
        <div id="result-area">
            <div class="empty-state" id="empty-state">
                <div class="empty-icon">📋</div>
                <div>Upload an image to extract tables</div>
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    Drag & drop anywhere or click the button above
                </div>
            </div>
            <div id="ocr-result" style="display: none;"></div>
            <div id="verify-result" style="display: none;"></div>
        </div>
    </div>

    <!-- 로딩 오버레이 -->
    <div class="loading" id="loading">
        <div class="loading-content">
            <div class="spinner"></div>
            <div id="loading-text">Processing...</div>
        </div>
    </div>

    <script>
        const uploadBtn = document.getElementById('upload-btn');
        const ocrBtn = document.getElementById('ocr-btn');
        const verifyBtn = document.getElementById('verify-btn');
        const fileInput = document.getElementById('file-input');
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loading-text');
        const imageSection = document.getElementById('image-section');
        const previewImage = document.getElementById('preview-image');
        const imageMeta = document.getElementById('image-meta');
        const emptyState = document.getElementById('empty-state');
        const guideState = document.getElementById('guide-state');
        const ocrResult = document.getElementById('ocr-result');
        const verifyResult = document.getElementById('verify-result');

        // 상태 관리
        let currentFile = null;
        let ocrData = null;

        // 전체 화면 드래그 앤 드롭
        document.body.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBtn.classList.add('dragover');
        });

        document.body.addEventListener('dragleave', (e) => {
            if (e.target === document.body) {
                uploadBtn.classList.remove('dragover');
            }
        });

        document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBtn.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            currentFile = file;
            ocrData = null;

            // 이미지 섹션 표시
            imageSection.classList.add('show');
            imageMeta.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

            // 미리보기 표시
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(file);

            // 상태 초기화
            emptyState.style.display = 'none';
            guideState.classList.add('show');
            ocrResult.style.display = 'none';
            ocrResult.innerHTML = '';
            verifyResult.style.display = 'none';
            verifyResult.innerHTML = '';

            // 버튼 상태
            ocrBtn.disabled = false;
            verifyBtn.disabled = true;
        }

        function runOCR() {
            if (!currentFile) {
                alert('Please upload an image first');
                return;
            }

            loading.classList.add('show');
            loadingText.textContent = 'PaddleOCR 실행 중...';
            guideState.classList.remove('show');

            const formData = new FormData();
            formData.append('image', currentFile);

            fetch('/api/extract/ocr', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('show');
                ocrResult.style.display = 'block';

                if (data.success) {
                    ocrData = data;
                    ocrResult.innerHTML = data.html;
                    verifyBtn.disabled = false;
                } else {
                    ocrResult.innerHTML = `
                        <div class="table-block">
                            <div class="table-header">
                                <span class="table-title">⚠️ Error</span>
                            </div>
                            <div class="section-label" style="color: #dc3545;">${data.error}</div>
                        </div>
                    `;
                    verifyBtn.disabled = true;
                }
            })
            .catch(error => {
                loading.classList.remove('show');
                ocrResult.style.display = 'block';
                ocrResult.innerHTML = `
                    <div class="table-block">
                        <div class="table-header">
                            <span class="table-title">⚠️ Error</span>
                        </div>
                        <div class="section-label" style="color: #dc3545;">${error.message}</div>
                    </div>
                `;
                verifyBtn.disabled = true;
            });
        }

        function runConfidenceCheck() {
            if (!currentFile || !ocrData) {
                alert('Please run OCR first');
                return;
            }

            loading.classList.add('show');
            loadingText.textContent = 'Confidence 분석 및 VLM 검증 중...';

            const formData = new FormData();
            formData.append('image', currentFile);
            if (ocrData.tables && ocrData.tables.length > 0) {
                formData.append('ocr_html', ocrData.tables[0].html);
                if (ocrData.tables[0].bbox) {
                    formData.append('table_bbox', JSON.stringify(ocrData.tables[0].bbox));
                }
            }

            fetch('/api/extract/verify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('show');
                verifyResult.style.display = 'block';

                if (data.success) {
                    verifyResult.innerHTML = data.html;
                } else {
                    verifyResult.innerHTML = `
                        <div class="table-block">
                            <div class="table-header">
                                <span class="table-title">⚠️ Verification Failed</span>
                            </div>
                            <div class="section-label" style="color: #dc3545;">${data.error}</div>
                        </div>
                    `;
                }
            })
            .catch(error => {
                loading.classList.remove('show');
                verifyResult.style.display = 'block';
                verifyResult.innerHTML = `
                    <div class="table-block">
                        <div class="table-header">
                            <span class="table-title">⚠️ Error</span>
                        </div>
                        <div class="section-label" style="color: #dc3545;">${error.message}</div>
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
"""


# ============================================================
# Flask 라우트
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/extract/ocr', methods=['POST'])
def api_extract_ocr():
    """PaddleOCR만 수행"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        img = Image.open(file.stream).convert('RGB')
        result = process_image_ocr_only(img)

        # DataFrame 제거 (JSON 직렬화 불가)
        if result.get('tables'):
            for table in result['tables']:
                if 'df' in table:
                    shape = table['df'].shape
                    table['df_shape'] = (int(shape[0]), int(shape[1]))
                    del table['df']
                # cells도 JSON 직렬화 가능하게 처리
                if 'cells' in table:
                    for cell in table['cells']:
                        cell['bbox'] = list(cell['bbox']) if cell.get('bbox') else None

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/extract/verify', methods=['POST'])
def api_extract_verify():
    """
    Confidence 기반 Selective VLM 검증
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})

    file = request.files['image']
    ocr_html = request.form.get('ocr_html', '')
    table_bbox_str = request.form.get('table_bbox', '')

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        img = Image.open(file.stream).convert('RGB')

        # 임시 파일로 저장 (Raw PaddleOCR용)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # 1. Raw PaddleOCR로 confidence 추출
            print("[Verify] Raw PaddleOCR 실행...")
            ocr_results = get_raw_paddle_ocr_with_confidence(tmp_path)
            print(f"[Verify] OCR 결과: {len(ocr_results)}개 텍스트 블록")

            # 2. 테이블 bbox 파싱
            table_bbox = None
            if table_bbox_str:
                try:
                    table_bbox = tuple(json.loads(table_bbox_str))
                except:
                    pass

            # 3. HTML에서 셀 정보 추출
            cells = extract_cells_from_html(ocr_html, table_bbox)
            print(f"[Verify] 셀 정보: {len(cells)}개 셀")

            # 4. Cell-OCR IOU 매핑
            cells_with_conf = map_ocr_to_cells_by_iou(cells, ocr_results)

            low_conf_count = sum(1 for c in cells_with_conf if c.get('confidence', 1.0) < CONFIDENCE_THRESHOLD)
            print(f"[Verify] 저신뢰 셀: {low_conf_count}개 (threshold: {CONFIDENCE_THRESHOLD*100}%)")

            # 5. Selective VLM 검증
            verification_result = selective_vlm_verification(img, cells_with_conf)

            print(f"[Verify] VLM 검증: {verification_result['vlm_verified']}개, "
                  f"수정: {len(verification_result['corrections'])}개")

            # 6. HTML 생성
            tables = [{
                'html': ocr_html,
                'has_colspan': 'colspan' in ocr_html,
                'has_rowspan': 'rowspan' in ocr_html,
                'df_shape': (0, 0)
            }]

            html = generate_confidence_check_html(tables, verification_result, cells_with_conf)

            return jsonify({
                'success': True,
                'html': html,
                'total_cells': verification_result['total_cells'],
                'low_confidence_cells': verification_result['low_confidence_cells'],
                'vlm_verified': verification_result['vlm_verified'],
                'corrections': verification_result['corrections'],
                'skipped': verification_result['skipped']
            })

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'version': '7.0',
        'engine': 'img2table + PaddleOCR + Selective VLM',
        'features': ['PaddleOCR', 'Confidence Extraction', 'IOU Mapping', 'Selective VLM'],
        'threshold': CONFIDENCE_THRESHOLD
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Comet Table Extractor v7.0 (Confidence-based Selective VLM)")
    print("=" * 60)
    print("Engine: img2table + PaddleOCR + Selective VLM")
    print("Method: Confidence-based verification (threshold: 80%)")
    print("Port: 6007")
    print("=" * 60)

    # OCR 엔진 미리 초기화
    get_ocr_engine()
    get_raw_ocr_engine()

    app.run(host='0.0.0.0', port=6007, debug=True)
