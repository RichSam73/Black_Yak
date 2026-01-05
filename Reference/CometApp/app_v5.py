# -*- coding: utf-8 -*-
"""
Comet + ERP Table Extractor v5.0
================================
- 전체 화면 활용 UI (테이블 크게 표시)
- img2table: 테이블 구조 추출 (colspan/rowspan 자동 감지)
- PaddleOCR: 텍스트 인식
- **v5.0 변경사항**: Dual OCR 전용 (VLM 기반 코드 완전 제거)
  - OCR 1: PaddleOCR (img2table 기반)
  - OCR 2: granite3.2-vision (Ollama Vision 모델)
  - 두 OCR 결과를 셀 단위로 비교하여 불일치 셀 감지
  - difflib fuzzy matching으로 유사도 계산
- 포트: 6006
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['JSON_AS_ASCII'] = False  # 한글 유니코드 출력 지원

# ============================================================
# Ollama 설정 (Dual OCR 비교용)
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
SECOND_OCR_MODEL = "granite3.2-vision:2b"  # 두 번째 OCR 엔진 (Vision 지원)

# Dual OCR 비교 설정
SIMILARITY_THRESHOLD = 0.85  # 85% 이상 유사하면 일치로 판단

# 전역 OCR 인스턴스
_ocr_engine = None


def get_ocr_engine():
    """img2table용 PaddleOCR 엔진 초기화 (한글+영어 지원)"""
    global _ocr_engine
    if _ocr_engine is None:
        print("[init] PaddleOCR engine for img2table (korean)...")
        _ocr_engine = Img2TablePaddleOCR(lang="korean")
        print("[init] PaddleOCR engine ready")
    return _ocr_engine


# ============================================================
# Dual OCR 함수들 (PaddleOCR vs granite3.2-vision)
# ============================================================

def second_ocr_full_table(image: Image.Image) -> list:
    """
    granite3.2-vision으로 전체 테이블 이미지를 읽고 2D 배열로 반환

    Args:
        image: 테이블 이미지

    Returns:
        2D 배열 [[row1_cells], [row2_cells], ...]
    """
    try:
        # base64 변환
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = """이 테이블 이미지에서 모든 텍스트를 추출하세요.
각 행을 새 줄로, 각 셀을 | 로 구분하세요.

예시 출력:
헤더1 | 헤더2 | 헤더3
값1 | 값2 | 값3
값4 | 값5 | 값6

빈 셀은 (empty)로 표시하세요.
테이블 내용만 출력, 설명 없음."""

        print(f"[Dual OCR] granite3.2-vision 전체 테이블 OCR 시작...")
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": SECOND_OCR_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"num_predict": 4096, "temperature": 0}
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('response', '').strip()
            print(f"[Dual OCR] 응답 길이: {len(raw_text)}")

            # 파싱: 줄 단위 → 셀 단위
            table_2d = []
            for line in raw_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('---'):
                    continue
                cells = [c.strip() for c in line.split('|')]
                # (empty) → 빈 문자열
                cells = ['' if c.lower() in ['(empty)', 'empty', ''] else c for c in cells]
                if cells:
                    table_2d.append(cells)

            print(f"[Dual OCR] 추출된 행: {len(table_2d)}")
            return table_2d

        print(f"[Dual OCR] API 오류: {response.status_code}")
        return []

    except Exception as e:
        print(f"[Dual OCR] 오류: {e}")
        return []


def compare_ocr_results(paddle_table: list, second_table: list) -> list:
    """
    PaddleOCR 결과와 second OCR 결과를 셀 단위로 비교

    Args:
        paddle_table: PaddleOCR 2D 배열
        second_table: second OCR 2D 배열

    Returns:
        불일치 셀 목록: [{"row": int, "col": int, "paddle": str, "second": str, "similarity": float}]
    """
    from difflib import SequenceMatcher

    differences = []

    if not paddle_table or not second_table:
        return differences

    max_rows = max(len(paddle_table), len(second_table))

    for row_idx in range(max_rows):
        paddle_row = paddle_table[row_idx] if row_idx < len(paddle_table) else []
        second_row = second_table[row_idx] if row_idx < len(second_table) else []

        max_cols = max(len(paddle_row), len(second_row))

        for col_idx in range(max_cols):
            paddle_cell = str(paddle_row[col_idx]).strip() if col_idx < len(paddle_row) else ""
            second_cell = str(second_row[col_idx]).strip() if col_idx < len(second_row) else ""

            # 둘 다 비어있으면 스킵
            if not paddle_cell and not second_cell:
                continue

            # 정확히 일치하면 스킵
            if paddle_cell == second_cell:
                continue

            # fuzzy matching으로 유사도 계산
            similarity = SequenceMatcher(None, paddle_cell, second_cell).ratio()

            # 유사도가 임계값 미만이면 불일치로 판단
            if similarity < SIMILARITY_THRESHOLD:
                differences.append({
                    "row": row_idx,
                    "col": col_idx,
                    "paddle": paddle_cell,
                    "second": second_cell,
                    "similarity": round(similarity, 3),
                    "type": "mismatch"
                })
                print(f"[Dual OCR] 불일치 발견: ({row_idx},{col_idx}) "
                      f"PaddleOCR='{paddle_cell}' vs granite='{second_cell}' "
                      f"(유사도: {similarity:.2%})")

    return differences


def dual_ocr_check(image: Image.Image, paddle_table: list) -> dict:
    """
    Dual OCR 비교 검사 수행

    Args:
        image: 원본 이미지
        paddle_table: PaddleOCR로 추출한 2D 테이블

    Returns:
        {
            "success": bool,
            "differences": list,  # 불일치 셀 목록
            "paddle_cells": int,  # PaddleOCR 셀 수
            "second_cells": int,  # second OCR 셀 수
            "mismatch_count": int,  # 불일치 셀 수
            "second_table": list,  # granite-vision 추출 결과
            "error": str or None
        }
    """
    try:
        print("[Dual OCR] granite3.2-vision으로 테이블 재추출 시작...")

        # second OCR로 테이블 추출
        second_table = second_ocr_full_table(image)

        if not second_table:
            return {
                "success": False,
                "differences": [],
                "paddle_cells": sum(len(row) for row in paddle_table),
                "second_cells": 0,
                "mismatch_count": 0,
                "second_table": [],
                "error": "granite3.2-vision OCR 결과 없음"
            }

        # 두 OCR 결과 비교
        differences = compare_ocr_results(paddle_table, second_table)

        paddle_cells = sum(len(row) for row in paddle_table)
        second_cells = sum(len(row) for row in second_table)

        print(f"[Dual OCR] 비교 완료: PaddleOCR {paddle_cells}셀, "
              f"granite3.2-vision {second_cells}셀, 불일치 {len(differences)}셀")

        return {
            "success": True,
            "differences": differences,
            "paddle_cells": paddle_cells,
            "second_cells": second_cells,
            "mismatch_count": len(differences),
            "second_table": second_table,
            "error": None
        }

    except Exception as e:
        print(f"[Dual OCR] 오류: {e}")
        return {
            "success": False,
            "differences": [],
            "paddle_cells": sum(len(row) for row in paddle_table) if paddle_table else 0,
            "second_cells": 0,
            "mismatch_count": 0,
            "second_table": [],
            "error": str(e)
        }


# ============================================================
# 테이블 추출 함수
# ============================================================

def extract_tables_from_image(image: Image.Image) -> list:
    """img2table을 사용하여 이미지에서 테이블 추출"""
    ocr = get_ocr_engine()

    # PIL Image를 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

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
            bbox = None
            if hasattr(table, 'bbox') and table.bbox is not None:
                b = table.bbox
                bbox = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))

            result = {
                'index': idx,
                'html': table.html,
                'df': table.df,
                'bbox': bbox,
                'has_colspan': 'colspan' in table.html,
                'has_rowspan': 'rowspan' in table.html,
            }
            results.append(result)

        return results

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def html_to_2d_array(html: str) -> list:
    """
    img2table HTML을 2D 배열로 변환 (Positional Anchoring용)
    colspan/rowspan 고려하여 셀 위치 정확히 매핑
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return []

    rows = table.find_all('tr')
    if not rows:
        return []

    # 최대 열 수 계산
    max_cols = 0
    for row in rows:
        cols = 0
        for cell in row.find_all(['td', 'th']):
            colspan = int(cell.get('colspan', 1))
            cols += colspan
        max_cols = max(max_cols, cols)

    # 2D 배열 초기화 (None으로)
    grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]

    # 셀 배치 (colspan/rowspan 고려)
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            # 이미 채워진 셀 건너뛰기 (rowspan으로 인해)
            while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1

            if col_idx >= max_cols:
                break

            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            text = cell.get_text(strip=True)

            # 셀 값 배치
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < len(grid) and col_idx + c < max_cols:
                        if r == 0 and c == 0:
                            grid[row_idx + r][col_idx + c] = text
                        else:
                            grid[row_idx + r][col_idx + c] = ""  # 병합된 셀은 빈 문자열

            col_idx += colspan

    # None을 빈 문자열로 변환
    return [[cell if cell is not None else "" for cell in row] for row in grid]


# ============================================================
# HTML 생성 함수
# ============================================================

def generate_ocr_result_html(tables: list) -> str:
    """
    OCR만 수행한 결과를 HTML로 렌더링 (v5 - OCR 전용)
    """
    if not tables:
        return "<p>No tables detected</p>"

    html_parts = []

    for table in tables:
        # df가 있으면 shape 정보 사용
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

        # OCR 테이블만 표시
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


def generate_highlighted_html(ocr_html: str, differences: list) -> str:
    """
    차이점을 하이라이트한 HTML 생성

    - 차이 있는 셀: 노란색 배경
    - PaddleOCR 값: 취소선 (빨간색)
    - granite 값: 굵은 글씨 (녹색)
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(ocr_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ocr_html

    # 차이점을 (row, col) -> diff 매핑
    diff_map = {(d["row"], d["col"]): d for d in differences}

    rows = table.find_all('tr')

    # 각 셀의 실제 위치 추적 (colspan/rowspan 고려)
    max_cols = 0
    for row in rows:
        cols = sum(int(cell.get('colspan', 1)) for cell in row.find_all(['td', 'th']))
        max_cols = max(max_cols, cols)

    # 셀 위치 그리드 생성
    cell_grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            # 이미 채워진 셀 건너뛰기
            while col_idx < max_cols and cell_grid[row_idx][col_idx] is not None:
                col_idx += 1

            if col_idx >= max_cols:
                break

            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))

            # 그리드에 셀 배치
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < len(cell_grid) and col_idx + c < max_cols:
                        cell_grid[row_idx + r][col_idx + c] = (row_idx, col_idx, cell)

            # 이 셀에 차이가 있는지 확인
            if (row_idx, col_idx) in diff_map:
                diff = diff_map[(row_idx, col_idx)]

                # 기존 클래스에 diff-cell 추가
                existing_class = cell.get('class', [])
                if isinstance(existing_class, str):
                    existing_class = [existing_class]
                existing_class.append('diff-cell')
                cell['class'] = existing_class

                # data-diff 속성 추가 (툴팁용)
                cell['data-diff'] = f"PaddleOCR: {diff['ocr']} → granite: {diff['ai']}"

                # 셀 내용 변경 (PaddleOCR 취소선 + granite 굵게)
                paddle_value = diff['ocr'] if diff['ocr'] else '(empty)'
                granite_value = diff['ai'] if diff['ai'] else '(empty)'

                paddle_span = soup.new_tag('span')
                paddle_span['class'] = 'ocr-value'
                paddle_span.string = paddle_value

                arrow = soup.new_tag('span')
                arrow['class'] = 'diff-arrow'
                arrow.string = ' → '

                granite_span = soup.new_tag('span')
                granite_span['class'] = 'ai-value'
                granite_span.string = granite_value

                cell.clear()
                cell.append(paddle_span)
                cell.append(arrow)
                cell.append(granite_span)

            col_idx += colspan

    return str(table)


def generate_dual_ocr_result_html(tables: list, dual_result: dict, differences: list) -> str:
    """
    Dual OCR 결과를 HTML로 렌더링 (v5 - Dual OCR 전용)
    """
    if not tables:
        return "<p>No tables to check</p>"

    html_parts = []

    for table in tables:
        # df가 있으면 shape 정보 사용
        if 'df' in table and hasattr(table['df'], 'shape'):
            cols = len(table['df'].columns)
            rows = len(table['df'])
        elif 'df_shape' in table:
            rows, cols = table['df_shape']
        else:
            cols = rows = '?'

        # Dual OCR 결과 표시
        if dual_result and dual_result.get('success'):
            diff_count = len(differences)
            paddle_cells = dual_result.get('paddle_cells', 0)
            granite_cells = dual_result.get('second_cells', 0)

            # 하이라이트 적용
            highlighted_html = generate_highlighted_html(table['html'], differences)

            # 불일치 상세 목록
            diff_details = ""
            if diff_count > 0:
                diff_items = ["<li><strong>🔍 OCR 불일치 셀:</strong></li>"]
                for d in differences[:15]:  # 최대 15개
                    similarity_pct = d.get('similarity', 0) * 100
                    diff_items.append(
                        f"<li style='margin-left:20px'>Row {d['row']+1}, Col {d['col']+1}: "
                        f"<span class='ocr-value'>{d['ocr']}</span> → "
                        f"<span class='ai-value'>{d['ai']}</span> "
                        f"<span style='color:#888'>(유사도: {similarity_pct:.0f}%)</span></li>"
                    )
                if diff_count > 15:
                    diff_items.append(f"<li style='margin-left:20px'>... 외 {diff_count - 15}개</li>")
                diff_details = f"<ul class='diff-list'>{''.join(diff_items)}</ul>"

            # 결과 메시지
            if diff_count == 0:
                status_msg = "✅ 두 OCR 결과 일치 - 신뢰도 높음"
                status_class = "success"
            else:
                status_msg = f"⚠️ 불일치 <span class='diff-count'>{diff_count}</span>개 발견 (검토 필요)"
                status_class = "has-diff"

            html_parts.append(f"""
            <div class="table-block section-diff">
                <div class="table-header">
                    <span class="table-title">🔄 Dual OCR Comparison</span>
                    <span class="table-meta">PaddleOCR {paddle_cells}셀 vs granite-vision {granite_cells}셀</span>
                </div>
                <div class="section-label {status_class}">
                    {status_msg}
                </div>
                {diff_details}
                <div class="table-content">
                    {highlighted_html}
                </div>
            </div>
            """)

        elif dual_result and dual_result.get('error'):
            html_parts.append(f"""
            <div class="table-block section-diff">
                <div class="table-header">
                    <span class="table-title">⚠️ Dual OCR Failed</span>
                </div>
                <div class="ai-status error">오류: {dual_result['error']}</div>
            </div>
            """)

    return '\n'.join(html_parts)


def process_image_ocr_only(img: Image.Image) -> dict:
    """
    OCR만 수행 (v5 - 버튼 분리용)
    AI 보정 없이 빠르게 결과 반환
    """
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
# UI HTML 템플릿 (v5 - Dual OCR 전용)
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comet Table Extractor v5.0</title>
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

        /* 헤더 - 버튼 3개 포함 */
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
            background: #e94560;
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

        .compare-btn {
            background: #0f3460;
            border: 2px solid #ffc107;
        }

        .compare-btn:hover:not(:disabled) {
            background: #ffc107;
            color: #0f3460;
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

        /* 메인 컨테이너 - 수직 스크롤 */
        .main-container {
            margin-top: 60px;
            padding: 20px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }

        /* 이미지 섹션 - 맨 위 */
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
            border-top: 4px solid #e94560;
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

        /* 처리 상태 표시 */
        .ai-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 13px;
            margin-bottom: 15px;
        }

        .ai-status.processing {
            background: rgba(0, 217, 255, 0.2);
            color: #00d9ff;
        }

        .ai-status.complete {
            background: rgba(40, 167, 69, 0.2);
            color: #28a745;
        }

        .ai-status.error {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }

        /* 섹션 스타일 */
        .section-original .table-header {
            background: #0f3460;
            border-left: 4px solid #00d9ff;
        }

        .section-diff .table-header {
            background: #3d2c00;
            border-left: 4px solid #ffc107;
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
                <span class="version">v5.0</span>
            </div>
            <div class="btn-group">
                <input type="file" id="file-input" accept="image/*">
                <button class="action-btn upload-btn" id="upload-btn" onclick="document.getElementById('file-input').click()">
                    📤 Upload
                </button>
                <button class="action-btn ocr-btn" id="ocr-btn" disabled onclick="runOCR()">
                    🔍 Run OCR
                </button>
                <button class="action-btn compare-btn" id="compare-btn" disabled onclick="runDualOCR()">
                    🔄 Dual OCR 비교
                </button>
            </div>
        </div>
        <div class="tech-badges">
            <span class="tech-badge">img2table</span>
            <span class="tech-badge">PaddleOCR</span>
            <span class="tech-badge">granite-vision</span>
            <span class="tech-badge">Dual OCR</span>
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
            <div id="compare-result" style="display: none;"></div>
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
        const compareBtn = document.getElementById('compare-btn');
        const fileInput = document.getElementById('file-input');
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loading-text');
        const imageSection = document.getElementById('image-section');
        const previewImage = document.getElementById('preview-image');
        const imageMeta = document.getElementById('image-meta');
        const emptyState = document.getElementById('empty-state');
        const guideState = document.getElementById('guide-state');
        const ocrResult = document.getElementById('ocr-result');
        const compareResult = document.getElementById('compare-result');

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
            compareResult.style.display = 'none';
            compareResult.innerHTML = '';

            // 버튼 상태
            ocrBtn.disabled = false;
            compareBtn.disabled = true;
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
                    compareBtn.disabled = false;
                } else {
                    ocrResult.innerHTML = `
                        <div class="table-block">
                            <div class="table-header">
                                <span class="table-title">⚠️ Error</span>
                            </div>
                            <div class="ai-status error">${data.error}</div>
                        </div>
                    `;
                    compareBtn.disabled = true;
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
                        <div class="ai-status error">${error.message}</div>
                    </div>
                `;
                compareBtn.disabled = true;
            });
        }

        function runDualOCR() {
            if (!currentFile || !ocrData) {
                alert('Please run OCR first');
                return;
            }

            loading.classList.add('show');
            loadingText.textContent = 'granite-vision으로 재추출 중... (약 30초)';

            const formData = new FormData();
            formData.append('image', currentFile);
            if (ocrData.tables && ocrData.tables.length > 0) {
                formData.append('ocr_html', ocrData.tables[0].html);
            }

            fetch('/api/extract/dual', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('show');
                compareResult.style.display = 'block';

                if (data.success) {
                    compareResult.innerHTML = data.html;
                } else {
                    compareResult.innerHTML = `
                        <div class="table-block">
                            <div class="table-header">
                                <span class="table-title">⚠️ Dual OCR Failed</span>
                            </div>
                            <div class="ai-status error">${data.error}</div>
                        </div>
                    `;
                }
            })
            .catch(error => {
                loading.classList.remove('show');
                compareResult.style.display = 'block';
                compareResult.innerHTML = `
                    <div class="table-block">
                        <div class="table-header">
                            <span class="table-title">⚠️ Error</span>
                        </div>
                        <div class="ai-status error">${error.message}</div>
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
    """
    PaddleOCR만 수행 (v5 - 버튼 분리)
    """
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

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/extract/dual', methods=['POST'])
def api_extract_dual():
    """
    Dual OCR 비교 수행 (v5 - PaddleOCR vs granite-vision)
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})

    file = request.files['image']
    ocr_html = request.form.get('ocr_html', '')

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        img = Image.open(file.stream).convert('RGB')

        # OCR HTML을 2D 배열로 변환
        if not ocr_html:
            return jsonify({'success': False, 'error': 'OCR HTML required. Please run OCR first.'})

        paddle_2d = html_to_2d_array(ocr_html)

        # Dual OCR 비교 (PaddleOCR vs granite-vision)
        dual_result = dual_ocr_check(img, paddle_2d)

        if not dual_result.get('success'):
            return jsonify({
                'success': False,
                'error': dual_result.get('error', 'Dual OCR check failed')
            })

        # 불일치 셀을 differences 형식으로 변환
        differences = []
        for diff in dual_result.get('differences', []):
            differences.append({
                "row": diff['row'],
                "col": diff['col'],
                "ocr": diff['paddle'],
                "ai": diff['second'],
                "type": "mismatch",
                "similarity": diff.get('similarity', 0)
            })

        # HTML 생성
        tables = [{
            'html': ocr_html,
            'has_colspan': 'colspan' in ocr_html,
            'has_rowspan': 'rowspan' in ocr_html,
            'df_shape': (0, 0)
        }]

        html = generate_dual_ocr_result_html(tables, dual_result, differences)

        return jsonify({
            'success': True,
            'html': html,
            'differences': differences,
            'diff_count': len(differences),
            'paddle_cells': dual_result.get('paddle_cells', 0),
            'second_cells': dual_result.get('second_cells', 0)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'version': '5.0',
        'engine': 'img2table + PaddleOCR + granite-vision (Dual OCR Only)',
        'features': ['PaddleOCR', 'Dual OCR Comparison', 'Fuzzy Matching']
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Comet Table Extractor v5.0 (Dual OCR Only)")
    print("=" * 60)
    print("Engine: img2table + PaddleOCR + granite3.2-vision")
    print("Method: Dual OCR Comparison (VLM 코드 완전 제거)")
    print("Port: 6006")
    print("=" * 60)

    # OCR 엔진 미리 초기화
    get_ocr_engine()

    app.run(host='0.0.0.0', port=6006, debug=True)
