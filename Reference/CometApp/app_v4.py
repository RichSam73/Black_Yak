# -*- coding: utf-8 -*-
"""
Comet + ERP Table Extractor v4.2
================================
- 전체 화면 활용 UI (테이블 크게 표시)
- img2table: 테이블 구조 추출 (colspan/rowspan 자동 감지)
- PaddleOCR: 텍스트 인식
- **v4.2 변경사항**: Dual OCR 비교 방식 (Vision 모델 사용)
  - OCR 1: PaddleOCR (img2table 기반)
  - OCR 2: granite3.2-vision (Ollama Vision 모델)
  - 두 OCR 결과를 셀 단위로 비교하여 불일치 셀 감지
  - difflib fuzzy matching으로 유사도 계산
- 포트: 6005
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

# Ollama 설정 (Dual OCR 비교용)
OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen2.5vl:latest"  # VLM 모델 (백업용)
SECOND_OCR_MODEL = "granite3.2-vision:2b"  # 두 번째 OCR 엔진 (Vision 지원)

# Dual OCR 비교 설정
SIMILARITY_THRESHOLD = 0.85  # 85% 이상 유사하면 일치로 판단

# AI 누락/한글철자 검사 프롬프트 (Round 25 - 타겟 검사 방식)
AI_SPELLING_CHECK_PROMPT = """아래 OCR 추출 결과에서 **한글 철자 오류**만 찾아주세요.

OCR 추출 한글 텍스트 목록:
{korean_texts}

다음 유형의 오류만 찾으세요:
1. 유사 글자 혼동: 헹→행, 맷→맛, 잇→있, 랑→량 등
2. 자모 오류: ㄱ↔ㅋ, ㄷ↔ㅌ, ㅂ↔ㅍ 혼동
3. 받침 오류: 받침 누락 또는 잘못된 받침

**숫자, 영어, 특수문자는 무시하세요. OCR 숫자는 정확합니다.**

출력 형식 (JSON):
{{
  "corrections": [
    {{"original": "잘못된글자", "corrected": "올바른글자", "reason": "이유"}}
  ]
}}

오류가 없으면: {{"corrections": []}}
JSON만 출력, 설명 없음."""

AI_MISSING_CHECK_PROMPT = """이 테이블 이미지에서 OCR이 누락한 셀이 있는지 확인해주세요.

OCR이 빈칸으로 인식한 위치들:
{empty_cells}

각 빈칸 위치를 이미지에서 확인하고, 실제로 텍스트가 있는데 누락된 경우만 알려주세요.

**숫자가 있는 셀은 OCR이 정확하니 무시하세요.**
**원래 빈칸인 셀은 무시하세요.**

출력 형식 (JSON):
{{
  "missing": [
    {{"row": 행번호, "col": 열번호, "value": "실제값", "reason": "이유"}}
  ]
}}

누락이 없으면: {{"missing": []}}
JSON만 출력, 설명 없음."""

# 전체 테이블 비교 프롬프트 (Round 26 - 검증 전용 방식으로 개선)
# Round 26 핵심 변경: AI에게 "생성"이 아닌 "검증만" 요청
# - placeholder 텍스트 제거 (VLM 환각 방지)
# - 구체적인 셀 위치 지정으로 명확한 질문
AI_FULL_TABLE_CHECK_PROMPT = """당신은 테이블 이미지를 검증하는 전문가입니다.
아래 OCR 결과에서 빈칸인 셀들이 있습니다. 각 셀을 이미지에서 확인하고 실제로 빈칸인지 답해주세요.

=== 빈칸으로 감지된 셀들 ===
{empty_cells_info}

=== 작업 ===
위 각 셀에 대해:
1. 이미지에서 해당 위치를 확인하세요
2. 실제로 빈칸이면 SKIP
3. 텍스트가 있으면 해당 텍스트를 알려주세요

=== 출력 형식 (JSON) ===
{{"corrections": []}}

corrections 배열에는 실제 텍스트가 있는 셀만 포함:
- row: 행 번호 (정수)
- col: 열 번호 (정수)
- value: 이미지에서 읽은 실제 텍스트

모든 셀이 실제로 빈칸이면 빈 배열을 반환하세요."""

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


def second_ocr_cell(image: Image.Image, cell_bbox: tuple) -> str:
    """
    granite3.2-vision으로 특정 셀 영역의 텍스트 추출

    Args:
        image: 전체 이미지
        cell_bbox: (x1, y1, x2, y2) 셀 좌표

    Returns:
        OCR 결과 텍스트
    """
    try:
        # 셀 영역 크롭
        x1, y1, x2, y2 = cell_bbox
        cell_img = image.crop((x1, y1, x2, y2))

        # base64 변환
        buffered = io.BytesIO()
        cell_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": SECOND_OCR_MODEL,
                "prompt": "Extract all text from this image. Output ONLY the text, nothing else.",
                "images": [img_base64],
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0}
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()

        return ""
    except Exception as e:
        print(f"[second-ocr] Error: {e}")
        return ""


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

        print(f"[second-ocr] 전체 테이블 OCR 시작...")
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
            print(f"[second-ocr] 응답 길이: {len(raw_text)}")

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

            print(f"[second-ocr] 추출된 행: {len(table_2d)}")
            return table_2d

        print(f"[second-ocr] API 오류: {response.status_code}")
        return []

    except Exception as e:
        print(f"[second-ocr] 오류: {e}")
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
                      f"PaddleOCR='{paddle_cell}' vs second='{second_cell}' "
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
                "error": "second OCR 결과 없음"
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
            "error": str(e)
        }


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


def extract_korean_texts(table_2d: list) -> list:
    """
    2D 테이블에서 한글이 포함된 텍스트만 추출

    Returns:
        [{"row": int, "col": int, "text": str}, ...]
    """
    import re
    korean_pattern = re.compile(r'[가-힣]+')

    korean_texts = []
    for row_idx, row in enumerate(table_2d):
        for col_idx, cell in enumerate(row):
            if cell and korean_pattern.search(str(cell)):
                korean_texts.append({
                    "row": row_idx,
                    "col": col_idx,
                    "text": str(cell)
                })
    return korean_texts


def extract_empty_cells(table_2d: list) -> list:
    """
    2D 테이블에서 빈 셀 위치 추출

    Returns:
        [{"row": int, "col": int}, ...]
    """
    empty_cells = []
    for row_idx, row in enumerate(table_2d):
        for col_idx, cell in enumerate(row):
            if not cell or str(cell).strip() == "":
                empty_cells.append({
                    "row": row_idx,
                    "col": col_idx
                })
    return empty_cells


def ai_check_korean_spelling(korean_texts: list) -> dict:
    """
    한글 텍스트 목록에서 철자 오류 검사 (이미지 없이 텍스트만)

    Args:
        korean_texts: [{"row": int, "col": int, "text": str}, ...]

    Returns:
        {
            "success": bool,
            "corrections": [{"original": str, "corrected": str, "reason": str}],
            "error": str or None
        }
    """
    if not korean_texts:
        return {"success": True, "corrections": [], "error": None}

    # 텍스트 목록 생성
    text_list = "\n".join([f"- Row {t['row']+1}, Col {t['col']+1}: {t['text']}" for t in korean_texts])

    prompt = AI_SPELLING_CHECK_PROMPT.format(korean_texts=text_list)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 4096}
            },
            timeout=180  # 72B 모델용 타임아웃 (3분)
        )

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('response', '{}')

            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return {
                        "success": True,
                        "corrections": data.get("corrections", []),
                        "error": None
                    }
                except json.JSONDecodeError as e:
                    return {"success": False, "corrections": [], "error": f"JSON parse error: {e}"}

        return {"success": False, "corrections": [], "error": f"API error: {response.status_code}"}

    except Exception as e:
        return {"success": False, "corrections": [], "error": str(e)}


def ai_check_missing_cells(image: Image.Image, empty_cells: list) -> dict:
    """
    이미지와 빈 셀 위치를 AI에게 보내서 누락 검사

    Args:
        image: PIL Image
        empty_cells: [{"row": int, "col": int}, ...]

    Returns:
        {
            "success": bool,
            "missing": [{"row": int, "col": int, "value": str, "reason": str}],
            "error": str or None
        }
    """
    if not empty_cells:
        return {"success": True, "missing": [], "error": None}

    # 이미지 base64 변환
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 빈 셀 위치 목록 생성
    empty_list = "\n".join([f"- Row {c['row']+1}, Col {c['col']+1}" for c in empty_cells[:50]])  # 최대 50개

    prompt = AI_MISSING_CHECK_PROMPT.format(empty_cells=empty_list)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"num_predict": 4096}
            },
            timeout=180  # 72B 모델용 타임아웃 (3분)
        )

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('response', '{}')

            # JSON 추출
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return {
                        "success": True,
                        "missing": data.get("missing", []),
                        "error": None
                    }
                except json.JSONDecodeError as e:
                    return {"success": False, "missing": [], "error": f"JSON parse error: {e}"}

        return {"success": False, "missing": [], "error": f"API error: {response.status_code}"}

    except Exception as e:
        return {"success": False, "missing": [], "error": str(e)}


def ai_full_table_check(image: Image.Image, table_2d: list) -> dict:
    """
    빈 셀 검증 AI 검사 (Round 26 - 검증 전용 방식)

    핵심 변경 (VLM 환각 방지):
    - 전체 테이블 전송 → 빈 셀만 전송
    - AI에게 "생성" 요청 → "검증만" 요청
    - placeholder 텍스트 제거

    Returns:
        {
            "success": bool,
            "corrections": [...],  # 수정 사항
            "empty_cells_checked": int,  # 검사한 빈 셀 수
            "error": str or None
        }
    """
    # Round 26: 빈 셀만 추출하여 AI에게 전달
    empty_cells = []
    for row_idx, row in enumerate(table_2d):
        for col_idx, cell in enumerate(row):
            if not cell or str(cell).strip() == "":
                empty_cells.append({
                    "row": row_idx,
                    "col": col_idx
                })

    # 빈 셀이 없으면 검사 불필요
    if not empty_cells:
        print(f"[AI Full Check] 빈 셀 없음 - AI 검사 생략")
        return {"success": True, "corrections": [], "empty_cells_checked": 0, "error": None}

    # 빈 셀 정보를 명확하게 전달 (최대 30개로 제한)
    empty_cells_limited = empty_cells[:30]
    empty_cells_info = "\n".join([
        f"셀 [{c['row']+1}행, {c['col']+1}열]: 현재 빈칸"
        for c in empty_cells_limited
    ])

    print(f"[AI Full Check] 빈 셀 {len(empty_cells)}개 중 {len(empty_cells_limited)}개 검사")

    # 이미지 base64 변환
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    prompt = AI_FULL_TABLE_CHECK_PROMPT.format(empty_cells_info=empty_cells_info)

    # Round 26: 간소화된 JSON 스키마 (value만 필요)
    AI_FULL_TABLE_FORMAT = {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer"},
                        "col": {"type": "integer"},
                        "value": {"type": "string"}
                    },
                    "required": ["row", "col", "value"]
                }
            }
        },
        "required": ["corrections"]
    }

    try:
        print(f"[AI Full Check] Ollama API 호출 중... (모델: {VISION_MODEL})")
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "format": AI_FULL_TABLE_FORMAT,
                "options": {
                    "num_predict": 2048,  # 간소화된 응답으로 충분
                    "temperature": 0
                }
            },
            timeout=180
        )

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('response', '{}')

            print(f"[AI Full Check] 응답 수신: 길이={len(raw_text)}")
            print(f"[AI Full Check] 응답 내용: {raw_text[:500]}")

            try:
                data = json.loads(raw_text)
                corrections = data.get("corrections", [])

                # Round 26: 환각 필터링 - placeholder 텍스트 감지
                valid_corrections = []
                hallucination_keywords = ["OCR값", "올바른값", "missing", "wrong", "실제값", "이유"]

                for c in corrections:
                    value = str(c.get("value", "")).strip()
                    # 빈 값이나 환각 키워드 포함된 값 필터링
                    if value and not any(kw in value for kw in hallucination_keywords):
                        valid_corrections.append(c)
                    else:
                        print(f"[AI Full Check] 환각 필터링: row={c.get('row')}, col={c.get('col')}, value='{value}'")

                print(f"[AI Full Check] 유효한 수정사항: {len(valid_corrections)}개 (환각 제거: {len(corrections) - len(valid_corrections)}개)")

                return {
                    "success": True,
                    "corrections": valid_corrections,
                    "empty_cells_checked": len(empty_cells_limited),
                    "error": None
                }
            except json.JSONDecodeError as e:
                print(f"[AI Full Check] JSON 파싱 오류: {e}")
                return {"success": False, "corrections": [], "empty_cells_checked": 0, "error": f"JSON parse error: {e}"}

        print(f"[AI Full Check] API 오류: {response.status_code}")
        return {"success": False, "corrections": [], "empty_cells_checked": 0, "error": f"API error: {response.status_code}"}

    except Exception as e:
        print(f"[AI Full Check] 예외 발생: {e}")
        return {"success": False, "corrections": [], "empty_cells_checked": 0, "error": str(e)}


def ai_targeted_check(image: Image.Image, table_2d: list) -> dict:
    """
    AI 검사 (Round 26 - 빈 셀 검증 전용 방식)

    Returns:
        {
            "success": bool,
            "spelling_corrections": [...],  # 한글 철자 수정 (현재 미사용)
            "missing_cells": [...],          # 누락된 셀 (빈칸 → 실제값)
            "empty_cells_checked": int,      # 검사한 빈 셀 수
            "error": str or None
        }
    """
    # 빈 셀 검증 방식 사용
    result = ai_full_table_check(image, table_2d)

    # Round 26: 새로운 형식으로 변환
    # corrections: [{"row": 1, "col": 2, "value": "실제텍스트"}]
    spelling_corrections = []  # 현재 미사용 (빈 셀 검증에 집중)
    missing_cells = []

    for corr in result.get("corrections", []):
        # Round 26: 간소화된 형식 - row, col, value만 있음
        row = corr.get("row", 0)
        col = corr.get("col", 0)
        value = corr.get("value", "")

        # 1-based → 0-based 변환 (AI가 1-based로 응답)
        missing_cells.append({
            "row": row - 1 if row > 0 else row,
            "col": col - 1 if col > 0 else col,
            "value": value,
            "reason": "AI가 이미지에서 발견"
        })

    return {
        "success": result.get("success", False),
        "spelling_corrections": spelling_corrections,
        "missing_cells": missing_cells,
        "total_corrections": len(missing_cells),
        "empty_cells_checked": result.get("empty_cells_checked", 0),
        "error": result.get("error")
    }


def apply_corrections_to_table(table_2d: list, spelling_corrections: list, missing_cells: list) -> tuple:
    """
    테이블에 AI 수정사항 적용

    Args:
        table_2d: 원본 2D 테이블
        spelling_corrections: 철자 수정 목록
        missing_cells: 누락 셀 목록

    Returns:
        (corrected_table_2d, differences_list)
    """
    import copy
    corrected = copy.deepcopy(table_2d)
    differences = []

    # 1. 철자 수정 적용
    # corrections: [{"original": "헹거루프", "corrected": "행거루프", "reason": "..."}]
    for corr in spelling_corrections:
        original = corr.get("original", "")
        corrected_text = corr.get("corrected", "")

        if not original or not corrected_text:
            continue

        # 테이블에서 해당 텍스트 찾아서 교체
        for row_idx, row in enumerate(corrected):
            for col_idx, cell in enumerate(row):
                if str(cell) == original:
                    corrected[row_idx][col_idx] = corrected_text
                    differences.append({
                        "row": row_idx,
                        "col": col_idx,
                        "ocr": original,
                        "ai": corrected_text,
                        "type": "spelling"
                    })

    # 2. 누락 셀 채우기 (Round 27 버그 수정: 이미 0-based이므로 변환하지 않음)
    # missing: [{"row": 3, "col": 5, "value": "에리안", "reason": "..."}]
    for missing in missing_cells:
        row_idx = missing.get("row", 0)  # 이미 0-based
        col_idx = missing.get("col", 0)  # 이미 0-based
        value = missing.get("value", "")

        if row_idx >= 0 and row_idx < len(corrected):
            if col_idx >= 0 and col_idx < len(corrected[row_idx]):
                old_value = corrected[row_idx][col_idx]
                corrected[row_idx][col_idx] = value
                differences.append({
                    "row": row_idx,
                    "col": col_idx,
                    "ocr": old_value if old_value else "(empty)",
                    "ai": value,
                    "type": "missing"
                })

    return corrected, differences


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


def compare_tables(ocr_table: list, ai_table: list) -> list:
    """
    OCR 테이블과 AI 테이블 비교하여 차이점 반환 (Round 24 개선)

    **개선된 비교 로직**:
    1. AI 결과가 dict이면 rows만 사용 (headers 별도 처리 안 함)
    2. 빈 문자열 vs 빈 문자열은 차이로 취급하지 않음
    3. 한글 OCR 오류만 교정 (숫자 데이터는 OCR이 더 정확할 수 있음)

    Args:
        ocr_table: img2table에서 추출한 2D 배열
        ai_table: AI에서 추출한 결과 (dict 또는 list)

    Returns:
        [{"row": int, "col": int, "ocr": str, "ai": str}, ...]
    """
    differences = []

    # AI 결과가 dict이면 rows만 추출
    if isinstance(ai_table, dict):
        ai_rows = ai_table.get("rows", [])
    else:
        ai_rows = ai_table

    if not ocr_table or not ai_rows:
        return differences

    max_rows = max(len(ocr_table), len(ai_rows))

    for row_idx in range(max_rows):
        ocr_row = ocr_table[row_idx] if row_idx < len(ocr_table) else []
        ai_row = ai_rows[row_idx] if row_idx < len(ai_rows) else []

        max_cols = max(len(ocr_row), len(ai_row))

        for col_idx in range(max_cols):
            ocr_cell = ocr_row[col_idx] if col_idx < len(ocr_row) else ""
            ai_cell = ai_row[col_idx] if col_idx < len(ai_row) else ""

            # 정규화 후 비교
            ocr_normalized = str(ocr_cell).strip()
            ai_normalized = str(ai_cell).strip()

            # 둘 다 비어있으면 차이 아님
            if not ocr_normalized and not ai_normalized:
                continue

            # AI가 빈값을 반환했는데 OCR에 값이 있으면 -> OCR 신뢰 (차이 무시)
            # 숫자 데이터의 경우 OCR이 더 정확할 수 있음
            if ocr_normalized and not ai_normalized:
                # 단, OCR 값이 숫자인 경우에만 OCR 신뢰
                if ocr_normalized.replace(',', '').replace('.', '').isdigit():
                    continue

            # 실제 차이 발견
            if ocr_normalized != ai_normalized:
                differences.append({
                    "row": row_idx,
                    "col": col_idx,
                    "ocr": ocr_normalized,
                    "ai": ai_normalized
                })

    return differences


def generate_highlighted_html(ocr_html: str, differences: list, ai_data: dict) -> str:
    """
    차이점을 하이라이트한 HTML 생성 (Round 24 합의 - Unified Diff)

    - 차이 있는 셀: 노란색 배경
    - OCR 값: 취소선 (빨간색)
    - AI 값: 굵은 글씨 (녹색)
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
                cell['data-diff'] = f"OCR: {diff['ocr']} → AI: {diff['ai']}"

                # 셀 내용 변경 (OCR 취소선 + AI 굵게)
                ocr_value = diff['ocr'] if diff['ocr'] else '(empty)'
                ai_value = diff['ai'] if diff['ai'] else '(empty)'

                new_content = soup.new_tag('span')

                ocr_span = soup.new_tag('span')
                ocr_span['class'] = 'ocr-value'
                ocr_span.string = ocr_value

                arrow = soup.new_tag('span')
                arrow['class'] = 'diff-arrow'
                arrow.string = ' → '

                ai_span = soup.new_tag('span')
                ai_span['class'] = 'ai-value'
                ai_span.string = ai_value

                cell.clear()
                cell.append(ocr_span)
                cell.append(arrow)
                cell.append(ai_span)

            col_idx += colspan

    return str(table)


def validate_table_with_ai(image: Image.Image, table_html: str) -> dict:
    """AI를 사용하여 테이블 검증"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    prompt = f"""Analyze this table image and the extracted HTML.
Check for:
1. Sum/total errors (do row/column totals match?)
2. Missing data
3. OCR errors

Extracted HTML:
{table_html[:2000]}

Respond in JSON format:
{{
    "is_valid": true/false,
    "errors": ["error1", "error2"],
    "warnings": ["warning1"],
    "corrections": {{"cell_location": "correct_value"}}
}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"num_predict": 4096}
            },
            timeout=180  # 72B 모델용 타임아웃 (3분)
        )

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('response', '{}')

            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', raw_text)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"is_valid": True, "errors": [], "warnings": ["AI validation skipped"]}

    except Exception as e:
        return {"is_valid": True, "errors": [], "warnings": [f"AI validation failed: {str(e)}"]}


def generate_final_table_html(ocr_html: str, differences: list, ai_data: dict) -> str:
    """
    최종 완성본 테이블 생성 - AI 보정값을 적용한 최종 결과

    Args:
        ocr_html: 원본 OCR HTML
        differences: OCR vs AI 차이점 목록
        ai_data: AI 추출 결과

    Returns:
        보정 완료된 최종 테이블 HTML
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(ocr_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ocr_html

    # 차이점을 (row, col) -> ai_value 매핑
    correction_map = {(d["row"], d["col"]): d["ai"] for d in differences}

    rows = table.find_all('tr')

    # 최대 열 수 계산
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

            # 이 셀에 보정이 필요한지 확인
            if (row_idx, col_idx) in correction_map:
                ai_value = correction_map[(row_idx, col_idx)]

                # 보정된 셀 표시 (녹색 배경)
                existing_class = cell.get('class', [])
                if isinstance(existing_class, str):
                    existing_class = [existing_class]
                existing_class.append('corrected-cell')
                cell['class'] = existing_class

                # AI 값으로 교체
                cell.clear()
                cell.string = ai_value if ai_value else ""

            col_idx += colspan

    return str(table)


def generate_ocr_result_html(tables: list) -> str:
    """
    OCR만 수행한 결과를 HTML로 렌더링 (v4 - OCR 전용)

    Args:
        tables: img2table 추출 결과
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
                <span class="table-title">📋 OCR Result</span>
                <span class="table-meta">{table_meta}</span>
            </div>
            <div class="section-label">img2table + PaddleOCR 결과</div>
            <div class="table-content">
                {table['html']}
            </div>
        </div>
        """)

    return '\n'.join(html_parts)


def generate_ai_result_html(tables: list, ai_result: dict, differences: list) -> str:
    """
    AI 검사 결과를 HTML로 렌더링 (v4 - AI 전용)

    Args:
        tables: img2table 추출 결과
        ai_result: AI 검사 결과
        differences: 차이점 목록
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

        # AI 검사 결과 표시
        if ai_result and ai_result.get('success'):
            differences = differences or []
            diff_count = len(differences)

            # 유형별 분류
            spelling_diffs = [d for d in differences if d.get('type') == 'spelling']
            missing_diffs = [d for d in differences if d.get('type') == 'missing']

            # 하이라이트 적용
            highlighted_html = generate_highlighted_html(table['html'], differences, ai_result)

            # 검사 통계
            korean_checked = ai_result.get('korean_texts_checked', 0)
            empty_checked = ai_result.get('empty_cells_checked', 0)

            diff_details = ""
            if diff_count > 0:
                diff_items = []

                # 철자 오류
                if spelling_diffs:
                    diff_items.append("<li><strong>📝 한글 철자 오류:</strong></li>")
                    for d in spelling_diffs[:10]:
                        diff_items.append(
                            f"<li style='margin-left:20px'>Row {d['row']+1}, Col {d['col']+1}: "
                            f"<span class='ocr-value'>{d['ocr']}</span> → <span class='ai-value'>{d['ai']}</span></li>"
                        )

                # 누락 셀
                if missing_diffs:
                    diff_items.append("<li><strong>📥 누락된 셀:</strong></li>")
                    for d in missing_diffs[:10]:
                        diff_items.append(
                            f"<li style='margin-left:20px'>Row {d['row']+1}, Col {d['col']+1}: "
                            f"<span class='ocr-value'>{d['ocr']}</span> → <span class='ai-value'>{d['ai']}</span></li>"
                        )

                diff_details = f"<ul class='diff-list'>{''.join(diff_items)}</ul>"

            # AI 검사 결과 메시지
            if diff_count == 0:
                status_msg = "✅ 오류 없음 - OCR 결과가 정확합니다"
                status_class = "success"
            else:
                status_msg = f"철자오류 <span class='diff-count'>{len(spelling_diffs)}</span>개 | 누락 <span class='diff-count'>{len(missing_diffs)}</span>개 발견"
                status_class = "has-diff"

            html_parts.append(f"""
            <div class="table-block section-diff">
                <div class="table-header">
                    <span class="table-title">🔍 AI Verification Result</span>
                    <span class="table-meta">한글 {korean_checked}개 검사 | 빈셀 {empty_checked}개 검사</span>
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

            # 최종 완성본 (차이점이 있을 때만)
            if diff_count > 0:
                final_html = generate_final_table_html(table['html'], differences, ai_result)

                html_parts.append(f"""
                <div class="table-block section-final">
                    <div class="table-header">
                        <span class="table-title">✅ Final Corrected Table</span>
                        <span class="table-meta">AI 보정 적용 완료</span>
                    </div>
                    <div class="section-label">최종 완성본 (보정된 셀: 녹색 배경)</div>
                    <div class="table-content">
                        {final_html}
                    </div>
                </div>
                """)

        elif ai_result and ai_result.get('error'):
            html_parts.append(f"""
            <div class="table-block section-diff">
                <div class="table-header">
                    <span class="table-title">⚠️ AI Verification Failed</span>
                </div>
                <div class="ai-status error">오류: {ai_result['error']}</div>
            </div>
            """)

    return '\n'.join(html_parts)


def process_image_ocr_only(img: Image.Image) -> dict:
    """
    OCR만 수행 (v4 - 버튼 분리용)
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


# 새로운 UI 레이아웃 - v4: OCR/AI 버튼 분리
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comet Table Extractor v4.0</title>
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

        .ai-btn {
            background: #0f3460;
            border: 2px solid #28a745;
        }

        .ai-btn:hover:not(:disabled) {
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

        /* AI 처리 상태 표시 */
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

        .ai-status .spinner-small {
            width: 16px;
            height: 16px;
            border: 2px solid #0f3460;
            border-top: 2px solid #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        /* Diff 요약 박스 */
        .diff-summary {
            background: rgba(255, 243, 205, 0.3);
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .diff-summary h4 {
            color: #ffc107;
            margin-bottom: 10px;
        }

        .diff-count {
            font-size: 24px;
            font-weight: bold;
            color: #ffc107;
        }

        /* 3단 수직 배치 섹션 스타일 */
        .section-original .table-header {
            background: #0f3460;
            border-left: 4px solid #00d9ff;
        }

        .section-diff .table-header {
            background: #3d2c00;
            border-left: 4px solid #ffc107;
        }

        .section-final .table-header {
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

        .section-label .diff-count {
            font-size: 16px;
            color: #ffc107;
        }

        /* 보정된 셀 스타일 (최종본) */
        .corrected-cell {
            background: rgba(40, 167, 69, 0.3) !important;
            border: 1px solid #28a745 !important;
        }

        /* 차이점 목록 */
        .diff-list {
            max-height: 150px;
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
    <!-- 상단 헤더 - 버튼 3개 포함 -->
    <div class="header">
        <div class="header-left">
            <div class="logo">
                <h1>Comet Table Extractor</h1>
                <span class="version">v4.0</span>
            </div>
            <div class="btn-group">
                <input type="file" id="file-input" accept="image/*">
                <button class="action-btn upload-btn" id="upload-btn" onclick="document.getElementById('file-input').click()">
                    📤 Upload
                </button>
                <button class="action-btn ocr-btn" id="ocr-btn" disabled onclick="runOCR()">
                    🔍 Run OCR
                </button>
                <button class="action-btn ai-btn" id="ai-btn" disabled onclick="runAI()">
                    🔄 Compare OCR
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

    <!-- 메인 컨테이너 - 수직 스크롤 -->
    <div class="main-container">
        <!-- 이미지 섹션 - 맨 위 (업로드 후 표시) -->
        <div class="image-section" id="image-section">
            <div class="section-header">
                <span class="section-title">🖼️ Uploaded Image</span>
                <span class="section-meta" id="image-meta"></span>
            </div>
            <div class="image-content">
                <img id="preview-image" class="preview-image">
            </div>
        </div>

        <!-- 안내 상태 (이미지 업로드 후 OCR 전) -->
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
            <div id="ai-result" style="display: none;"></div>
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
        const aiBtn = document.getElementById('ai-btn');
        const fileInput = document.getElementById('file-input');
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loading-text');
        const imageSection = document.getElementById('image-section');
        const previewImage = document.getElementById('preview-image');
        const imageMeta = document.getElementById('image-meta');
        const emptyState = document.getElementById('empty-state');
        const guideState = document.getElementById('guide-state');
        const ocrResult = document.getElementById('ocr-result');
        const aiResult = document.getElementById('ai-result');

        // 상태 관리
        let currentFile = null;
        let ocrData = null;  // OCR 결과 저장

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
            aiResult.style.display = 'none';
            aiResult.innerHTML = '';

            // 버튼 상태
            ocrBtn.disabled = false;
            aiBtn.disabled = true;
        }

        function runOCR() {
            if (!currentFile) {
                alert('Please upload an image first');
                return;
            }

            loading.classList.add('show');
            loadingText.textContent = 'Running OCR...';
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
                    ocrData = data;  // OCR 결과 저장
                    ocrResult.innerHTML = data.html;
                    aiBtn.disabled = false;  // AI 버튼 활성화
                } else {
                    ocrResult.innerHTML = `
                        <div class="validation-box error">
                            <strong>Error</strong>
                            <p>${data.error}</p>
                        </div>
                    `;
                    aiBtn.disabled = true;
                }
            })
            .catch(error => {
                loading.classList.remove('show');
                ocrResult.style.display = 'block';
                ocrResult.innerHTML = `
                    <div class="validation-box error">
                        <strong>Error</strong>
                        <p>${error.message}</p>
                    </div>
                `;
                aiBtn.disabled = true;
            });
        }

        function runAI() {
            if (!currentFile || !ocrData) {
                alert('Please run OCR first');
                return;
            }

            loading.classList.add('show');
            loadingText.textContent = 'Comparing with granite-vision...';

            const formData = new FormData();
            formData.append('image', currentFile);
            // OCR HTML 전달 (AI 검사용)
            if (ocrData.tables && ocrData.tables.length > 0) {
                formData.append('ocr_html', ocrData.tables[0].html);
            }

            fetch('/api/extract/ai', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('show');
                aiResult.style.display = 'block';

                if (data.success) {
                    // AI 결과 HTML 생성 요청
                    renderAIResult(data);
                } else {
                    aiResult.innerHTML = `
                        <div class="validation-box error">
                            <strong>Error</strong>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            })
            .catch(error => {
                loading.classList.remove('show');
                aiResult.style.display = 'block';
                aiResult.innerHTML = `
                    <div class="validation-box error">
                        <strong>Error</strong>
                        <p>${error.message}</p>
                    </div>
                `;
            });
        }

        function renderAIResult(data) {
            // AI 결과 HTML 렌더링 요청
            const formData = new FormData();
            formData.append('image', currentFile);
            formData.append('ai_result', JSON.stringify(data.ai_result));
            formData.append('differences', JSON.stringify(data.differences));
            if (ocrData.tables && ocrData.tables.length > 0) {
                formData.append('ocr_html', ocrData.tables[0].html);
            }

            fetch('/api/render/ai', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(renderData => {
                if (renderData.success) {
                    aiResult.innerHTML = renderData.html;
                } else {
                    // Fallback: 간단한 결과 표시
                    const diffCount = data.diff_count || 0;
                    if (diffCount === 0) {
                        aiResult.innerHTML = `
                            <div class="table-block section-diff">
                                <div class="table-header">
                                    <span class="table-title">🔍 AI Verification Result</span>
                                </div>
                                <div class="section-label success">✅ 오류 없음 - OCR 결과가 정확합니다</div>
                            </div>
                        `;
                    } else {
                        aiResult.innerHTML = `
                            <div class="table-block section-diff">
                                <div class="table-header">
                                    <span class="table-title">🔍 AI Verification Result</span>
                                </div>
                                <div class="section-label has-diff">${diffCount}개 수정 사항 발견</div>
                            </div>
                        `;
                    }
                }
            })
            .catch(error => {
                // Fallback
                const diffCount = data.diff_count || 0;
                aiResult.innerHTML = `
                    <div class="table-block section-diff">
                        <div class="table-header">
                            <span class="table-title">🔍 AI Verification Result</span>
                        </div>
                        <div class="section-label">${diffCount}개 수정 사항 발견</div>
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/extract/ocr', methods=['POST'])
def api_extract_ocr():
    """
    OCR만 수행 (v4 - 버튼 분리)
    AI 보정 없이 빠르게 결과 반환
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


@app.route('/api/extract/ai', methods=['POST'])
def api_extract_ai():
    """
    Dual OCR 비교 수행 (v4.2 - Vision 모델 사용)

    변경 사항:
    - VLM (Qwen2.5-VL) 대신 granite3.2-vision 사용
    - PaddleOCR vs granite3.2-vision 결과 비교
    - difflib fuzzy matching으로 불일치 셀 감지
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

        # Dual OCR 비교 (PaddleOCR vs deepseek-ocr)
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
                "ocr": diff['paddle'],  # PaddleOCR 값
                "ai": diff['second'],  # second OCR 값
                "type": "mismatch",
                "similarity": diff.get('similarity', 0)
            })

        return jsonify({
            'success': True,
            'ai_result': {
                'success': True,
                'method': 'dual_ocr',
                'paddle_cells': dual_result.get('paddle_cells', 0),
                'second_cells': dual_result.get('second_cells', 0),
                'mismatch_count': dual_result.get('mismatch_count', 0),
                'korean_texts_checked': 0,  # 호환성 유지
                'empty_cells_checked': 0    # 호환성 유지
            },
            'differences': differences,
            'diff_count': len(differences),
            'correction_type': 'dual_ocr'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/render/ai', methods=['POST'])
def api_render_ai():
    """
    AI 결과 HTML 렌더링 (v4 - 클라이언트 요청용)
    """
    try:
        ocr_html = request.form.get('ocr_html', '')
        ai_result_json = request.form.get('ai_result', '{}')
        differences_json = request.form.get('differences', '[]')

        if not ocr_html:
            return jsonify({'success': False, 'error': 'OCR HTML required'})

        ai_result = json.loads(ai_result_json)
        differences = json.loads(differences_json)

        # 테이블 구조 생성
        tables = [{
            'html': ocr_html,
            'has_colspan': 'colspan' in ocr_html,
            'has_rowspan': 'rowspan' in ocr_html,
            'df_shape': (0, 0)
        }]

        html = generate_ai_result_html(tables, ai_result, differences)

        return jsonify({
            'success': True,
            'html': html
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'version': '4.1',
        'engine': 'img2table + PaddleOCR + granite-vision (Dual OCR)',
        'features': ['OCR', 'Dual OCR Comparison', 'Fuzzy Matching', 'Separated Buttons']
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Comet Table Extractor v4.2")
    print("=" * 60)
    print("Engine: img2table + PaddleOCR + granite3.2-vision (Dual OCR)")
    print("Port: 6005")
    print("New: Dual OCR comparison (PaddleOCR vs granite-vision)")
    print("=" * 60)

    # OCR 엔진 미리 초기화
    get_ocr_engine()

    app.run(host='0.0.0.0', port=6005, debug=True)
