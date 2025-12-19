"""
스마트 테이블 추출기 v4
Perplexity Comet 방식: 원본 PDF 이미지 위에 OCR 텍스트를 투명 오버레이로 배치
사용자가 텍스트를 선택/복사할 수 있게 함 (100% 정확도)

v4 변경사항:
- Comet 방식 OCR 오버레이 시스템 추가
- PDF 이미지 + OCR 텍스트 좌표 기반 투명 오버레이
- HTML/CSS로 텍스트 선택 가능하게 렌더링

v3 변경사항:
- Granite3.2-vision VLM 추가 (문서 이해 특화 모델)
- Ollama 기반 로컬 VLM 추출 모드
- Table Transformer는 폴백으로 유지

v2 변경사항:
- PaddleOCR 사용 (Tesseract 대비 한글 인식률 향상)
- PP-OCRv5 모델 적용
"""

import fitz
from PIL import Image
import io
import os
import json
import re
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 환경 변수 설정 (PaddleOCR 연결 체크 스킵)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Ollama VLM 설정
OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None

# PaddleOCR 설정
PADDLEOCR_AVAILABLE = False
_paddle_ocr = None

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None

# Tesseract OCR 설정 (폴백용)
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSDATA_DIR = str(Path(__file__).parent / "tessdata")
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None

# OCR 사용 가능 여부
OCR_AVAILABLE = PADDLEOCR_AVAILABLE or TESSERACT_AVAILABLE

# Table Transformer 설정
TABLE_TRANSFORMER_AVAILABLE = False
try:
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TABLE_TRANSFORMER_AVAILABLE = True
except ImportError:
    torch = None
    AutoImageProcessor = None
    TableTransformerForObjectDetection = None


# 모델 캐시 (한 번만 로드)
_detection_model = None
_detection_processor = None
_structure_model = None
_structure_processor = None


def get_paddle_ocr():
    """PaddleOCR 인스턴스 가져오기 (싱글톤)"""
    global _paddle_ocr
    if _paddle_ocr is None and PADDLEOCR_AVAILABLE:
        _paddle_ocr = PaddleOCR(lang='korean')
    return _paddle_ocr


def load_detection_model():
    """테이블 감지 모델 로드 (캐시)"""
    global _detection_model, _detection_processor
    if _detection_model is None:
        _detection_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        _detection_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
    return _detection_processor, _detection_model


def load_structure_model():
    """테이블 구조 인식 모델 로드 (캐시)"""
    global _structure_model, _structure_processor
    if _structure_model is None:
        _structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        _structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
    return _structure_processor, _structure_model


def pdf_page_to_image(pdf_bytes: bytes, page_num: int = 0, zoom: float = 2.0) -> Image.Image:
    """PDF 페이지를 PIL Image로 변환"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    doc.close()
    return img


def detect_tables(image: Image.Image, threshold: float = 0.7) -> list:
    """이미지에서 테이블 영역 감지"""
    if not TABLE_TRANSFORMER_AVAILABLE:
        return []

    processor, model = load_detection_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = [int(x) for x in box.tolist()]
        tables.append({
            "score": round(score.item(), 3),
            "label": model.config.id2label[label.item()],
            "box": box_coords  # [x1, y1, x2, y2]
        })

    return tables


def detect_table_structure(table_image: Image.Image, threshold: float = 0.5) -> dict:
    """테이블 이미지에서 셀/행/열 구조 감지"""
    if not TABLE_TRANSFORMER_AVAILABLE:
        return {"cells": [], "rows": [], "columns": []}

    processor, model = load_structure_model()
    inputs = processor(images=table_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([table_image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    cells = []
    rows = []
    columns = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = [int(x) for x in box.tolist()]
        label_name = model.config.id2label[label.item()]

        item = {
            "score": round(score.item(), 3),
            "box": box_coords
        }

        if "cell" in label_name:
            cells.append(item)
        elif "row" in label_name:
            rows.append(item)
        elif "column" in label_name:
            columns.append(item)

    return {
        "cells": cells,
        "rows": rows,
        "columns": columns
    }


def ocr_cell_paddle(image: Image.Image, box: list) -> str:
    """PaddleOCR로 셀 텍스트 추출"""
    ocr = get_paddle_ocr()
    if ocr is None:
        return ""

    x1, y1, x2, y2 = box
    cell_img = image.crop((x1, y1, x2, y2))

    # PIL Image를 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cell_img.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = ocr.predict(tmp_path)
        os.unlink(tmp_path)

        if result:
            texts = []
            for res in result:
                if 'rec_texts' in res:
                    texts.extend(res['rec_texts'])
            return ' '.join(texts).strip()
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return ""

    return ""


def ocr_cell_tesseract(image: Image.Image, box: list) -> str:
    """Tesseract OCR로 셀 텍스트 추출 (폴백)"""
    if not TESSERACT_AVAILABLE:
        return ""

    x1, y1, x2, y2 = box
    cell_img = image.crop((x1, y1, x2, y2))

    # OCR 실행 (한글+영어)
    text = pytesseract.image_to_string(cell_img, lang='kor+eng').strip()
    return text


def ocr_cell(image: Image.Image, box: list) -> str:
    """셀 이미지에서 OCR로 텍스트 추출 (PaddleOCR 우선, Tesseract 폴백)"""
    if PADDLEOCR_AVAILABLE:
        return ocr_cell_paddle(image, box)
    elif TESSERACT_AVAILABLE:
        return ocr_cell_tesseract(image, box)
    return ""


def ocr_full_image_paddle(image: Image.Image) -> list:
    """PaddleOCR로 전체 이미지 OCR (더 빠른 방식)"""
    ocr = get_paddle_ocr()
    if ocr is None:
        return []

    # PIL Image를 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = ocr.predict(tmp_path)
        os.unlink(tmp_path)

        if result:
            ocr_results = []
            for res in result:
                if 'rec_texts' in res and 'rec_polys' in res:
                    texts = res['rec_texts']
                    polys = res['rec_polys']
                    scores = res.get('rec_scores', [1.0] * len(texts))

                    for text, poly, score in zip(texts, polys, scores):
                        # poly는 4개의 점 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        if len(poly) >= 4:
                            x_coords = [p[0] for p in poly]
                            y_coords = [p[1] for p in poly]
                            # numpy int16 → Python int 변환 (JSON 직렬화 호환)
                            box = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                            ocr_results.append({
                                "text": text,
                                "box": box,
                                "score": float(score) if hasattr(score, 'item') else score
                            })
            return ocr_results
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return []


def organize_cells_to_table(cells: list, table_width: int, table_height: int) -> list:
    """셀들을 행/열 구조로 정리하여 2D 테이블 생성"""
    if not cells:
        return []

    # 셀들을 y 좌표(행)로 먼저 정렬, 같은 행 내에서는 x 좌표(열)로 정렬
    sorted_cells = sorted(cells, key=lambda c: (c["box"][1], c["box"][0]))

    # 행 그룹 생성 (y 좌표가 비슷한 셀들을 같은 행으로)
    rows = []
    current_row = []
    current_y = None
    y_tolerance = 20  # 같은 행으로 간주할 y 오차 범위

    for cell in sorted_cells:
        y = cell["box"][1]

        if current_y is None:
            current_y = y
            current_row = [cell]
        elif abs(y - current_y) <= y_tolerance:
            current_row.append(cell)
        else:
            # 새로운 행 시작
            rows.append(sorted(current_row, key=lambda c: c["box"][0]))
            current_row = [cell]
            current_y = y

    if current_row:
        rows.append(sorted(current_row, key=lambda c: c["box"][0]))

    # 텍스트만 추출하여 2D 리스트로 변환
    table_data = []
    for row in rows:
        row_data = [cell.get("text", "") for cell in row]
        table_data.append(row_data)

    return table_data


def extract_smart_tables(pdf_bytes: bytes, progress_callback=None, use_paddle=True) -> dict:
    """
    스마트 테이블 추출 (Comet 방식)
    1. PDF를 이미지로 변환
    2. Table Transformer로 테이블 영역 감지
    3. 각 테이블의 셀 구조 인식
    4. PaddleOCR/Tesseract로 각 셀의 텍스트 추출
    5. 구조화된 테이블 데이터 반환

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백 함수 (page, total, message)
        use_paddle: PaddleOCR 사용 여부 (False면 Tesseract 사용)
    """
    if not TABLE_TRANSFORMER_AVAILABLE:
        return {
            "tables": [],
            "error": "Table Transformer가 설치되지 않았습니다.",
            "is_ai_extracted": False
        }

    if not OCR_AVAILABLE:
        return {
            "tables": [],
            "error": "OCR(PaddleOCR 또는 Tesseract)이 설치되지 않았습니다.",
            "is_ai_extracted": False
        }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # OCR 엔진 정보
    ocr_engine = "PaddleOCR" if (use_paddle and PADDLEOCR_AVAILABLE) else "Tesseract"

    result = {
        "tables": [],
        "is_ai_extracted": True,
        "total_pages": total_pages,
        "ocr_engine": ocr_engine
    }

    for page_num in range(total_pages):
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"페이지 {page_num + 1}/{total_pages} 처리 중...")

        # PDF 페이지를 이미지로 변환
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # 2x 확대
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        page_image = Image.open(io.BytesIO(img_bytes))

        # 테이블 영역 감지
        detected_tables = detect_tables(page_image, threshold=0.7)

        for table_idx, table_info in enumerate(detected_tables):
            if progress_callback:
                progress_callback(
                    page_num + 1, total_pages,
                    f"페이지 {page_num + 1} - 테이블 {table_idx + 1} 구조 분석 중..."
                )

            # 테이블 영역 크롭
            x1, y1, x2, y2 = table_info["box"]
            table_image = page_image.crop((x1, y1, x2, y2))

            # 셀 구조 인식
            structure = detect_table_structure(table_image, threshold=0.5)
            cells = structure["cells"]

            if not cells:
                continue

            # 각 셀에서 OCR로 텍스트 추출
            if progress_callback:
                progress_callback(
                    page_num + 1, total_pages,
                    f"페이지 {page_num + 1} - 테이블 {table_idx + 1} OCR 중... ({ocr_engine})"
                )

            for cell in cells:
                if use_paddle and PADDLEOCR_AVAILABLE:
                    cell["text"] = ocr_cell_paddle(table_image, cell["box"])
                else:
                    cell["text"] = ocr_cell_tesseract(table_image, cell["box"])

            # 셀들을 2D 테이블 구조로 정리
            table_data = organize_cells_to_table(
                cells,
                table_image.width,
                table_image.height
            )

            if table_data:
                result["tables"].append({
                    "page": page_num + 1,
                    "table_index": table_idx + 1,
                    "confidence": table_info["score"],
                    "data": table_data,
                    "cell_count": len(cells),
                    "row_count": len(table_data),
                    "col_count": max(len(row) for row in table_data) if table_data else 0
                })

    doc.close()
    return result


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """PDF가 스캔(이미지 기반)인지 확인"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(min(3, len(doc))):  # 처음 3페이지만 확인
        page = doc[page_num]
        text = page.get_text().strip()

        if len(text) > 50:  # 텍스트가 있으면 텍스트 PDF
            doc.close()
            return False

    doc.close()
    return True


# ============================================================
# VLM (Vision Language Model) 기반 테이블 추출 - Comet 방식
# ============================================================

def check_ollama_model(model_name: str = "granite3.2-vision") -> bool:
    """Ollama에서 모델 사용 가능 여부 확인"""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        result = ollama.list()
        # ollama.list()는 models 속성을 가진 객체 반환
        models_list = getattr(result, 'models', None)
        if models_list is None and isinstance(result, dict):
            models_list = result.get('models', [])
        if models_list is None:
            models_list = []

        model_names = []
        for m in models_list:
            # Model 객체 또는 딕셔너리 처리
            name = getattr(m, 'model', None)
            if name is None and isinstance(m, dict):
                name = m.get('model')
            if name:
                model_names.append(name.split(':')[0])

        return model_name.split(':')[0] in model_names
    except Exception as e:
        print(f"Ollama 모델 확인 오류: {e}")
        return False


def extract_table_with_vlm(image_path: str, model: str = "granite3.2-vision") -> dict:
    """
    VLM을 사용하여 이미지에서 테이블 데이터 추출
    Perplexity Comet 방식: AI가 이미지를 보고 직접 테이블 내용을 이해

    Args:
        image_path: 이미지 파일 경로
        model: 사용할 Ollama VLM 모델

    Returns:
        추출된 테이블 데이터 딕셔너리
    """
    if not OLLAMA_AVAILABLE:
        return {"error": "Ollama가 설치되지 않았습니다.", "raw_text": ""}

    # VLM 프롬프트 (ellipsis ... 제거하여 JSON 파싱 오류 방지)
    prompt = '''This is a garment manufacturing document (WORK SHEET).
Extract ALL text and data from this document image.

Focus on these key fields if present:
- STYLE NO / Style Number
- SEASON
- COLOR WAY (all color codes like BK, NA, SV and their names)
- GOODS KIND / Product type
- BRAND
- FACTORY
- ORDER NO
- DELIVERY DATE
- Any table data with rows and columns

Return the data as valid JSON. Example format:
{
    "fields": {
        "STYLE_NO": "ABC123",
        "SEASON": "2024FW",
        "COLOR_WAY": [{"code": "BK", "name": "BLACK"}, {"code": "NA", "name": "NAVY"}],
        "GOODS_KIND": "JACKET",
        "BRAND": "BRAND_NAME"
    },
    "tables": [
        {
            "title": "Size Chart",
            "headers": ["SIZE", "CHEST", "LENGTH"],
            "rows": [["S", "100", "65"], ["M", "105", "67"]]
        }
    ],
    "raw_text": "all visible text"
}

IMPORTANT:
- Return ONLY valid JSON, no ellipsis (...) or placeholder text
- Only include fields that are actually visible in the image
- If no tables found, use empty array: "tables": []'''

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }],
            options={'num_predict': 4000, 'temperature': 0.1}
        )

        content = response['message']['content']

        # JSON 추출 시도
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # VLM이 ellipsis를 포함했을 경우 제거 (JSON 파싱 오류 방지)
            # 패턴: , ...] 또는 , ...]
            json_str = re.sub(r',\s*\.\.\.(\s*[\]\}])', r'\1', json_str)
            # 패턴: [..., ...] -> [...] (배열 끝의 ellipsis)
            json_str = re.sub(r'\.\.\.\s*,?\s*(?=[\]\}])', '', json_str)

            try:
                data = json.loads(json_str)
                data['raw_response'] = content
                return data
            except json.JSONDecodeError as e:
                # 디버그용: 파싱 실패 시 원본 저장
                return {
                    "raw_text": content,
                    "fields": {},
                    "tables": [],
                    "parse_error": str(e),
                    "json_attempted": json_str[:500]
                }

        # JSON 파싱 실패 시 텍스트로 반환
        return {
            "raw_text": content,
            "fields": {},
            "tables": []
        }

    except Exception as e:
        return {
            "error": str(e),
            "raw_text": ""
        }


def extract_vlm_tables(pdf_bytes: bytes, progress_callback=None, model: str = "granite3.2-vision") -> dict:
    """
    VLM(Vision Language Model)을 사용하여 PDF에서 테이블 추출
    Perplexity Comet 방식: AI가 문서를 보고 직접 이해하여 데이터 추출

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백 함수 (page, total, message)
        model: 사용할 VLM 모델 (기본: granite3.2-vision)

    Returns:
        추출 결과 딕셔너리
    """
    if not OLLAMA_AVAILABLE:
        return {
            "tables": [],
            "error": "Ollama가 설치되지 않았습니다. 'pip install ollama' 실행 필요",
            "is_ai_extracted": False,
            "extraction_method": "vlm"
        }

    if not check_ollama_model(model):
        return {
            "tables": [],
            "error": f"모델 '{model}'이 설치되지 않았습니다. 'ollama pull {model}' 실행 필요",
            "is_ai_extracted": False,
            "extraction_method": "vlm"
        }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    result = {
        "tables": [],
        "pages": [],
        "is_ai_extracted": True,
        "total_pages": total_pages,
        "extraction_method": "vlm",
        "model": model
    }

    for page_num in range(total_pages):
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"페이지 {page_num + 1}/{total_pages} VLM 분석 중...")

        # PDF 페이지를 이미지로 변환
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # 2x 확대로 더 선명하게
        pix = page.get_pixmap(matrix=mat)

        # 임시 파일로 저장 (Windows 호환)
        tmp_path = os.path.join(tempfile.gettempdir(), f"vlm_page_{page_num}.png")
        pix.save(tmp_path)

        try:
            # VLM으로 추출
            vlm_result = extract_table_with_vlm(tmp_path, model)

            # 결과 저장
            page_data = {
                "page": page_num + 1,
                "fields": vlm_result.get("fields", {}),
                "tables": vlm_result.get("tables", []),
                "raw_text": vlm_result.get("raw_text", "")
            }

            if "error" in vlm_result:
                page_data["error"] = vlm_result["error"]

            result["pages"].append(page_data)

            # 테이블 데이터를 표준 형식으로 변환
            for table_idx, table in enumerate(vlm_result.get("tables", [])):
                table_data = []

                # 헤더가 있으면 추가
                if table.get("headers"):
                    table_data.append(table["headers"])

                # 행 데이터 추가
                for row in table.get("rows", []):
                    table_data.append(row)

                if table_data:
                    result["tables"].append({
                        "page": page_num + 1,
                        "table_index": table_idx + 1,
                        "confidence": 0.95,  # VLM은 신뢰도를 직접 제공하지 않음
                        "data": table_data,
                        "title": table.get("title", ""),
                        "row_count": len(table_data),
                        "col_count": max(len(row) for row in table_data) if table_data else 0
                    })

            # 필드 데이터도 테이블로 변환 (키-값 쌍)
            fields = vlm_result.get("fields", {})
            if fields:
                field_table = []
                for key, value in fields.items():
                    if isinstance(value, list):
                        # COLOR_WAY 등 리스트 형태 처리
                        for item in value:
                            if isinstance(item, dict):
                                field_table.append([key, f"{item.get('code', '')} - {item.get('name', '')}"])
                            else:
                                field_table.append([key, str(item)])
                    else:
                        field_table.append([key, str(value)])

                if field_table:
                    result["tables"].insert(0, {
                        "page": page_num + 1,
                        "table_index": 0,
                        "confidence": 0.95,
                        "data": [["필드", "값"]] + field_table,
                        "title": "문서 필드",
                        "row_count": len(field_table) + 1,
                        "col_count": 2
                    })

        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    doc.close()
    return result


# ============================================================
# 통합 추출 함수
# ============================================================

# ============================================================
# Comet 방식 OCR 오버레이 시스템
# PDF 원본 이미지 + 투명 OCR 텍스트 레이어 = 선택 가능한 텍스트
# ============================================================

import base64


def extract_ocr_with_coordinates(pdf_bytes: bytes, progress_callback=None, zoom: float = 2.0) -> dict:
    """
    Comet 방식: PDF 페이지별로 원본 이미지 + OCR 텍스트 좌표 추출

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백 함수 (page, total, message)
        zoom: 이미지 확대 배율 (기본 2.0)

    Returns:
        {
            "pages": [
                {
                    "page": 1,
                    "image_base64": "...",  # PNG 이미지 base64
                    "width": 1200,
                    "height": 1600,
                    "ocr_results": [
                        {"text": "텍스트", "box": [x1, y1, x2, y2], "score": 0.99},
                        ...
                    ]
                },
                ...
            ],
            "total_pages": 6,
            "extraction_method": "comet"
        }
    """
    if not PADDLEOCR_AVAILABLE:
        return {
            "pages": [],
            "error": "PaddleOCR가 설치되지 않았습니다.",
            "extraction_method": "comet"
        }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    result = {
        "pages": [],
        "total_pages": total_pages,
        "extraction_method": "comet",
        "ocr_engine": "PaddleOCR PP-OCRv5"
    }

    for page_num in range(total_pages):
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"페이지 {page_num + 1}/{total_pages} OCR 처리 중...")

        # PDF 페이지를 이미지로 변환
        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # 이미지를 PIL Image로 변환
        img = Image.open(io.BytesIO(img_bytes))

        # PaddleOCR로 텍스트 + 좌표 추출
        ocr_results = ocr_full_image_paddle(img)

        # 이미지를 base64로 인코딩
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        page_data = {
            "page": page_num + 1,
            "image_base64": img_base64,
            "width": img.width,
            "height": img.height,
            "ocr_results": ocr_results
        }

        result["pages"].append(page_data)

    doc.close()
    return result


def generate_comet_overlay_html(page_data: dict, scale: float = 1.0) -> str:
    """
    Comet 방식 HTML 오버레이 생성
    원본 이미지 위에 투명한 선택 가능 텍스트 레이어 배치

    Args:
        page_data: extract_ocr_with_coordinates의 페이지 데이터
        scale: 표시 스케일 (기본 1.0)

    Returns:
        HTML 문자열 (이미지 + 투명 텍스트 오버레이)
    """
    img_base64 = page_data["image_base64"]
    width = page_data["width"]
    height = page_data["height"]
    ocr_results = page_data["ocr_results"]
    page_num = page_data["page"]

    # 스케일 적용
    display_width = int(width * scale)
    display_height = int(height * scale)

    # 텍스트 오버레이 요소 생성
    text_elements = []
    for item in ocr_results:
        text = item["text"]
        box = item["box"]  # [x1, y1, x2, y2]

        # 좌표 스케일 적용
        x1 = int(box[0] * scale)
        y1 = int(box[1] * scale)
        x2 = int(box[2] * scale)
        y2 = int(box[3] * scale)

        box_width = x2 - x1
        box_height = y2 - y1

        # 폰트 크기 계산 (박스 높이 기준)
        font_size = max(8, int(box_height * 0.8))

        # HTML 특수문자 이스케이프
        escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        text_elements.append(f'''
            <span style="
                position: absolute;
                left: {x1}px;
                top: {y1}px;
                width: {box_width}px;
                height: {box_height}px;
                font-size: {font_size}px;
                line-height: {box_height}px;
                color: transparent;
                background: transparent;
                cursor: text;
                user-select: text;
                -webkit-user-select: text;
                overflow: hidden;
                white-space: nowrap;
            " data-text="{escaped_text}">{escaped_text}</span>
        ''')

    # 전체 HTML 생성
    html = f'''
    <div class="comet-page-container" style="
        position: relative;
        width: {display_width}px;
        height: {display_height}px;
        margin: 10px auto;
        border: 1px solid #ddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    ">
        <!-- 원본 이미지 (배경) -->
        <img src="data:image/png;base64,{img_base64}"
             style="
                 position: absolute;
                 top: 0;
                 left: 0;
                 width: {display_width}px;
                 height: {display_height}px;
                 pointer-events: none;
             "
             alt="페이지 {page_num}"
        />

        <!-- OCR 텍스트 오버레이 (투명, 선택 가능) -->
        <div class="comet-text-layer" style="
            position: absolute;
            top: 0;
            left: 0;
            width: {display_width}px;
            height: {display_height}px;
            z-index: 10;
        ">
            {''.join(text_elements)}
        </div>
    </div>
    '''

    return html


def generate_comet_full_html(comet_data: dict, scale: float = 0.8) -> str:
    """
    전체 PDF에 대한 Comet HTML 생성 (모든 페이지)

    Args:
        comet_data: extract_ocr_with_coordinates의 전체 결과
        scale: 표시 스케일 (기본 0.8 = 80%)

    Returns:
        전체 HTML 문자열
    """
    pages = comet_data.get("pages", [])
    total_pages = comet_data.get("total_pages", len(pages))

    page_htmls = []
    for page_data in pages:
        page_html = generate_comet_overlay_html(page_data, scale)
        page_num = page_data["page"]

        page_htmls.append(f'''
            <div class="comet-page-wrapper" style="margin-bottom: 20px;">
                <h3 style="text-align: center; color: #333; margin: 10px 0;">
                    페이지 {page_num} / {total_pages}
                </h3>
                {page_html}
            </div>
        ''')

    # 전체 HTML 문서
    full_html = f'''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>Comet OCR 오버레이</title>
        <style>
            body {{
                font-family: 'Malgun Gothic', sans-serif;
                background-color: #f5f5f5;
                padding: 20px;
                margin: 0;
            }}
            .comet-header {{
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .comet-info {{
                text-align: center;
                color: #666;
                margin-bottom: 20px;
            }}
            .comet-page-wrapper {{
                background: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            /* 텍스트 선택 시 하이라이트 */
            .comet-text-layer span::selection {{
                background: rgba(0, 120, 215, 0.3);
                color: transparent;
            }}
        </style>
    </head>
    <body>
        <div class="comet-header">
            <h1>📄 Comet OCR 오버레이 뷰어</h1>
            <p>텍스트를 드래그하여 선택 → Ctrl+C로 복사</p>
        </div>

        <div class="comet-info">
            <p>총 {total_pages} 페이지 | OCR 엔진: PaddleOCR PP-OCRv5</p>
        </div>

        <div class="comet-pages">
            {''.join(page_htmls)}
        </div>
    </body>
    </html>
    '''

    return full_html


# ============================================================
# Comet + Table Transformer 하이브리드 방식
# Table Transformer로 테이블 영역 감지 → 각 영역별 Comet OCR
# ============================================================

def extract_comet_with_table_detection(pdf_bytes: bytes, progress_callback=None,
                                        detection_threshold: float = 0.5) -> dict:
    """
    Comet + Table Transformer 하이브리드 방식
    1. Table Transformer로 테이블 영역(bbox) 감지
    2. 각 테이블 영역에 대해 Comet OCR 적용
    3. 테이블별로 구조화된 데이터 반환

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백 함수
        detection_threshold: 테이블 감지 임계값 (기본 0.5)

    Returns:
        {
            "tables": [
                {
                    "page": 1,
                    "table_index": 1,
                    "table_name": "COLOR/SIZE QTY",  # 감지된 테이블 제목
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.95,
                    "data": [[...], [...], ...],
                    "row_count": 10,
                    "col_count": 5,
                    "ocr_results": [...]  # 원본 OCR 데이터
                },
                ...
            ],
            "pages": [...],
            "comet_html": [...],
            "total_tables": 5,
            "extraction_method": "comet_hybrid"
        }
    """
    if not PADDLEOCR_AVAILABLE:
        return {
            "tables": [],
            "error": "PaddleOCR가 설치되지 않았습니다.",
            "extraction_method": "comet_hybrid"
        }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    result = {
        "tables": [],
        "pages": [],
        "comet_html": [],
        "total_pages": total_pages,
        "total_tables": 0,
        "is_ai_extracted": True,
        "extraction_method": "comet_hybrid",
        "ocr_engine": "PaddleOCR PP-OCRv5",
        "table_detector": "Table Transformer" if TABLE_TRANSFORMER_AVAILABLE else "Y-coordinate clustering"
    }

    table_index_global = 0

    for page_num in range(total_pages):
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"페이지 {page_num + 1}/{total_pages} 처리 중...")

        # PDF 페이지를 이미지로 변환
        page = doc[page_num]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        # 전체 페이지 OCR 수행
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"페이지 {page_num + 1} OCR 처리 중...")

        page_ocr_results = ocr_full_image_paddle(img)

        # 이미지 base64 인코딩
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # 페이지 데이터 저장
        page_data = {
            "page": page_num + 1,
            "image_base64": img_base64,
            "width": img.width,
            "height": img.height,
            "ocr_results": page_ocr_results
        }
        result["pages"].append(page_data)

        # Comet HTML 생성
        page_html = generate_comet_overlay_html(page_data, scale=0.8)
        result["comet_html"].append({
            "page": page_num + 1,
            "html": page_html
        })

        # 테이블 영역 감지
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"페이지 {page_num + 1} 테이블 영역 감지 중...")

        if TABLE_TRANSFORMER_AVAILABLE:
            # Table Transformer로 테이블 영역 감지
            detected_tables = detect_tables(img, threshold=detection_threshold)
        else:
            # Table Transformer 없으면 전체 페이지를 하나의 테이블로 처리
            detected_tables = [{
                "score": 0.99,
                "label": "table",
                "box": [0, 0, img.width, img.height]
            }]

        if not detected_tables:
            # 테이블이 감지되지 않으면 Y 좌표 클러스터링으로 테이블 영역 추정
            detected_tables = estimate_table_regions_from_ocr(page_ocr_results, img.width, img.height)

        # 각 테이블 영역별로 처리
        for table_idx, table_info in enumerate(detected_tables):
            table_index_global += 1
            table_box = table_info["box"]  # [x1, y1, x2, y2]
            table_score = table_info.get("score", 0.99)

            # 테이블 영역 내의 OCR 결과만 필터링
            table_ocr = filter_ocr_by_bbox(page_ocr_results, table_box)

            if not table_ocr:
                continue

            # 테이블 제목 추정 (테이블 영역 위쪽 텍스트)
            table_name = estimate_table_name(page_ocr_results, table_box)

            # OCR 결과를 행/열 구조로 변환
            table_data = ocr_to_table_structure(table_ocr, table_box)

            if table_data:
                result["tables"].append({
                    "page": page_num + 1,
                    "table_index": table_idx + 1,
                    "table_index_global": table_index_global,
                    "table_name": table_name,
                    "bbox": table_box,
                    "confidence": float(table_score),
                    "data": table_data,
                    "row_count": len(table_data),
                    "col_count": max(len(row) for row in table_data) if table_data else 0,
                    "ocr_results": table_ocr,
                    "extraction_method": "comet_hybrid"
                })

    doc.close()
    result["total_tables"] = len(result["tables"])

    return result


def filter_ocr_by_bbox(ocr_results: list, bbox: list, margin: int = 5) -> list:
    """
    바운딩 박스 내의 OCR 결과만 필터링

    Args:
        ocr_results: OCR 결과 리스트
        bbox: [x1, y1, x2, y2] 테이블 영역
        margin: 마진 (픽셀)

    Returns:
        필터링된 OCR 결과
    """
    x1, y1, x2, y2 = bbox
    filtered = []

    for item in ocr_results:
        box = item["box"]
        # OCR 박스의 중심점이 테이블 영역 내에 있는지 확인
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        if (x1 - margin <= center_x <= x2 + margin and
            y1 - margin <= center_y <= y2 + margin):
            # 상대 좌표로 변환
            filtered.append({
                "text": item["text"],
                "box": [
                    item["box"][0] - x1,  # 상대 x1
                    item["box"][1] - y1,  # 상대 y1
                    item["box"][2] - x1,  # 상대 x2
                    item["box"][3] - y1   # 상대 y2
                ],
                "box_absolute": item["box"],  # 절대 좌표 유지
                "score": item.get("score", 1.0)
            })

    return filtered


def estimate_table_name(ocr_results: list, table_bbox: list, search_height: int = 60) -> str:
    """
    테이블 영역 위쪽에서 테이블 제목 추정

    Args:
        ocr_results: 페이지 전체 OCR 결과
        table_bbox: 테이블 바운딩 박스 [x1, y1, x2, y2]
        search_height: 테이블 위쪽 검색 높이 (픽셀)

    Returns:
        추정된 테이블 제목
    """
    x1, y1, x2, y2 = table_bbox

    # 테이블 위쪽 영역에서 텍스트 찾기
    candidates = []
    for item in ocr_results:
        box = item["box"]
        text_center_y = (box[1] + box[3]) / 2
        text_center_x = (box[0] + box[2]) / 2

        # 테이블 위쪽 search_height 픽셀 내에 있고, x 범위가 겹치는 텍스트
        if (y1 - search_height <= text_center_y < y1 and
            x1 <= text_center_x <= x2):
            candidates.append({
                "text": item["text"],
                "y": box[1]
            })

    if candidates:
        # 가장 아래쪽(테이블에 가까운) 텍스트 선택
        candidates.sort(key=lambda x: x["y"], reverse=True)
        return candidates[0]["text"]

    return ""


def estimate_table_regions_from_ocr(ocr_results: list, page_width: int, page_height: int,
                                     gap_threshold: int = 50) -> list:
    """
    OCR 결과의 Y 좌표 분포를 분석하여 테이블 영역 추정
    큰 Y 갭이 있으면 테이블 구분으로 판단

    Args:
        ocr_results: OCR 결과 리스트
        page_width: 페이지 너비
        page_height: 페이지 높이
        gap_threshold: 테이블 구분 갭 임계값 (픽셀)

    Returns:
        추정된 테이블 영역 리스트
    """
    if not ocr_results:
        return []

    # Y 좌표로 정렬
    sorted_items = sorted(ocr_results, key=lambda x: x["box"][1])

    # 테이블 영역 그룹화
    groups = []
    current_group = [sorted_items[0]]

    for i in range(1, len(sorted_items)):
        prev_y2 = current_group[-1]["box"][3]  # 이전 항목의 y2
        curr_y1 = sorted_items[i]["box"][1]     # 현재 항목의 y1

        if curr_y1 - prev_y2 > gap_threshold:
            # 큰 갭 발견 - 새 그룹 시작
            groups.append(current_group)
            current_group = [sorted_items[i]]
        else:
            current_group.append(sorted_items[i])

    if current_group:
        groups.append(current_group)

    # 각 그룹을 테이블 영역으로 변환
    tables = []
    for group in groups:
        if len(group) < 3:  # 최소 3개 텍스트가 있어야 테이블로 간주
            continue

        # 그룹의 바운딩 박스 계산
        x1 = min(item["box"][0] for item in group)
        y1 = min(item["box"][1] for item in group)
        x2 = max(item["box"][2] for item in group)
        y2 = max(item["box"][3] for item in group)

        # 마진 추가
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(page_width, x2 + margin)
        y2 = min(page_height, y2 + margin)

        tables.append({
            "score": 0.9,
            "label": "table_estimated",
            "box": [x1, y1, x2, y2]
        })

    return tables


def ocr_to_table_structure(ocr_results: list, table_bbox: list,
                            y_tolerance: int = 15, x_gap_threshold: int = 30) -> list:
    """
    OCR 결과를 테이블 행/열 구조로 변환

    Args:
        ocr_results: 필터링된 OCR 결과 (상대 좌표)
        table_bbox: 테이블 바운딩 박스
        y_tolerance: 같은 행으로 간주할 Y 오차
        x_gap_threshold: 열 구분을 위한 X 갭 임계값

    Returns:
        2D 테이블 데이터 [[row1_col1, row1_col2, ...], [row2_col1, ...], ...]
    """
    if not ocr_results:
        return []

    # Y 좌표로 정렬 후 행 그룹화
    sorted_items = sorted(ocr_results, key=lambda x: (x["box"][1], x["box"][0]))

    rows = []
    current_row = []
    current_y = None

    for item in sorted_items:
        y = item["box"][1]

        if current_y is None:
            current_y = y
            current_row = [item]
        elif abs(y - current_y) <= y_tolerance:
            current_row.append(item)
        else:
            # 새 행 시작 - 현재 행을 X 좌표로 정렬
            rows.append(sorted(current_row, key=lambda x: x["box"][0]))
            current_row = [item]
            current_y = y

    if current_row:
        rows.append(sorted(current_row, key=lambda x: x["box"][0]))

    # 열 위치 추정 (X 좌표 클러스터링)
    all_x_positions = []
    for row in rows:
        for item in row:
            all_x_positions.append(item["box"][0])

    if not all_x_positions:
        return []

    # X 좌표 클러스터링으로 열 경계 찾기
    column_boundaries = estimate_column_boundaries(all_x_positions, x_gap_threshold)

    # 각 행을 열에 맞춰 정렬
    table_data = []
    for row in rows:
        row_data = assign_to_columns(row, column_boundaries)
        if any(cell.strip() for cell in row_data):  # 빈 행 제외
            table_data.append(row_data)

    return table_data


def estimate_column_boundaries(x_positions: list, gap_threshold: int = 30) -> list:
    """
    X 좌표 리스트에서 열 경계 추정

    Args:
        x_positions: X 좌표 리스트
        gap_threshold: 열 구분 갭 임계값

    Returns:
        열 경계 리스트 [(start, end), ...]
    """
    if not x_positions:
        return []

    sorted_x = sorted(set(x_positions))

    # 클러스터링
    clusters = []
    current_cluster = [sorted_x[0]]

    for i in range(1, len(sorted_x)):
        if sorted_x[i] - sorted_x[i-1] <= gap_threshold:
            current_cluster.append(sorted_x[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_x[i]]

    if current_cluster:
        clusters.append(current_cluster)

    # 각 클러스터의 경계
    boundaries = []
    for cluster in clusters:
        start = min(cluster) - gap_threshold // 2
        end = max(cluster) + gap_threshold // 2
        boundaries.append((start, end))

    return boundaries


def assign_to_columns(row_items: list, column_boundaries: list) -> list:
    """
    행의 항목들을 열에 할당

    Args:
        row_items: 행의 OCR 항목들
        column_boundaries: 열 경계 리스트

    Returns:
        열에 맞춰 정렬된 텍스트 리스트
    """
    if not column_boundaries:
        return [item["text"] for item in row_items]

    # 각 열에 해당하는 텍스트 찾기
    row_data = [""] * len(column_boundaries)

    for item in row_items:
        x = item["box"][0]
        text = item["text"]

        # 가장 가까운 열 찾기
        best_col = 0
        min_dist = float('inf')

        for col_idx, (start, end) in enumerate(column_boundaries):
            col_center = (start + end) / 2
            dist = abs(x - col_center)
            if dist < min_dist:
                min_dist = dist
                best_col = col_idx

        # 이미 값이 있으면 합치기
        if row_data[best_col]:
            row_data[best_col] += " " + text
        else:
            row_data[best_col] = text

    return row_data


def extract_comet_tables(pdf_bytes: bytes, progress_callback=None) -> dict:
    """
    Comet 방식으로 PDF에서 테이블 데이터 추출
    OCR 결과를 구조화하여 테이블 형태로 반환

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백

    Returns:
        테이블 추출 결과 (기존 형식과 호환)
    """
    # OCR 좌표 추출
    comet_data = extract_ocr_with_coordinates(pdf_bytes, progress_callback)

    if "error" in comet_data:
        return {
            "tables": [],
            "error": comet_data["error"],
            "is_ai_extracted": True,
            "extraction_method": "comet"
        }

    result = {
        "tables": [],
        "pages": [],
        "comet_html": [],  # Comet HTML 오버레이
        "is_ai_extracted": True,
        "total_pages": comet_data["total_pages"],
        "extraction_method": "comet",
        "ocr_engine": comet_data.get("ocr_engine", "PaddleOCR")
    }

    for page_data in comet_data["pages"]:
        page_num = page_data["page"]
        ocr_results = page_data["ocr_results"]

        # Comet HTML 오버레이 생성
        page_html = generate_comet_overlay_html(page_data, scale=0.8)
        result["comet_html"].append({
            "page": page_num,
            "html": page_html
        })

        # OCR 결과를 행/열 구조로 정리 (기존 호환)
        if ocr_results:
            # Y 좌표로 행 그룹화
            sorted_items = sorted(ocr_results, key=lambda x: (x["box"][1], x["box"][0]))

            rows = []
            current_row = []
            current_y = None
            y_tolerance = 15  # 같은 행으로 간주할 Y 오차

            for item in sorted_items:
                y = item["box"][1]

                if current_y is None:
                    current_y = y
                    current_row = [item]
                elif abs(y - current_y) <= y_tolerance:
                    current_row.append(item)
                else:
                    # 새 행 시작
                    rows.append(sorted(current_row, key=lambda x: x["box"][0]))
                    current_row = [item]
                    current_y = y

            if current_row:
                rows.append(sorted(current_row, key=lambda x: x["box"][0]))

            # 테이블 데이터 생성
            table_data = []
            for row in rows:
                row_texts = [item["text"] for item in row]
                if any(row_texts):  # 빈 행 제외
                    table_data.append(row_texts)

            if table_data:
                result["tables"].append({
                    "page": page_num,
                    "table_index": 1,
                    "confidence": 0.99,  # OCR 기반이므로 높은 신뢰도
                    "data": table_data,
                    "row_count": len(table_data),
                    "col_count": max(len(row) for row in table_data) if table_data else 0,
                    "extraction_method": "comet_ocr"
                })

        # 페이지별 원본 데이터 저장
        result["pages"].append({
            "page": page_num,
            "ocr_results": ocr_results,
            "image_base64": page_data["image_base64"],
            "width": page_data["width"],
            "height": page_data["height"]
        })

    return result


def extract_tables_auto(pdf_bytes: bytes, progress_callback=None, method: str = "auto") -> dict:
    """
    자동으로 최적의 방법으로 테이블 추출

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백
        method: 추출 방법
            - "auto": 자동 선택 (Comet > Table Transformer)
            - "comet": Comet 방식 (OCR 오버레이) - 권장
            - "vlm": VLM(Granite3.2-vision) 사용
            - "table_transformer": Table Transformer + OCR 사용

    Returns:
        추출 결과 딕셔너리
    """
    if method == "auto":
        # Comet 방식 우선 (가장 정확)
        if PADDLEOCR_AVAILABLE:
            method = "comet"
        elif OLLAMA_AVAILABLE and check_ollama_model("granite3.2-vision"):
            method = "vlm"
        elif TABLE_TRANSFORMER_AVAILABLE:
            method = "table_transformer"
        else:
            return {
                "tables": [],
                "error": "사용 가능한 추출 방법이 없습니다. PaddleOCR, Ollama 또는 Table Transformer를 설치해주세요.",
                "is_ai_extracted": False
            }

    if method == "comet":
        return extract_comet_tables(pdf_bytes, progress_callback)
    elif method == "vlm":
        return extract_vlm_tables(pdf_bytes, progress_callback)
    else:
        return extract_smart_tables(pdf_bytes, progress_callback)


# 테스트 코드
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("스마트 테이블 추출기 v3 테스트")
    print("=" * 60)
    print(f"\n엔진 상태:")
    print(f"  Ollama VLM: {'사용 가능' if OLLAMA_AVAILABLE else '사용 불가'}")
    print(f"  Granite3.2-vision: {'설치됨' if check_ollama_model('granite3.2-vision') else '미설치'}")
    print(f"  PaddleOCR: {'사용 가능' if PADDLEOCR_AVAILABLE else '사용 불가'}")
    print(f"  Tesseract: {'사용 가능' if TESSERACT_AVAILABLE else '사용 불가'}")
    print(f"  Table Transformer: {'사용 가능' if TABLE_TRANSFORMER_AVAILABLE else '사용 불가'}")

    # 테스트 PDF
    pdf_path = "e:/Antigravity/Black_Yak/제로스팟 다운자켓#1 오더 등록 작지 1BYPAWU005-M-1.pdf"

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        print(f"\nPDF: {pdf_path}")
        print(f"Is Scanned PDF: {is_scanned_pdf(pdf_bytes)}")

        def progress(page, total, msg):
            print(f"  [{page}/{total}] {msg}")

        # VLM 테스트
        if OLLAMA_AVAILABLE and check_ollama_model("granite3.2-vision"):
            print("\n" + "=" * 60)
            print("VLM 추출 테스트 (Granite3.2-vision)")
            print("=" * 60)
            result = extract_vlm_tables(pdf_bytes, progress_callback=progress)

            print(f"\n결과:")
            print(f"  추출 방법: {result.get('extraction_method', 'N/A')}")
            print(f"  모델: {result.get('model', 'N/A')}")
            print(f"  발견된 테이블 수: {len(result['tables'])}")

            for table in result["tables"]:
                print(f"\n  페이지 {table['page']}, 테이블 {table['table_index']}:")
                print(f"    제목: {table.get('title', 'N/A')}")
                print(f"    크기: {table['row_count']} 행 x {table['col_count']} 열")
                print(f"    데이터:")
                for row in table["data"][:5]:
                    print(f"      {row}")

            # 페이지별 필드 정보
            for page_data in result.get("pages", []):
                if page_data.get("fields"):
                    print(f"\n  페이지 {page_data['page']} 추출 필드:")
                    for key, value in page_data["fields"].items():
                        print(f"    {key}: {value}")

        # Table Transformer 테스트 (비교용)
        print("\n" + "=" * 60)
        print("Table Transformer 추출 테스트 (비교용)")
        print("=" * 60)
        result = extract_smart_tables(pdf_bytes, progress_callback=progress, use_paddle=True)

        print(f"\n결과:")
        print(f"  OCR 엔진: {result.get('ocr_engine', 'N/A')}")
        print(f"  발견된 테이블 수: {len(result['tables'])}")

        for table in result["tables"][:2]:  # 처음 2개만
            print(f"\n  페이지 {table['page']}, 테이블 {table['table_index']}:")
            print(f"    신뢰도: {table['confidence']}")
            print(f"    크기: {table['row_count']} 행 x {table['col_count']} 열")
            print(f"    샘플 데이터 (처음 3행):")
            for row in table["data"][:3]:
                print(f"      {row}")
    else:
        print(f"\nPDF 파일을 찾을 수 없습니다: {pdf_path}")
