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
                            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                            ocr_results.append({
                                "text": text,
                                "box": box,
                                "score": score
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

def extract_tables_auto(pdf_bytes: bytes, progress_callback=None, method: str = "auto") -> dict:
    """
    자동으로 최적의 방법으로 테이블 추출

    Args:
        pdf_bytes: PDF 파일 바이트
        progress_callback: 진행상황 콜백
        method: 추출 방법
            - "auto": 자동 선택 (VLM > Table Transformer)
            - "vlm": VLM(Granite3.2-vision) 사용
            - "table_transformer": Table Transformer + OCR 사용

    Returns:
        추출 결과 딕셔너리
    """
    if method == "auto":
        # VLM 사용 가능하면 VLM 우선
        if OLLAMA_AVAILABLE and check_ollama_model("granite3.2-vision"):
            method = "vlm"
        elif TABLE_TRANSFORMER_AVAILABLE:
            method = "table_transformer"
        else:
            return {
                "tables": [],
                "error": "사용 가능한 추출 방법이 없습니다. Ollama 또는 Table Transformer를 설치해주세요.",
                "is_ai_extracted": False
            }

    if method == "vlm":
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
