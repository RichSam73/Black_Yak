"""
Table Transformer를 사용하여 이미지에서 테이블 구조를 감지하고
OCR로 각 셀의 내용을 추출하는 테스트 스크립트
"""

import fitz
from PIL import Image
import io
from pathlib import Path
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import pytesseract
import os

# Tesseract 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_DIR = str(Path(__file__).parent / "tessdata")
os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

# PDF 파일 경로
PDF_PATH = "e:/Antigravity/Black_Yak/제로스팟 다운자켓#1 오더 등록 작지 1BYPAWU005-M-1.pdf"

def pdf_page_to_image(pdf_path: str, page_num: int = 0, zoom: float = 2.0) -> Image.Image:
    """PDF 페이지를 이미지로 변환"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    doc.close()
    return img

def detect_tables(image: Image.Image):
    """Table Transformer로 테이블 영역 감지"""
    print("Table Transformer 모델 로딩 중...")

    # 테이블 감지 모델 로드
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    # 이미지 처리
    inputs = processor(images=image, return_tensors="pt")

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 결과 후처리
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        tables.append({
            "score": round(score.item(), 3),
            "label": model.config.id2label[label.item()],
            "box": box  # [x1, y1, x2, y2]
        })
        print(f"감지됨: {model.config.id2label[label.item()]} (신뢰도: {score.item():.3f}), 좌표: {box}")

    return tables

def detect_table_structure(image: Image.Image, table_box: list):
    """Table Transformer로 테이블 내부 구조(행/열) 감지"""
    print("\n테이블 구조 분석 중...")

    # 테이블 영역 크롭
    x1, y1, x2, y2 = table_box
    table_img = image.crop((x1, y1, x2, y2))

    # 구조 인식 모델 로드
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    # 추론
    inputs = processor(images=table_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # 결과 후처리
    target_sizes = torch.tensor([table_img.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    cells = []
    rows = []
    columns = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]

        item = {
            "score": round(score.item(), 3),
            "label": label_name,
            "box": box
        }

        if label_name == "table cell":
            cells.append(item)
        elif label_name == "table row":
            rows.append(item)
        elif label_name == "table column":
            columns.append(item)

    print(f"감지된 요소: 셀 {len(cells)}개, 행 {len(rows)}개, 열 {len(columns)}개")

    return {
        "cells": cells,
        "rows": rows,
        "columns": columns,
        "table_image": table_img
    }

def extract_cell_text(table_img: Image.Image, cells: list) -> list:
    """각 셀에서 OCR로 텍스트 추출"""
    print("\nOCR로 셀 텍스트 추출 중...")

    results = []
    for cell in cells:
        x1, y1, x2, y2 = cell["box"]
        cell_img = table_img.crop((x1, y1, x2, y2))

        # OCR 실행
        text = pytesseract.image_to_string(cell_img, lang='kor+eng').strip()

        results.append({
            "box": cell["box"],
            "text": text,
            "score": cell["score"]
        })

    return results

def main():
    print(f"PDF 파일: {PDF_PATH}")
    print("=" * 60)

    # 1. PDF 첫 페이지를 이미지로 변환
    print("\n1. PDF를 이미지로 변환 중...")
    image = pdf_page_to_image(PDF_PATH, page_num=0, zoom=2.0)
    print(f"   이미지 크기: {image.size}")

    # 이미지 저장 (디버깅용)
    image.save("e:/Antigravity/Black_Yak/test_page1.png")
    print("   test_page1.png 저장됨")

    # 2. 테이블 영역 감지
    print("\n2. 테이블 영역 감지 중...")
    tables = detect_tables(image)

    if not tables:
        print("테이블이 감지되지 않았습니다.")
        return

    # 3. 첫 번째 테이블의 구조 분석
    print("\n3. 테이블 구조 분석...")
    first_table = tables[0]
    structure = detect_table_structure(image, first_table["box"])

    # 4. 셀 텍스트 추출
    if structure["cells"]:
        cell_texts = extract_cell_text(structure["table_image"], structure["cells"])

        print("\n4. 추출된 셀 텍스트:")
        print("-" * 40)
        for ct in cell_texts[:20]:  # 처음 20개만 출력
            if ct["text"]:
                print(f"  [{ct['box']}] {ct['text']}")

        if len(cell_texts) > 20:
            print(f"  ... 외 {len(cell_texts) - 20}개 더 있음")
    else:
        print("셀이 감지되지 않았습니다.")

if __name__ == "__main__":
    main()
