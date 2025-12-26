"""
PaddleOCR PP-StructureV3 테스트
- 테이블 구조 인식 + OCR 통합
- Comet과 유사한 End-to-end 방식
"""

import fitz
from PIL import Image
import io
import os
import sys

# 한글 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

# PDF 파일 경로
PDF_PATH = "e:/Antigravity/Black_Yak/제로스팟 다운자켓#1 오더 등록 작지 1BYPAWU005-M-1.pdf"

def pdf_page_to_image(pdf_path: str, page_num: int = 0, zoom: float = 2.0) -> str:
    """PDF 페이지를 이미지 파일로 저장"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    output_path = f"e:/Antigravity/Black_Yak/test_page_{page_num+1}.png"
    pix.save(output_path)
    doc.close()

    print(f"페이지 {page_num+1} 이미지 저장: {output_path}")
    return output_path


def test_paddleocr_table():
    """PaddleOCR PP-StructureV3로 테이블 추출 테스트"""
    print("=" * 60)
    print("PaddleOCR PP-StructureV3 테이블 추출 테스트")
    print("=" * 60)

    # 1. PDF 첫 페이지를 이미지로 변환
    print("\n1. PDF를 이미지로 변환 중...")
    img_path = pdf_page_to_image(PDF_PATH, page_num=0, zoom=2.0)

    # 2. PaddleX 테이블 인식 파이프라인 사용
    print("\n2. PaddleX 테이블 인식 파이프라인 초기화 중...")

    from paddlex import create_pipeline

    # 테이블 인식 파이프라인 생성
    pipeline = create_pipeline(pipeline="table_recognition")

    # 3. 테이블 추출
    print("\n3. 테이블 구조 분석 중...")
    result = pipeline.predict(img_path)

    # 4. 결과 분석
    print("\n4. 추출 결과:")
    print("-" * 40)

    for res in result:
        print(f"\n입력 파일: {res['input_path']}")

        # 테이블 정보
        if 'table_res_list' in res:
            tables = res['table_res_list']
            print(f"발견된 테이블 수: {len(tables)}")

            for i, table in enumerate(tables):
                print(f"\n테이블 #{i+1}:")
                if 'html' in table:
                    html = table['html']
                    print(f"  HTML 길이: {len(html)} 문자")
                    print(f"  HTML 미리보기: {html[:300]}...")

        # 전체 결과 출력
        print(f"\n전체 결과 키: {res.keys()}")

    return result


def test_paddleocr_simple():
    """간단한 OCR 테스트 (최신 API - predict 사용)"""
    print("\n" + "=" * 60)
    print("PaddleOCR 기본 OCR 테스트 (PP-OCRv5)")
    print("=" * 60)

    # PDF 첫 페이지 이미지
    img_path = "e:/Antigravity/Black_Yak/test_page_1.png"

    if not os.path.exists(img_path):
        img_path = pdf_page_to_image(PDF_PATH, page_num=0, zoom=2.0)

    print("\nPaddleOCR 초기화 중...")
    from paddleocr import PaddleOCR

    # 기본 OCR (한국어) - 최신 API
    ocr = PaddleOCR(lang='korean')

    print("OCR 실행 중 (predict 메서드)...")
    result = ocr.predict(img_path)

    print("\n추출된 텍스트 (처음 30개):")
    print("-" * 40)

    count = 0
    if result:
        for res in result:
            if 'rec_texts' in res:
                texts = res['rec_texts']
                scores = res.get('rec_scores', [0]*len(texts))
                for text, score in zip(texts, scores):
                    if count >= 30:
                        break
                    print(f"  [{score:.2f}] {text}")
                    count += 1
            elif 'text' in res:
                # 다른 형식의 결과 처리
                text = res['text']
                score = res.get('score', 0)
                if count < 30:
                    print(f"  [{score:.2f}] {text}")
                    count += 1

        print(f"\n총 결과 객체 수: {len(result)}")
    else:
        print("결과 없음")

    # 결과 구조 확인
    if result:
        print(f"\n결과 구조 (첫 번째 항목의 키):")
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            if isinstance(first, dict):
                print(f"  {first.keys()}")
            else:
                print(f"  타입: {type(first)}")


if __name__ == "__main__":
    # PP-StructureV3 테이블 추출 테스트
    try:
        result = test_paddleocr_table()
    except Exception as e:
        print(f"\nPP-StructureV3 오류: {e}")
        import traceback
        traceback.print_exc()

    # 기본 OCR 테스트
    print("\n" + "=" * 60)
    try:
        test_paddleocr_simple()
    except Exception as e:
        print(f"\n기본 OCR 오류: {e}")
        import traceback
        traceback.print_exc()
