"""
PaddleOCR 간단 테스트 - 기본 OCR만 테스트
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


def test_paddleocr_simple():
    """간단한 OCR 테스트"""
    print("=" * 60)
    print("PaddleOCR 기본 OCR 테스트 (PP-OCRv5)")
    print("=" * 60)

    # PDF 첫 페이지 이미지
    img_path = "e:/Antigravity/Black_Yak/test_page_1.png"

    if not os.path.exists(img_path):
        img_path = pdf_page_to_image(PDF_PATH, page_num=0, zoom=2.0)

    print(f"\n이미지 경로: {img_path}")
    print("\nPaddleOCR 초기화 중...")

    # 환경 변수 설정
    os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

    from paddleocr import PaddleOCR

    # 기본 OCR (한국어) - 최신 API
    ocr = PaddleOCR(lang='korean')

    print("OCR 실행 중...")
    result = ocr.predict(img_path)

    print("\n추출된 텍스트:")
    print("-" * 40)

    if result:
        for res in result:
            print(f"\n결과 키: {res.keys() if isinstance(res, dict) else type(res)}")

            if isinstance(res, dict):
                if 'rec_texts' in res:
                    texts = res['rec_texts']
                    scores = res.get('rec_scores', [0]*len(texts))
                    for i, (text, score) in enumerate(zip(texts[:30], scores[:30])):
                        print(f"  [{i+1}] [{score:.2f}] {text}")
                else:
                    # 다른 형식
                    for key, value in res.items():
                        if key in ['input_path']:
                            continue
                        print(f"  {key}: {str(value)[:100]}...")
    else:
        print("결과 없음")


if __name__ == "__main__":
    try:
        test_paddleocr_simple()
    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()
