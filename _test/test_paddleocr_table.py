"""
PaddleOCR 테이블 구조 추출 테스트
PP-StructureV3 테이블 인식 파이프라인
"""

import fitz
from PIL import Image
import io
import os
import sys
import json

# 한글 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

# 환경 변수 설정 (연결 체크 스킵)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

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


def test_table_recognition():
    """PaddleX 테이블 인식 파이프라인 테스트"""
    print("=" * 60)
    print("PaddleX 테이블 인식 파이프라인 테스트")
    print("=" * 60)

    # PDF 첫 페이지 이미지
    img_path = "e:/Antigravity/Black_Yak/test_page_1.png"

    if not os.path.exists(img_path):
        img_path = pdf_page_to_image(PDF_PATH, page_num=0, zoom=2.0)

    print(f"\n이미지 경로: {img_path}")
    print("\n테이블 인식 파이프라인 초기화 중...")

    from paddlex import create_pipeline

    # 테이블 인식 파이프라인 생성
    pipeline = create_pipeline(pipeline="table_recognition")

    print("테이블 구조 분석 중...")
    result = pipeline.predict(img_path)

    print("\n추출 결과:")
    print("-" * 40)

    tables_found = 0
    for res in result:
        print(f"\n입력 파일: {res.get('input_path', 'N/A')}")

        # 결과 키 확인
        print(f"결과 키: {res.keys()}")

        # 레이아웃 정보
        if 'layout_res' in res:
            layout = res['layout_res']
            print(f"\n레이아웃 요소 수: {len(layout) if isinstance(layout, list) else 'N/A'}")

        # 테이블 정보
        if 'table_res_list' in res:
            tables = res['table_res_list']
            tables_found = len(tables)
            print(f"\n발견된 테이블 수: {tables_found}")

            for i, table in enumerate(tables):
                print(f"\n테이블 #{i+1}:")
                print(f"  키: {table.keys()}")

                if 'html' in table:
                    html = table['html']
                    print(f"  HTML 길이: {len(html)} 문자")
                    # HTML 일부 출력
                    print(f"  HTML 미리보기:")
                    print(f"    {html[:500]}...")

                if 'cell_box_list' in table:
                    print(f"  셀 수: {len(table['cell_box_list'])}")

                if 'rec_res' in table:
                    print(f"  OCR 결과 수: {len(table['rec_res'])}")

    # 결과를 JSON으로 저장
    if tables_found > 0:
        output_file = "e:/Antigravity/Black_Yak/paddleocr_table_result.json"
        save_result = {
            "tables_found": tables_found,
            "tables": []
        }

        for res in result:
            if 'table_res_list' in res:
                for table in res['table_res_list']:
                    if 'html' in table:
                        save_result["tables"].append({
                            "html": table['html'][:2000],  # HTML 일부만
                            "cell_count": len(table.get('cell_box_list', [])),
                        })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {output_file}")

    return result


if __name__ == "__main__":
    try:
        result = test_table_recognition()
    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()
