"""
img2table 라이브러리 테스트
- Merged cells 자동 처리
- PaddleOCR 연동
"""
from pathlib import Path
from img2table.document import PDF
from img2table.ocr import PaddleOCR

def test_img2table(pdf_path: str):
    """img2table로 PDF 테이블 추출"""
    print(f"\n{'='*60}")
    print(f"img2table TEST: {Path(pdf_path).name}")
    print(f"{'='*60}")

    # OCR 엔진 초기화 (한국어)
    print("\n[1] PaddleOCR 초기화...")
    ocr = PaddleOCR(lang="korean")

    # PDF 문서 로드
    print("[2] PDF 로드...")
    pdf = PDF(pdf_path)

    # 테이블 추출
    print("[3] 테이블 추출 중...")
    tables = pdf.extract_tables(
        ocr=ocr,
        implicit_rows=True,      # 암시적 행 감지
        implicit_columns=True,   # 암시적 열 감지
        borderless_tables=False, # 선 없는 테이블 감지 안함
        min_confidence=50        # OCR 최소 신뢰도
    )

    print(f"\n[결과] {len(tables)} 테이블 발견")

    # 각 테이블 출력
    for i, table in enumerate(tables):
        print(f"\n{'─'*50}")
        print(f"Table {i+1}:")
        print(f"  - Page: {table.page}")
        print(f"  - BBox: {table.bbox}")

        # DataFrame 변환
        df = table.df
        if df is not None and not df.empty:
            print(f"  - Shape: {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"\n{df.to_string()}")
        else:
            print("  - Empty table")

    return tables


def test_specific_page(pdf_path: str, page_num: int = 0):
    """특정 페이지만 테스트"""
    print(f"\n{'='*60}")
    print(f"img2table TEST (Page {page_num}): {Path(pdf_path).name}")
    print(f"{'='*60}")

    ocr = PaddleOCR(lang="korean")
    pdf = PDF(pdf_path, pages=[page_num])

    # extract_tables는 dict 반환: {page_num: [Table, ...]}
    result = pdf.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=False,
        min_confidence=50
    )

    print(f"\n[결과 구조] {type(result)}")
    print(f"[결과 내용] {result}")

    # dict인 경우 처리
    if isinstance(result, dict):
        for page, tables in result.items():
            print(f"\n=== Page {page}: {len(tables)} 테이블 ===")
            for i, table in enumerate(tables):
                print(f"\n{'─'*50}")
                print(f"Table {i+1}: {type(table)}")

                # Table 객체 속성 확인
                if hasattr(table, 'df'):
                    df = table.df
                    if df is not None and not df.empty:
                        print(f"  - Shape: {df.shape[0]} rows x {df.shape[1]} cols")
                        print(f"\n--- First 10 rows ---")
                        print(df.head(10).to_string())
                elif hasattr(table, 'to_dataframe'):
                    df = table.to_dataframe()
                    print(f"  - Shape: {df.shape[0]} rows x {df.shape[1]} cols")
                    print(df.head(10).to_string())
                else:
                    print(f"  - Attributes: {dir(table)}")
    else:
        print(f"  Unexpected result type: {type(result)}")

    return result


if __name__ == "__main__":
    pdf_path = r"E:\Antigravity\Black_Yak\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"

    # 첫 페이지만 테스트
    test_specific_page(pdf_path, page_num=0)
