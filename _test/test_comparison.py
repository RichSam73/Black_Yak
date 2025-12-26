"""
img2table vs Grid-First 비교 테스트
COLOR/SIZE QTY 테이블 추출 정확도 비교
"""
import sys
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

# Grid-First 함수 import
sys.path.insert(0, str(Path(__file__).parent))
from test_grid_first import find_tables, extract_table_from_region, print_table

# img2table
from img2table.document import PDF
from img2table.ocr import PaddleOCR


def compare_methods(pdf_path: str):
    """두 방식 비교"""
    print(f"\n{'='*70}")
    print("METHOD COMPARISON: img2table vs Grid-First")
    print(f"PDF: {Path(pdf_path).name}")
    print(f"{'='*70}")

    # ===== 1. Grid-First 방식 =====
    print("\n" + "="*35)
    print("1. GRID-FIRST METHOD")
    print("="*35)

    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    tables_gf = find_tables(img, min_cells=10)
    print(f"\nFound {len(tables_gf)} tables")

    for i, t in enumerate(tables_gf):
        result = extract_table_from_region(img, t)
        print(f"\n--- Table {i+1} ---")
        print(f"Box: {t['box']}")
        print(f"Structure: {t['rows']}x{t['cols']}")
        print_table(result["data"], f"Grid-First Table {i+1}")

    doc.close()

    # ===== 2. img2table 방식 =====
    print("\n" + "="*35)
    print("2. IMG2TABLE METHOD")
    print("="*35)

    ocr = PaddleOCR(lang="korean")
    pdf = PDF(pdf_path, pages=[0])

    result = pdf.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=False,
        min_confidence=50
    )

    for page_num, tables in result.items():
        print(f"\nPage {page_num}: {len(tables)} tables")

        for i, table in enumerate(tables):
            df = table.df
            if df is not None and not df.empty:
                print(f"\n--- Table {i+1} ---")
                print(f"Shape: {df.shape[0]}x{df.shape[1]}")

                # COLOR/SIZE QTY 부분만 추출 (대략 row 4-13)
                print("\n[COLOR/SIZE QTY 영역 - rows 4~13]")
                subset = df.iloc[4:14, :15]  # 처음 15열만
                print(subset.to_string())


def extract_color_size_table_only(pdf_path: str):
    """COLOR/SIZE QTY 테이블만 정확히 추출 비교"""
    print(f"\n{'='*70}")
    print("COLOR/SIZE QTY TABLE EXTRACTION COMPARISON")
    print(f"{'='*70}")

    # Grid-First - 특정 영역
    print("\n--- Grid-First (특정 영역) ---")
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # COLOR/SIZE QTY 영역 (2x 스케일 좌표)
    region = (110, 186, 976, 430)
    from test_grid_first import count_cells_in_region

    num_rows, num_cols, row_bounds, col_bounds = count_cells_in_region(img, list(region), min_line_len=15)
    print(f"Detected: {num_rows}x{num_cols} = {num_rows * num_cols} cells")

    table_info = {
        "box": list(region),
        "rows": num_rows,
        "cols": num_cols,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds
    }
    result_gf = extract_table_from_region(img, table_info)
    print_table(result_gf["data"], "Grid-First Result")

    doc.close()

    # img2table - 전체 페이지에서 해당 영역 필터링
    print("\n--- img2table (전체 페이지) ---")
    ocr = PaddleOCR(lang="korean")
    pdf = PDF(pdf_path, pages=[0])

    result = pdf.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True)

    for page_num, tables in result.items():
        for table in tables:
            df = table.df
            # COLOR/SIZE QTY 행 찾기
            for idx, row in df.iterrows():
                if any("COLOR/SIZE" in str(v) for v in row.values if v):
                    print(f"\nCOLOR/SIZE QTY 시작 행: {idx}")
                    # 해당 행부터 10행 출력
                    subset = df.iloc[idx:idx+10, :15]
                    print(subset.to_string())
                    break


if __name__ == "__main__":
    pdf_path = r"E:\Antigravity\Black_Yak\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"

    # 비교 실행
    extract_color_size_table_only(pdf_path)
