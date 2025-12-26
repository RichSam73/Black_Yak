"""
Grid-First Table Extraction - 자동 테이블 감지 버전
큰 박스 안에 셀 10개 이상 = 테이블

1. 큰 닫힌 사각형들 찾기
2. 각 사각형 안의 선 개수로 셀 수 계산
3. 셀 10개 이상이면 테이블로 인정
4. 테이블 영역 내 격자 + OCR 매핑
"""
import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from smart_table_extractor import ocr_full_image_paddle


def find_boxes(img: Image.Image, min_area: int = 5000, max_area_ratio: float = 0.8) -> list:
    """
    닫힌 사각형들 찾기 (내부 박스 포함)
    - min_area: 최소 면적
    - max_area_ratio: 이미지 대비 최대 면적 비율 (전체 페이지 외곽선 제외용)
    """
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # RETR_TREE로 내부 contour도 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.width * img.height
    max_area = img_area * max_area_ratio

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 너무 작거나 너무 큰 contour 제외
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # 가로세로 비율 체크 - 너무 길쭉한건 제외
        aspect = max(w, h) / (min(w, h) + 1)
        if aspect > 20:
            continue

        boxes.append({"box": [x, y, x+w, y+h], "area": area, "w": w, "h": h})

    # 면적 기준 내림차순
    boxes.sort(key=lambda b: b["area"], reverse=True)

    # 중복 박스 제거 (거의 같은 위치의 박스)
    filtered = []
    for box in boxes:
        is_dup = False
        for existing in filtered:
            # IoU 기반 중복 체크
            if box_iou(box["box"], existing["box"]) > 0.9:
                is_dup = True
                break
        if not is_dup:
            filtered.append(box)

    return filtered


def box_iou(box1: list, box2: list) -> float:
    """두 박스의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter)


def count_cells_in_region(img: Image.Image, box: list, min_line_len: int = 30) -> tuple:
    """
    영역 내 가로선/세로선 수 → 셀 수 계산
    Returns: (num_rows, num_cols, row_bounds, col_bounds)
    """
    x1, y1, x2, y2 = box
    cropped = img.crop((x1, y1, x2, y2))

    img_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape

    # 가로선 찾기
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines, axis=1)
    h_coords = np.where(h_proj > min_line_len)[0]
    row_bounds = group_coords(h_coords)

    # 세로선 찾기
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0)
    v_coords = np.where(v_proj > min_line_len)[0]
    col_bounds = group_coords(v_coords)

    num_rows = max(0, len(row_bounds) - 1)
    num_cols = max(0, len(col_bounds) - 1)

    return num_rows, num_cols, row_bounds, col_bounds


def group_coords(coords: np.ndarray, gap: int = 5) -> list:
    """연속 좌표 그룹화"""
    if len(coords) == 0:
        return []

    groups = []
    current = [coords[0]]

    for c in coords[1:]:
        if c - current[-1] <= gap:
            current.append(c)
        else:
            groups.append(int(np.mean(current)))
            current = [c]

    groups.append(int(np.mean(current)))
    return groups


def find_tables(img: Image.Image, min_cells: int = 10) -> list:
    """
    테이블 영역 자동 감지
    - 큰 박스 안에 셀 min_cells개 이상이면 테이블
    """
    boxes = find_boxes(img)

    print(f"\n=== Box Detection ===")
    print(f"Found {len(boxes)} boxes")

    tables = []
    for i, box_info in enumerate(boxes):
        box = box_info["box"]
        num_rows, num_cols, row_bounds, col_bounds = count_cells_in_region(img, box)
        num_cells = num_rows * num_cols

        status = "[TABLE]" if num_cells >= min_cells else "[skip]"
        print(f"Box {i+1}: {box} -> {num_rows}x{num_cols}={num_cells} cells {status}")

        if num_cells >= min_cells:
            tables.append({
                "box": box,
                "rows": num_rows,
                "cols": num_cols,
                "row_bounds": row_bounds,
                "col_bounds": col_bounds
            })

    return tables


def extract_table_from_region(img: Image.Image, table_info: dict) -> dict:
    """테이블 영역에서 데이터 추출"""
    box = table_info["box"]
    x1, y1, x2, y2 = box

    # 크롭
    cropped = img.crop((x1, y1, x2, y2))

    row_bounds = table_info["row_bounds"]
    col_bounds = table_info["col_bounds"]
    num_rows = len(row_bounds) - 1
    num_cols = len(col_bounds) - 1

    # 빈 테이블 생성
    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    # OCR
    ocr_results = ocr_full_image_paddle(cropped)

    # 매핑
    for ocr in ocr_results:
        ocr_box = ocr.get("box", [])
        text = ocr.get("text", "").strip()

        if not text or len(ocr_box) < 4:
            continue

        cx = (ocr_box[0] + ocr_box[2]) / 2
        cy = (ocr_box[1] + ocr_box[3]) / 2

        row_idx = find_index(cy, row_bounds)
        col_idx = find_index(cx, col_bounds)

        if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
            if table[row_idx][col_idx]:
                table[row_idx][col_idx] += " " + text
            else:
                table[row_idx][col_idx] = text

    return {
        "data": table,
        "box": box,
        "rows": num_rows,
        "cols": num_cols,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds
    }


def find_index(value: float, bounds: list) -> int:
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    return -1


def print_table(table: list, title: str = ""):
    if not table:
        print("Empty table")
        return

    if title:
        print(f"\n=== {title} ===")
    print(f"Size: {len(table)} x {len(table[0])}")

    for i, row in enumerate(table[:12]):
        cells = [str(c)[:10].ljust(10) for c in row[:10]]
        suffix = "..." if len(row) > 10 else ""
        print(f"R{i:2d}: |" + "|".join(cells) + f"|{suffix}")

    if len(table) > 12:
        print(f"... ({len(table) - 12} more rows)")


def test_pdf_auto(pdf_path: str, min_cells: int = 10):
    """PDF에서 테이블 자동 감지"""
    import fitz

    print(f"\n{'='*50}")
    print(f"AUTO TABLE DETECTION: {Path(pdf_path).name}")
    print(f"Min cells: {min_cells}")
    print(f"{'='*50}")

    doc = fitz.open(pdf_path)
    page = doc[0]

    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 테이블 자동 감지
    tables = find_tables(img, min_cells=min_cells)
    print(f"\n=== Found {len(tables)} tables ===")

    results = []
    for i, table_info in enumerate(tables):
        print(f"\n--- Extracting Table {i+1} ---")
        print(f"Box: {table_info['box']}")
        print(f"Structure: {table_info['rows']} x {table_info['cols']}")

        result = extract_table_from_region(img, table_info)
        print_table(result["data"], f"Table {i+1}")
        results.append(result)

    # 시각화
    visualize_tables(img, tables, "auto_detection.png")

    doc.close()
    return results


def visualize_tables(img: Image.Image, tables: list, output_path: str):
    """테이블 영역 시각화"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for i, t in enumerate(tables):
        x1, y1, x2, y2 = t["box"]
        # 테이블 외곽 (녹색 굵은선)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # 라벨
        cv2.putText(img_cv, f"Table {i+1}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img_cv)
    print(f"\nVisualization: {output_path}")


def test_specific_region(pdf_path: str, region: tuple):
    """특정 영역만 테스트"""
    import fitz

    print(f"\n{'='*50}")
    print(f"SPECIFIC REGION TEST: {Path(pdf_path).name}")
    print(f"Region: {region}")
    print(f"{'='*50}")

    doc = fitz.open(pdf_path)
    page = doc[0]

    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    x1, y1, x2, y2 = region
    cropped = img.crop((x1, y1, x2, y2))

    # 셀 수 계산 (min_line_len=15로 더 짧은 선도 감지)
    num_rows, num_cols, row_bounds, col_bounds = count_cells_in_region(img, list(region), min_line_len=15)
    print(f"Detected: {num_rows} rows x {num_cols} cols = {num_rows * num_cols} cells")
    print(f"Row bounds: {row_bounds}")
    print(f"Col bounds: {col_bounds}")

    # 테이블 정보 구성
    table_info = {
        "box": list(region),
        "rows": num_rows,
        "cols": num_cols,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds
    }

    # 추출
    result = extract_table_from_region(img, table_info)
    print_table(result["data"], "Extracted Table")

    # 시각화
    img_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    for y in row_bounds:
        cv2.line(img_cv, (0, y), (img_cv.shape[1], y), (0, 0, 255), 1)
    for x in col_bounds:
        cv2.line(img_cv, (x, 0), (x, img_cv.shape[0]), (255, 0, 0), 1)
    cv2.imwrite("region_grid.png", img_cv)
    print("Grid visualization: region_grid.png")

    doc.close()
    return result


if __name__ == "__main__":
    import sys

    pdf_path = r"E:\Antigravity\Black_Yak\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"

    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        # 자동 감지 모드
        test_pdf_auto(pdf_path, min_cells=10)
    else:
        # 특정 영역 테스트 (COLOR/SIZE QTY 테이블)
        # 원본 좌표 약 (55, 93, 488, 215) -> 2x = (110, 186, 976, 430)
        test_specific_region(pdf_path, region=(110, 186, 976, 430))
