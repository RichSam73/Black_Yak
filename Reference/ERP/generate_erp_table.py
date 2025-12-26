"""
ERP용 테이블 생성 스크립트
- Comet 오버레이 HTML 아래에 ERP 전송용 구조화 테이블 추가
- 입력: 이미지 파일
- 출력: Comet 오버레이 + ERP 테이블이 합쳐진 HTML

Reference/table2.html 구조 참조:
- 13열 고정 (COLOR CODE, NAME, 6개 사이즈, 4개 빈 열, TOTAL)
- 데이터 행 + 3개 빈 행 + TOTAL 행
"""
from PIL import Image
import cv2
import numpy as np
import os
import re
import base64
import easyocr

# 전역 OCR 인스턴스 (재사용)
_easyocr_reader = None

def get_easyocr():
    """EasyOCR 인스턴스 싱글톤"""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("  [EasyOCR 초기화 중...]")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


# =============================================================================
# Grid-First 핵심 함수들
# =============================================================================

def grid_find_boxes(img: Image.Image, min_area: int = 1000, max_area_ratio: float = 0.95) -> list:
    """닫힌 사각형들 찾기"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.width * img.height
    max_area = img_area * max_area_ratio

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / (min(w, h) + 1)
        if aspect > 20:
            continue

        boxes.append({"box": [x, y, x+w, y+h], "area": area, "w": w, "h": h})

    boxes.sort(key=lambda b: b["area"], reverse=True)

    filtered = []
    for box in boxes:
        is_dup = False
        for existing in filtered:
            if grid_box_iou(box["box"], existing["box"]) > 0.9:
                is_dup = True
                break
        if not is_dup:
            filtered.append(box)

    return filtered


def grid_box_iou(box1: list, box2: list) -> float:
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


def grid_group_coords(coords: np.ndarray, gap: int = 5) -> list:
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


def grid_count_cells_in_region(img: Image.Image, box: list, min_line_len: int = 20) -> tuple:
    """영역 내 가로선/세로선 수 -> 셀 수 계산"""
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
    row_bounds = grid_group_coords(h_coords)

    # 세로선 찾기
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0)
    v_coords = np.where(v_proj > min_line_len)[0]
    col_bounds = grid_group_coords(v_coords)

    num_rows = max(0, len(row_bounds) - 1)
    num_cols = max(0, len(col_bounds) - 1)

    return num_rows, num_cols, row_bounds, col_bounds


def grid_find_index(value: float, bounds: list) -> int:
    """bounds 리스트에서 value가 속하는 인덱스 찾기"""
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    return -1


def ocr_image_easyocr(image: Image.Image) -> list:
    """EasyOCR로 이미지 OCR"""
    reader = get_easyocr()
    img_array = np.array(image)

    try:
        results = reader.readtext(
            img_array,
            min_size=5,
            text_threshold=0.5,
            low_text=0.3,
            contrast_ths=0.1,
            adjust_contrast=0.5,
        )

        ocr_results = []
        for (bbox, text, score) in results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            box = [int(min(x_coords)), int(min(y_coords)),
                   int(max(x_coords)), int(max(y_coords))]

            ocr_results.append({
                "text": text,
                "box": box,
                "score": float(score)
            })

        return ocr_results
    except Exception as e:
        print(f"EasyOCR 오류: {e}")
        return []


def grid_extract_table(img: Image.Image, table_info: dict) -> list:
    """Grid-First 방식: 테이블 영역에서 데이터 추출"""
    box = table_info["box"]
    x1, y1, x2, y2 = box

    cropped = img.crop((x1, y1, x2, y2))

    row_bounds = table_info["row_bounds"]
    col_bounds = table_info["col_bounds"]
    num_rows = len(row_bounds) - 1
    num_cols = len(col_bounds) - 1

    print(f"  Grid 감지: {num_rows}행 x {num_cols}열")

    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    ocr_results = ocr_image_easyocr(cropped)
    print(f"  OCR 결과: {len(ocr_results)}개 텍스트")

    for ocr in ocr_results:
        ocr_box = ocr.get("box", [])
        text = ocr.get("text", "").strip()

        if not text or len(ocr_box) < 4:
            continue

        cx = (ocr_box[0] + ocr_box[2]) / 2
        cy = (ocr_box[1] + ocr_box[3]) / 2

        row_idx = grid_find_index(cy, row_bounds)
        col_idx = grid_find_index(cx, col_bounds)

        if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
            if table[row_idx][col_idx]:
                table[row_idx][col_idx] += " " + text
            else:
                table[row_idx][col_idx] = text

    return table


# =============================================================================
# 테이블 파싱 및 HTML 생성
# =============================================================================

def generate_generic_erp_table_html(table_2d: list) -> str:
    """Grid-First 2D 테이블을 그대로 ERP용 HTML 테이블로 변환

    양식에 관계없이 Grid-First 결과를 그대로 HTML로 출력
    - COLOR/SIZE QTY, SUB MATERIAL INFORMATION 등 모든 양식 지원
    """

    if not table_2d or len(table_2d) == 0:
        return '<p>테이블 데이터가 없습니다.</p>'

    num_cols = max(len(row) for row in table_2d)

    html = '''
    <table class="erp-table">
'''

    for row_idx, row in enumerate(table_2d):
        html += '        <tr>\n'

        for col_idx in range(num_cols):
            cell = row[col_idx] if col_idx < len(row) else ''
            cell = cell.strip() if cell else ''

            # 첫 행은 헤더 스타일
            if row_idx == 0:
                css_class = 'header'
            # 두 번째 행도 서브헤더로 처리 (보통 컬럼명)
            elif row_idx == 1:
                css_class = 'sub-header'
            # 빈 셀
            elif not cell:
                css_class = 'empty-cell'
            # TOTAL 포함 행
            elif 'TOTAL' in ' '.join([c for c in row if c]).upper():
                css_class = 'total-row-cell'
            # 숫자 데이터 (콤마 포함)
            elif cell.replace(',', '').replace('.', '').isdigit():
                css_class = 'data-cell'
            else:
                css_class = ''

            if css_class:
                html += f'            <td class="{css_class}">{cell}</td>\n'
            else:
                html += f'            <td>{cell}</td>\n'

        html += '        </tr>\n'

    html += '    </table>\n'

    return html




def generate_comet_overlay_section(image_path: str, ocr_results: list) -> str:
    """Comet 오버레이 섹션 생성"""

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    img = Image.open(image_path)
    width, height = img.size

    text_spans = []
    for item in ocr_results:
        x1, y1, x2, y2 = item["box"]
        text = item["text"].replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        score = item.get("score", 1.0)
        font_size = max(10, int((y2 - y1) * 0.8))

        text_spans.append(f'''
            <span class="ocr-text" style="
                left: {x1}px;
                top: {y1}px;
                width: {x2-x1}px;
                height: {y2-y1}px;
                font-size: {font_size}px;
            " title="score: {score:.3f}">{text}</span>''')

    section = f'''
    <div class="section">
        <h2>1. Comet 방식 테이블 추출</h2>
        <p class="info">
            텍스트를 드래그하여 선택/복사할 수 있습니다.<br>
            OCR 결과: <strong>{len(ocr_results)}개</strong> 텍스트 감지
        </p>

        <div class="comet-container" id="container">
            <img class="comet-image"
                 src="data:image/png;base64,{img_base64}"
                 width="{width}" height="{height}"
                 alt="Original Table Image">
            <div class="comet-overlay">
                {"".join(text_spans)}
            </div>
        </div>

        <div class="controls">
            <label>
                <input type="checkbox" id="debugMode" onchange="toggleDebug()">
                디버그 모드 (텍스트 영역 표시)
            </label>
        </div>
    </div>
'''
    return section


def generate_combined_html(image_path: str, ocr_results: list, table_2d: list) -> str:
    """Comet 오버레이 + ERP 테이블 통합 HTML 생성

    table_2d: Grid-First로 추출한 2D 배열 (양식 무관)
    """

    comet_section = generate_comet_overlay_section(image_path, ocr_results)
    erp_table = generate_generic_erp_table_html(table_2d)

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>테이블 추출 - Comet + ERP</title>
    <style>
        body {{
            margin: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        .section {{
            background: #fff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .info {{
            color: #666;
            margin-bottom: 15px;
        }}

        /* Comet 오버레이 스타일 */
        .comet-container {{
            position: relative;
            display: inline-block;
            border: 2px solid #333;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            background: #fff;
        }}
        .comet-image {{
            display: block;
            pointer-events: none;
        }}
        .comet-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .ocr-text {{
            position: absolute;
            color: transparent;
            user-select: text;
            cursor: text;
            line-height: 1.2;
            font-family: Arial, sans-serif;
        }}
        .ocr-text::selection {{
            background: rgba(0, 120, 215, 0.3);
        }}
        .debug-mode .ocr-text {{
            background: rgba(255, 0, 0, 0.1);
            border: 1px dashed rgba(255, 0, 0, 0.3);
        }}
        .controls {{
            margin-top: 15px;
        }}
        .controls label {{
            cursor: pointer;
        }}

        /* ERP 테이블 스타일 */
        .erp-table {{
            border-collapse: collapse;
            font-size: 12px;
            border: 2px solid #000;
            margin-top: 20px;
        }}
        .erp-table td {{
            border: 1px solid #000;
            padding: 5px 10px;
            text-align: center;
            height: 24px;
            background: #fff;
            color: #000;
        }}
        .erp-table .header {{
            font-weight: bold;
            background: #f0f0f0;
        }}
        .erp-table .sub-header {{
            font-weight: bold;
            background: #f8f8f8;
        }}
        .erp-table .color-code {{
            font-weight: bold;
        }}
        .erp-table .color-name {{
            font-weight: bold;
        }}
        .erp-table .data-cell {{
            text-align: right;
        }}
        .erp-table .total-row td {{
            font-weight: bold;
            background: #f0f0f0;
        }}
        .erp-table .total-col {{
            font-weight: bold;
            text-align: right;
        }}
        .erp-table .empty-cell {{
            background: #fff;
        }}
        .erp-table .total-row-cell {{
            font-weight: bold;
            background: #f0f0f0;
        }}

        /* 복사 버튼 */
        .copy-btn {{
            margin-top: 10px;
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .copy-btn:hover {{
            background: #0056b3;
        }}
        .copy-success {{
            color: #28a745;
            margin-left: 10px;
            display: none;
        }}
    </style>
</head>
<body>
    <h1>테이블 추출 결과</h1>

    {comet_section}

    <div class="section">
        <h2>2. ERP 전송용 테이블</h2>
        <p class="info">아래 테이블을 복사하여 ERP 시스템에 붙여넣을 수 있습니다.</p>

        {erp_table}

        <button class="copy-btn" onclick="copyTable()">테이블 복사</button>
        <span class="copy-success" id="copySuccess">복사 완료!</span>
    </div>

    <script>
        function toggleDebug() {{
            const container = document.getElementById('container');
            const checkbox = document.getElementById('debugMode');
            if (checkbox.checked) {{
                container.classList.add('debug-mode');
            }} else {{
                container.classList.remove('debug-mode');
            }}
        }}

        function copyTable() {{
            const table = document.querySelector('.erp-table');
            const range = document.createRange();
            range.selectNode(table);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();

            const successMsg = document.getElementById('copySuccess');
            successMsg.style.display = 'inline';
            setTimeout(() => {{
                successMsg.style.display = 'none';
            }}, 2000);
        }}
    </script>
</body>
</html>'''

    return html


def main(image_path: str, output_path: str = None):
    """메인 함수 - Comet 오버레이 + ERP 테이블 통합 HTML 생성"""

    print(f"=" * 60)
    print(f"ERP용 테이블 생성 (Comet + ERP 통합)")
    print(f"이미지: {image_path}")
    print(f"=" * 60)

    # 이미지 로드
    img = Image.open(image_path)
    print(f"이미지 크기: {img.width} x {img.height}")

    # 1. Grid 구조 분석
    print(f"\n[1단계] 테이블 격자 구조 감지...")
    full_box = [0, 0, img.width, img.height]
    num_rows, num_cols, row_bounds, col_bounds = grid_count_cells_in_region(img, full_box)

    print(f"  격자: {num_rows}행 x {num_cols}열")

    if num_rows < 2 or num_cols < 2:
        print("테이블 격자를 찾을 수 없습니다.")
        return

    table_info = {
        "box": full_box,
        "rows": num_rows,
        "cols": num_cols,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds
    }

    # 2. 전체 이미지 OCR
    print(f"\n[2단계] 전체 이미지 OCR...")
    all_ocr_results = ocr_image_easyocr(img)
    print(f"  OCR 결과: {len(all_ocr_results)}개 텍스트")

    # 3. Grid 기반 셀 매핑
    print(f"\n[3단계] Grid 기반 셀 매핑...")
    table_2d = grid_extract_table(img, table_info)

    # 4. 테이블 구조 확인
    print(f"\n[4단계] Grid-First 테이블 구조 확인...")
    non_empty_rows = sum(1 for row in table_2d if any(cell.strip() for cell in row))
    print(f"  총 행: {len(table_2d)}, 데이터 있는 행: {non_empty_rows}")

    # 처음 5개 행 미리보기
    print(f"\n테이블 미리보기 (처음 5행):")
    for i, row in enumerate(table_2d[:5]):
        row_texts = [cell.strip() for cell in row if cell.strip()]
        print(f"  Row {i}: {row_texts[:6]}{'...' if len(row_texts) > 6 else ''}")

    # 파일 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, f"{base_name}_erp.html")

    # 5. 통합 HTML 생성
    print(f"\n[5단계] 통합 HTML 생성...")
    html = generate_combined_html(image_path, all_ocr_results, table_2d)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n" + "=" * 60)
    print(f"[완료] ERP용 HTML 생성됨")
    print(f"  출력 파일: {output_path}")
    print(f"=" * 60)

    # 브라우저에서 열기
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(output_path))

    return html


if __name__ == "__main__":
    # 테스트: Submaterial_information.png
    image_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"
    main(image_path)
