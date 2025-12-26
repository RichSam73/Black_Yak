"""
이미지에서 테이블 데이터를 OCR로 추출하여 HTML로 변환하는 스크립트
Grid-First 방식: 격자선 먼저 감지 → 셀 경계 확정 → OCR 텍스트 매핑

방안 A+B 구현:
- EasyOCR: PaddleOCR의 콤마/숫자 인식 문제 해결
- Comet 오버레이: 원본 이미지 + 투명 텍스트로 누락 없는 선택/복사
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
        # GPU 사용 불가 시 자동으로 CPU 사용, verbose=False로 다운로드 출력 억제
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


# =============================================================================
# Grid-First 핵심 함수들 (smart_table_extractor.py에서 이식)
# =============================================================================

def grid_find_boxes(img: Image.Image, min_area: int = 1000, max_area_ratio: float = 0.95) -> list:
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
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / (min(w, h) + 1)
        if aspect > 20:
            continue

        boxes.append({"box": [x, y, x+w, y+h], "area": area, "w": w, "h": h})

    boxes.sort(key=lambda b: b["area"], reverse=True)

    # 중복 박스 제거
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
    """
    영역 내 가로선/세로선 수 -> 셀 수 계산
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


def grid_find_tables(img: Image.Image, min_cells: int = 6) -> list:
    """
    테이블 영역 자동 감지
    - 큰 박스 안에 셀 min_cells개 이상이면 테이블
    """
    boxes = grid_find_boxes(img)
    print(f"  발견된 박스: {len(boxes)}개")

    tables = []
    for i, box_info in enumerate(boxes):
        box = box_info["box"]
        num_rows, num_cols, row_bounds, col_bounds = grid_count_cells_in_region(img, box)
        num_cells = num_rows * num_cols
        print(f"    박스 {i}: {box}, {num_rows}행 x {num_cols}열 = {num_cells}셀")

        if num_cells >= min_cells:
            tables.append({
                "box": box,
                "rows": num_rows,
                "cols": num_cols,
                "row_bounds": row_bounds,
                "col_bounds": col_bounds
            })

    return tables


def grid_find_index(value: float, bounds: list) -> int:
    """bounds 리스트에서 value가 속하는 인덱스 찾기"""
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    return -1


def ocr_image_easyocr(image: Image.Image) -> list:
    """EasyOCR로 이미지 OCR (좌표 + 신뢰도 포함)

    PaddleOCR의 콤마/숫자 인식 문제 해결을 위해 EasyOCR 사용
    - bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 형식
    - 반환: [{"text": str, "box": [x1,y1,x2,y2], "score": float}, ...]
    """
    reader = get_easyocr()

    # PIL Image → numpy array
    img_array = np.array(image)

    try:
        # 작은 텍스트 감지를 위해 파라미터 조정
        # - min_size: 작은 텍스트 감지 (기본값 10 → 5)
        # - text_threshold: 텍스트 확률 임계값 낮춤 (기본값 0.7 → 0.5)
        # - low_text: 저해상도 텍스트 감지 개선 (기본값 0.4 → 0.3)
        # - width_ths: 문자 병합 너비 (기본값 0.5)
        results = reader.readtext(
            img_array,
            min_size=5,              # 작은 텍스트 감지
            text_threshold=0.5,      # 텍스트 확률 임계값 낮춤
            low_text=0.3,            # 저해상도 텍스트 감지 개선
            contrast_ths=0.1,        # 대비 임계값 낮춤
            adjust_contrast=0.5,     # 대비 조정
        )

        ocr_results = []
        for (bbox, text, score) in results:
            # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
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
    """
    Grid-First 방식: 테이블 영역에서 데이터 추출
    1. 격자선 기반 셀 경계 확정
    2. 각 셀 내에서 OCR 텍스트 매핑
    """
    box = table_info["box"]
    x1, y1, x2, y2 = box

    # 테이블 영역 크롭
    cropped = img.crop((x1, y1, x2, y2))

    row_bounds = table_info["row_bounds"]
    col_bounds = table_info["col_bounds"]
    num_rows = len(row_bounds) - 1
    num_cols = len(col_bounds) - 1

    print(f"  Grid 감지: {num_rows}행 x {num_cols}열")
    print(f"  Row bounds: {row_bounds}")
    print(f"  Col bounds: {col_bounds}")

    # 빈 테이블 생성
    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    # OCR 실행 (크롭된 영역에 대해) - EasyOCR 사용
    ocr_results = ocr_image_easyocr(cropped)
    print(f"  OCR 결과 (EasyOCR): {len(ocr_results)}개 텍스트")

    # OCR 결과를 격자 셀에 매핑
    for ocr in ocr_results:
        ocr_box = ocr.get("box", [])
        text = ocr.get("text", "").strip()
        score = ocr.get("score", 1.0)

        if not text or len(ocr_box) < 4:
            continue

        # 텍스트 중심점 계산
        cx = (ocr_box[0] + ocr_box[2]) / 2
        cy = (ocr_box[1] + ocr_box[3]) / 2

        # 해당 셀 찾기
        row_idx = grid_find_index(cy, row_bounds)
        col_idx = grid_find_index(cx, col_bounds)

        if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
            if table[row_idx][col_idx]:
                table[row_idx][col_idx] += " " + text
            else:
                table[row_idx][col_idx] = text
            print(f"    [{row_idx},{col_idx}] = '{text}' (score: {score:.3f})")

    return table


# =============================================================================
# 테이블 구조 파싱 및 HTML 생성
# =============================================================================

def parse_table_data(table: list) -> dict:
    """Grid-First로 추출된 2D 테이블을 구조화된 데이터로 파싱"""

    if not table or len(table) < 3:
        return None

    result = {
        'title': '',
        'headers': [],
        'data': [],
        'totals': [],
        'grand_total': ''
    }

    for i, row in enumerate(table):
        # 빈 문자열 제거하고 실제 값만 추출
        row_texts = [cell.strip() for cell in row if cell.strip()]
        print(f"Row {i}: {row_texts}")

        # 첫 번째 행: 타이틀
        if i == 0:
            result['title'] = ' '.join(row_texts)

        # 두 번째 행: 헤더 (SIZE 값들)
        elif i == 1:
            for text in row_texts:
                if text.isdigit() and len(text) == 3:
                    result['headers'].append(text)

        # TOTAL 행
        elif 'TOTAL' in row_texts:
            values = []
            for text in row_texts:
                if text != 'TOTAL':
                    text_clean = text.replace(',', '')
                    if text_clean.isdigit():
                        values.append(text)
            if values:
                result['grand_total'] = values[-1]
                result['totals'] = values[:-1]

        # 데이터 행 (색상 코드로 시작)
        elif len(row_texts) >= 2 and len(row_texts[0]) == 2 and row_texts[0].isupper():
            color_code = row_texts[0]
            color_name = row_texts[1]
            values = []
            total = ''

            for text in row_texts[2:]:
                text_clean = text.replace(',', '')
                if text_clean.isdigit():
                    values.append(text)

            if values:
                total = values[-1]
                values = values[:-1]

            result['data'].append({
                'code': color_code,
                'name': color_name,
                'values': values,
                'total': total
            })

    # Grand Total 검증 (데이터 행 합계로 교차 검증)
    if result['data'] and result['grand_total']:
        data_total_sum = 0
        for d in result['data']:
            total_clean = d['total'].replace(',', '')
            if total_clean.isdigit():
                data_total_sum += int(total_clean)

        grand_clean = result['grand_total'].replace(',', '')
        if grand_clean.isdigit() and data_total_sum > 0:
            if data_total_sum != int(grand_clean):
                print(f"  [검증] Grand Total OCR 오류 감지: '{result['grand_total']}' -> 계산값: {data_total_sum:,}")
                result['grand_total'] = f"{data_total_sum:,}"

    return result


def generate_html(table_data: dict, empty_cols: int = 4, empty_rows: int = 3) -> str:
    """테이블 데이터로 HTML 생성"""

    headers = table_data['headers']
    data = table_data['data']
    totals = table_data['totals']
    grand_total = table_data['grand_total']
    title = table_data['title']

    # 전체 열 수 계산
    total_cols = 2 + len(headers) + empty_cols + 1

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
            background: #fff;
        }}
        table {{
            border-collapse: collapse;
            font-size: 12px;
            border: 2px solid #000;
        }}
        td {{
            border: 1px solid #000;
            padding: 5px 10px;
            text-align: center;
            height: 24px;
            background: #fff;
            color: #000;
        }}
        .header {{
            font-weight: bold;
        }}
        .sub-header {{
            font-weight: bold;
        }}
        .color-code {{
            font-weight: bold;
        }}
        .color-name {{
            font-weight: bold;
        }}
        .data-cell {{
            text-align: right;
        }}
        .total-row td {{
            font-weight: bold;
        }}
        .total-col {{
            font-weight: bold;
            text-align: right;
        }}
    </style>
</head>
<body>
    <table>
        <!-- 1행: 헤더 (전체 병합) -->
        <tr>
            <td colspan="{total_cols}" class="header">{title}</td>
        </tr>
        <!-- 2행: 서브 헤더 -->
        <tr>
            <td class="sub-header" colspan="2">COLOR / SIZE</td>
"""

    # 사이즈 헤더 추가
    for h in headers:
        html += f'            <td class="sub-header">{h}</td>\n'

    # 빈 열 추가
    for _ in range(empty_cols):
        html += '            <td class="empty-cell"></td>\n'

    # TOTAL 헤더
    html += '            <td class="sub-header">TOTAL</td>\n'
    html += '        </tr>\n'

    # 데이터 행
    for row in data:
        html += '        <tr>\n'
        html += f'            <td class="color-code">{row["code"]}</td>\n'
        html += f'            <td class="color-name">{row["name"]}</td>\n'

        # 값 추가
        for val in row["values"]:
            if val:
                html += f'            <td class="data-cell">{val}</td>\n'
            else:
                html += '            <td class="empty-cell"></td>\n'

        # 빈 열 추가
        empty_data_cols = len(headers) - len(row["values"]) + empty_cols
        for _ in range(empty_data_cols):
            html += '            <td class="empty-cell"></td>\n'

        # TOTAL
        html += f'            <td class="total-col">{row["total"]}</td>\n'
        html += '        </tr>\n'

    # 빈 행
    for i in range(empty_rows):
        html += f'        <!-- 빈 행 {i+1} -->\n'
        html += '        <tr>\n'
        for _ in range(total_cols):
            html += '            <td class="empty-cell"></td>\n'
        html += '        </tr>\n'

    # TOTAL 행
    html += '        <!-- TOTAL 행 -->\n'
    html += '        <tr class="total-row">\n'
    html += '            <td colspan="2">TOTAL</td>\n'

    for val in totals:
        html += f'            <td>{val}</td>\n'

    # 빈 열
    remaining_cols = len(headers) - len(totals)
    for _ in range(remaining_cols + empty_cols):
        html += '            <td></td>\n'

    html += f'            <td class="total-col">{grand_total}</td>\n'
    html += '        </tr>\n'

    html += """    </table>
</body>
</html>
"""

    return html


# =============================================================================
# Comet 오버레이 HTML 생성 (방안 B)
# =============================================================================

def generate_comet_overlay_html(image_path: str, ocr_results: list) -> str:
    """
    원본 이미지 + 투명 텍스트 오버레이 HTML 생성 (진정한 Comet 방식)

    핵심 원리:
    - 배경 레이어: 원본 이미지 (pointer-events: none)
    - 오버레이 레이어: OCR 좌표에 맞춰 투명 <span> 배치
    - 사용자가 보는 것: 원본 이미지
    - 사용자가 선택/복사하는 것: 투명 텍스트 레이어

    Args:
        image_path: 원본 이미지 경로
        ocr_results: OCR 결과 리스트 [{"text": str, "box": [x1,y1,x2,y2], "score": float}, ...]

    Returns:
        HTML 문자열
    """
    # 1. 원본 이미지를 base64로 변환
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    # 이미지 크기 가져오기
    img = Image.open(image_path)
    width, height = img.size

    # 2. OCR 결과 → 투명 스팬 생성
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

    # 3. HTML 조합
    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Comet Overlay - Table</title>
    <style>
        body {{
            margin: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        h2 {{
            color: #333;
        }}
        .info {{
            margin-bottom: 15px;
            color: #666;
        }}
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
        /* 선택 시 하이라이트 */
        .ocr-text::selection {{
            background: rgba(0, 120, 215, 0.3);
        }}
        /* 디버그 모드: 텍스트 영역 표시 */
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
    </style>
</head>
<body>
    <h2>🔮 Comet 방식 테이블 추출</h2>
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
    </script>
</body>
</html>'''

    return html


def main(image_path: str, output_path: str = None):
    """
    메인 함수 - Grid-First 방식으로 테이블 추출 + Comet 오버레이

    출력물:
    1. 구조화된 HTML 테이블 (output_path)
    2. Comet 오버레이 HTML (output_path.replace('.html', '_comet.html'))
    """

    print(f"=" * 60)
    print(f"Grid-First 테이블 추출 + Comet 오버레이")
    print(f"이미지: {image_path}")
    print(f"=" * 60)

    # 이미지 로드
    img = Image.open(image_path)
    print(f"이미지 크기: {img.width} x {img.height}")

    # 1. Grid-First: 전체 이미지를 테이블로 간주하고 격자 분석
    print(f"\n[1단계] 테이블 격자 구조 감지...")

    # 전체 이미지 영역에서 격자 구조 분석
    full_box = [0, 0, img.width, img.height]
    num_rows, num_cols, row_bounds, col_bounds = grid_count_cells_in_region(img, full_box)

    print(f"  격자 분석 결과: {num_rows}행 x {num_cols}열")
    print(f"  Row bounds: {row_bounds}")
    print(f"  Col bounds: {col_bounds}")

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

    print(f"테이블 구조: {table_info['rows']}행 x {table_info['cols']}열")

    # 2. 전체 이미지 OCR (Comet용)
    print(f"\n[2단계] 전체 이미지 OCR (EasyOCR)...")
    all_ocr_results = ocr_image_easyocr(img)
    print(f"  전체 OCR 결과: {len(all_ocr_results)}개 텍스트")
    for ocr in all_ocr_results:
        print(f"    '{ocr['text']}' (score: {ocr['score']:.3f})")

    # 3. Grid 기반 셀 매핑
    print(f"\n[3단계] Grid 기반 셀 매핑...")
    table_2d = grid_extract_table(img, table_info)

    # 4. 테이블 데이터 파싱
    print(f"\n[4단계] 테이블 데이터 파싱...")
    table_data = parse_table_data(table_2d)

    if not table_data:
        print("테이블 데이터 파싱 실패")
        return

    print(f"\n파싱 결과:")
    print(f"  Title: {table_data['title']}")
    print(f"  Headers: {table_data['headers']}")
    print(f"  Data rows: {len(table_data['data'])}")
    for d in table_data['data']:
        print(f"    {d['code']} {d['name']}: {d['values']} -> {d['total']}")
    print(f"  Totals: {table_data['totals']}")
    print(f"  Grand Total: {table_data['grand_total']}")

    # 파일 경로 설정
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_output.html")

    comet_output_path = output_path.replace('.html', '_comet.html')

    # 5. 출력 1: 구조화된 HTML 테이블
    print(f"\n[5단계] HTML 생성...")
    html_table = generate_html(table_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_table)
    print(f"  [출력 1] 구조화 테이블: {output_path}")

    # 6. 출력 2: Comet 오버레이 HTML
    print(f"\n[6단계] Comet 오버레이 HTML 생성...")
    html_comet = generate_comet_overlay_html(image_path, all_ocr_results)

    with open(comet_output_path, 'w', encoding='utf-8') as f:
        f.write(html_comet)
    print(f"  [출력 2] Comet 오버레이: {comet_output_path}")

    print(f"\n" + "=" * 60)
    print(f"[완료] 두 가지 HTML 파일 생성됨")
    print(f"  1. 구조화 테이블: {output_path}")
    print(f"  2. Comet 오버레이: {comet_output_path}")
    print(f"=" * 60)

    # 브라우저에서 열기 (둘 다)
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(output_path))
    webbrowser.open('file://' + os.path.realpath(comet_output_path))

    return html_table, html_comet


if __name__ == "__main__":
    # 이미지 경로
    image_path = r"E:\Antigravity\Black_Yak\Reference\BY_Original_Table.png"
    output_path = r"E:\Antigravity\Black_Yak\Reference\BY_Original_Table_output.html"

    main(image_path, output_path)
