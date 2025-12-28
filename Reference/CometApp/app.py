"""
Comet + ERP 통합 웹 앱
- PaddleOCR 사용 (한글 인식 정확도 향상)
- Comet 오버레이 + ERP 테이블 동시 제공
- Grid 감지 방식으로 셀 매핑
"""
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
import io
from paddleocr import PaddleOCR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# 전역 OCR 인스턴스
_paddle_ocr = None

def get_paddleocr():
    """PaddleOCR 인스턴스 싱글톤 (한글)"""
    global _paddle_ocr
    if _paddle_ocr is None:
        print("  [PaddleOCR 초기화 중... (한글)]")
        _paddle_ocr = PaddleOCR(lang='korean')
    return _paddle_ocr


def ocr_image_paddle(image: Image.Image) -> list:
    """PaddleOCR로 이미지 OCR"""
    ocr = get_paddleocr()
    img_array = np.array(image)

    try:
        # 새로운 predict API 사용
        results = ocr.predict(img_array)

        ocr_results = []
        if results:
            # 새 API 결과 구조: list of dict with 'rec_texts', 'rec_scores', 'dt_polys' 등
            for result in results:
                rec_texts = result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', [])
                dt_polys = result.get('dt_polys', [])

                for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, dt_polys)):
                    if not text.strip():
                        continue

                    # poly는 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형식
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    box = [int(min(x_coords)), int(min(y_coords)),
                           int(max(x_coords)), int(max(y_coords))]

                    ocr_results.append({
                        "text": text,
                        "box": box,
                        "score": float(score)
                    })

        return ocr_results
    except Exception as e:
        print(f"PaddleOCR 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


# =============================================================================
# Grid-First 핵심 함수들
# =============================================================================

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


def grid_count_cells_in_region(img: Image.Image, box: list, min_line_len: int = 40) -> tuple:
    """영역 내 가로선/세로선 수 -> 셀 수 계산"""
    x1, y1, x2, y2 = box
    cropped = img.crop((x1, y1, x2, y2))

    img_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape

    # 가로선 찾기 (이미지 너비의 30% 이상인 선만)
    h_min_len = max(min_line_len, int(w * 0.3))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines, axis=1)
    h_coords = np.where(h_proj > h_min_len)[0]
    row_bounds = grid_group_coords(h_coords, gap=10)

    # 세로선 찾기 (이미지 높이의 20% 이상인 선만)
    v_min_len = max(min_line_len, int(h * 0.2))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_min_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0)
    v_coords = np.where(v_proj > v_min_len)[0]
    col_bounds = grid_group_coords(v_coords, gap=10)

    num_rows = max(0, len(row_bounds) - 1)
    num_cols = max(0, len(col_bounds) - 1)

    print(f"[Grid] {num_rows}행 x {num_cols}열")
    print(f"[Grid] row_bounds: {row_bounds}")
    print(f"[Grid] col_bounds: {col_bounds}")

    return num_rows, num_cols, row_bounds, col_bounds


def grid_find_index(value: float, bounds: list) -> int:
    """bounds 리스트에서 value가 속하는 인덱스 찾기"""
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    return -1


def grid_extract_table_from_ocr(ocr_results: list, row_bounds: list, col_bounds: list) -> list:
    """OCR 결과를 Grid 셀에 매핑하여 2D 테이블 생성"""
    num_rows = len(row_bounds) - 1
    num_cols = len(col_bounds) - 1

    if num_rows <= 0 or num_cols <= 0:
        return []

    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for ocr in ocr_results:
        ocr_box = ocr.get("box", [])
        text = ocr.get("text", "").strip()

        if not text or len(ocr_box) < 4:
            continue

        # 중심점 기반 매핑
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
# HTML 생성
# =============================================================================

def generate_erp_table_html(table_2d: list) -> str:
    """Grid-First 2D 테이블을 ERP용 HTML 테이블로 변환"""

    if not table_2d or len(table_2d) == 0:
        return '<p style="color: #ff6b6b;">테이블 격자를 감지하지 못했습니다. Comet 탭에서 직접 텍스트를 복사해주세요.</p>'

    num_cols = max(len(row) for row in table_2d)

    html = '<table class="erp-table">\n'

    for row_idx, row in enumerate(table_2d):
        html += '<tr>\n'

        for col_idx in range(num_cols):
            cell = row[col_idx] if col_idx < len(row) else ''
            cell = cell.strip() if cell else ''

            if row_idx == 0:
                css_class = 'header'
            elif row_idx == 1:
                css_class = 'sub-header'
            elif not cell:
                css_class = 'empty-cell'
            elif 'TOTAL' in ' '.join([c for c in row if c]).upper():
                css_class = 'total-row-cell'
            elif cell.replace(',', '').replace('.', '').replace('-', '').isdigit():
                css_class = 'data-cell'
            else:
                css_class = ''

            if css_class:
                html += f'<td class="{css_class}">{cell}</td>\n'
            else:
                html += f'<td>{cell}</td>\n'

        html += '</tr>\n'

    html += '</table>\n'

    return html


def process_image(img: Image.Image, img_base64: str) -> dict:
    """이미지 처리 - Comet OCR + Grid 기반 ERP 테이블"""

    width, height = img.size

    # 1. PaddleOCR 수행 (한 번만!)
    ocr_results = ocr_image_paddle(img)

    # 2. Grid 구조 분석
    full_box = [0, 0, width, height]
    num_rows, num_cols, row_bounds, col_bounds = grid_count_cells_in_region(img, full_box)

    # 3. OCR 결과를 Grid 셀에 매핑
    has_grid = num_rows >= 2 and num_cols >= 2
    if has_grid:
        table_2d = grid_extract_table_from_ocr(ocr_results, row_bounds, col_bounds)
        grid_info = f"{num_rows}행 x {num_cols}열"
    else:
        table_2d = []
        grid_info = "격자 없음"

    # 4. Comet 텍스트 스팬 생성
    text_spans = []
    for item in ocr_results:
        x1, y1, x2, y2 = item["box"]
        text = item["text"].replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        score = item.get("score", 1.0)
        font_size = max(10, int((y2 - y1) * 0.8))

        text_spans.append({
            "x": x1, "y": y1,
            "width": x2 - x1, "height": y2 - y1,
            "text": text, "score": score,
            "fontSize": font_size
        })

    # 5. ERP 테이블 HTML 생성
    erp_table_html = generate_erp_table_html(table_2d)

    return {
        "success": True,
        "width": width,
        "height": height,
        "image_base64": img_base64,
        "ocr_count": len(ocr_results),
        "grid_info": grid_info,
        "has_grid": has_grid,
        "text_spans": text_spans,
        "erp_table_html": erp_table_html
    }


# =============================================================================
# HTML 템플릿
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comet + ERP 테이블 추출 (PaddleOCR)</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #4ecca3;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.2em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            color: #a0a0a0;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .engine-badge {
            display: inline-block;
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }

        /* 업로드 영역 */
        .upload-section {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #4ecca3;
            border-radius: 16px;
            padding: 60px 40px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(78, 204, 163, 0.05);
        }
        .upload-area:hover {
            border-color: #45b393;
            background: rgba(78, 204, 163, 0.1);
            transform: scale(1.01);
        }
        .upload-area.dragover {
            border-color: #e94560;
            background: rgba(233, 69, 96, 0.1);
        }
        .upload-icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        .upload-text {
            font-size: 20px;
            color: #fff;
            margin-bottom: 10px;
        }
        .upload-hint {
            font-size: 14px;
            color: #888;
        }
        #fileInput {
            display: none;
        }

        /* 로딩 */
        .loading {
            display: none;
            text-align: center;
            padding: 60px;
            color: #fff;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(78, 204, 163, 0.2);
            border-top: 4px solid #4ecca3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            font-size: 18px;
            color: #ccc;
        }

        /* 결과 영역 */
        .result-section {
            display: none;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        .result-section.active {
            display: block;
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            color: #4ecca3;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(78, 204, 163, 0.3);
        }
        .info-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        .info-badge {
            display: inline-block;
            background: rgba(78, 204, 163, 0.2);
            color: #4ecca3;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            border: 1px solid rgba(78, 204, 163, 0.3);
        }

        /* 탭 */
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 14px 28px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px 12px 0 0;
            cursor: pointer;
            font-size: 15px;
            color: #ccc;
            transition: all 0.2s;
        }
        .tab:hover {
            background: rgba(255,255,255,0.15);
        }
        .tab.active {
            background: linear-gradient(135deg, #4ecca3, #45b393);
            color: white;
            border-color: transparent;
        }
        .tab-content {
            display: none;
            background: rgba(0,0,0,0.2);
            border-radius: 0 12px 12px 12px;
            padding: 20px;
        }
        .tab-content.active {
            display: block;
        }

        /* Comet 컨테이너 */
        .comet-wrapper {
            overflow: auto;
            max-height: 70vh;
            border: 2px solid #333;
            border-radius: 8px;
            background: #1a1a1a;
        }
        .comet-container {
            position: relative;
            display: inline-block;
        }
        .comet-image {
            display: block;
            pointer-events: none;
        }
        .comet-overlay {
            position: absolute;
            top: 0;
            left: 0;
        }
        .ocr-text {
            position: absolute;
            color: transparent;
            user-select: text;
            cursor: text;
            line-height: 1.2;
        }
        .ocr-text::selection {
            background: rgba(78, 204, 163, 0.4);
        }
        .debug-mode .ocr-text {
            background: rgba(78, 204, 163, 0.2);
            border: 1px solid rgba(78, 204, 163, 0.5);
        }

        /* ERP 테이블 */
        .erp-wrapper {
            overflow: auto;
            max-height: 70vh;
            background: #fff;
            border-radius: 8px;
            padding: 20px;
        }
        .erp-table {
            border-collapse: collapse;
            font-size: 13px;
            border: 2px solid #000;
            width: auto;
            min-width: 100%;
        }
        .erp-table td {
            border: 1px solid #000;
            padding: 6px 12px;
            text-align: center;
            height: 28px;
            background: #fff;
            white-space: nowrap;
            color: #000;
        }
        .erp-table .header {
            font-weight: bold;
            background: #d0d0d0;
        }
        .erp-table .sub-header {
            font-weight: bold;
            background: #e8e8e8;
        }
        .erp-table .data-cell {
            text-align: right;
        }
        .erp-table .empty-cell {
            background: #fff;
        }
        .erp-table .total-row-cell {
            font-weight: bold;
            background: #f0f0f0;
        }

        /* 버튼 */
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #4ecca3, #45b393);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(78, 204, 163, 0.4);
        }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .btn-secondary:hover {
            background: rgba(255,255,255,0.15);
        }
        .btn-success {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
        }
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        }

        /* 컨트롤 */
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
            align-items: center;
        }
        .controls label {
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ccc;
            font-size: 14px;
        }
        .controls input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #4ecca3;
        }

        /* 토스트 */
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #4ecca3, #45b393);
            color: white;
            padding: 15px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(78, 204, 163, 0.4);
            display: none;
            z-index: 1000;
            font-weight: 600;
        }
        .toast.show {
            display: block;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* 안내 텍스트 */
        .help-text {
            color: #888;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .help-text strong {
            color: #4ecca3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Comet + ERP 테이블 추출 <span class="engine-badge">PaddleOCR</span></h1>
        <p class="subtitle">이미지에서 텍스트를 추출하고 ERP 테이블로 변환합니다</p>

        <!-- 업로드 섹션 -->
        <div class="upload-section" id="uploadSection">
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">🖼️</div>
                <div class="upload-text">이미지를 드래그하거나 클릭하여 업로드</div>
                <div class="upload-hint">PNG, JPG, JPEG 지원 (최대 16MB)</div>
            </div>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <!-- 로딩 -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">PaddleOCR 처리 중... 잠시만 기다려주세요</div>
        </div>

        <!-- 결과 섹션 -->
        <div class="result-section" id="resultSection">
            <div class="info-bar">
                <button class="btn btn-secondary" onclick="resetUpload()">🔄 새 이미지</button>
                <span id="imageInfo"></span>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('comet')">1️⃣ Comet 오버레이</button>
                <button class="tab" onclick="switchTab('erp')">2️⃣ ERP 테이블</button>
            </div>

            <!-- Comet 탭 -->
            <div class="tab-content active" id="cometTab">
                <div class="section-title">📝 Comet 방식 텍스트 추출</div>
                <p class="help-text">
                    <strong>사용법:</strong> 이미지 위의 텍스트를 드래그하여 선택하고 <strong>Ctrl+C</strong>로 복사하세요.
                </p>
                <div class="controls">
                    <label>
                        <input type="checkbox" id="debugMode" onchange="toggleDebug()">
                        디버그 모드 (텍스트 영역 표시)
                    </label>
                </div>
                <div class="comet-wrapper">
                    <div class="comet-container" id="cometContainer">
                        <img class="comet-image" id="cometImage">
                        <div class="comet-overlay" id="cometOverlay"></div>
                    </div>
                </div>
            </div>

            <!-- ERP 탭 -->
            <div class="tab-content" id="erpTab">
                <div class="section-title">📋 ERP 전송용 테이블</div>
                <p class="help-text">
                    <strong>사용법:</strong> 아래 테이블을 복사하여 ERP 시스템에 붙여넣을 수 있습니다.
                </p>
                <div class="controls">
                    <button class="btn btn-success" onclick="copyTable()">📋 테이블 복사</button>
                </div>
                <div class="erp-wrapper" id="erpTableContainer"></div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const uploadSection = document.getElementById('uploadSection');
        const resultSection = document.getElementById('resultSection');

        // 드래그 앤 드롭
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                processFile(file);
            }
        });

        function processFile(file) {
            uploadSection.style.display = 'none';
            loading.classList.add('active');

            const formData = new FormData();
            formData.append('image', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('active');
                if (data.error) {
                    showToast('오류: ' + data.error, true);
                    uploadSection.style.display = 'block';
                } else {
                    displayResult(data);
                }
            })
            .catch(error => {
                loading.classList.remove('active');
                showToast('오류가 발생했습니다', true);
                uploadSection.style.display = 'block';
            });
        }

        function displayResult(data) {
            // 이미지 정보
            document.getElementById('imageInfo').innerHTML =
                `<span class="info-badge">📐 ${data.width} x ${data.height}</span>` +
                `<span class="info-badge">📝 ${data.ocr_count}개 텍스트</span>` +
                `<span class="info-badge">📊 ${data.grid_info}</span>`;

            // Comet 이미지
            const cometImage = document.getElementById('cometImage');
            cometImage.src = 'data:image/png;base64,' + data.image_base64;
            cometImage.width = data.width;
            cometImage.height = data.height;

            // OCR 텍스트 오버레이
            const overlay = document.getElementById('cometOverlay');
            overlay.innerHTML = '';
            overlay.style.width = data.width + 'px';
            overlay.style.height = data.height + 'px';

            data.text_spans.forEach(span => {
                const el = document.createElement('span');
                el.className = 'ocr-text';
                el.style.left = span.x + 'px';
                el.style.top = span.y + 'px';
                el.style.width = span.width + 'px';
                el.style.height = span.height + 'px';
                el.style.fontSize = span.fontSize + 'px';
                el.title = 'score: ' + span.score.toFixed(3);
                el.textContent = span.text;
                overlay.appendChild(el);
            });

            // ERP 테이블
            document.getElementById('erpTableContainer').innerHTML = data.erp_table_html;

            resultSection.classList.add('active');
        }

        function resetUpload() {
            resultSection.classList.remove('active');
            uploadSection.style.display = 'block';
            fileInput.value = '';
            // 탭 초기화
            switchTab('comet');
        }

        function toggleDebug() {
            const container = document.getElementById('cometContainer');
            const checkbox = document.getElementById('debugMode');
            if (checkbox.checked) {
                container.classList.add('debug-mode');
            } else {
                container.classList.remove('debug-mode');
            }
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            if (tabName === 'comet') {
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('cometTab').classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('erpTab').classList.add('active');
            }
        }

        function copyTable() {
            const table = document.querySelector('.erp-table');
            if (!table) {
                showToast('복사할 테이블이 없습니다', true);
                return;
            }

            const range = document.createRange();
            range.selectNode(table);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();

            showToast('테이블 복사 완료!');
        }

        function showToast(message, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.style.background = isError
                ? 'linear-gradient(135deg, #e94560, #ff6b6b)'
                : 'linear-gradient(135deg, #4ecca3, #45b393)';
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "이미지를 선택해주세요."})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "이미지를 선택해주세요."})

    try:
        # 이미지 읽기
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Base64 인코딩
        img_base64 = base64.b64encode(img_bytes).decode()

        # 처리 (PaddleOCR + Grid 매핑)
        result = process_image(img, img_base64)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"처리 오류: {str(e)}"})


if __name__ == '__main__':
    print("=" * 50)
    print("Comet + ERP 테이블 추출 (PaddleOCR)")
    print("http://localhost:5001 에서 접속하세요")
    print("=" * 50)
    app.run(debug=True, port=5001)
