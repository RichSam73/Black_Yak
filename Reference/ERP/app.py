"""
ERP 테이블 추출 웹 앱
- 이미지 업로드 → OCR → Comet 오버레이 + ERP 테이블 생성
"""
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import cv2
import numpy as np
import os
import base64
import io
import easyocr

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# 전역 OCR 인스턴스
_easyocr_reader = None

def get_easyocr():
    """EasyOCR 인스턴스 싱글톤 (한글+영어)"""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("  [EasyOCR 초기화 중... (한글+영어)]")
        _easyocr_reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    return _easyocr_reader


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

    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    ocr_results = ocr_image_easyocr(cropped)

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
# HTML 생성
# =============================================================================

def generate_generic_erp_table_html(table_2d: list) -> str:
    """Grid-First 2D 테이블을 ERP용 HTML 테이블로 변환"""

    if not table_2d or len(table_2d) == 0:
        return '<p>테이블 데이터가 없습니다.</p>'

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
            elif cell.replace(',', '').replace('.', '').isdigit():
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
    """이미지 처리 및 결과 반환"""

    width, height = img.size

    # 1. Grid 구조 분석
    full_box = [0, 0, width, height]
    num_rows, num_cols, row_bounds, col_bounds = grid_count_cells_in_region(img, full_box)

    if num_rows < 2 or num_cols < 2:
        return {"error": "테이블 격자를 찾을 수 없습니다."}

    table_info = {
        "box": full_box,
        "rows": num_rows,
        "cols": num_cols,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds
    }

    # 2. 전체 이미지 OCR
    all_ocr_results = ocr_image_easyocr(img)

    # 3. Grid 기반 셀 매핑
    table_2d = grid_extract_table(img, table_info)

    # 4. OCR 텍스트 스팬 생성
    text_spans = []
    for item in all_ocr_results:
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
    erp_table_html = generate_generic_erp_table_html(table_2d)

    return {
        "success": True,
        "width": width,
        "height": height,
        "image_base64": img_base64,
        "ocr_count": len(all_ocr_results),
        "grid_info": f"{num_rows}행 x {num_cols}열",
        "text_spans": text_spans,
        "erp_table_html": erp_table_html
    }


# =============================================================================
# Flask 라우트
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERP 테이블 추출기</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        /* 업로드 영역 */
        .upload-section {
            background: white;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 60px 40px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        .upload-area.dragover {
            border-color: #28a745;
            background: #e8f5e9;
        }
        .upload-icon {
            font-size: 64px;
            margin-bottom: 20px;
        }
        .upload-text {
            font-size: 18px;
            color: #666;
            margin-bottom: 10px;
        }
        .upload-hint {
            font-size: 14px;
            color: #999;
        }
        #fileInput {
            display: none;
        }

        /* 로딩 */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 결과 영역 */
        .result-section {
            display: none;
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .result-section.active {
            display: block;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .info-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            margin-right: 10px;
        }

        /* Comet 컨테이너 */
        .comet-container {
            position: relative;
            display: inline-block;
            border: 2px solid #333;
            margin: 20px 0;
            overflow: auto;
            max-width: 100%;
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
            background: rgba(0, 120, 215, 0.3);
        }
        .debug-mode .ocr-text {
            background: rgba(255, 0, 0, 0.15);
            border: 1px dashed rgba(255, 0, 0, 0.5);
        }

        /* ERP 테이블 */
        .erp-table {
            border-collapse: collapse;
            font-size: 12px;
            border: 2px solid #000;
            margin: 20px 0;
            width: 100%;
            overflow-x: auto;
            display: block;
        }
        .erp-table td {
            border: 1px solid #000;
            padding: 5px 10px;
            text-align: center;
            height: 24px;
            background: #fff;
            white-space: nowrap;
        }
        .erp-table .header {
            font-weight: bold;
            background: #e8e8e8;
        }
        .erp-table .sub-header {
            font-weight: bold;
            background: #f0f0f0;
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
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5a6fd6;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-success:hover {
            background: #218838;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }

        /* 컨트롤 */
        .controls {
            margin: 15px 0;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .controls label {
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        /* 토스트 메시지 */
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #28a745;
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            display: none;
            z-index: 1000;
        }
        .toast.show {
            display: block;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* 탭 */
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 24px;
            background: #f0f0f0;
            border: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 ERP 테이블 추출기</h1>

        <!-- 업로드 섹션 -->
        <div class="upload-section" id="uploadSection">
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📤</div>
                <div class="upload-text">이미지를 드래그하거나 클릭하여 업로드</div>
                <div class="upload-hint">지원 형식: PNG, JPG, JPEG (최대 16MB)</div>
            </div>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <!-- 로딩 -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>이미지 분석 중... (OCR 처리에 시간이 걸릴 수 있습니다)</div>
        </div>

        <!-- 결과 섹션 -->
        <div class="result-section" id="resultSection">
            <div class="controls">
                <button class="btn btn-secondary" onclick="resetUpload()">🔄 새 이미지 업로드</button>
                <span id="imageInfo"></span>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('comet')">1. Comet 오버레이</button>
                <button class="tab" onclick="switchTab('erp')">2. ERP 테이블</button>
            </div>

            <!-- Comet 탭 -->
            <div class="tab-content active" id="cometTab">
                <div class="section-title">Comet 방식 테이블 추출</div>
                <p>텍스트를 드래그하여 선택/복사할 수 있습니다.</p>
                <div class="controls">
                    <label>
                        <input type="checkbox" id="debugMode" onchange="toggleDebug()">
                        디버그 모드 (텍스트 영역 표시)
                    </label>
                </div>
                <div class="comet-container" id="cometContainer">
                    <img class="comet-image" id="cometImage">
                    <div class="comet-overlay" id="cometOverlay"></div>
                </div>
            </div>

            <!-- ERP 탭 -->
            <div class="tab-content" id="erpTab">
                <div class="section-title">ERP 전송용 테이블</div>
                <p>아래 테이블을 복사하여 ERP 시스템에 붙여넣을 수 있습니다.</p>
                <div class="controls">
                    <button class="btn btn-success" onclick="copyTable()">📋 테이블 복사</button>
                </div>
                <div id="erpTableContainer"></div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast">복사 완료!</div>

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
                    alert(data.error);
                    uploadSection.style.display = 'block';
                } else {
                    displayResult(data);
                }
            })
            .catch(error => {
                loading.classList.remove('active');
                alert('오류가 발생했습니다: ' + error);
                uploadSection.style.display = 'block';
            });
        }

        function displayResult(data) {
            // 이미지 정보
            document.getElementById('imageInfo').innerHTML =
                `<span class="info-badge">크기: ${data.width} x ${data.height}</span>` +
                `<span class="info-badge">OCR: ${data.ocr_count}개 텍스트</span>` +
                `<span class="info-badge">Grid: ${data.grid_info}</span>`;

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
            if (!table) return;

            const range = document.createRange();
            range.selectNode(table);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();

            showToast('복사 완료!');
        }

        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 2000);
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

        # 처리
        result = process_image(img, img_base64)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"처리 오류: {str(e)}"})


if __name__ == '__main__':
    print("=" * 50)
    print("ERP 테이블 추출 웹 앱")
    print("http://localhost:5000 에서 접속하세요")
    print("=" * 50)
    app.run(debug=True, port=5000)
