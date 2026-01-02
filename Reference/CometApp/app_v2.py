# -*- coding: utf-8 -*-
"""
Comet + ERP Table Extractor v2.0
================================
- img2table: 테이블 구조 추출 (colspan/rowspan 자동 감지)
- PaddleOCR: 텍스트 인식
- Qwen2.5-VL: AI 검증 (선택적)
- 포트: 6003

아키텍처:
1. img2table로 테이블 구조 + OCR 동시 추출
2. HTML 출력 (colspan/rowspan 포함)
3. AI 검증 (합계 오류, 헤더-값 불일치 감지)
"""

from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
import io
import os
import json
import requests

# img2table imports
from img2table.document import Image as Img2TableImage
from img2table.ocr import PaddleOCR as Img2TablePaddleOCR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['JSON_AS_ASCII'] = False  # 한글 유니코드 출력 지원

# Ollama 설정 (AI 검증용)
OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen2.5vl"

# 전역 OCR 인스턴스
_ocr_engine = None


def get_ocr_engine():
    """img2table용 PaddleOCR 엔진 초기화 (한글+영어 지원)"""
    global _ocr_engine
    if _ocr_engine is None:
        print("[init] PaddleOCR engine for img2table (korean)...")
        _ocr_engine = Img2TablePaddleOCR(lang="korean")
        print("[init] PaddleOCR engine ready")
    return _ocr_engine


def extract_tables_from_image(image: Image.Image) -> list:
    """
    img2table을 사용하여 이미지에서 테이블 추출

    Args:
        image: PIL Image 객체

    Returns:
        list of dict: 각 테이블의 HTML, DataFrame, bbox 정보
    """
    ocr = get_ocr_engine()

    # PIL Image를 임시 파일로 저장 (img2table은 파일 경로 필요)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        # img2table로 테이블 추출
        doc = Img2TableImage(src=tmp_path)
        tables = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,       # 암시적 행 감지
            borderless_tables=False,  # 테두리 있는 테이블만
            min_confidence=50         # 최소 신뢰도
        )

        results = []
        for idx, table in enumerate(tables):
            # BBox 객체를 tuple로 변환
            bbox = None
            if hasattr(table, 'bbox') and table.bbox is not None:
                b = table.bbox
                bbox = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))

            result = {
                'index': idx,
                'html': table.html,
                'df': table.df,
                'bbox': bbox,
                'has_colspan': 'colspan' in table.html,
                'has_rowspan': 'rowspan' in table.html,
            }
            results.append(result)

        return results

    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def validate_table_with_ai(image: Image.Image, table_html: str) -> dict:
    """
    AI를 사용하여 테이블 검증 (합계 오류, 데이터 불일치 감지)

    Args:
        image: PIL Image 객체
        table_html: 추출된 HTML 테이블

    Returns:
        dict: 검증 결과 (is_valid, errors, warnings)
    """
    # 이미지를 base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    prompt = f"""Analyze this table image and the extracted HTML.
Check for:
1. Sum/total errors (do row/column totals match?)
2. Missing data
3. OCR errors

Extracted HTML:
{table_html[:2000]}

Respond in JSON format:
{{
    "is_valid": true/false,
    "errors": ["error1", "error2"],
    "warnings": ["warning1"],
    "corrections": {{"cell_location": "correct_value"}}
}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('response', '{}')

            # JSON 파싱 시도
            try:
                # JSON 블록 추출
                import re
                json_match = re.search(r'\{[\s\S]*\}', raw_text)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"is_valid": True, "errors": [], "warnings": ["AI validation skipped"]}

    except Exception as e:
        return {"is_valid": True, "errors": [], "warnings": [f"AI validation failed: {str(e)}"]}


def generate_result_html(tables: list, validation: dict = None) -> str:
    """
    추출된 테이블을 HTML로 렌더링

    Args:
        tables: extract_tables_from_image의 결과
        validation: AI 검증 결과 (선택적)

    Returns:
        str: HTML 문자열
    """
    if not tables:
        return "<p>No tables detected</p>"

    html_parts = []

    for table in tables:
        # 테이블 헤더
        html_parts.append(f"""
        <div class="table-container">
            <h3>Table {table['index'] + 1}</h3>
            <p class="table-info">
                Columns: {len(table['df'].columns)} |
                Rows: {len(table['df'])} |
                colspan: {'Yes' if table['has_colspan'] else 'No'} |
                rowspan: {'Yes' if table['has_rowspan'] else 'No'}
            </p>
            <div class="table-wrapper">
                {table['html']}
            </div>
        </div>
        """)

    # 검증 결과 추가
    if validation:
        if validation.get('errors'):
            html_parts.append(f"""
            <div class="validation-errors">
                <h4>Errors</h4>
                <ul>
                    {''.join(f'<li>{e}</li>' for e in validation['errors'])}
                </ul>
            </div>
            """)
        if validation.get('warnings'):
            html_parts.append(f"""
            <div class="validation-warnings">
                <h4>Warnings</h4>
                <ul>
                    {''.join(f'<li>{w}</li>' for w in validation['warnings'])}
                </ul>
            </div>
            """)

    return '\n'.join(html_parts)


def process_image(img: Image.Image, img_base64: str) -> dict:
    """
    이미지 처리 메인 함수

    Args:
        img: PIL Image 객체
        img_base64: base64 인코딩된 이미지

    Returns:
        dict: 처리 결과
    """
    result = {
        'success': False,
        'tables': [],
        'html': '',
        'validation': None,
        'error': None
    }

    try:
        # 1. 테이블 추출
        tables = extract_tables_from_image(img)

        if not tables:
            result['error'] = 'No tables detected in image'
            return result

        result['tables'] = tables
        result['success'] = True

        # 2. AI 검증 (첫 번째 테이블만)
        if tables:
            validation = validate_table_with_ai(img, tables[0]['html'])
            result['validation'] = validation

        # 3. HTML 생성
        result['html'] = generate_result_html(tables, result['validation'])

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comet Table Extractor v2.0</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .version-badge {
            background: #4CAF50;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 14px;
            margin-left: 10px;
        }
        .upload-area {
            background: white;
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #4CAF50;
        }
        .upload-area.dragover {
            border-color: #4CAF50;
            background: #e8f5e9;
        }
        #file-input {
            display: none;
        }
        .btn {
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .btn:hover {
            background: #45a049;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result-container {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            flex: 1;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .panel h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .image-preview {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .table-container {
            margin-bottom: 20px;
        }
        .table-info {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .table-wrapper {
            overflow-x: auto;
        }
        .table-wrapper table {
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
        }
        .table-wrapper td, .table-wrapper th {
            border: 1px solid #333;
            padding: 8px;
            text-align: center;
        }
        .table-wrapper th {
            background: #f0f0f0;
        }
        .validation-errors {
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin-top: 15px;
            border-radius: 4px;
        }
        .validation-warnings {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin-top: 15px;
            border-radius: 4px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .tech-badge {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comet Table Extractor <span class="version-badge">v2.0</span></h1>
        <p>
            <span class="tech-badge">img2table</span>
            <span class="tech-badge">PaddleOCR</span>
            <span class="tech-badge">Qwen2.5-VL</span>
        </p>

        <div class="upload-area" id="upload-area">
            <input type="file" id="file-input" accept="image/*">
            <p>Drag & drop an image here or click to select</p>
            <button class="btn" onclick="document.getElementById('file-input').click()">
                Select Image
            </button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Extracting tables...</p>
        </div>

        <div class="result-container" id="result-container" style="display: none;">
            <div class="panel">
                <h2>Original Image</h2>
                <img id="preview-image" class="image-preview" alt="Preview">
            </div>
            <div class="panel">
                <h2>Extracted Tables</h2>
                <div id="result-html"></div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('result-container');
        const previewImage = document.getElementById('preview-image');
        const resultHtml = document.getElementById('result-html');

        // Drag & drop
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
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(file);

            // Upload
            uploadImage(file);
        }

        function uploadImage(file) {
            loading.style.display = 'block';
            resultContainer.style.display = 'none';

            const formData = new FormData();
            formData.append('image', file);

            fetch('/api/extract', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                resultContainer.style.display = 'flex';

                if (data.success) {
                    resultHtml.innerHTML = data.html;
                } else {
                    resultHtml.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                resultContainer.style.display = 'flex';
                resultHtml.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/extract', methods=['POST'])
def api_extract():
    """이미지에서 테이블 추출 API"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        # 이미지 로드
        img = Image.open(file.stream).convert('RGB')

        # base64 인코딩
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 처리
        result = process_image(img, img_base64)

        # DataFrame은 JSON 직렬화 불가하므로 제거
        if result.get('tables'):
            for table in result['tables']:
                if 'df' in table:
                    shape = table['df'].shape
                    table['df_shape'] = (int(shape[0]), int(shape[1]))
                    del table['df']

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/health')
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'ok',
        'version': '2.0',
        'engine': 'img2table + PaddleOCR'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Comet Table Extractor v2.0")
    print("=" * 60)
    print("Engine: img2table + PaddleOCR + Qwen2.5-VL")
    print("Port: 6003")
    print("=" * 60)

    # OCR 엔진 미리 초기화
    get_ocr_engine()

    app.run(host='0.0.0.0', port=6003, debug=True)
