# -*- coding: utf-8 -*-
"""
PDF Translator - 한글 텍스트를 다국어로 번역하는 웹앱
- Flask 기반 웹 인터페이스
- PaddleOCR + VLM (qwen2.5vl) 사용
- 지원 언어: 영어, 베트남어, 중국어, 일본어
"""

import os
import sys
import io
import json
import base64
import tempfile
import requests
from datetime import datetime
from flask import Flask, render_template_string, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR
import cv2
import fitz  # PyMuPDF

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)

# 설정
OLLAMA_URL = "http://localhost:11434/api/generate"
UPLOAD_FOLDER = tempfile.gettempdir()
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 언어별 설정
LANGUAGE_CONFIG = {
    "english": {
        "name": "English",
        "code": "en",
        "prompt_lang": "English"
    },
    "vietnamese": {
        "name": "Tiếng Việt",
        "code": "vi",
        "prompt_lang": "Vietnamese"
    },
    "chinese": {
        "name": "中文",
        "code": "zh",
        "prompt_lang": "Chinese (Simplified)"
    },
    "japanese": {
        "name": "日本語",
        "code": "ja",
        "prompt_lang": "Japanese"
    }
}

# 의류 전문 용어 사전 (한글 → 다국어)
GARMENT_DICT = {
    "english": {
        "남성": "Men's", "여성": "Women's", "자켓": "Jacket", "다운자켓": "Down Jacket",
        "후드": "Hood", "에리": "Collar", "봉제": "Sewing", "작업": "Work",
        "원단": "Fabric", "안감": "Lining", "겉감": "Shell", "소매": "Sleeve",
        "밑단": "Hem", "어깨": "Shoulder", "가슴": "Chest", "허리": "Waist",
        "지퍼": "Zipper", "스토퍼": "Stopper", "고리": "Loop", "테이프": "Tape",
        "앞판": "Front Panel", "뒷판": "Back Panel", "로고": "LOGO",
        "벨크로": "Velcro", "밴드": "Band", "아일렛": "Eyelet", "스트링": "String",
        "주머니": "Pocket", "포켓": "Pocket", "메인": "Main", "라벨": "Label"
    },
    "vietnamese": {
        "남성": "Nam", "여성": "Nữ", "자켓": "Áo khoác", "다운자켓": "Áo phao",
        "후드": "Mũ trùm", "에리": "Cổ áo", "봉제": "May", "작업": "Công việc",
        "원단": "Vải", "안감": "Lót", "겉감": "Vỏ ngoài", "소매": "Tay áo",
        "밑단": "Gấu áo", "어깨": "Vai", "가슴": "Ngực", "허리": "Eo",
        "지퍼": "Khóa kéo", "스토퍼": "Nút chặn", "고리": "Vòng", "테이프": "Băng dính",
        "앞판": "Thân trước", "뒷판": "Thân sau", "로고": "Logo",
        "벨크로": "Velcro", "밴드": "Dây đai", "아일렛": "Lỗ xỏ dây", "스트링": "Dây rút",
        "주머니": "Túi", "포켓": "Túi", "메인": "Chính", "라벨": "Nhãn"
    },
    "chinese": {
        "남성": "男士", "여성": "女士", "자켓": "夹克", "다운자켓": "羽绒服",
        "후드": "连帽", "에리": "领子", "봉제": "缝纫", "작업": "工作",
        "원단": "面料", "안감": "里料", "겉감": "外层", "소매": "袖子",
        "밑단": "下摆", "어깨": "肩部", "가슴": "胸部", "허리": "腰部",
        "지퍼": "拉链", "스토퍼": "止扣", "고리": "环扣", "테이프": "胶带",
        "앞판": "前片", "뒷판": "后片", "로고": "标志",
        "벨크로": "魔术贴", "밴드": "松紧带", "아일렛": "鸡眼", "스트링": "抽绳",
        "주머니": "口袋", "포켓": "口袋", "메인": "主要", "라벨": "标签"
    },
    "japanese": {
        "남성": "メンズ", "여성": "レディース", "자켓": "ジャケット", "다운자켓": "ダウンジャケット",
        "후드": "フード", "에리": "襟", "봉제": "縫製", "작업": "作業",
        "원단": "生地", "안감": "裏地", "겉감": "表地", "소매": "袖",
        "밑단": "裾", "어깨": "肩", "가슴": "胸", "허리": "ウエスト",
        "지퍼": "ジッパー", "스토퍼": "ストッパー", "고리": "ループ", "테이프": "テープ",
        "앞판": "前身頃", "뒷판": "後身頃", "로고": "ロゴ",
        "벨크로": "ベルクロ", "밴드": "バンド", "아일렛": "アイレット", "스트링": "ストリング",
        "주머니": "ポケット", "포켓": "ポケット", "메인": "メイン", "라벨": "ラベル"
    }
}

# OCR 엔진 초기화 (싱글톤)
ocr_engine = None

def get_ocr_engine():
    global ocr_engine
    if ocr_engine is None:
        print("[init] PaddleOCR engine (korean)...")
        ocr_engine = PaddleOCR(use_textline_orientation=True, lang="korean")
        print("[init] PaddleOCR engine ready")
    return ocr_engine


def pdf_to_images(pdf_path, zoom=2.0):
    """PDF를 이미지로 변환"""
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img_path = os.path.join(UPLOAD_FOLDER, f"page_{page_num + 1}.png")
        pix.save(img_path)
        images.append(img_path)

    doc.close()
    return images


def get_ocr_results(image_path):
    """PaddleOCR로 텍스트와 위치 추출"""
    ocr = get_ocr_engine()
    result = ocr.predict(image_path)

    texts = []
    if result:
        for item in result:
            if isinstance(item, dict):
                rec_texts = item.get('rec_text', item.get('rec_texts', []))
                rec_scores = item.get('rec_score', item.get('rec_scores', []))
                dt_polys = item.get('dt_polys', [])

                if isinstance(rec_texts, str):
                    rec_texts = [rec_texts]
                    rec_scores = [rec_scores]
                    dt_polys = [dt_polys]

                for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                    text_str = str(text)
                    # 한글이 포함된 텍스트만 추출
                    if any('\uac00' <= c <= '\ud7a3' for c in text_str):
                        bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
                        texts.append({
                            "bbox": bbox,
                            "text": text_str,
                            "confidence": float(score) if score else 1.0
                        })

    return texts


def translate_with_dict(korean_text, target_lang):
    """사전 기반 번역"""
    result = korean_text
    if target_lang in GARMENT_DICT:
        for kor, trans in GARMENT_DICT[target_lang].items():
            result = result.replace(kor, trans)
    return result


def translate_with_vlm(image_path, texts, target_lang):
    """VLM으로 이미지 컨텍스트와 함께 번역"""
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    translations = []

    # 모든 한글 텍스트를 한 번에 번역 요청
    korean_list = [item["text"] for item in texts]
    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(korean_list)])

    prompt = f"""This is a garment/clothing technical specification image (tech pack).
Translate the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.
Keep translations SHORT and professional. Only respond with numbered translations in {lang_config['prompt_lang']}.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (same numbering, SHORT answers only):"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "qwen2.5vl:latest",
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()

            # 응답 파싱
            lines = response_text.split("\n")
            trans_dict = {}
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    parts = line.split(".", 1)
                    if len(parts) == 2:
                        idx = int(parts[0]) - 1
                        trans = parts[1].strip()
                        if idx < len(korean_list):
                            trans_dict[idx] = trans

            # 결과 매핑
            for i, item in enumerate(texts):
                if i in trans_dict:
                    translated = trans_dict[i]
                else:
                    translated = translate_with_dict(item["text"], target_lang)

                translations.append({
                    **item,
                    "translated": translated
                })
        else:
            # fallback: 사전 번역
            for item in texts:
                translated = translate_with_dict(item["text"], target_lang)
                translations.append({**item, "translated": translated})

    except Exception as e:
        print(f"VLM error: {e}")
        for item in texts:
            translated = translate_with_dict(item["text"], target_lang)
            translations.append({**item, "translated": translated})

    return translations


def replace_text_in_image(image_path, translations, output_path):
    """이미지에서 한글 영역을 지우고 번역된 텍스트로 교체"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 제목 영역 처리
    title_items = [item for item in translations if min(p[1] for p in item["bbox"]) < 25]
    if title_items:
        title_y_max = max(max(p[1] for p in item["bbox"]) for item in title_items) + 5
        cv2.rectangle(img, (0, 0), (width, int(title_y_max)), (255, 255, 255), -1)

    # 한글 영역을 배경색으로 덮기
    for item in translations:
        bbox = item["bbox"]
        pts = np.array(bbox, dtype=np.int32)

        x_min = max(0, int(min(p[0] for p in bbox)) - 5)
        y_min = max(0, int(min(p[1] for p in bbox)) - 5)
        x_max = min(width, int(max(p[0] for p in bbox)) + 5)
        y_max = min(height, int(max(p[1] for p in bbox)) + 5)

        border_pixels = []
        for x in range(x_min, x_max):
            if y_min > 0:
                border_pixels.append(img[y_min-1, x])
            if y_max < height:
                border_pixels.append(img[min(y_max, height-1), x])

        if border_pixels:
            bg_color = np.mean(border_pixels, axis=0).astype(np.uint8)
        else:
            bg_color = np.array([255, 255, 255], dtype=np.uint8)

        # 확장된 영역 채우기
        expanded_pts = pts.copy().astype(np.float64)
        center = np.mean(pts, axis=0)
        for i in range(len(expanded_pts)):
            direction = expanded_pts[i] - center
            expanded_pts[i] = expanded_pts[i] + direction * 0.35

        cv2.fillPoly(img, [expanded_pts.astype(np.int32)], bg_color.tolist())

        x1 = max(0, int(min(p[0] for p in bbox)) - 5)
        y1 = max(0, int(min(p[1] for p in bbox)) - 3)
        x2 = min(width, int(max(p[0] for p in bbox)) + 5)
        y2 = min(height, int(max(p[1] for p in bbox)) + 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color.tolist(), -1)
        cv2.rectangle(img, (x1-2, y1-2), (x2+2, y2+2), bg_color.tolist(), -1)

    # PIL로 변환하여 텍스트 삽입
    img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    font_sizes = [11, 10, 9, 8, 7]

    for item in translations:
        bbox = item["bbox"]
        translated_text = item["translated"]

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        box_width = max(xs) - min(xs)

        x = int(min(xs))
        y = int(min(ys))

        font = None
        text_width = 0
        for size in font_sizes:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except:
                font = ImageFont.load_default()
                break

            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]

            if text_width <= box_width * 1.5:
                break

        if text_width > box_width * 2:
            words = translated_text.split()
            if len(words) > 3:
                translated_text = " ".join(words[:3]) + "..."

        draw.text((x, y), translated_text, fill=(0, 0, 0), font=font)

    img_result.save(output_path)
    return output_path


# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Translator - 의류 기술서 번역</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-area:hover {
            background: #f0f4ff;
            border-color: #764ba2;
        }
        .upload-area.dragover {
            background: #e8edff;
            border-color: #764ba2;
        }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .language-select {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .lang-btn {
            padding: 12px 25px;
            border: 2px solid #667eea;
            border-radius: 25px;
            background: white;
            color: #667eea;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1em;
        }
        .lang-btn:hover, .lang-btn.active {
            background: #667eea;
            color: white;
        }
        .status {
            text-align: center;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            display: none;
        }
        .status.processing {
            display: block;
            background: #fff3cd;
            color: #856404;
        }
        .status.success {
            display: block;
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            display: block;
            background: #f8d7da;
            color: #721c24;
        }
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .result-item {
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.2s;
        }
        .result-item:hover {
            transform: scale(1.02);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .result-item img {
            width: 100%;
            display: block;
        }
        .result-item .download {
            display: block;
            text-align: center;
            padding: 10px;
            background: #667eea;
            color: white;
            text-decoration: none;
        }
        .file-info {
            text-align: center;
            color: #666;
            margin: 10px 0;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 PDF Translator</h1>
        <p class="subtitle">의류 기술서 (Tech Pack) 한글 → 다국어 번역</p>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" id="dropZone">
                <p style="font-size: 3em; margin-bottom: 10px;">📁</p>
                <p style="font-size: 1.2em; margin-bottom: 10px;">PDF 또는 이미지 파일을 드래그하거나 클릭하세요</p>
                <p style="color: #999;">지원 형식: PDF, PNG, JPG</p>
                <input type="file" id="fileInput" name="file" accept=".pdf,.png,.jpg,.jpeg">
            </div>

            <div class="file-info" id="fileInfo"></div>

            <p style="text-align: center; margin: 20px 0; font-weight: bold;">번역 언어 선택:</p>
            <div class="language-select">
                <button type="button" class="lang-btn active" data-lang="english">🇺🇸 English</button>
                <button type="button" class="lang-btn" data-lang="vietnamese">🇻🇳 Tiếng Việt</button>
                <button type="button" class="lang-btn" data-lang="chinese">🇨🇳 中文</button>
                <button type="button" class="lang-btn" data-lang="japanese">🇯🇵 日本語</button>
            </div>
            <input type="hidden" name="target_lang" id="targetLang" value="english">

            <div style="text-align: center; margin-top: 30px;">
                <button type="submit" class="btn" id="translateBtn" disabled>
                    🚀 번역 시작
                </button>
            </div>
        </form>

        <div class="status" id="status"></div>

        <div class="results" id="results"></div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const translateBtn = document.getElementById('translateBtn');
        const langBtns = document.querySelectorAll('.lang-btn');
        const targetLang = document.getElementById('targetLang');
        const status = document.getElementById('status');
        const results = document.getElementById('results');

        // 드래그 앤 드롭
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileInfo();
            }
        });

        fileInput.addEventListener('change', updateFileInfo);

        function updateFileInfo() {
            if (fileInput.files.length) {
                const file = fileInput.files[0];
                fileInfo.textContent = `선택된 파일: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                translateBtn.disabled = false;
            } else {
                fileInfo.textContent = '';
                translateBtn.disabled = true;
            }
        }

        // 언어 선택
        langBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                langBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                targetLang.value = btn.dataset.lang;
            });
        });

        // 폼 제출
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('target_lang', targetLang.value);

            translateBtn.disabled = true;
            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span>번역 중... (VLM 처리로 1-2분 소요될 수 있습니다)';
            results.innerHTML = '';

            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    status.className = 'status success';
                    status.textContent = `✅ 번역 완료! ${data.files.length}개 페이지 처리됨`;

                    results.innerHTML = data.files.map(file => `
                        <div class="result-item">
                            <img src="/output/${file}" alt="${file}">
                            <a href="/download/${file}" class="download">📥 다운로드</a>
                        </div>
                    `).join('');
                } else {
                    status.className = 'status error';
                    status.textContent = `❌ 오류: ${data.error}`;
                }
            } catch (err) {
                status.className = 'status error';
                status.textContent = `❌ 오류: ${err.message}`;
            }

            translateBtn.disabled = false;
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/translate', methods=['POST'])
def translate():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "파일이 없습니다"})

        file = request.files['file']
        target_lang = request.form.get('target_lang', 'english')

        if file.filename == '':
            return jsonify({"success": False, "error": "파일이 선택되지 않았습니다"})

        # 파일 저장
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # PDF인 경우 이미지로 변환
        if filename.lower().endswith('.pdf'):
            image_paths = pdf_to_images(filepath)
        else:
            image_paths = [filepath]

        # 각 이미지 처리
        output_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] Processing: {img_path}")

            # OCR
            texts = get_ocr_results(img_path)
            print(f"  Found {len(texts)} Korean texts")

            if texts:
                # 번역
                translations = translate_with_vlm(img_path, texts, target_lang)

                # 이미지 교체
                output_filename = f"translated_{timestamp}_page{i+1}_{target_lang}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                replace_text_in_image(img_path, translations, output_path)
                output_files.append(output_filename)
            else:
                print(f"  No Korean text found, skipping...")

        return jsonify({"success": True, "files": output_files})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/output/<filename>')
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(OUTPUT_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    print("=" * 60)
    print("PDF Translator - 의류 기술서 번역 앱")
    print("=" * 60)
    print("Engine: PaddleOCR + VLM (qwen2.5vl)")
    print("Languages: English, Vietnamese, Chinese, Japanese")
    print("Port: 6008")
    print("=" * 60)

    # OCR 엔진 미리 로드
    get_ocr_engine()

    app.run(host='0.0.0.0', port=6008, debug=True)
