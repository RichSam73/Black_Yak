# -*- coding: utf-8 -*-
"""
PDF Translator - 한글 텍스트를 다국어로 번역하는 웹앱
- Flask 기반 웹 인터페이스
- PaddleOCR + VLM (qwen2.5vl) 사용
- 지원 언어: 영어, 베트남어, 중국어, 인도네시아어, 벵골어
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
    "indonesian": {
        "name": "Bahasa Indonesia",
        "code": "id",
        "prompt_lang": "Indonesian"
    },
    "bengali": {
        "name": "বাংলা",
        "code": "bn",
        "prompt_lang": "Bengali"
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
    "indonesian": {
        "남성": "Pria", "여성": "Wanita", "자켓": "Jaket", "다운자켓": "Jaket Bulu",
        "후드": "Tudung", "에리": "Kerah", "봉제": "Jahit", "작업": "Kerja",
        "원단": "Kain", "안감": "Lapisan Dalam", "겉감": "Lapisan Luar", "소매": "Lengan",
        "밑단": "Keliman", "어깨": "Bahu", "가슴": "Dada", "허리": "Pinggang",
        "지퍼": "Ritsleting", "스토퍼": "Penghenti", "고리": "Lingkaran", "테이프": "Pita",
        "앞판": "Panel Depan", "뒷판": "Panel Belakang", "로고": "Logo",
        "벨크로": "Velcro", "밴드": "Pita Elastis", "아일렛": "Lubang Tali", "스트링": "Tali Serut",
        "주머니": "Saku", "포켓": "Saku", "메인": "Utama", "라벨": "Label"
    },
    "bengali": {
        "남성": "পুরুষ", "여성": "মহিলা", "자켓": "জ্যাকেট", "다운자켓": "ডাউন জ্যাকেট",
        "후드": "হুড", "에리": "কলার", "봉제": "সেলাই", "작업": "কাজ",
        "원단": "কাপড়", "안감": "আস্তরণ", "겉감": "বাইরের স্তর", "소매": "হাতা",
        "밑단": "নিচের প্রান্ত", "어깨": "কাঁধ", "가슴": "বুক", "허리": "কোমর",
        "지퍼": "জিপার", "스토퍼": "স্টপার", "고리": "লুপ", "테이프": "টেপ",
        "앞판": "সামনের প্যানেল", "뒷판": "পিছনের প্যানেল", "로고": "লোগো",
        "벨크로": "ভেলক্রো", "밴드": "ব্যান্ড", "아일렛": "আইলেট", "스트링": "স্ট্রিং",
        "주머니": "পকেট", "포켓": "পকেট", "메인": "প্রধান", "라벨": "লেবেল"
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
            padding: 5px;
        }
        .container {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 6px;
            padding: 5px 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            min-height: calc(100vh - 10px);
        }
        .header-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 8px;
            flex-wrap: nowrap;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }
        .header-row h1 {
            color: #333;
            font-size: 1.2em;
            margin: 0;
            white-space: nowrap;
        }
        .header-row .subtitle {
            color: #666;
            font-size: 0.7em;
            margin: 0;
            white-space: nowrap;
        }
        .lang-btn {
            padding: 3px 6px;
            border: 2px solid #667eea;
            border-radius: 10px;
            background: white;
            color: #667eea;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.65em;
            white-space: nowrap;
        }
        .lang-btn:hover, .lang-btn.active {
            background: #667eea;
            color: white;
        }
        .file-select-btn {
            padding: 3px 8px;
            border: 2px solid #28a745;
            border-radius: 10px;
            background: white;
            color: #28a745;
            cursor: pointer;
            font-size: 0.65em;
            white-space: nowrap;
            transition: all 0.3s;
        }
        .file-select-btn:hover {
            background: #28a745;
            color: white;
        }
        .file-select-btn.has-file {
            background: #28a745;
            color: white;
        }
        .translate-btn {
            padding: 3px 10px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            font-size: 0.65em;
            white-space: nowrap;
            transition: all 0.3s;
        }
        .translate-btn:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .translate-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        input[type="file"] { display: none; }

        /* 에디터 레이아웃 */
        .editor-container {
            display: none;
            height: calc(100vh - 70px);
            position: relative;
        }
        .editor-container.active {
            display: flex;
        }

        /* 좌측: 페이지 프리뷰 */
        .preview-panel {
            flex: 1;
            min-width: 200px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* 리사이저 (드래그 핸들) */
        .resizer {
            width: 8px;
            background: #e0e0e0;
            cursor: col-resize;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
            border-radius: 4px;
            margin: 0 3px;
        }
        .resizer:hover {
            background: #667eea;
        }
        .resizer:active {
            background: #764ba2;
        }
        .resizer::after {
            content: '⋮';
            color: #999;
            font-size: 14px;
        }
        .resizer:hover::after {
            color: white;
        }
        .preview-header {
            background: #f8f9fa;
            padding: 8px 12px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .preview-header .page-info {
            font-weight: bold;
            color: #333;
        }
        .preview-nav {
            display: flex;
            gap: 5px;
        }
        .preview-nav button {
            padding: 4px 10px;
            border: 1px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8em;
        }
        .preview-nav button:hover {
            background: #667eea;
            color: white;
        }
        .preview-nav button:disabled {
            border-color: #ccc;
            color: #ccc;
            cursor: not-allowed;
            background: white;
        }
        .preview-image {
            flex: 1;
            overflow: auto;
            padding: 10px;
            background: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }
        .preview-image img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        /* 우측: 번역 테이블 */
        .translation-panel {
            width: 450px;
            min-width: 300px;
            flex-shrink: 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .translation-header {
            background: #f8f9fa;
            padding: 8px 12px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .translation-header .title {
            font-weight: bold;
            color: #333;
        }
        .confirm-btn {
            padding: 5px 15px;
            border: none;
            border-radius: 5px;
            background: #28a745;
            color: white;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.3s;
        }
        .confirm-btn:hover {
            background: #218838;
        }
        .confirm-btn.confirmed {
            background: #6c757d;
        }
        .translation-table-wrapper {
            flex: 1;
            overflow: auto;
            padding: 10px;
        }
        .translation-table {
            width: 100%;
            border-collapse: collapse;
        }
        .translation-table th {
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        .translation-table th:first-child {
            width: 30px;
        }
        .translation-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }
        .translation-table tr:hover {
            background: #f8f9fa;
        }
        .translation-table .idx {
            color: #999;
            font-size: 0.85em;
            text-align: center;
        }
        .translation-table .korean {
            color: #333;
            font-size: 0.9em;
        }
        .translation-table .trans-input {
            width: 100%;
            padding: 6px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9em;
            transition: border-color 0.3s;
        }
        .translation-table .trans-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        .translation-table .trans-input.modified {
            border-color: #ffc107;
            background: #fffde7;
        }

        /* 상태 메시지 */
        .status {
            text-align: center;
            padding: 8px;
            margin: 8px 0;
            border-radius: 6px;
            display: none;
            font-size: 0.9em;
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

        /* 결과 그리드 */
        .results {
            display: none;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }
        .results.active {
            display: grid;
        }
        .result-item {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
        }
        .result-item:hover {
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .result-item img {
            width: 100%;
            display: block;
        }
        .result-item .download {
            display: block;
            text-align: center;
            padding: 8px;
            background: #667eea;
            color: white;
            text-decoration: none;
            font-size: 0.85em;
        }

        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 초기 안내 */
        .initial-guide {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .initial-guide .icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        .initial-guide h2 {
            color: #333;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg">
        <input type="hidden" id="targetLang" value="english">

        <div class="header-row">
            <h1>📄 PDF Translator</h1>
            <span class="subtitle">한글→다국어</span>
            <button type="button" class="lang-btn active" data-lang="english">🇺🇸EN</button>
            <button type="button" class="lang-btn" data-lang="vietnamese">🇻🇳VI</button>
            <button type="button" class="lang-btn" data-lang="chinese">🇨🇳中</button>
            <button type="button" class="lang-btn" data-lang="indonesian">🇮🇩ID</button>
            <button type="button" class="lang-btn" data-lang="bengali">🇧🇩BN</button>
            <button type="button" class="file-select-btn" id="fileSelectBtn">📁 파일선택</button>
            <button type="button" class="translate-btn" id="translateBtn" disabled>🚀 번역</button>
        </div>

        <div class="status" id="status"></div>

        <!-- 초기 안내 화면 -->
        <div class="initial-guide" id="initialGuide">
            <div class="icon">📂</div>
            <h2>파일을 선택하세요</h2>
            <p>PDF 또는 이미지 파일을 선택하면 자동으로 OCR이 실행됩니다</p>
        </div>

        <!-- 에디터 컨테이너 -->
        <div class="editor-container" id="editorContainer">
            <!-- 좌측: 페이지 프리뷰 -->
            <div class="preview-panel" id="previewPanel">
                <div class="preview-header">
                    <span class="page-info" id="pageInfo">페이지 1 / 1</span>
                    <div class="preview-nav">
                        <button id="prevPageBtn" disabled>◀ 이전</button>
                        <button id="nextPageBtn" disabled>다음 ▶</button>
                    </div>
                </div>
                <div class="preview-image">
                    <img id="previewImg" src="" alt="페이지 프리뷰">
                </div>
            </div>

            <!-- 리사이저 핸들 -->
            <div class="resizer" id="resizer"></div>

            <!-- 우측: 번역 테이블 -->
            <div class="translation-panel" id="translationPanel">
                <div class="translation-header">
                    <span class="title">번역 편집</span>
                    <button class="confirm-btn" id="confirmBtn">✅ 확정</button>
                </div>
                <div class="translation-table-wrapper">
                    <table class="translation-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>한국어 (원본)</th>
                                <th>번역</th>
                            </tr>
                        </thead>
                        <tbody id="translationBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- 결과 표시 -->
        <div class="results" id="results"></div>
    </div>

    <script>
        // 상태 변수
        let currentPage = 0;
        let totalPages = 0;
        let pagesData = [];  // [{image: base64, texts: [...], translations: [...], confirmed: bool}]

        // DOM 요소
        const fileInput = document.getElementById('fileInput');
        const fileSelectBtn = document.getElementById('fileSelectBtn');
        const translateBtn = document.getElementById('translateBtn');
        const langBtns = document.querySelectorAll('.lang-btn');
        const targetLang = document.getElementById('targetLang');
        const status = document.getElementById('status');
        const initialGuide = document.getElementById('initialGuide');
        const editorContainer = document.getElementById('editorContainer');
        const previewImg = document.getElementById('previewImg');
        const pageInfo = document.getElementById('pageInfo');
        const prevPageBtn = document.getElementById('prevPageBtn');
        const nextPageBtn = document.getElementById('nextPageBtn');
        const translationBody = document.getElementById('translationBody');
        const confirmBtn = document.getElementById('confirmBtn');
        const results = document.getElementById('results');

        // 파일 선택 버튼
        fileSelectBtn.addEventListener('click', () => fileInput.click());

        // 파일 선택 시 자동 OCR
        fileInput.addEventListener('change', async () => {
            if (!fileInput.files.length) return;

            const file = fileInput.files[0];
            fileSelectBtn.classList.add('has-file');
            fileSelectBtn.textContent = '✅ ' + file.name.substring(0, 8) + (file.name.length > 8 ? '...' : '');

            // OCR 실행
            await loadAndProcessFile(file);
        });

        // 언어 선택
        langBtns.forEach(btn => {
            btn.addEventListener('click', async () => {
                langBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                targetLang.value = btn.dataset.lang;

                // 이미 파일이 로드되어 있으면 재번역
                if (pagesData.length > 0) {
                    await retranslateCurrentPage();
                }
            });
        });

        // 파일 로드 및 OCR 처리
        async function loadAndProcessFile(file) {
            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span>파일 분석 중... OCR 처리 중입니다';
            initialGuide.style.display = 'none';
            editorContainer.classList.remove('active');
            results.classList.remove('active');

            const formData = new FormData();
            formData.append('file', file);
            formData.append('target_lang', targetLang.value);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    pagesData = data.pages;
                    totalPages = pagesData.length;
                    currentPage = 0;

                    status.className = 'status success';
                    status.textContent = `✅ ${totalPages}페이지 분석 완료! 번역을 편집하세요.`;

                    // 에디터 표시
                    editorContainer.classList.add('active');
                    translateBtn.disabled = false;

                    // 첫 페이지 표시
                    showPage(0);
                } else {
                    status.className = 'status error';
                    status.textContent = `❌ 오류: ${data.error}`;
                    initialGuide.style.display = 'block';
                }
            } catch (err) {
                status.className = 'status error';
                status.textContent = `❌ 오류: ${err.message}`;
                initialGuide.style.display = 'block';
            }
        }

        // 페이지 표시
        function showPage(pageIdx) {
            if (pageIdx < 0 || pageIdx >= totalPages) return;

            currentPage = pageIdx;
            const page = pagesData[pageIdx];

            // 이미지 표시
            previewImg.src = 'data:image/png;base64,' + page.image;

            // 페이지 정보
            pageInfo.textContent = `페이지 ${pageIdx + 1} / ${totalPages}`;

            // 네비게이션 버튼
            prevPageBtn.disabled = pageIdx === 0;
            nextPageBtn.disabled = pageIdx === totalPages - 1;

            // 번역 테이블 갱신
            updateTranslationTable(page);

            // 확정 버튼 상태
            if (page.confirmed) {
                confirmBtn.textContent = '✔ 확정됨';
                confirmBtn.classList.add('confirmed');
            } else {
                confirmBtn.textContent = '✅ 확정';
                confirmBtn.classList.remove('confirmed');
            }
        }

        // 번역 테이블 갱신
        function updateTranslationTable(page) {
            translationBody.innerHTML = '';

            if (!page.translations || page.translations.length === 0) {
                translationBody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#999;padding:30px;">한글 텍스트가 없습니다</td></tr>';
                return;
            }

            page.translations.forEach((item, idx) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="idx">${idx + 1}</td>
                    <td class="korean">${escapeHtml(item.text)}</td>
                    <td>
                        <input type="text" class="trans-input"
                               data-idx="${idx}"
                               data-original="${escapeHtml(item.translated)}"
                               value="${escapeHtml(item.translated)}">
                    </td>
                `;
                translationBody.appendChild(tr);
            });

            // 입력 필드 이벤트
            translationBody.querySelectorAll('.trans-input').forEach(input => {
                input.addEventListener('input', (e) => {
                    const original = e.target.dataset.original;
                    if (e.target.value !== original) {
                        e.target.classList.add('modified');
                    } else {
                        e.target.classList.remove('modified');
                    }
                    // 수정 시 확정 해제
                    pagesData[currentPage].confirmed = false;
                    confirmBtn.textContent = '✅ 확정';
                    confirmBtn.classList.remove('confirmed');
                });
            });
        }

        // HTML 이스케이프
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // 페이지 네비게이션
        prevPageBtn.addEventListener('click', () => {
            saveCurrentTranslations();
            showPage(currentPage - 1);
        });

        nextPageBtn.addEventListener('click', () => {
            saveCurrentTranslations();
            showPage(currentPage + 1);
        });

        // 현재 페이지 번역 저장
        function saveCurrentTranslations() {
            const inputs = translationBody.querySelectorAll('.trans-input');
            inputs.forEach(input => {
                const idx = parseInt(input.dataset.idx);
                if (pagesData[currentPage].translations[idx]) {
                    pagesData[currentPage].translations[idx].translated = input.value;
                }
            });
        }

        // 확정 버튼
        confirmBtn.addEventListener('click', () => {
            saveCurrentTranslations();
            pagesData[currentPage].confirmed = true;
            confirmBtn.textContent = '✔ 확정됨';
            confirmBtn.classList.add('confirmed');

            // 모든 입력 필드 modified 클래스 제거
            translationBody.querySelectorAll('.trans-input').forEach(input => {
                input.classList.remove('modified');
                input.dataset.original = input.value;
            });

            status.className = 'status success';
            status.textContent = `✅ 페이지 ${currentPage + 1} 번역 확정됨`;
        });

        // 현재 페이지 재번역
        async function retranslateCurrentPage() {
            if (pagesData.length === 0) return;

            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span>언어 변경 중... 재번역 중입니다';

            try {
                const response = await fetch('/retranslate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        page_idx: currentPage,
                        target_lang: targetLang.value,
                        texts: pagesData[currentPage].translations.map(t => ({
                            text: t.text,
                            bbox: t.bbox
                        }))
                    })
                });

                const data = await response.json();
                if (data.success) {
                    pagesData[currentPage].translations = data.translations;
                    pagesData[currentPage].confirmed = false;
                    showPage(currentPage);
                    status.className = 'status success';
                    status.textContent = `✅ 재번역 완료`;
                }
            } catch (err) {
                status.className = 'status error';
                status.textContent = `❌ 재번역 오류: ${err.message}`;
            }
        }

        // 리사이저 드래그 기능
        const resizer = document.getElementById('resizer');
        const previewPanel = document.getElementById('previewPanel');
        const translationPanel = document.getElementById('translationPanel');

        let isResizing = false;

        resizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            const containerRect = editorContainer.getBoundingClientRect();
            const containerWidth = containerRect.width;
            const mouseX = e.clientX - containerRect.left;

            // 번역 패널 너비 = 컨테이너 오른쪽 끝에서 마우스 위치까지
            const newTranslationWidth = containerWidth - mouseX - 8; // 8px for resizer

            // 최소/최대 제한
            if (newTranslationWidth >= 300 && newTranslationWidth <= containerWidth - 300) {
                translationPanel.style.width = newTranslationWidth + 'px';
            }
        });

        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });

        // 최종 번역 버튼
        translateBtn.addEventListener('click', async () => {
            // 현재 페이지 저장
            saveCurrentTranslations();

            // 미확정 페이지 확인
            const unconfirmed = pagesData.filter(p => !p.confirmed).length;
            if (unconfirmed > 0) {
                if (!confirm(`${unconfirmed}개 페이지가 미확정입니다. 계속 진행하시겠습니까?`)) {
                    return;
                }
            }

            translateBtn.disabled = true;
            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span>최종 번역본 생성 중...';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        pages: pagesData,
                        target_lang: targetLang.value
                    })
                });

                const data = await response.json();

                if (data.success) {
                    status.className = 'status success';
                    status.textContent = `✅ 번역 완료! ${data.files.length}개 파일 생성됨`;

                    // 결과 표시
                    editorContainer.classList.remove('active');
                    results.classList.add('active');
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


# 임시 저장소: 세션별 이미지 경로
temp_image_paths = {}


@app.route('/analyze', methods=['POST'])
def analyze():
    """파일 업로드 + OCR + 초기 번역"""
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

        # 세션 ID 생성
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_paths[session_id] = image_paths

        # 각 페이지 분석
        pages = []
        for i, img_path in enumerate(image_paths):
            print(f"[Analyze {i+1}/{len(image_paths)}] {img_path}")

            # 이미지를 base64로 인코딩
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()

            # OCR
            texts = get_ocr_results(img_path)
            print(f"  Found {len(texts)} Korean texts")

            # 번역
            translations = []
            if texts:
                translations = translate_with_vlm(img_path, texts, target_lang)

            pages.append({
                "image": image_base64,
                "image_path": img_path,
                "translations": translations,
                "confirmed": False
            })

        return jsonify({
            "success": True,
            "session_id": session_id,
            "pages": pages
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/retranslate', methods=['POST'])
def retranslate():
    """언어 변경 시 재번역"""
    try:
        data = request.get_json()
        target_lang = data.get('target_lang', 'english')
        texts = data.get('texts', [])

        # 사전 기반 번역 (VLM 없이 빠르게)
        translations = []
        for item in texts:
            translated = translate_with_dict(item['text'], target_lang)
            translations.append({
                "text": item['text'],
                "bbox": item['bbox'],
                "translated": translated
            })

        return jsonify({
            "success": True,
            "translations": translations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/generate', methods=['POST'])
def generate():
    """최종 번역 이미지 생성"""
    try:
        data = request.get_json()
        pages = data.get('pages', [])
        target_lang = data.get('target_lang', 'english')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []

        for i, page in enumerate(pages):
            print(f"[Generate {i+1}/{len(pages)}]")

            # base64 이미지를 임시 파일로 저장
            image_data = base64.b64decode(page['image'])
            temp_img_path = os.path.join(UPLOAD_FOLDER, f"temp_gen_{timestamp}_{i}.png")
            with open(temp_img_path, 'wb') as f:
                f.write(image_data)

            translations = page.get('translations', [])

            if translations:
                # 이미지에 번역 적용
                output_filename = f"translated_{timestamp}_page{i+1}_{target_lang}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                replace_text_in_image(temp_img_path, translations, output_path)
                output_files.append(output_filename)
            else:
                print(f"  No translations, skipping...")

            # 임시 파일 삭제
            try:
                os.remove(temp_img_path)
            except:
                pass

        return jsonify({
            "success": True,
            "files": output_files
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/translate', methods=['POST'])
def translate():
    """기존 호환용: 바로 번역"""
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
    print("Languages: English, Vietnamese, Chinese, Indonesian, Bengali")
    print("Port: 6008")
    print("=" * 60)

    # OCR 엔진 미리 로드
    get_ocr_engine()

    app.run(host='0.0.0.0', port=6008, debug=True)
