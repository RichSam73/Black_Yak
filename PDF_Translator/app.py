# -*- coding: utf-8 -*-
"""
PDF Translator - 한글 텍스트를 다국어로 번역하는 웹앱
- Flask 기반 웹 인터페이스
- PaddleOCR + VLM (qwen2.5vl) 사용
- 지원 언어: 영어, 베트남어, 중국어, 인도네시아어, 벵골어
"""

# 버전 정보
VERSION = "1.6.0"
VERSION_DATE = "2026-01-09"
VERSION_NOTES = """
v1.6.0 (2026-01-09)
- ★ Gemini 배치 번역: 모든 페이지 텍스트를 1회 API 호출로 번역 (Free Tier 최적화)
- AI 모델 선택: Gemini 2.0 Flash, GPT-4o, GPT-4o-mini 지원
- API 키 입력 필드 추가

v1.5.0 (2026-01-08)
- ★ 세로 텍스트 지원: 높이>너비×2 → 글자를 세로로 배치
- ★ 진행 상황 표시: OCR/번역 단계별 실시간 진행률 + 경과시간

v1.4.4 (2026-01-08)
- 선 보존: 마진 최소화 (15% → 1px)로 테이블 선 침범 방지

v1.4.2 (2026-01-08)
- 텍스트 완전 삭제: Inpainting 대신 배경색으로 직접 덮어쓰기
- 어두운 배경 지원: 배경 밝기 감지 → 자동으로 흰색/검정 텍스트 선택

v1.3.0 (2026-01-08)
- 배경색 샘플링 방식 적용: bbox 주변 가장자리에서 배경색 감지
- 글자 높이에 비례한 동적 마진 (최소 5px, 높이의 20%)
- 인페인팅 대신 배경색으로 자연스럽게 채우기

v1.2.1 (2026-01-07)
- 텍스트 지우기 단순화: 흰색으로 확실하게 덮어쓰기 (인페인팅 제거)
- 마진 확대: 글자 높이의 15-20%로 충분히 덮음
- 안정성 향상: 복잡한 인페인팅 대신 단순한 방식 채택

v1.2.0 (2026-01-07)
- 인페인팅 기술 시도 (문제 발생으로 롤백)

v1.1.0 (2026-01-06)
- 텍스트 지우기 개선: 글자에서 떨어진 영역에서 배경색 샘플링
- 마진 확장: 글자 높이에 비례한 동적 마진으로 완전히 지움
- 배경색 감지 개선: 5-10픽셀 떨어진 곳에서 샘플링하여 글자 색상 혼입 방지

v1.0.0 (2026-01-06)
- 미리보기 기능 추가: 번역 결과를 내보내기 전 미리보기 가능
- 텍스트 영역 지우기: 한국어 텍스트를 배경색으로 지우고 번역 텍스트 삽입
- 배경색 자동 감지: 테두리 픽셀 샘플링으로 흰색 계열 우선 감지
- 미리보기 캐시: 페이지별 캐시로 성능 최적화
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
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# AI 모델 설정
AI_MODELS = {
    "ollama": {
        "models": ["qwen2.5vl:latest", "llava:latest", "bakllava:latest"],
        "default": "qwen2.5vl:latest"
    },
    "claude": {
        "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "default": "claude-sonnet-4-20250514"
    },
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "default": "gpt-4o"
    },
    "gemini": {
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "default": "gemini-2.0-flash"
    }
}
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
        # 전처리 비활성화: bbox 좌표가 원본 이미지와 정확히 일치하도록 함
        ocr_engine = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,  # 문서 방향 분류 끄기
            use_doc_unwarping=False,             # 문서 왜곡 보정 끄기
            use_textline_orientation=False       # 텍스트라인 방향 분류 끄기
        )
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

    # ★ 핵심 수정: 이미지를 RGB numpy 배열로 변환하여 전달
    # PaddleOCR은 RGB 형식을 기대하므로, 파일 경로 대신 RGB 배열 전달
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = ocr.predict(img_rgb)

    texts = []
    if result:
        for item in result:
            rec_texts = []
            rec_scores = []
            dt_polys = []

            # OCRResult 객체 처리 (새 PaddleOCR API)
            if hasattr(item, 'rec_texts'):
                rec_texts = item.rec_texts or []
                rec_scores = item.rec_scores or []
                # dt_polys 사용 (원본 detection 좌표 - 더 정확함)
                dt_polys = item.dt_polys if hasattr(item, 'dt_polys') and item.dt_polys is not None else []
            elif isinstance(item, dict):
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


def translate_with_claude(image_path, texts, target_lang, api_key, model=None):
    """Claude API로 이미지 컨텍스트와 함께 번역"""
    if model is None:
        model = AI_MODELS["claude"]["default"]
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
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            CLAUDE_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("content", [{}])[0].get("text", "").strip()

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
            print(f"Claude API error: {response.status_code} - {response.text}")
            # fallback: 사전 번역
            for item in texts:
                translated = translate_with_dict(item["text"], target_lang)
                translations.append({**item, "translated": translated})

    except Exception as e:
        print(f"Claude API error: {e}")
        for item in texts:
            translated = translate_with_dict(item["text"], target_lang)
            translations.append({**item, "translated": translated})

    return translations


def translate_with_openai(image_path, texts, target_lang, api_key, model=None):
    """OpenAI GPT-4 Vision API로 이미지 컨텍스트와 함께 번역"""
    if model is None:
        model = AI_MODELS["openai"]["default"]
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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

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
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            # fallback: 사전 번역
            for item in texts:
                translated = translate_with_dict(item["text"], target_lang)
                translations.append({**item, "translated": translated})

    except Exception as e:
        print(f"OpenAI API error: {e}")
        for item in texts:
            translated = translate_with_dict(item["text"], target_lang)
            translations.append({**item, "translated": translated})

    return translations


def translate_batch_with_gemini(all_pages_texts, target_lang, api_key, model=None):
    """Google Gemini API로 모든 페이지의 텍스트를 한 번에 번역 (배치 모드)

    Args:
        all_pages_texts: [{page_idx: int, texts: [{text, bbox}, ...]}, ...]
        target_lang: 번역 대상 언어
        api_key: Gemini API 키
        model: Gemini 모델명

    Returns:
        {page_idx: [translated_texts], ...}
    """
    if model is None:
        model = AI_MODELS["gemini"]["default"]
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    # 모든 페이지의 텍스트를 하나의 리스트로 합침 (페이지 구분 포함)
    all_korean = []
    page_text_counts = []  # 각 페이지별 텍스트 개수

    for page_data in all_pages_texts:
        page_texts = page_data["texts"]
        page_text_counts.append(len(page_texts))
        for item in page_texts:
            all_korean.append(item["text"])

    if not all_korean:
        return {page_data["page_idx"]: [] for page_data in all_pages_texts}

    # 전체 텍스트를 번호로 조인
    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(all_korean)])

    prompt = f"""This is a garment/clothing technical specification document (tech pack).
Translate ALL the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.
Keep translations SHORT and professional. Only respond with numbered translations in {lang_config['prompt_lang']}.
There are {len(all_korean)} items total from multiple pages. Translate ALL of them.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (same numbering 1-{len(all_korean)}, SHORT answers only):"""

    try:
        url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 8192}
        }

        print(f"[Batch Translation] Sending {len(all_korean)} texts to Gemini...")

        response = requests.post(url, headers=headers, json=payload, timeout=180)

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

            # 응답 파싱
            lines = response_text.split("\n")
            trans_dict = {}
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    parts = line.split(".", 1)
                    if len(parts) == 2:
                        try:
                            idx = int(parts[0]) - 1
                            trans = parts[1].strip()
                            if 0 <= idx < len(all_korean):
                                trans_dict[idx] = trans
                        except ValueError:
                            continue

            print(f"[Batch Translation] Got {len(trans_dict)}/{len(all_korean)} translations")

            # 페이지별로 결과 분배
            result_by_page = {}
            current_idx = 0

            for page_data in all_pages_texts:
                page_idx = page_data["page_idx"]
                page_texts = page_data["texts"]
                page_translations = []

                for item in page_texts:
                    if current_idx in trans_dict:
                        translated = trans_dict[current_idx]
                    else:
                        translated = translate_with_dict(item["text"], target_lang)

                    page_translations.append({
                        **item,
                        "translated": translated
                    })
                    current_idx += 1

                result_by_page[page_idx] = page_translations

            return result_by_page
        else:
            print(f"Gemini Batch API error: {response.status_code} - {response.text}")
            # fallback: 사전 번역
            return _fallback_batch_translation(all_pages_texts, target_lang)

    except Exception as e:
        print(f"Gemini Batch API error: {e}")
        return _fallback_batch_translation(all_pages_texts, target_lang)


def _fallback_batch_translation(all_pages_texts, target_lang):
    """배치 번역 실패 시 사전 번역으로 fallback"""
    result_by_page = {}
    for page_data in all_pages_texts:
        page_idx = page_data["page_idx"]
        page_translations = []
        for item in page_data["texts"]:
            translated = translate_with_dict(item["text"], target_lang)
            page_translations.append({**item, "translated": translated})
        result_by_page[page_idx] = page_translations
    return result_by_page


def translate_with_gemini(image_path, texts, target_lang, api_key, model=None):
    """Google Gemini API로 이미지 컨텍스트와 함께 번역"""
    if model is None:
        model = AI_MODELS["gemini"]["default"]
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
        # Gemini API URL에 모델명과 API 키 추가
        url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 4096
            }
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

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
            print(f"Gemini API error: {response.status_code} - {response.text}")
            # fallback: 사전 번역
            for item in texts:
                translated = translate_with_dict(item["text"], target_lang)
                translations.append({**item, "translated": translated})

    except Exception as e:
        print(f"Gemini API error: {e}")
        for item in texts:
            translated = translate_with_dict(item["text"], target_lang)
            translations.append({**item, "translated": translated})

    return translations


def translate_with_vlm(image_path, texts, target_lang, ai_engine="ollama", api_key=None, model=None):
    """VLM으로 이미지 컨텍스트와 함께 번역 (Ollama, Claude, GPT-4, Gemini)"""

    # Claude API 선택 시
    if ai_engine == "claude" and api_key:
        return translate_with_claude(image_path, texts, target_lang, api_key, model)

    # OpenAI GPT-4 API 선택 시
    if ai_engine == "openai" and api_key:
        return translate_with_openai(image_path, texts, target_lang, api_key, model)

    # Google Gemini API 선택 시
    if ai_engine == "gemini" and api_key:
        return translate_with_gemini(image_path, texts, target_lang, api_key, model)

    # 기본: Ollama
    if model is None:
        model = AI_MODELS["ollama"]["default"]
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
                "model": model,
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


def get_background_color(img, bbox, height, width):
    """bbox 주변의 배경색을 샘플링 (글자에서 떨어진 영역에서 샘플링)"""
    # bbox 경계 계산
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    box_height = y_max - y_min
    box_width = x_max - x_min

    # 샘플링 거리: bbox에서 5-10픽셀 떨어진 곳 (글자가 없는 영역)
    sample_dist = max(5, min(10, box_height // 3))

    border_pixels = []

    # 상단 바깥 영역 (bbox 위 sample_dist~sample_dist+3 픽셀)
    sample_y = y_min - sample_dist
    if sample_y >= 3:
        for x in range(max(0, x_min), min(width, x_max)):
            for dy in range(3):
                if sample_y - dy >= 0:
                    border_pixels.append(img[sample_y - dy, x])

    # 하단 바깥 영역
    sample_y = y_max + sample_dist
    if sample_y < height - 3:
        for x in range(max(0, x_min), min(width, x_max)):
            for dy in range(3):
                if sample_y + dy < height:
                    border_pixels.append(img[sample_y + dy, x])

    # 좌측 바깥 영역
    sample_x = x_min - sample_dist
    if sample_x >= 3:
        for y in range(max(0, y_min), min(height, y_max)):
            for dx in range(3):
                if sample_x - dx >= 0:
                    border_pixels.append(img[y, sample_x - dx])

    # 우측 바깥 영역
    sample_x = x_max + sample_dist
    if sample_x < width - 3:
        for y in range(max(0, y_min), min(height, y_max)):
            for dx in range(3):
                if sample_x + dx < width:
                    border_pixels.append(img[y, sample_x + dx])

    if border_pixels:
        # 흰색/밝은 계열 픽셀만 필터링 (RGB 각 채널이 180 이상)
        bright_pixels = [p for p in border_pixels if all(c >= 180 for c in p)]
        if bright_pixels:
            # 가장 밝은 픽셀들의 평균 사용
            bg_color = np.mean(bright_pixels, axis=0).astype(np.uint8)
        else:
            # 밝은 픽셀이 없으면 전체 평균
            bg_color = np.mean(border_pixels, axis=0).astype(np.uint8)
    else:
        bg_color = np.array([255, 255, 255], dtype=np.uint8)

    return bg_color


def get_background_color_from_edges(img, bbox, margin=10):
    """bbox 주변 가장자리에서 배경색 샘플링"""
    from collections import Counter

    height, width = img.shape[:2]
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    edge_pixels = []

    # 상단 가장자리 (margin 픽셀 위)
    sample_y = max(0, y_min - margin)
    for x in range(max(0, x_min), min(width, x_max)):
        pixel = tuple(img[sample_y, x])
        edge_pixels.append(pixel)

    # 하단 가장자리 (margin 픽셀 아래)
    sample_y = min(height - 1, y_max + margin)
    for x in range(max(0, x_min), min(width, x_max)):
        pixel = tuple(img[sample_y, x])
        edge_pixels.append(pixel)

    # 좌측 가장자리 (margin 픽셀 왼쪽)
    sample_x = max(0, x_min - margin)
    for y in range(max(0, y_min), min(height, y_max)):
        pixel = tuple(img[y, sample_x])
        edge_pixels.append(pixel)

    # 우측 가장자리 (margin 픽셀 오른쪽)
    sample_x = min(width - 1, x_max + margin)
    for y in range(max(0, y_min), min(height, y_max)):
        pixel = tuple(img[y, sample_x])
        edge_pixels.append(pixel)

    if edge_pixels:
        # 가장 많이 등장하는 색상 선택
        most_common = Counter(edge_pixels).most_common(1)[0][0]
        return most_common

    return (255, 255, 255)  # 기본값: 흰색


def erase_text_region(img, bbox):
    """텍스트 영역만 지우기 (선은 건드리지 않음) - v1.4.4"""
    height, width = img.shape[:2]

    # bbox 경계 계산
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    # ★ 마진 없이 bbox 그대로 사용 (선 침범 방지)
    # 또는 아주 작은 마진만 (1픽셀)
    margin = 1
    x_min_ext = max(0, x_min - margin)
    y_min_ext = max(0, y_min - margin)
    x_max_ext = min(width, x_max + margin)
    y_max_ext = min(height, y_max + margin)

    # 배경색 샘플링
    bg_color = sample_background_color(img, bbox, height, width)

    # ★ bbox 내부만 배경색으로 채우기 (선은 bbox 바깥이므로 안전)
    cv2.rectangle(img, (x_min_ext, y_min_ext), (x_max_ext, y_max_ext), bg_color, -1)

    return img, bg_color


def sample_background_color(img, bbox, height, width):
    """bbox 바깥 영역에서 배경색 샘플링"""
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    box_height = y_max - y_min
    sample_dist = max(3, min(8, box_height // 4))  # 샘플링 거리

    samples = []

    # 상단 바깥
    if y_min - sample_dist >= 0:
        for x in range(max(0, x_min), min(width, x_max), 2):
            samples.append(img[y_min - sample_dist, x])

    # 하단 바깥
    if y_max + sample_dist < height:
        for x in range(max(0, x_min), min(width, x_max), 2):
            samples.append(img[y_max + sample_dist, x])

    # 좌측 바깥
    if x_min - sample_dist >= 0:
        for y in range(max(0, y_min), min(height, y_max), 2):
            samples.append(img[y, x_min - sample_dist])

    # 우측 바깥
    if x_max + sample_dist < width:
        for y in range(max(0, y_min), min(height, y_max), 2):
            samples.append(img[y, x_max + sample_dist])

    if samples:
        # 중간값 사용 (노이즈에 강함)
        samples_array = np.array(samples)
        bg_color = np.median(samples_array, axis=0).astype(np.uint8)
        return tuple(map(int, bg_color))

    return (255, 255, 255)


def get_text_color_for_background(bg_color):
    """배경색에 따라 적절한 텍스트 색상 반환 (밝은 배경 → 검정, 어두운 배경 → 흰색)"""
    # BGR to grayscale luminance
    if isinstance(bg_color, (list, tuple, np.ndarray)):
        # OpenCV BGR 순서
        b, g, r = bg_color[0], bg_color[1], bg_color[2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        luminance = bg_color

    # 밝기 임계값: 128 (중간값)
    if luminance < 128:
        return (255, 255, 255)  # 어두운 배경 → 흰색 텍스트
    else:
        return (0, 0, 0)  # 밝은 배경 → 검정 텍스트


def is_vertical_text(bbox):
    """세로 텍스트 여부 판단 - 높이가 너비의 2배 이상이면 세로"""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    return box_height > box_width * 2


def draw_vertical_text(draw, text, x, y, font, fill, box_width, box_height):
    """세로 텍스트 그리기 - 글자를 하나씩 세로로 배치"""
    # 글자당 높이 계산
    char_height = box_height / max(len(text), 1)
    
    # 폰트 크기 조정 (글자당 공간에 맞게)
    font_size = min(int(char_height * 0.9), int(box_width * 0.9))
    font_size = max(font_size, 6)  # 최소 6px
    
    try:
        adjusted_font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            adjusted_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            adjusted_font = font
    
    # 각 글자를 세로로 배치
    current_y = y
    for char in text:
        # 글자 중앙 정렬 (x축)
        char_bbox = draw.textbbox((0, 0), char, font=adjusted_font)
        char_width = char_bbox[2] - char_bbox[0]
        char_x = x + (box_width - char_width) // 2
        
        draw.text((char_x, current_y), char, fill=fill, font=adjusted_font)
        current_y += char_height


def replace_text_in_image(image_path, translations, output_path):
    """이미지에서 한글 영역을 지우고 번역된 텍스트로 교체 - v1.4.2"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 1단계: 모든 텍스트 영역을 배경색으로 지우고, 배경색 저장
    bg_colors = {}  # bbox별 배경색 저장
    for i, item in enumerate(translations):
        bbox = item["bbox"]
        img, bg_color = erase_text_region(img, bbox)
        bg_colors[i] = bg_color

    # 2단계: PIL로 변환하여 번역 텍스트 삽입
    img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    # 폰트 크기 (고정 폰트, 크기만 조절)
    font_sizes = [12, 11, 10, 9, 8, 7]

    for i, item in enumerate(translations):
        bbox = item["bbox"]
        translated_text = item["translated"]

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)

        # 왼쪽 상단에서 시작 (왼쪽 정렬)
        x = int(min(xs))
        y = int(min(ys))

        # 적절한 폰트 크기 선택
        font = None
        text_width = 0
        selected_text_height = 0
        for size in font_sizes:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
                except:
                    font = ImageFont.load_default()
                    break

            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            selected_text_height = text_bbox[3] - text_bbox[1]

            # 높이가 원본 박스에 맞으면 OK (너비는 오른쪽으로 확장 가능)
            if selected_text_height <= box_height * 1.2:
                break

        # 세로 중앙 정렬
        text_bbox_actual = draw.textbbox((0, 0), translated_text, font=font)
        actual_text_height = text_bbox_actual[3] - text_bbox_actual[1]
        y_center = int(min(ys)) + box_height // 2
        y_adjusted = y_center - actual_text_height // 2 - text_bbox_actual[1]

        # ★ 배경색에 따라 텍스트 색상 결정 (어두운 배경 → 흰색 텍스트)
        bg_color = bg_colors.get(i, (255, 255, 255))
        text_color = get_text_color_for_background(bg_color)
        # PIL은 RGB 순서이므로 BGR→RGB 변환
        text_color_rgb = (text_color[2], text_color[1], text_color[0]) if text_color == (255, 255, 255) else text_color

        # ★ 세로 텍스트 판정 및 처리 (v1.5.0)
        if is_vertical_text(bbox):
            draw_vertical_text(draw, translated_text, x, y, font, text_color_rgb, box_width, box_height)
        else:
            draw.text((x, y_adjusted), translated_text, fill=text_color_rgb, font=font)

    img_result.save(output_path)
    return output_path


def generate_preview_image(image_base64, translations):
    """미리보기 이미지 생성 (메모리에서 처리) - v1.4.2"""
    # base64 이미지를 numpy 배열로 변환
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    # 1단계: 모든 텍스트 영역을 배경색으로 지우고, 배경색 저장
    bg_colors = {}
    for i, item in enumerate(translations):
        bbox = item["bbox"]
        img, bg_color = erase_text_region(img, bbox)
        bg_colors[i] = bg_color

    # 2단계: PIL로 변환하여 번역 텍스트 삽입
    img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    font_sizes = [12, 11, 10, 9, 8, 7]

    for i, item in enumerate(translations):
        bbox = item["bbox"]
        translated_text = item.get("translated", item.get("text", ""))

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)

        x = int(min(xs))
        y = int(min(ys))

        font = None
        selected_text_height = 0
        for size in font_sizes:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
                except:
                    font = ImageFont.load_default()
                    break

            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            selected_text_height = text_bbox[3] - text_bbox[1]

            if selected_text_height <= box_height * 1.2:
                break

        # 세로 중앙 정렬
        text_bbox_actual = draw.textbbox((0, 0), translated_text, font=font)
        actual_text_height = text_bbox_actual[3] - text_bbox_actual[1]
        y_center = int(min(ys)) + box_height // 2
        y_adjusted = y_center - actual_text_height // 2 - text_bbox_actual[1]

        # ★ 배경색에 따라 텍스트 색상 결정
        bg_color = bg_colors.get(i, (255, 255, 255))
        text_color = get_text_color_for_background(bg_color)
        text_color_rgb = (text_color[2], text_color[1], text_color[0]) if text_color == (255, 255, 255) else text_color

        # ★ 세로 텍스트 판정 및 처리 (v1.5.0)
        if is_vertical_text(bbox):
            draw_vertical_text(draw, translated_text, x, y, font, text_color_rgb, box_width, box_height)
        else:
            draw.text((x, y_adjusted), translated_text, fill=text_color_rgb, font=font)

    # 결과를 base64로 반환
    buffer = io.BytesIO()
    img_result.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


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
        .version-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.6em;
            font-weight: bold;
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
        .settings-btn {
            padding: 3px 8px;
            border: 2px solid #6c757d;
            border-radius: 10px;
            background: white;
            color: #6c757d;
            cursor: pointer;
            font-size: 0.7em;
            transition: all 0.3s;
        }
        .settings-btn:hover {
            background: #6c757d;
            color: white;
        }
        input[type="file"] { display: none; }

        /* 모달 스타일 */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal-content {
            background: white;
            border-radius: 12px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .modal-header h2 {
            margin: 0;
            font-size: 1.2em;
        }
        .modal-close {
            background: none;
            border: none;
            color: white;
            font-size: 1.5em;
            cursor: pointer;
            opacity: 0.8;
        }
        .modal-close:hover {
            opacity: 1;
        }
        .modal-body {
            padding: 20px;
        }
        .modal-footer {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
        }
        .setting-group {
            margin-bottom: 20px;
        }
        .setting-group label {
            display: block;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        .setting-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
        }
        .setting-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .setting-hint {
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }
        .api-key-input-wrapper {
            display: flex;
            gap: 8px;
        }
        .api-key-input-wrapper input {
            flex: 1;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }
        .api-key-input-wrapper input:focus {
            outline: none;
            border-color: #667eea;
        }
        .toggle-visibility {
            padding: 10px 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-size: 1em;
        }
        .toggle-visibility:hover {
            background: #f0f0f0;
        }
        .setting-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .setting-info h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .setting-info ul {
            margin: 0;
            padding-left: 20px;
        }
        .setting-info li {
            margin-bottom: 5px;
            font-size: 0.9em;
            color: #555;
        }
        .btn-primary {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        .btn-primary:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            padding: 10px 20px;
            border: 2px solid #6c757d;
            border-radius: 8px;
            background: white;
            color: #6c757d;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        .btn-secondary:hover {
            background: #6c757d;
            color: white;
        }

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
            width: 12px;
            background: linear-gradient(90deg, #ddd 0%, #bbb 50%, #ddd 100%);
            cursor: col-resize;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            border-radius: 6px;
            margin: 0 4px;
            position: relative;
        }
        .resizer:hover {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
            width: 14px;
        }
        .resizer:active {
            background: #764ba2;
        }
        .resizer::before {
            content: '';
            position: absolute;
            left: 3px;
            top: 50%;
            transform: translateY(-50%);
            width: 2px;
            height: 40px;
            background: rgba(255,255,255,0.5);
            border-radius: 1px;
        }
        .resizer::after {
            content: '';
            position: absolute;
            right: 3px;
            top: 50%;
            transform: translateY(-50%);
            width: 2px;
            height: 40px;
            background: rgba(255,255,255,0.5);
            border-radius: 1px;
        }
        .preview-header {
            background: #f8f9fa;
            padding: 8px 12px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .preview-header .page-info {
            font-weight: bold;
            color: #333;
        }
        .preview-toggle {
            display: flex;
            gap: 2px;
            background: #e9ecef;
            padding: 2px;
            border-radius: 6px;
        }
        .toggle-btn {
            padding: 4px 10px;
            border: none;
            background: transparent;
            color: #666;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            transition: all 0.2s;
        }
        .toggle-btn:hover {
            background: rgba(102, 126, 234, 0.1);
        }
        .toggle-btn.active {
            background: #667eea;
            color: white;
        }
        .toggle-btn.loading {
            opacity: 0.6;
            cursor: wait;
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
            <span class="version-badge">v{{ version }}</span>
            <span class="subtitle">한글→다국어</span>
            <button type="button" class="lang-btn active" data-lang="english">🇺🇸EN</button>
            <button type="button" class="lang-btn" data-lang="vietnamese">🇻🇳VI</button>
            <button type="button" class="lang-btn" data-lang="chinese">🇨🇳中</button>
            <button type="button" class="lang-btn" data-lang="indonesian">🇮🇩ID</button>
            <button type="button" class="lang-btn" data-lang="bengali">🇧🇩BN</button>
            <button type="button" class="file-select-btn" id="fileSelectBtn">📁 파일선택</button>
            <button type="button" class="translate-btn" id="translateBtn" disabled>🚀 번역</button>
            <button type="button" class="settings-btn" id="settingsBtn">⚙️</button>
        </div>

        <!-- 설정 모달 -->
        <div class="modal-overlay" id="settingsModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>⚙️ AI 설정</h2>
                    <button class="modal-close" id="closeSettings">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="setting-group">
                        <label>AI 엔진 선택</label>
                        <select id="aiEngineSelect">
                            <option value="ollama" selected>🖥️ Ollama (로컬) - 무료</option>
                            <option value="claude">🟣 Claude API</option>
                            <option value="openai">🟢 OpenAI GPT-4</option>
                            <option value="gemini">🔵 Google Gemini</option>
                        </select>
                        <p class="setting-hint">Ollama는 로컬에서 실행되며 API 키가 필요 없습니다.</p>
                    </div>

                    <div class="setting-group api-key-group" id="apiKeyGroup" style="display: none;">
                        <label id="apiKeyLabel">API Key</label>
                        <div class="api-key-input-wrapper">
                            <input type="password" id="apiKeyInput" placeholder="API 키를 입력하세요">
                            <button type="button" class="toggle-visibility" id="toggleApiKey">👁️</button>
                        </div>
                        <p class="setting-hint" id="apiKeyHint">
                            API 키는 브라우저에만 저장되며 서버로 전송되지 않습니다.
                        </p>
                    </div>

                    <div class="setting-group" id="modelGroup">
                        <label>모델 선택</label>
                        <select id="modelSelect">
                            <!-- JavaScript로 동적 생성 -->
                        </select>
                        <p class="setting-hint" id="modelHint">선택한 AI 엔진에서 사용할 모델을 선택하세요.</p>
                    </div>

                    <div class="setting-info">
                        <h4>AI별 특징</h4>
                        <ul>
                            <li><strong>Ollama</strong>: 무료, 로컬 실행, 인터넷 불필요</li>
                            <li><strong>Claude</strong>: 높은 정확도, Vision 지원</li>
                            <li><strong>GPT-4</strong>: 범용성 높음, Vision 지원</li>
                            <li><strong>Gemini</strong>: 빠른 속도, 비용 저렴</li>
                        </ul>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-secondary" id="cancelSettings">취소</button>
                    <button type="button" class="btn-primary" id="saveSettings">저장</button>
                </div>
            </div>
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
                    <div class="preview-toggle">
                        <button class="toggle-btn active" id="showOriginal">📄 원본</button>
                        <button class="toggle-btn" id="showPreview">🔄 미리보기</button>
                    </div>
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

        // 미리보기 토글 버튼
        const showOriginalBtn = document.getElementById('showOriginal');
        const showPreviewBtn = document.getElementById('showPreview');

        // 미리보기 상태
        let isPreviewMode = false;
        let previewCache = {};  // 페이지별 미리보기 캐시

        // 설정 관련 요소
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const closeSettings = document.getElementById('closeSettings');
        const cancelSettings = document.getElementById('cancelSettings');
        const saveSettings = document.getElementById('saveSettings');
        const aiEngineSelect = document.getElementById('aiEngineSelect');
        const apiKeyGroup = document.getElementById('apiKeyGroup');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const apiKeyLabel = document.getElementById('apiKeyLabel');
        const apiKeyHint = document.getElementById('apiKeyHint');
        const toggleApiKey = document.getElementById('toggleApiKey');
        const modelSelect = document.getElementById('modelSelect');
        const modelHint = document.getElementById('modelHint');

        // AI 설정 상태 (localStorage에서 로드)
        let currentAiEngine = localStorage.getItem('pdf_translator_ai_engine') || 'ollama';
        let currentModel = localStorage.getItem('pdf_translator_model') || '';
        let apiKeys = JSON.parse(localStorage.getItem('pdf_translator_api_keys') || '{}');

        // AI 모델 정보
        const aiModels = {
            ollama: {
                models: ['qwen2.5vl:latest', 'llava:latest', 'bakllava:latest'],
                default: 'qwen2.5vl:latest',
                hints: {
                    'qwen2.5vl:latest': '다국어 지원, Vision 강력 (권장)',
                    'llava:latest': '경량 모델, 빠른 속도',
                    'bakllava:latest': 'LLaVA 기반, 균형잡힌 성능'
                }
            },
            claude: {
                models: ['claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
                default: 'claude-sonnet-4-20250514',
                hints: {
                    'claude-sonnet-4-20250514': '최신 모델, 고성능 (권장)',
                    'claude-3-5-sonnet-20241022': '안정적인 성능',
                    'claude-3-haiku-20240307': '빠르고 저렴'
                }
            },
            openai: {
                models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
                default: 'gpt-4o',
                hints: {
                    'gpt-4o': '최신 멀티모달, 고성능 (권장)',
                    'gpt-4o-mini': '저렴하고 빠름',
                    'gpt-4-turbo': '안정적, Vision 지원'
                }
            },
            gemini: {
                models: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
                default: 'gemini-2.0-flash',
                hints: {
                    'gemini-2.0-flash': '최신 모델, 빠르고 저렴 (권장)',
                    'gemini-1.5-flash': '빠른 속도, 비용 효율',
                    'gemini-1.5-pro': '고성능, 복잡한 작업용'
                }
            }
        };

        // API 키 힌트 정보
        const apiKeyInfo = {
            claude: {
                label: 'Claude API Key',
                hint: 'Anthropic Console에서 발급받은 API 키를 입력하세요.',
                placeholder: 'sk-ant-...'
            },
            openai: {
                label: 'OpenAI API Key',
                hint: 'OpenAI Platform에서 발급받은 API 키를 입력하세요.',
                placeholder: 'sk-...'
            },
            gemini: {
                label: 'Google Gemini API Key',
                hint: 'Google AI Studio에서 발급받은 API 키를 입력하세요.',
                placeholder: 'AIza...'
            }
        };

        // 초기 설정 UI 업데이트
        function initSettings() {
            aiEngineSelect.value = currentAiEngine;
            updateApiKeyVisibility();
            updateModelOptions();
        }

        // API 키 입력 필드 표시/숨김
        function updateApiKeyVisibility() {
            const engine = aiEngineSelect.value;
            if (engine === 'ollama') {
                apiKeyGroup.style.display = 'none';
            } else {
                apiKeyGroup.style.display = 'block';
                const info = apiKeyInfo[engine];
                apiKeyLabel.textContent = info.label;
                apiKeyHint.textContent = info.hint;
                apiKeyInput.placeholder = info.placeholder;
                apiKeyInput.value = apiKeys[engine] || '';
            }
        }

        // 모델 선택 옵션 업데이트
        function updateModelOptions() {
            const engine = aiEngineSelect.value;
            const modelInfo = aiModels[engine];

            // 기존 옵션 제거
            modelSelect.innerHTML = '';

            // 새 옵션 추가
            modelInfo.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                if (model === modelInfo.default) {
                    option.textContent += ' (기본)';
                }
                modelSelect.appendChild(option);
            });

            // 저장된 모델이 있으면 선택, 없으면 기본값
            const savedModel = localStorage.getItem(`pdf_translator_model_${engine}`);
            if (savedModel && modelInfo.models.includes(savedModel)) {
                modelSelect.value = savedModel;
            } else {
                modelSelect.value = modelInfo.default;
            }

            // 힌트 업데이트
            updateModelHint();
        }

        // 모델 힌트 업데이트
        function updateModelHint() {
            const engine = aiEngineSelect.value;
            const model = modelSelect.value;
            const hint = aiModels[engine].hints[model] || '';
            modelHint.textContent = hint;
        }

        // 설정 모달 열기
        settingsBtn.addEventListener('click', () => {
            initSettings();
            settingsModal.classList.add('active');
        });

        // 설정 모달 닫기
        function closeModal() {
            settingsModal.classList.remove('active');
        }
        closeSettings.addEventListener('click', closeModal);
        cancelSettings.addEventListener('click', closeModal);
        settingsModal.addEventListener('click', (e) => {
            if (e.target === settingsModal) closeModal();
        });

        // AI 엔진 변경 시
        aiEngineSelect.addEventListener('change', () => {
            updateApiKeyVisibility();
            updateModelOptions();
        });

        // 모델 변경 시
        modelSelect.addEventListener('change', updateModelHint);

        // API 키 표시/숨김 토글
        toggleApiKey.addEventListener('click', () => {
            if (apiKeyInput.type === 'password') {
                apiKeyInput.type = 'text';
                toggleApiKey.textContent = '🙈';
            } else {
                apiKeyInput.type = 'password';
                toggleApiKey.textContent = '👁️';
            }
        });

        // 설정 저장
        saveSettings.addEventListener('click', () => {
            const engine = aiEngineSelect.value;
            const model = modelSelect.value;
            currentAiEngine = engine;
            currentModel = model;

            // localStorage에 저장
            localStorage.setItem('pdf_translator_ai_engine', engine);
            localStorage.setItem(`pdf_translator_model_${engine}`, model);

            // API 키 저장 (Ollama 제외)
            if (engine !== 'ollama' && apiKeyInput.value) {
                apiKeys[engine] = apiKeyInput.value;
                localStorage.setItem('pdf_translator_api_keys', JSON.stringify(apiKeys));
            }

            closeModal();
            status.className = 'status success';
            status.textContent = `✅ ${getEngineName(engine)} - ${model} 설정 완료`;
        });

        // 엔진 이름 반환
        function getEngineName(engine) {
            const names = {
                ollama: 'Ollama (로컬)',
                claude: 'Claude',
                openai: 'GPT-4',
                gemini: 'Gemini'
            };
            return names[engine] || engine;
        }

        // 현재 API 키 가져오기
        function getCurrentApiKey() {
            if (currentAiEngine === 'ollama') return null;
            return apiKeys[currentAiEngine] || null;
        }

        // 현재 모델 가져오기
        function getCurrentModel() {
            const savedModel = localStorage.getItem(`pdf_translator_model_${currentAiEngine}`);
            if (savedModel) return savedModel;
            return aiModels[currentAiEngine]?.default || null;
        }

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

                // 이미 파일이 로드되어 있으면 전체 페이지 재번역
                if (pagesData.length > 0) {
                    await retranslateAllPages();
                }
            });
        });

        // 파일 로드 및 OCR 처리
        let progressInterval = null;
        
        async function loadAndProcessFile(file) {
            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span>파일 분석 중... 시작 중';
            initialGuide.style.display = 'none';
            editorContainer.classList.remove('active');
            results.classList.remove('active');

            // ★ 진행 상황 폴링 시작
            progressInterval = setInterval(async () => {
                try {
                    const progRes = await fetch('/progress');
                    const prog = await progRes.json();
                    if (prog.stage) {
                        status.innerHTML = `<span class="spinner"></span>${prog.stage} (${prog.current}/${prog.total}) - ${prog.detail} [${prog.elapsed}]`;
                    }
                } catch (e) {}
            }, 500);

            const formData = new FormData();
            formData.append('file', file);
            formData.append('target_lang', targetLang.value);
            formData.append('ai_engine', currentAiEngine);
            formData.append('model', getCurrentModel());
            const apiKey = getCurrentApiKey();
            if (apiKey) {
                formData.append('api_key', apiKey);
            }

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                // ★ 폴링 중지
                if (progressInterval) {
                    clearInterval(progressInterval);
                    progressInterval = null;
                }

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
                // ★ 에러 시에도 폴링 중지
                if (progressInterval) {
                    clearInterval(progressInterval);
                    progressInterval = null;
                }
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

            // 미리보기 모드에 따라 이미지 표시
            if (isPreviewMode) {
                showPreviewImage(pageIdx);
            } else {
                previewImg.src = 'data:image/png;base64,' + page.image;
            }

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

        // 미리보기 이미지 로드
        async function showPreviewImage(pageIdx, forceRefresh = false) {
            const page = pagesData[pageIdx];

            console.log('[Preview Debug] pageIdx:', pageIdx);
            console.log('[Preview Debug] translations:', page.translations);
            console.log('[Preview Debug] translations length:', page.translations ? page.translations.length : 'undefined');

            // 캐시에 있으면 바로 표시 (강제 새로고침이 아닌 경우)
            if (!forceRefresh && previewCache[pageIdx]) {
                console.log('[Preview Debug] Using cached preview');
                previewImg.src = 'data:image/png;base64,' + previewCache[pageIdx];
                return;
            }

            // 번역 데이터가 없으면 원본 표시
            if (!page.translations || page.translations.length === 0) {
                console.log('[Preview Debug] No translations, showing original image');
                previewImg.src = 'data:image/png;base64,' + page.image;
                return;
            }

            // 로딩 표시
            showPreviewBtn.classList.add('loading');
            showPreviewBtn.textContent = '⏳ 생성중...';

            try {
                const response = await fetch('/generate_preview', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image: page.image,
                        translations: page.translations
                    })
                });

                const data = await response.json();

                if (data.success) {
                    previewCache[pageIdx] = data.preview;
                    previewImg.src = 'data:image/png;base64,' + data.preview;
                } else {
                    console.error('Preview generation failed:', data.error);
                    previewImg.src = 'data:image/png;base64,' + page.image;
                }
            } catch (error) {
                console.error('Preview error:', error);
                previewImg.src = 'data:image/png;base64,' + page.image;
            } finally {
                showPreviewBtn.classList.remove('loading');
                showPreviewBtn.textContent = '🔄 미리보기';
            }
        }

        // 미리보기 캐시 초기화 (번역 수정 시)
        function invalidatePreviewCache(pageIdx) {
            delete previewCache[pageIdx];
        }

        // 원본/미리보기 토글
        showOriginalBtn.addEventListener('click', () => {
            if (!isPreviewMode) return;
            isPreviewMode = false;
            showOriginalBtn.classList.add('active');
            showPreviewBtn.classList.remove('active');
            const page = pagesData[currentPage];
            previewImg.src = 'data:image/png;base64,' + page.image;
        });

        showPreviewBtn.addEventListener('click', (e) => {
            const forceRefresh = e.shiftKey;  // Shift+클릭으로 강제 새로고침
            if (forceRefresh) {
                console.log('[Preview Debug] Force refresh requested');
                delete previewCache[currentPage];  // 캐시 삭제
            }
            if (isPreviewMode && !forceRefresh) return;
            isPreviewMode = true;
            showPreviewBtn.classList.add('active');
            showOriginalBtn.classList.remove('active');
            showPreviewImage(currentPage, forceRefresh);
        });

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

                    // 미리보기 캐시 무효화
                    invalidatePreviewCache(currentPage);
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

        // 모든 페이지 재번역 (언어 변경 시)
        async function retranslateAllPages() {
            if (pagesData.length === 0) return;

            // 미리보기 캐시 전체 초기화
            previewCache = {};

            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span>언어 변경 중... 전체 페이지 재번역 중입니다';

            try {
                // 모든 페이지 재번역
                for (let i = 0; i < pagesData.length; i++) {
                    status.innerHTML = `<span class="spinner"></span>재번역 중... (${i + 1}/${pagesData.length})`;

                    const response = await fetch('/retranslate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            page_idx: i,
                            target_lang: targetLang.value,
                            ai_engine: currentAiEngine,
                            model: getCurrentModel(),
                            api_key: getCurrentApiKey(),
                            image: pagesData[i].image,
                            texts: pagesData[i].translations.map(t => ({
                                text: t.text,
                                bbox: t.bbox
                            }))
                        })
                    });

                    const data = await response.json();
                    if (data.success) {
                        pagesData[i].translations = data.translations;
                        pagesData[i].confirmed = false;
                    }
                }

                // 현재 페이지 다시 표시
                showPage(currentPage);
                status.className = 'status success';
                status.textContent = `✅ 전체 ${pagesData.length}페이지 재번역 완료`;
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
    return render_template_string(HTML_TEMPLATE, version=VERSION)


# 임시 저장소: 세션별 이미지 경로
temp_image_paths = {}

# ★ 진행 상황 추적 (v1.4.5)
progress_status = {
    "stage": "",           # 현재 단계
    "current": 0,          # 현재 진행
    "total": 0,            # 전체
    "detail": "",          # 세부 정보
    "start_time": None     # 시작 시간
}

def update_progress(stage, current, total, detail=""):
    """진행 상황 업데이트"""
    global progress_status
    progress_status["stage"] = stage
    progress_status["current"] = current
    progress_status["total"] = total
    progress_status["detail"] = detail
    if current == 0:
        progress_status["start_time"] = datetime.now()


@app.route('/progress', methods=['GET'])
def get_progress():
    """진행 상황 조회 API"""
    elapsed = ""
    if progress_status["start_time"]:
        delta = datetime.now() - progress_status["start_time"]
        elapsed = f"{int(delta.total_seconds())}초 경과"
    
    return jsonify({
        "stage": progress_status["stage"],
        "current": progress_status["current"],
        "total": progress_status["total"],
        "detail": progress_status["detail"],
        "elapsed": elapsed
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """파일 업로드 + OCR + 초기 번역"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "파일이 없습니다"})

        file = request.files['file']
        target_lang = request.form.get('target_lang', 'english')
        ai_engine = request.form.get('ai_engine', 'ollama')
        api_key = request.form.get('api_key', None)
        model = request.form.get('model', None)

        print(f"[AI Engine] {ai_engine}, [Model] {model}")

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

        # 각 페이지 분석 (2단계: OCR 먼저, 번역은 배치로)
        pages = []
        total_pages = len(image_paths)
        all_pages_data = []  # OCR 결과 임시 저장

        # ===== 1단계: 모든 페이지 OCR =====
        for i, img_path in enumerate(image_paths):
            print(f"[OCR {i+1}/{total_pages}] {img_path}")

            # ★ 진행 상황: OCR
            update_progress("OCR", i+1, total_pages, f"페이지 {i+1}/{total_pages} OCR 처리 중...")

            # 이미지를 base64로 인코딩
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()

            # OCR
            texts = get_ocr_results(img_path)
            print(f"  Found {len(texts)} Korean texts")

            all_pages_data.append({
                "page_idx": i,
                "img_path": img_path,
                "image_base64": image_base64,
                "texts": texts
            })

        # ===== 2단계: 배치 번역 (1회 API 호출) =====
        total_texts = sum(len(p["texts"]) for p in all_pages_data)
        update_progress("번역", 1, 1, f"전체 {total_texts}개 텍스트 일괄 번역 중... (1회 API 호출)")
        print(f"[Batch Translation] Total {total_texts} texts from {total_pages} pages")

        # Gemini 배치 번역 사용 (Free Tier 최적화)
        if ai_engine == "gemini" and api_key and total_texts > 0:
            batch_input = [{"page_idx": p["page_idx"], "texts": p["texts"]} for p in all_pages_data]
            translations_by_page = translate_batch_with_gemini(batch_input, target_lang, api_key, model)

            # 결과 조합
            for page_data in all_pages_data:
                page_idx = page_data["page_idx"]
                translations = translations_by_page.get(page_idx, [])
                pages.append({
                    "image": page_data["image_base64"],
                    "image_path": page_data["img_path"],
                    "translations": translations,
                    "confirmed": False
                })
        else:
            # 기존 방식: 페이지별 번역 (Ollama, OpenAI 등)
            for page_data in all_pages_data:
                update_progress("번역", page_data["page_idx"]+1, total_pages,
                               f"페이지 {page_data['page_idx']+1}/{total_pages} - {len(page_data['texts'])}개 텍스트 번역 중...")

                translations = []
                if page_data["texts"]:
                    translations = translate_with_vlm(page_data["img_path"], page_data["texts"],
                                                      target_lang, ai_engine, api_key, model)

                pages.append({
                    "image": page_data["image_base64"],
                    "image_path": page_data["img_path"],
                    "translations": translations,
                    "confirmed": False
                })

        # ★ 진행 상황: 완료
        update_progress("완료", total_pages, total_pages, "분석 완료!")

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
    """언어 변경 시 재번역 (VLM 사용)"""
    try:
        data = request.get_json()
        target_lang = data.get('target_lang', 'english')
        ai_engine = data.get('ai_engine', 'ollama')
        api_key = data.get('api_key', None)
        model = data.get('model', None)
        image_base64 = data.get('image', None)
        texts = data.get('texts', [])

        print(f"[Retranslate] AI Engine: {ai_engine}, Model: {model}, Target: {target_lang}")

        # 이미지가 있고 AI 엔진을 사용하는 경우 VLM으로 번역
        if image_base64 and texts:
            # base64 이미지를 임시 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_img_path = os.path.join(UPLOAD_FOLDER, f"temp_retrans_{timestamp}.png")
            with open(temp_img_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))

            try:
                # VLM으로 번역 (선택된 AI 엔진 사용)
                text_items = [{"text": item['text'], "bbox": item['bbox']} for item in texts]
                translations = translate_with_vlm(temp_img_path, text_items, target_lang, ai_engine, api_key, model)
            finally:
                # 임시 파일 삭제
                try:
                    os.remove(temp_img_path)
                except:
                    pass
        else:
            # fallback: 사전 기반 번역
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


@app.route('/generate_preview', methods=['POST'])
def generate_preview():
    """번역된 이미지 미리보기 생성"""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        translations = data.get('translations', [])

        print(f"[generate_preview] Received {len(translations)} translations")
        for i, t in enumerate(translations[:3]):  # 처음 3개만 출력
            print(f"  [{i}] bbox: {t.get('bbox', 'N/A')}, text: {t.get('text', 'N/A')[:20]}...")

        if not image_base64:
            return jsonify({"success": False, "error": "이미지가 없습니다"})

        if not translations:
            print("[generate_preview] ERROR: No translations provided!")
            return jsonify({"success": False, "error": "번역 데이터가 없습니다"})

        # 미리보기 이미지 생성
        print("[generate_preview] Calling generate_preview_image...")
        preview_base64 = generate_preview_image(image_base64, translations)
        print("[generate_preview] Preview generated successfully")

        return jsonify({
            "success": True,
            "preview": preview_base64
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
    print(f"PDF Translator v{VERSION} - 의류 기술서 번역 앱")
    print("=" * 60)
    print(f"Version: {VERSION} ({VERSION_DATE})")
    print("Engine: PaddleOCR + VLM (qwen2.5vl)")
    print("Languages: English, Vietnamese, Chinese, Indonesian, Bengali")
    print("Port: 6008")
    print("=" * 60)

    # OCR 엔진 미리 로드
    get_ocr_engine()

    app.run(host='0.0.0.0', port=6008, debug=True)
