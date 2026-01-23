# -*- coding: utf-8 -*-
"""
PDF Translator - 한글 텍스트를 다국어로 번역하는 웹앱
- Flask 기반 웹 인터페이스
- PaddleOCR + VLM (qwen2.5vl) 사용
- 지원 언어: 영어, 베트남어, 중국어, 인도네시아어, 벵골어
"""

# 버전 정보
VERSION = "1.9.9"
VERSION_DATE = "2026-01-22"
VERSION_NOTES = """
v1.9.9 (2026-01-22)
- ★ object-fit: contain 좌표 보정: 화살표 스냅이 정확한 위치로 연결
- getActualImageSize() 함수로 실제 이미지 렌더링 크기/오프셋 계산
- getBoundingClientRect() 대신 실제 렌더링 치수 사용으로 좌표 정확도 향상

v1.9.8 (2026-01-22)
- ★ 줌 호환: 화살표 스냅이 확대/축소 상태에서도 정확히 동작
- 화살표 드래그 시 줌 레벨 고려한 좌표 변환
- 화살표 렌더링 시 이미지 렌더링 크기 기준 좌표 계산

v1.9.7 (2026-01-22)
- ★ 화살표 스냅 기능: 화살촉이 가장 가까운 텍스트 박스 중앙으로 자동 스냅
- 우클릭 드래그 시 bbox 중앙으로 정확히 지시선 연결

v1.9.6 (2026-01-20)
- ★ Gemini 프롬프트 강화: 괄호 설명 추가 금지
- ★ Google Vision 병합 gap 증가: 20→35px, 10→15px (더 적극적 병합)

v1.9.5 (2026-01-20)
- ★ Google Vision OCR 단어 병합: 인접 단어를 PaddleOCR처럼 그룹화
- "행거" + "루프" → "행거루프" 병합으로 번역 겹침 해결
- horizontal_gap=20px, vertical_threshold=10px 기준

v1.9.4 (2026-01-20)
- ★ API 키 보안 강화: 하드코딩 → 환경변수(.env) 방식
- .gitignore에 .env 추가 (GitHub 노출 방지)

v1.9.1 (2026-01-20)
- ★ 한글 폰트 수정: arial.ttf → malgun.ttf (맑은 고딕)
- 한글 텍스트가 □□□로 깨지는 문제 해결
- 다국어(한중일) 텍스트 렌더링 지원

v1.9.0 (2026-01-12)
- ★ 번역 완료 후 메모 추가 기능 (우클릭 메뉴)
- 메모 스타일: 크기/색/굵기/배경/테두리/투명도/줄바꿈
- 메모는 번역 미리보기 및 최종 출력에 반영

v1.8.2 (2026-01-11)
- ★ Placeholder 복원 강화: AI가 변형한 다양한 TERM 패턴 처리
- TERM_1, TERM 1, <TERM_1>, [TERM_1] 등 모든 변형 자동 복원
- 정규식 기반 유연한 패턴 매칭 추가

v1.8.1 (2026-01-11)
- ★ 성능 로그 추가: Batch OCR, Claude API, Gemini Batch, 병렬 번역 타이밍 출력
- ★ 확정 버튼 클릭 시 미리보기 즉시 갱신 복구
- 디버깅 및 성능 분석용 상세 로그

v1.8.0 (2026-01-10)
- ★ 사전 구조 통합: {"한글": {"full": "번역", "abbr": "약어"}} 
- ★ UI 약어 편집: 용어 사전에서 약어 직접 추가/수정 가능
- 하드코딩된 ABBREVIATIONS 제거, 사전 기반 약어 시스템으로 전환
- 장기적 확장성 개선 (category, note 등 필드 추가 용이)

v1.7.0 (2026-01-09)
- ★ 용어 사전 관리 기능: 의류 전문 용어 추가/수정/삭제 가능
- ★ 사전 후처리: AI 번역 후 사전 용어로 자동 교정 (일관성 향상)
- 📖 버튼 클릭으로 사전 관리 모달 오픈
- 언어별 탭 전환, 검색 기능, 실시간 저장
- JSON 파일(garment_dict.json)로 사전 데이터 분리

v1.6.1 (2026-01-09)
- Claude Opus 4.5 모델 추가
- 번역 프롬프트 강화: 모든 항목 번역 필수 규칙 적용
- 파싱 로직 개선: 다양한 번호 형식 지원 (1., 1), **1.**, 1:)

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
import re
import requests
import logging
import time
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed  # 병렬 처리용
from flask import Flask, render_template_string, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR
import cv2
import fitz  # PyMuPDF
from img2table.document import Image as Img2TableImage  # 테이블 감지용

# Google Cloud Vision OCR
from google.cloud import vision
from google.oauth2 import service_account

# ★ 환경변수에서 API 키 로드
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드

# ★ 로깅 설정 (겹침 감지 디버깅용)
LOG_FILE = os.path.join(os.path.dirname(__file__), 'overlap_debug.log')

# 전용 로거 생성 (Flask 로깅과 분리)
logger = logging.getLogger('overlap_debug')
logger.setLevel(logging.DEBUG)
logger.handlers = []  # 기존 핸들러 제거

# 파일 핸들러
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(file_handler)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(console_handler)

logger.propagate = False  # 부모 로거로 전파 방지

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)

# 설정
OLLAMA_URL = "http://localhost:11434/api/generate"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# ★ API 키 (환경변수에서 로드 - .env 파일 사용)
HARDCODED_API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "claude_sije": os.environ.get("CLAUDE_API_KEY_SIJE", ""),
    "claude_seam": os.environ.get("CLAUDE_API_KEY_SEAM", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", "")
}

# ★ Google Cloud Vision OCR 설정
VISION_KEY_PATH = os.path.join(os.path.dirname(__file__), "vision-key.json")
vision_client = None

def get_vision_client():
    """Google Cloud Vision 클라이언트 (lazy initialization)"""
    global vision_client
    if vision_client is None and os.path.exists(VISION_KEY_PATH):
        credentials = service_account.Credentials.from_service_account_file(VISION_KEY_PATH)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    return vision_client

def merge_adjacent_words(words, horizontal_gap=20, vertical_threshold=10, debug=True):
    """인접한 단어들을 하나의 텍스트 박스로 병합 (PaddleOCR 스타일)

    Args:
        words: [(text, bbox), ...] where bbox = [x1, y1, x2, y2]
        horizontal_gap: 이 픽셀 이내면 같은 그룹으로 병합
        vertical_threshold: y좌표 차이가 이 이내면 같은 줄로 판단
        debug: 디버그 로그 출력 여부

    Returns:
        list: [[bbox, (merged_text, confidence)], ...]
    """
    import datetime

    # 디버그 로그 파일
    log_path = os.path.join(os.path.dirname(__file__), "ocr_merge_debug.log")

    def log(msg):
        if debug:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    if not words:
        return []

    # ===== 디버그: 시작 =====
    if debug:
        log("\n" + "="*80)
        log(f"[{datetime.datetime.now()}] merge_adjacent_words 시작")
        log(f"설정: horizontal_gap={horizontal_gap}, vertical_threshold={vertical_threshold}")
        log(f"입력 단어 수: {len(words)}")
        log("-"*80)
        log("[원본 단어 목록 (정렬 전)]")
        for i, (text, bbox) in enumerate(words):
            x1, y1, x2, y2 = bbox
            cy = (y1 + y2) / 2
            log(f"  [{i:2d}] '{text}' | bbox=[{x1}, {y1}, {x2}, {y2}] | y_center={cy:.1f} | width={x2-x1}")

    # y좌표 기준으로 정렬 (같은 줄끼리 묶기)
    sorted_words = sorted(words, key=lambda w: (w[1][1], w[1][0]))  # y, x 순

    # ===== 디버그: 정렬 후 =====
    if debug:
        log("-"*80)
        log("[정렬 후 단어 목록 (y1, x1 순)]")
        for i, (text, bbox) in enumerate(sorted_words):
            x1, y1, x2, y2 = bbox
            cy = (y1 + y2) / 2
            log(f"  [{i:2d}] '{text}' | bbox=[{x1}, {y1}, {x2}, {y2}] | y_center={cy:.1f}")
        log("-"*80)
        log("[병합 과정]")

    merged_results = []
    current_group = [sorted_words[0]]
    group_num = 1

    if debug:
        t, b = sorted_words[0]
        log(f"  그룹#{group_num} 시작: '{t}'")

    for idx, word in enumerate(sorted_words[1:], start=1):
        text, bbox = word
        x1, y1, x2, y2 = bbox

        # 현재 그룹의 마지막 단어
        last_text, last_bbox = current_group[-1]
        last_x1, last_y1, last_x2, last_y2 = last_bbox

        # 같은 줄인지 확인 (y좌표 차이)
        y_center = (y1 + y2) / 2
        last_y_center = (last_y1 + last_y2) / 2
        y_diff = abs(y_center - last_y_center)
        same_line = y_diff <= vertical_threshold

        # 수평 간격 확인
        h_gap = x1 - last_x2
        close_enough = h_gap <= horizontal_gap and h_gap >= -5  # 약간 겹쳐도 허용

        # ===== 디버그: 각 비교 =====
        if debug:
            log(f"\n  --- 비교 [{idx}]: '{last_text}' vs '{text}' ---")
            log(f"      last: x1={last_x1}, y1={last_y1}, x2={last_x2}, y2={last_y2}, y_center={last_y_center:.1f}")
            log(f"      curr: x1={x1}, y1={y1}, x2={x2}, y2={y2}, y_center={y_center:.1f}")
            log(f"      y_diff={y_diff:.1f} (threshold={vertical_threshold}) → same_line={same_line}")
            log(f"      h_gap={h_gap} (range=[-5, {horizontal_gap}]) → close_enough={close_enough}")

        if same_line and close_enough:
            # 같은 그룹에 추가
            current_group.append(word)
            if debug:
                log(f"      ✅ 병합: 그룹#{group_num}에 '{text}' 추가")
        else:
            # 현재 그룹 병합 후 새 그룹 시작
            merged_results.append(_merge_word_group(current_group))
            if debug:
                group_texts = [w[0] for w in current_group]
                log(f"      ❌ 분리: 그룹#{group_num} 완료 → {group_texts}")
                if not same_line:
                    log(f"         이유: y_diff({y_diff:.1f}) > threshold({vertical_threshold})")
                if not close_enough:
                    if h_gap > horizontal_gap:
                        log(f"         이유: h_gap({h_gap}) > horizontal_gap({horizontal_gap})")
                    elif h_gap < -5:
                        log(f"         이유: h_gap({h_gap}) < -5 (역순 정렬 의심!)")
            current_group = [word]
            group_num += 1
            if debug:
                log(f"      그룹#{group_num} 시작: '{text}'")

    # 마지막 그룹 처리
    if current_group:
        merged_results.append(_merge_word_group(current_group))
        if debug:
            group_texts = [w[0] for w in current_group]
            log(f"\n  그룹#{group_num} 완료 (마지막) → {group_texts}")

    # ===== 디버그: 최종 결과 =====
    if debug:
        log("-"*80)
        log(f"[최종 병합 결과] {len(words)}개 → {len(merged_results)}개 그룹")
        for i, result in enumerate(merged_results):
            bbox, (text, conf) = result
            log(f"  그룹#{i+1}: '{text}'")
        log("="*80 + "\n")

    return merged_results


def _merge_word_group(group):
    """단어 그룹을 하나의 텍스트 박스로 병합"""
    # ★ bbox 확장 패딩 (한글 완전히 덮기 위해)
    PADDING_X = 2  # 좌우 확장
    PADDING_Y = 2  # 상하 확장
    
    if len(group) == 1:
        text, bbox = group[0]
        x1, y1, x2, y2 = bbox
        # 패딩 적용
        x1 = max(0, x1 - PADDING_X)
        y1 = max(0, y1 - PADDING_Y)
        x2 = x2 + PADDING_X
        y2 = y2 + PADDING_Y
        return [
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            (text, 0.99)
        ]
    
    # 텍스트 합치기 (x좌표 순으로 정렬)
    sorted_group = sorted(group, key=lambda w: w[1][0])
    merged_text = "".join([w[0] for w in sorted_group])
    
    # BBox 병합 (전체를 감싸는 박스) + 패딩
    all_x1 = max(0, min(w[1][0] for w in group) - PADDING_X)
    all_y1 = max(0, min(w[1][1] for w in group) - PADDING_Y)
    all_x2 = max(w[1][2] for w in group) + PADDING_X
    all_y2 = max(w[1][3] for w in group) + PADDING_Y
    
    bbox = [
        [all_x1, all_y1],
        [all_x2, all_y1],
        [all_x2, all_y2],
        [all_x1, all_y2]
    ]
    
    return [bbox, (merged_text, 0.99)]


def ocr_with_google_vision(image_path):
    """Google Cloud Vision API로 OCR 수행 (단어 병합 버전)

    Returns:
        list: PaddleOCR과 동일한 형식 [[bbox, (text, confidence)], ...]
              bbox = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    client = get_vision_client()
    if client is None:
        print("[Vision OCR] Client not available, falling back to PaddleOCR")
        return None

    start_time = time.time()

    with open(image_path, 'rb') as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        print(f"[Vision OCR] Error: {response.error.message}")
        return None

    texts = response.text_annotations

    if not texts:
        print("[Vision OCR] No text detected")
        return []

    # 첫 번째는 전체 텍스트이므로 스킵, 나머지가 개별 단어
    words = []
    for text in texts[1:]:
        vertices = text.bounding_poly.vertices
        x1 = vertices[0].x
        y1 = vertices[0].y
        x2 = vertices[1].x
        y2 = vertices[2].y
        words.append((text.description, [x1, y1, x2, y2]))

    # ★ 인접 단어 병합 (PaddleOCR 스타일) - gap 증가로 더 적극적 병합
    # horizontal_gap: 35→50 (숲+프린트 병합), vertical_threshold: 15→20 (한글 높이 차이)
    results = merge_adjacent_words(words, horizontal_gap=50, vertical_threshold=20)

    elapsed = time.time() - start_time
    print(f"[Vision OCR] Detected {len(texts)-1} words → Merged to {len(results)} groups in {elapsed:.2f}s")

    return results

# AI 모델 설정
AI_MODELS = {
    "ollama": {
        "models": ["qwen2.5vl:latest", "llava:latest", "bakllava:latest"],
        "default": "qwen2.5vl:latest"
    },
    "claude_sije": {
        "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "default": "claude-sonnet-4-20250514"
    },
    "claude_seam": {
        "models": ["claude-haiku-4-20250514", "claude-3-5-haiku-20241022", "claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        "default": "claude-haiku-4-20250514"
    },
    "openai": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "default": "gpt-4o-mini"
    },
    "gemini": {
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
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

# [레거시] 하드코딩된 약어 - 이제 garment_dict.json의 abbr 필드로 대체됨
# ABBREVIATIONS = {
#     "Garment Matching": "G.M",
#     "G Matching": "G.M",
#     "Accessory Matching": "A.M",
#     "A Matching": "A.M",
#     "Consumption": "Cons.",
#     "NaturalZipper": "Nat.Zip",
#     "Natural Zipper": "Nat.Zip",
#     "FrontZipper": "Fr.Zip",
#     "Front Zipper": "Fr.Zip",
#     "SidePocket": "Side Pkt",
#     "Side Pocket": "Side Pkt",
#     "Factory Handling": "Fact.Hdl",
#     "Hood/Hem": "Hd/Hm",
# }

# 의류 전문 용어 사전 파일 경로
GARMENT_DICT_FILE = os.path.join(os.path.dirname(__file__), "garment_dict.json")

def load_garment_dict():
    """JSON 파일에서 용어 사전 로드"""
    try:
        if os.path.exists(GARMENT_DICT_FILE):
            with open(GARMENT_DICT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load garment_dict.json: {e}")
    # 기본 사전 반환 (파일 없을 경우)
    return {
        "english": {"남성": "Men's", "여성": "Women's"},
        "vietnamese": {"남성": "Nam", "여성": "Nữ"},
        "chinese": {"남성": "男士", "여성": "女士"},
        "indonesian": {"남성": "Pria", "여성": "Wanita"},
        "bengali": {"남성": "পুরুষ", "여성": "মহিলা"}
    }

def save_garment_dict(data):
    """용어 사전을 JSON 파일로 저장"""
    try:
        with open(GARMENT_DICT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Error] Failed to save garment_dict.json: {e}")
        return False

# 용어 사전 로드 (전역)
GARMENT_DICT = load_garment_dict()

# OCR 엔진 초기화 (싱글톤)
ocr_engine = None

def get_ocr_engine():
    global ocr_engine
    if ocr_engine is None:
        import os
        # ★ 연결 체크 비활성화 (속도 향상)
        os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

        print("[init] PaddleOCR engine (korean, optimized)...")
        ocr_engine = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,  # 문서 방향 분류 끄기
            use_doc_unwarping=False,             # 문서 왜곡 보정 끄기
            use_textline_orientation=False,      # 텍스트라인 방향 분류 끄기
            text_det_thresh=0.2,                 # 텍스트 감지 임계값 (신규 파라미터)
            text_det_box_thresh=0.4,             # 박스 임계값 (신규 파라미터)
            device='cpu',                        # CPU 명시 (GPU 체크 생략)
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
                # 모든 텍스트 추출 (영어 포함) - 겹침 감지에 사용
                # has_korean 플래그로 번역 대상 여부 구분
                has_korean = any('\uac00' <= c <= '\ud7a3' for c in text_str)
                bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
                texts.append({
                    "bbox": bbox,
                    "text": text_str,
                    "confidence": float(score) if score else 1.0,
                    "has_korean": has_korean  # 한글 포함 여부 플래그
                })

    return texts


def get_ocr_results_batch(image_paths):
    """배치 OCR - 여러 이미지를 한번에 처리 (속도 향상)
    
    Args:
        image_paths: 이미지 경로 리스트
        
    Returns:
        list: 각 이미지별 OCR 결과 리스트
    """
    import time
    batch_start = time.time()
    
    ocr = get_ocr_engine()
    
    # 모든 이미지를 RGB numpy 배열로 변환
    load_start = time.time()
    images_rgb = []
    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images_rgb.append(img_rgb)
    load_time = time.time() - load_start
    print(f"[Batch OCR] Image loading: {load_time:.2f}s for {len(images_rgb)} images", flush=True)
    
    # 배치 OCR 실행
    ocr_start = time.time()
    print(f"[Batch OCR] Running OCR on {len(images_rgb)} images...", flush=True)
    results = ocr.predict(images_rgb)
    ocr_time = time.time() - ocr_start
    print(f"[Batch OCR] OCR inference: {ocr_time:.2f}s", flush=True)
    
    # 결과 파싱
    all_texts = []
    for page_idx, result in enumerate(results if results else []):
        texts = []
        rec_texts = []
        rec_scores = []
        dt_polys = []
        
        # OCRResult 객체 처리
        if hasattr(result, 'rec_texts'):
            rec_texts = result.rec_texts or []
            rec_scores = result.rec_scores or []
            dt_polys = result.dt_polys if hasattr(result, 'dt_polys') and result.dt_polys is not None else []
        elif isinstance(result, dict):
            rec_texts = result.get('rec_text', result.get('rec_texts', []))
            rec_scores = result.get('rec_score', result.get('rec_scores', []))
            dt_polys = result.get('dt_polys', [])
        
        if isinstance(rec_texts, str):
            rec_texts = [rec_texts]
            rec_scores = [rec_scores]
            dt_polys = [dt_polys]
        
        for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
            text_str = str(text)
            has_korean = any('\uac00' <= c <= '\ud7a3' for c in text_str)
            bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
            texts.append({
                "bbox": bbox,
                "text": text_str,
                "confidence": float(score) if score else 1.0,
                "has_korean": has_korean
            })
        
        all_texts.append(texts)
        print(f"  [Page {page_idx+1}] Found {len(texts)} texts", flush=True)
    
    total_time = time.time() - batch_start
    total_texts = sum(len(t) for t in all_texts)
    print(f"[Batch OCR] TOTAL: {total_time:.2f}s for {len(image_paths)} pages, {total_texts} texts", flush=True)

    return all_texts


def get_ocr_results_with_engine(image_paths, ocr_engine='paddleocr'):
    """OCR 엔진을 선택하여 배치 OCR 수행

    Args:
        image_paths: 이미지 경로 리스트
        ocr_engine: 'paddleocr' 또는 'google_vision'

    Returns:
        list: 각 이미지별 OCR 결과 리스트
    """
    import time

    if ocr_engine == 'google_vision':
        # Google Cloud Vision API 사용
        print(f"[Vision OCR] Processing {len(image_paths)} pages with Google Vision...", flush=True)
        batch_start = time.time()
        all_texts = []

        for page_idx, img_path in enumerate(image_paths):
            vision_result = ocr_with_google_vision(img_path)

            if vision_result is None:
                # Vision 실패 시 PaddleOCR fallback
                print(f"[Vision OCR] Page {page_idx+1}: Falling back to PaddleOCR", flush=True)
                paddle_result = get_ocr_results(img_path)
                all_texts.append(paddle_result)
            else:
                # Vision 결과를 표준 형식으로 변환
                texts = []
                for item in vision_result:
                    bbox, (text, confidence) = item
                    has_korean = any('\uac00' <= c <= '\ud7a3' for c in text)
                    texts.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": confidence,
                        "has_korean": has_korean
                    })
                all_texts.append(texts)
                print(f"  [Page {page_idx+1}] Vision: {len(texts)} texts", flush=True)

        total_time = time.time() - batch_start
        total_texts = sum(len(t) for t in all_texts)
        print(f"[Vision OCR] TOTAL: {total_time:.2f}s for {len(image_paths)} pages, {total_texts} texts", flush=True)
        return all_texts
    else:
        # PaddleOCR 배치 처리 (기존 함수 사용)
        return get_ocr_results_batch(image_paths)


def translate_with_dict(korean_text, target_lang):
    """사전 기반 번역 (fallback용)"""
    result = korean_text
    if target_lang in GARMENT_DICT:
        # 긴 용어부터 처리 (복합어 우선)
        sorted_terms = sorted(GARMENT_DICT[target_lang].items(), key=lambda x: len(x[0]), reverse=True)
        for kor, term_data in sorted_terms:
            if kor in result:
                # 새 구조: term_data = {"full": "번역", "abbr": "약어"}
                if isinstance(term_data, dict):
                    translation = term_data.get("full", "")
                else:
                    # 레거시 호환: 단순 문자열
                    translation = term_data
                if translation:
                    result = result.replace(kor, translation)
    return result


def apply_dict_preprocess(korean_text, target_lang):
    """AI 번역 전 사전 용어를 플레이스홀더로 대체 (Placeholder 방식)

    Args:
        korean_text: 원본 한글 텍스트
        target_lang: 대상 언어 (english, vietnamese 등)

    Returns:
        tuple: (플레이스홀더가 적용된 텍스트, 플레이스홀더 매핑 딕셔너리)

    Example:
        "23SS 행거루프 요척" → ("23SS 행거루프 <<TERM_1>>", {"<<TERM_1>>": "Consumption"})
    """
    if target_lang not in GARMENT_DICT:
        return korean_text, {}

    result = korean_text
    placeholder_map = {}  # {"<<TERM_1>>": "Consumption", ...}
    dict_terms = GARMENT_DICT[target_lang]

    # 긴 용어부터 처리 (복합어 우선: "후드탈부착" > "후드")
    sorted_terms = sorted(dict_terms.items(), key=lambda x: len(x[0]), reverse=True)

    term_idx = 1
    for korean_term, term_data in sorted_terms:
        if korean_term in result:
            placeholder = f"<<TERM_{term_idx}>>"
            result = result.replace(korean_term, placeholder)
            # 새 구조: term_data = {"full": "번역", "abbr": "약어"}
            if isinstance(term_data, dict):
                placeholder_map[placeholder] = term_data.get("full", "")
            else:
                # 레거시 호환: 단순 문자열인 경우
                placeholder_map[placeholder] = term_data
            term_idx += 1

    return result, placeholder_map


def detect_table_regions(image_path, max_avg_row_height=50):
    """img2table을 사용하여 테이블 영역 감지
    
    Args:
        image_path: 이미지 경로
        max_avg_row_height: 테이블로 인정할 최대 평균 행 높이 (기본 50px)
    
    Returns:
        list: 테이블 영역 bbox 리스트 [(x1, y1, x2, y2), ...]
    """
    try:
        img = Img2TableImage(src=image_path)
        tables = img.extract_tables()
        logger.info(f"[Table Detection] Found {len(tables)} raw tables")
        
        table_regions = []
        for idx, table in enumerate(tables):
            # 행 높이 계산
            if hasattr(table, 'content') and table.content:
                row_heights = []
                for row in table.content:
                    if row:
                        for cell in row:
                            if cell and hasattr(cell, 'bbox'):
                                cell_bbox = cell.bbox
                                if hasattr(cell_bbox, 'y1') and hasattr(cell_bbox, 'y2'):
                                    row_heights.append(cell_bbox.y2 - cell_bbox.y1)
                                break
                
                if row_heights:
                    avg_row_height = sum(row_heights) / len(row_heights)
                    logger.info(f"[Table Detection] Table #{idx} avg row height: {avg_row_height:.1f}px")
                    if avg_row_height > max_avg_row_height:
                        logger.info(f"[Table Detection] Table #{idx} skipped (height > {max_avg_row_height}px)")
                        continue
            
            # bbox 추출
            if hasattr(table, 'bbox'):
                bbox = table.bbox
                if hasattr(bbox, 'x1'):
                    table_regions.append((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                    logger.info(f"[Table Detection] Table #{idx} added: ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
        
        logger.info(f"[Table Detection] Final: {len(table_regions)} valid tables")
        return table_regions
    except Exception as e:
        logger.error(f"[Table Detection] Error: {e}")
        return []


def is_inside_table(bbox, table_regions):
    """텍스트 bbox가 테이블 영역 안에 있는지 확인"""
    if not table_regions:
        return False
    
    # bbox 중심점 계산
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    for (tx1, ty1, tx2, ty2) in table_regions:
        if tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2:
            return True
    return False


def restore_placeholders(translated_text, placeholder_map):
    """번역 결과에서 플레이스홀더를 사전 번역으로 복원

    Args:
        translated_text: AI가 번역한 텍스트 (플레이스홀더 포함)
        placeholder_map: 플레이스홀더 → 사전 번역 매핑

    Returns:
        str: 플레이스홀더가 사전 번역으로 대체된 최종 텍스트

    Example:
        ("23SS Hanger Loop <<TERM_1>>", {"<<TERM_1>>": "Consumption"})
        → "23SS Hanger Loop Consumption"
    """
    import re
    result = translated_text
    
    for placeholder, translation in placeholder_map.items():
        # 원본 placeholder (예: <<TERM_1>>)
        result = result.replace(placeholder, translation)
        
        # AI가 변형한 다양한 패턴도 처리
        # <<TERM_1>> 에서 숫자 추출
        match = re.search(r'TERM_(\d+)', placeholder)
        if match:
            term_num = match.group(1)
            # 다양한 변형 패턴 처리 (정확한 문자열 매칭)
            variations = [
                f"TERM_{term_num}",           # TERM_1 (꺾쇠 제거됨)
                f"TERM {term_num}",           # TERM 1 (언더스코어 제거됨)
                f"<TERM_{term_num}>",         # <TERM_1> (꺾쇠 하나만)
                f"[TERM_{term_num}]",         # [TERM_1] (대괄호로 변형)
                f"(TERM_{term_num})",         # (TERM_1) (괄호로 변형)
                f"{{TERM_{term_num}}}",       # {TERM_1} (중괄호로 변형)
                f"TERM{term_num}",            # TERM1 (언더스코어 완전 제거)
                f"Term_{term_num}",           # Term_1 (대소문자 변형)
                f"term_{term_num}",           # term_1 (소문자 변형)
            ]
            for var in variations:
                if var in result:
                    result = result.replace(var, translation)
            
            # 정규식으로 더 유연한 패턴 매칭 (공백, 특수문자 포함)
            # 예: "TERM _ 1", "TERM- 1", "TERM_1." 등
            flexible_patterns = [
                rf'<<\s*TERM[_\s-]*{term_num}\s*>>',  # << TERM_1 >> 등
                rf'<\s*TERM[_\s-]*{term_num}\s*>',    # < TERM_1 > 등
                rf'\[\s*TERM[_\s-]*{term_num}\s*\]',  # [ TERM_1 ] 등
                rf'\(\s*TERM[_\s-]*{term_num}\s*\)',  # ( TERM_1 ) 등
                rf'TERM[_\s-]*{term_num}(?![0-9])',   # TERM_1, TERM 1, TERM-1 (뒤에 숫자 없을 때만)
            ]
            for pattern in flexible_patterns:
                result = re.sub(pattern, translation, result, flags=re.IGNORECASE)
    
    return result


def apply_dict_postprocess(translated_text, original_korean, target_lang):
    """AI 번역 결과에 사전 용어 후처리 적용 (레거시 - 백업용)

    원본 한글에서 사전 용어가 있으면, 번역 결과에서 해당 부분을 사전 번역으로 교체
    Note: Placeholder 방식(apply_dict_preprocess + restore_placeholders)이 더 권장됨
    """
    if target_lang not in GARMENT_DICT:
        return translated_text

    result = translated_text
    dict_terms = GARMENT_DICT[target_lang]

    # 긴 용어부터 처리 (복합어 우선)
    sorted_terms = sorted(dict_terms.items(), key=lambda x: len(x[0]), reverse=True)

    for korean_term, correct_translation in sorted_terms:
        if korean_term in original_korean:
            # 원본에 해당 용어가 있으면, 번역 결과에서 잘못된 번역을 교체
            # 단, 이미 올바른 번역이 있으면 건너뜀
            if correct_translation not in result:
                # 흔한 오번역 패턴들을 사전 번역으로 교체
                result = result.replace(korean_term, correct_translation)

    return result


def translate_with_claude(image_path, texts, target_lang, api_key, model=None):
    """Claude API로 이미지 컨텍스트와 함께 번역 (Placeholder 방식 적용)"""
    import time
    api_start = time.time()

    print(f"[Claude] translate_with_claude called - texts: {len(texts)}, model: {model}", flush=True)
    logger.info(f"[Claude] translate_with_claude called - texts: {len(texts)}, model: {model}")
    if model is None:
        model = AI_MODELS["claude"]["default"]
    print(f"[Claude] Using model: {model}", flush=True)
    logger.info(f"[Claude] Using model: {model}")
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    translations = []

    # 한글 텍스트만 필터링 (인덱스 보존) - 영어만 있는 텍스트는 번역하지 않음
    korean_items = [(i, item) for i, item in enumerate(texts) if item.get("has_korean", True)]
    korean_list = [item["text"] for _, item in korean_items]
    korean_indices = [i for i, _ in korean_items]
    
    print(f"[Claude] Total texts: {len(texts)}, Korean texts to translate: {len(korean_list)}", flush=True)

    # ★ Placeholder 전처리: 사전 용어를 플레이스홀더로 대체
    preprocessed_list = []
    placeholder_maps = []  # 각 텍스트별 플레이스홀더 매핑 저장
    for korean_text in korean_list:
        processed_text, pmap = apply_dict_preprocess(korean_text, target_lang)
        preprocessed_list.append(processed_text)
        placeholder_maps.append(pmap)
        if pmap:
            print(f"[Claude] Preprocess: '{korean_text}' → '{processed_text}' (placeholders: {len(pmap)})", flush=True)

    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(preprocessed_list)])

    prompt = f"""This is a garment/clothing technical specification image (tech pack).
Translate ALL the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.

RULES:
- Translate EVERY item, even if it contains English or numbers
- Use FULL words only, do NOT abbreviate (e.g., "Consumption" not "Cons.", "Management" not "Mgmt.")
- Use format: "1. translation" (number + dot + space + translation)
- Do NOT skip any item
- Do NOT use markdown formatting like **bold**
- CRITICAL: <<TERM_N>> placeholders are pre-translated dictionary terms. Keep them EXACTLY as they are. Do NOT translate, modify, or replace them.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (translate ALL {len(korean_list)} items):"""

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

        print(f"[Claude] Calling API: {CLAUDE_API_URL}", flush=True)
        request_start = time.time()
        response = requests.post(
            CLAUDE_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        api_time = time.time() - request_start
        print(f"[Claude] API response status: {response.status_code} (took {api_time:.2f}s)", flush=True)
        logger.info(f"[Claude] API response status: {response.status_code} (took {api_time:.2f}s)")

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("content", [{}])[0].get("text", "").strip()
            print(f"[Claude] Raw response:\n{response_text}", flush=True)

            # 응답 파싱 (정규표현식으로 다양한 형식 지원)
            import re
            lines = response_text.split("\n")
            trans_dict = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 다양한 번호 형식 지원: "1. text", "1) text", "**1.** text", "1: text"
                match = re.match(r'^[*]*(\d+)[.*)\]:]+\s*[*]*\s*(.+)', line)
                if match:
                    idx = int(match.group(1)) - 1
                    trans = match.group(2).strip()
                    if idx < len(korean_list):
                        trans_dict[idx] = trans
                        print(f"[Claude] Parsed {idx+1}: {trans[:30]}...", flush=True)

            print(f"[Claude] Parsed {len(trans_dict)}/{len(korean_list)} translations", flush=True)

            # 결과 매핑 + 플레이스홀더 복원
            for i, item in enumerate(texts):
                if not item.get("has_korean", True):
                    # 영어 텍스트: 원본 유지 (번역하지 않음)
                    translations.append({**item, "translated": item["text"]})
                else:
                    # 한글 텍스트: 번역 결과 매핑
                    korean_idx = korean_indices.index(i)
                    if korean_idx in trans_dict:
                        translated = trans_dict[korean_idx]
                        # ★ 플레이스홀더를 사전 번역으로 복원
                        if placeholder_maps[korean_idx]:
                            translated = restore_placeholders(translated, placeholder_maps[korean_idx])
                            print(f"[Claude] Restored placeholders for item {i+1}: {translated[:50]}...", flush=True)
                    else:
                        translated = translate_with_dict(item["text"], target_lang)

                    translations.append({
                        **item,
                        "translated": translated
                    })
        else:
            print(f"[Claude] API error: {response.status_code} - {response.text}", flush=True)
            logger.error(f"[Claude] API error: {response.status_code} - {response.text}")
            # fallback: 사전 번역 (한글만), 영어는 원본 유지
            for item in texts:
                if item.get("has_korean", True):
                    translated = translate_with_dict(item["text"], target_lang)
                else:
                    translated = item["text"]  # 영어 원본 유지
                translations.append({**item, "translated": translated})

    except Exception as e:
        print(f"[Claude] Exception: {e}", flush=True)
        logger.error(f"[Claude] Exception: {e}")
        for item in texts:
            if item.get("has_korean", True):
                translated = translate_with_dict(item["text"], target_lang)
            else:
                translated = item["text"]  # 영어 원본 유지
            translations.append({**item, "translated": translated})

    total_time = time.time() - api_start
    print(f"[Claude] TOTAL: {total_time:.2f}s for {len(texts)} texts ({len(korean_list)} Korean)", flush=True)
    logger.info(f"[Claude] TOTAL: {total_time:.2f}s for {len(texts)} texts ({len(korean_list)} Korean)")
    return translations


def translate_with_openai(image_path, texts, target_lang, api_key, model=None):
    """OpenAI GPT-4 Vision API로 이미지 컨텍스트와 함께 번역 (Placeholder 방식 적용)"""
    if model is None:
        model = AI_MODELS["openai"]["default"]
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    translations = []

    # 한글 텍스트만 필터링 (인덱스 보존) - 영어만 있는 텍스트는 번역하지 않음
    korean_items = [(i, item) for i, item in enumerate(texts) if item.get("has_korean", True)]
    korean_list = [item["text"] for _, item in korean_items]
    korean_indices = [i for i, _ in korean_items]
    
    print(f"[OpenAI] Total texts: {len(texts)}, Korean texts to translate: {len(korean_list)}", flush=True)

    # ★ Placeholder 전처리: 사전 용어를 플레이스홀더로 대체
    preprocessed_list = []
    placeholder_maps = []
    for korean_text in korean_list:
        processed_text, pmap = apply_dict_preprocess(korean_text, target_lang)
        preprocessed_list.append(processed_text)
        placeholder_maps.append(pmap)

    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(preprocessed_list)])

    prompt = f"""This is a garment/clothing technical specification image (tech pack).
Translate ALL the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.

RULES:
- Translate EVERY item, even if it contains English or numbers
- Use FULL words only, do NOT abbreviate (e.g., "Consumption" not "Cons.", "Management" not "Mgmt.")
- Use format: "1. translation" (number + dot + space + translation)
- Do NOT skip any item
- Do NOT use markdown formatting like **bold**
- CRITICAL: <<TERM_N>> placeholders are pre-translated dictionary terms. Keep them EXACTLY as they are. Do NOT translate, modify, or replace them.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (translate ALL {len(korean_list)} items):"""

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

            # 결과 매핑 + 플레이스홀더 복원
            for i, item in enumerate(texts):
                if not item.get("has_korean", True):
                    # 영어 텍스트: 원본 유지 (번역하지 않음)
                    translations.append({**item, "translated": item["text"]})
                else:
                    # 한글 텍스트: 번역 결과 매핑
                    korean_idx = korean_indices.index(i)
                    if korean_idx in trans_dict:
                        translated = trans_dict[korean_idx]
                        # ★ 플레이스홀더를 사전 번역으로 복원
                        if placeholder_maps[korean_idx]:
                            translated = restore_placeholders(translated, placeholder_maps[korean_idx])
                    else:
                        translated = translate_with_dict(item["text"], target_lang)

                    translations.append({
                        **item,
                        "translated": translated
                    })
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            # fallback: 사전 번역 (한글만), 영어는 원본 유지
            for item in texts:
                if item.get("has_korean", True):
                    translated = translate_with_dict(item["text"], target_lang)
                else:
                    translated = item["text"]
                translations.append({**item, "translated": translated})

    except Exception as e:
        print(f"OpenAI API error: {e}")
        for item in texts:
            if item.get("has_korean", True):
                translated = translate_with_dict(item["text"], target_lang)
            else:
                translated = item["text"]
            translations.append({**item, "translated": translated})

    return translations


# ★ 배치 청크 크기: 순서 유지를 위해 100개씩 나눠서 처리
BATCH_CHUNK_SIZE = 200


def _translate_chunk_with_gemini(chunk_texts, lang_config, api_key, model, chunk_idx, total_chunks):
    """청크 단위로 Gemini API 호출 (내부 함수)"""
    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(chunk_texts)])

    prompt = f"""This is a garment/clothing technical specification document (tech pack).
Translate ALL the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.
Keep translations SHORT and professional. Only respond with numbered translations in {lang_config['prompt_lang']}.
There are {len(chunk_texts)} items. Translate ALL of them in the EXACT same order.
IMPORTANT: Keep <<TERM_N>> placeholders exactly as they are (do not translate them).
IMPORTANT: Respond with the EXACT SAME numbering (1 to {len(chunk_texts)}).
IMPORTANT: DO NOT add any explanations, notes, or comments in parentheses. Translation ONLY.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (numbering 1-{len(chunk_texts)}, SHORT answers only, NO explanations):"""

    url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 8192}
    }

    print(f"[Chunk {chunk_idx+1}/{total_chunks}] Sending {len(chunk_texts)} texts to Gemini...", flush=True)

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        result = response.json()
        response_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        # 응답 파싱
        lines = response_text.split("\n")
        chunk_dict = {}
        for line in lines:
            line = line.strip()
            match = re.match(r'^[\*\-\s]*(\d+)[\.\)\:\*\s]+(.+)', line)
            if match:
                try:
                    idx = int(match.group(1)) - 1
                    trans = match.group(2).strip().strip('*').strip()
                    if 0 <= idx < len(chunk_texts) and trans:
                        chunk_dict[idx] = trans
                except ValueError:
                    continue

        print(f"[Chunk {chunk_idx+1}/{total_chunks}] Got {len(chunk_dict)}/{len(chunk_texts)} translations", flush=True)
        return chunk_dict
    else:
        print(f"[Chunk {chunk_idx+1}/{total_chunks}] API error: {response.status_code}", flush=True)
        return {}


def translate_batch_with_gemini(all_pages_texts, target_lang, api_key, model=None):
    """Google Gemini API로 모든 페이지의 텍스트를 청크 단위로 번역 (배치 모드, Placeholder 방식)

    Args:
        all_pages_texts: [{page_idx: int, texts: [{text, bbox}, ...]}, ...]
        target_lang: 번역 대상 언어
        api_key: Gemini API 키
        model: Gemini 모델명

    Returns:
        {page_idx: [translated_texts], ...}
    """
    import time
    batch_start = time.time()

    if model is None:
        model = AI_MODELS["gemini"]["default"]
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    total_pages = len(all_pages_texts)
    total_texts = sum(len(p["texts"]) for p in all_pages_texts)
    print(f"[Gemini Batch] Starting batch translation: {total_pages} pages, {total_texts} texts", flush=True)

    # 모든 페이지의 텍스트를 하나의 리스트로 합침 (페이지 구분 포함)
    # ★ 모든 텍스트를 번역 API에 전송 (이전 작동 버전과 동일)
    all_korean = []
    all_placeholder_maps = []  # ★ 각 텍스트별 플레이스홀더 매핑
    page_text_counts = []  # ★ 페이지별 텍스트 개수 (결과 분배용)

    for page_data in all_pages_texts:
        page_texts = page_data["texts"]
        page_text_counts.append(len(page_texts))
        for item in page_texts:
            # ★ Placeholder 전처리
            processed_text, pmap = apply_dict_preprocess(item["text"], target_lang)
            all_korean.append(processed_text)
            all_placeholder_maps.append(pmap)

    if not all_korean:
        return {page_data["page_idx"]: [] for page_data in all_pages_texts}

    # ★ 청크 단위로 나눠서 번역 (순서 유지를 위해)
    trans_dict = {}
    total_chunks = (len(all_korean) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE
    print(f"[Gemini Batch] Splitting into {total_chunks} chunks of {BATCH_CHUNK_SIZE} texts each", flush=True)

    try:
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * BATCH_CHUNK_SIZE
            chunk_end = min(chunk_start + BATCH_CHUNK_SIZE, len(all_korean))
            chunk_texts = all_korean[chunk_start:chunk_end]

            # 청크 번역
            chunk_result = _translate_chunk_with_gemini(
                chunk_texts, lang_config, api_key, model, chunk_idx, total_chunks
            )

            # 결과를 전체 인덱스로 변환하여 저장
            for local_idx, trans in chunk_result.items():
                global_idx = chunk_start + local_idx
                trans_dict[global_idx] = trans

            # ★ 진행 상황 업데이트 (청크 완료 후)
            update_progress("번역", chunk_end, len(all_korean),
                          f"{chunk_end}/{len(all_korean)}개 번역 완료 (Gemini 배치)")

        print(f"[Gemini Batch] Total translations: {len(trans_dict)}/{len(all_korean)}", flush=True)

        # 페이지별로 결과 분배 (이전 작동 버전과 동일한 단순 로직)
        result_by_page = {}
        current_idx = 0

        for page_idx, page_data in enumerate(all_pages_texts):
            page_translations = []
            page_text_count = page_text_counts[page_idx]

            for i in range(page_text_count):
                item = page_data["texts"][i]
                if current_idx in trans_dict:
                    translated = trans_dict[current_idx]
                    pmap = all_placeholder_maps[current_idx]
                    # ★ 플레이스홀더를 사전 번역으로 복원
                    if pmap:
                        translated = restore_placeholders(translated, pmap)
                    # ★ 남은 플레이스홀더 제거 (Gemini가 잘못 복사한 경우)
                    if '<<TERM_' in translated:
                        translated = re.sub(r'<<TERM_\d+>>', '', translated).strip()
                else:
                    translated = item["text"]

                page_translations.append({
                    **item,
                    "translated": translated
                })
                current_idx += 1

            result_by_page[page_data["page_idx"]] = page_translations

        total_time = time.time() - batch_start
        print(f"[Gemini Batch] TOTAL: {total_time:.2f}s for {total_pages} pages, {total_texts} texts ({total_chunks} API calls)", flush=True)
        return result_by_page

    except Exception as e:
        print(f"Gemini Batch API error: {e}", flush=True)
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
    """Google Gemini API로 이미지 컨텍스트와 함께 번역 (Placeholder 방식 적용)"""
    if model is None:
        model = AI_MODELS["gemini"]["default"]
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    translations = []

    # 모든 한글 텍스트를 한 번에 번역 요청
    korean_list = [item["text"] for item in texts]

    # ★ Placeholder 전처리: 사전 용어를 플레이스홀더로 대체
    preprocessed_list = []
    placeholder_maps = []
    for korean_text in korean_list:
        processed_text, pmap = apply_dict_preprocess(korean_text, target_lang)
        preprocessed_list.append(processed_text)
        placeholder_maps.append(pmap)

    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(preprocessed_list)])

    prompt = f"""This is a garment/clothing technical specification image (tech pack).
Translate ALL the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.

RULES:
- Translate EVERY item, even if it contains English or numbers
- Use FULL words only, do NOT abbreviate (e.g., "Consumption" not "Cons.", "Management" not "Mgmt.")
- Use format: "1. translation" (number + dot + space + translation)
- Do NOT skip any item
- Do NOT use markdown formatting like **bold**
- CRITICAL: <<TERM_N>> placeholders are pre-translated dictionary terms. Keep them EXACTLY as they are. Do NOT translate, modify, or replace them.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (translate ALL {len(korean_list)} items):"""

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

            # 결과 매핑 + 플레이스홀더 복원
            for i, item in enumerate(texts):
                if i in trans_dict:
                    translated = trans_dict[i]
                    # ★ 플레이스홀더를 사전 번역으로 복원
                    if placeholder_maps[i]:
                        translated = restore_placeholders(translated, placeholder_maps[i])
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


def translate_pages_parallel(pages_data, target_lang, ai_engine, api_key, model, max_workers=3):
    """병렬 번역 - 여러 페이지를 동시에 번역 (Claude, OpenAI용)
    
    Args:
        pages_data: [{"page_idx": 0, "img_path": "...", "texts": [...]}, ...]
        target_lang: 대상 언어
        ai_engine: AI 엔진 (claude, openai)
        api_key: API 키
        model: 모델명
        max_workers: 동시 처리 스레드 수 (기본 3, API rate limit 고려)
        
    Returns:
        dict: {page_idx: translations, ...}
    """
    import time
    parallel_start = time.time()
    results = {}
    page_times = {}  # 각 페이지별 소요 시간
    
    def translate_single_page(page_data):
        """단일 페이지 번역 (스레드에서 실행)"""
        page_start = time.time()
        page_idx = page_data["page_idx"]
        img_path = page_data["img_path"]
        texts = page_data["texts"]
        
        if not texts:
            return page_idx, [], 0
        
        try:
            translations = translate_with_vlm(img_path, texts, target_lang, ai_engine, api_key, model)
            elapsed = time.time() - page_start
            print(f"  [Parallel] Page {page_idx+1} done - {len(translations)} texts in {elapsed:.2f}s", flush=True)
            return page_idx, translations, elapsed
        except Exception as e:
            elapsed = time.time() - page_start
            print(f"  [Parallel] Page {page_idx+1} ERROR in {elapsed:.2f}s: {e}", flush=True)
            # 에러 시 원본 텍스트 반환
            return page_idx, [{"bbox": t["bbox"], "text": t["text"], "translated": t["text"], 
                             "has_korean": t.get("has_korean", True)} for t in texts], elapsed
    
    total_texts = sum(len(p["texts"]) for p in pages_data)
    print(f"[Parallel Translation] Starting {len(pages_data)} pages ({total_texts} texts) with {max_workers} workers...", flush=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 페이지 번역 작업 제출
        futures = {executor.submit(translate_single_page, pd): pd["page_idx"] for pd in pages_data}
        
        # 완료되는 순서대로 결과 수집
        for future in as_completed(futures):
            page_idx, translations, elapsed = future.result()
            results[page_idx] = translations
            page_times[page_idx] = elapsed
    
    total_time = time.time() - parallel_start
    avg_time = sum(page_times.values()) / len(page_times) if page_times else 0
    print(f"[Parallel Translation] TOTAL: {total_time:.2f}s (avg per page: {avg_time:.2f}s, workers: {max_workers})", flush=True)
    return results


def translate_with_vlm(image_path, texts, target_lang, ai_engine="ollama", api_key=None, model=None):
    """VLM으로 이미지 컨텍스트와 함께 번역 (Ollama, Claude, GPT-4, Gemini)"""

    # Claude API 선택 시
    if ai_engine in ("claude_sije", "claude_seam") and api_key:
        return translate_with_claude(image_path, texts, target_lang, api_key, model)

    # OpenAI GPT-4 API 선택 시
    if ai_engine == "openai" and api_key:
        return translate_with_openai(image_path, texts, target_lang, api_key, model)

    # Google Gemini API 선택 시
    if ai_engine == "gemini" and api_key:
        return translate_with_gemini(image_path, texts, target_lang, api_key, model)

    # 기본: Ollama (Placeholder 방식 적용)
    if model is None:
        model = AI_MODELS["ollama"]["default"]
    lang_config = LANGUAGE_CONFIG.get(target_lang, LANGUAGE_CONFIG["english"])

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    translations = []

    # 모든 한글 텍스트를 한 번에 번역 요청
    korean_list = [item["text"] for item in texts]

    # ★ Placeholder 전처리: 사전 용어를 플레이스홀더로 대체
    preprocessed_list = []
    placeholder_maps = []
    for korean_text in korean_list:
        processed_text, pmap = apply_dict_preprocess(korean_text, target_lang)
        preprocessed_list.append(processed_text)
        placeholder_maps.append(pmap)

    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(preprocessed_list)])

    prompt = f"""This is a garment/clothing technical specification image (tech pack).
Translate ALL the following Korean texts to {lang_config['prompt_lang']}. These are garment industry terms.

RULES:
- Translate EVERY item, even if it contains English or numbers
- Use FULL words only, do NOT abbreviate (e.g., "Consumption" not "Cons.", "Management" not "Mgmt.")
- Use format: "1. translation" (number + dot + space + translation)
- Do NOT skip any item
- Do NOT use markdown formatting like **bold**
- CRITICAL: <<TERM_N>> placeholders are pre-translated dictionary terms. Keep them EXACTLY as they are. Do NOT translate, modify, or replace them.

Korean texts:
{korean_joined}

{lang_config['prompt_lang']} translations (translate ALL {len(korean_list)} items):"""

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

            # 응답 파싱 (정규표현식으로 다양한 형식 지원)
            import re
            lines = response_text.split("\n")
            trans_dict = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 다양한 번호 형식 지원: "1. text", "1) text", "**1.** text", "1: text"
                match = re.match(r'^[*]*(\d+)[.*)\]:]+\s*[*]*\s*(.+)', line)
                if match:
                    idx = int(match.group(1)) - 1
                    trans = match.group(2).strip()
                    if idx < len(korean_list):
                        trans_dict[idx] = trans

            # 결과 매핑 + 플레이스홀더 복원
            for i, item in enumerate(texts):
                if i in trans_dict:
                    translated = trans_dict[i]
                    # ★ 플레이스홀더를 사전 번역으로 복원
                    if placeholder_maps[i]:
                        translated = restore_placeholders(translated, placeholder_maps[i])
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

    # ★ 지우기 영역 설정 (Google Vision용)
    # margin_x = 3: X축 좌우 3px 확장 (한글 잔여 제거)
    # margin_y = -2: Y축 상하 2px 축소 (테이블 선 보호)
    margin_x = 3
    margin_y = -2
    x_min_ext = max(0, x_min - margin_x)
    y_min_ext = max(0, y_min - margin_y)  # y_min + 3 (아래로 축소)
    x_max_ext = min(width, x_max + margin_x)
    y_max_ext = min(height, y_max + margin_y)  # y_max - 3 (위로 축소)

    # 배경색 샘플링
    bg_color = sample_background_color(img, bbox, height, width)

    # ★ bbox 내부만 배경색으로 채우기 (선은 bbox 바깥이므로 안전)
    cv2.rectangle(img, (x_min_ext, y_min_ext), (x_max_ext, y_max_ext), bg_color, -1)

    return img, bg_color


def sample_background_color(img, bbox, height, width):
    """bbox 내부에서 가장 많이 등장하는 색상을 배경색으로 판단

    원리: 텍스트보다 배경 픽셀이 더 많으므로 최빈값 = 배경색
    """
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    # bbox 내부 픽셀 샘플링
    samples = []
    for y in range(max(0, y_min), min(height, y_max), 2):
        for x in range(max(0, x_min), min(width, x_max), 2):
            pixel = tuple(img[y, x].tolist())
            samples.append(pixel)

    if samples:
        # 가장 많이 등장하는 색상 = 배경색
        most_common = Counter(samples).most_common(1)[0][0]
        return most_common

    return (255, 255, 255)  # 기본: 흰색


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


def check_bbox_overlap(bbox1, bbox2):
    """두 bbox가 겹치는지 확인

    Args:
        bbox1: (x, y, width, height) 형태의 튜플
        bbox2: (x, y, width, height) 형태의 튜플

    Returns:
        bool: 겹치면 True
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 겹침 없음 조건 (하나라도 만족하면 겹치지 않음)
    if x1 + w1 <= x2 or x2 + w2 <= x1:  # 좌우로 분리
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:  # 상하로 분리
        return False
    return True


def abbreviate_text(text, used_abbreviations, target_lang="english"):
    """긴 텍스트를 약어로 축약 (사전 기반)

    Args:
        text: 원본 텍스트
        used_abbreviations: 사용된 약어 추적용 set (수정됨)
        target_lang: 대상 언어 (사전에서 약어 조회용)

    Returns:
        str: 축약된 텍스트
    """
    result = text
    
    # 사전에서 약어 조회
    if target_lang in GARMENT_DICT:
        lang_dict = GARMENT_DICT[target_lang]
        for korean_term, term_data in lang_dict.items():
            if isinstance(term_data, dict):
                full_text = term_data.get("full", "")
                abbr = term_data.get("abbr", "")
                if abbr and full_text in result:
                    result = result.replace(full_text, abbr)
                    used_abbreviations.add((abbr, full_text))  # (약어, 원문) 저장
    
    return result


def find_bottom_empty_area(image_height, all_bboxes, required_height=25):
    """이미지 하단에서 빈 공간 찾기

    Args:
        image_height: 이미지 전체 높이
        all_bboxes: 모든 텍스트 bbox 리스트 [(x, y, w, h), ...]
        required_height: 필요한 최소 높이 (기본 25px)

    Returns:
        int or None: 범례를 넣을 y좌표, 공간 없으면 None
    """
    if not all_bboxes:
        return image_height - required_height - 5

    # 모든 bbox 중 가장 아래 y좌표 찾기
    max_y = 0
    for bbox in all_bboxes:
        y = bbox[1] + bbox[3]  # y + height
        if y > max_y:
            max_y = y

    # 하단 여백이 required_height 이상이면 사용 가능
    if image_height - max_y >= required_height + 10:  # 10px 추가 마진
        return max_y + 5  # 마지막 텍스트 아래 5px

    return None  # 공간 없음


def render_legend(draw, used_abbreviations, image_width, legend_y, font_size=8):
    """범례를 이미지 하단 중앙에 렌더링

    Args:
        draw: PIL ImageDraw 객체
        used_abbreviations: {(약어, 원문), ...} set
        image_width: 이미지 너비
        legend_y: 범례를 넣을 y좌표
        font_size: 폰트 크기 (기본 8)
    """
    if not used_abbreviations:
        return

    # 범례 텍스트 생성: "* G.M=Garment Matching, A.M=Accessory Matching"
    legend_parts = [f"{abbr}={full}" for abbr, full in sorted(used_abbreviations)]
    legend_text = "* " + ", ".join(legend_parts)

    # 폰트 로드
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # 텍스트 너비 계산
    text_bbox = draw.textbbox((0, 0), legend_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    # 중앙 정렬
    x = (image_width - text_width) // 2

    # 회색으로 작게 렌더링
    draw.text((x, legend_y), legend_text, fill=(128, 128, 128), font=font)


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
        adjusted_font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        try:
            adjusted_font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
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


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def parse_hex_color(value, fallback=(0, 0, 0)):
    if not value:
        return fallback
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(value)
    color = str(value).strip().lstrip("#")
    if len(color) == 3:
        color = "".join([c * 2 for c in color])
    if len(color) != 6:
        return fallback
    try:
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def load_memo_font(font_size, bold=False):
    font_size = max(6, font_size)
    font_candidates = []
    if bold:
        font_candidates.extend(["arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"])
    font_candidates.extend(["malgun.ttf", "C:/Windows/Fonts/malgun.ttf"])

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text_to_lines(draw, text, font, max_width):
    max_width = max(1, int(max_width))

    def text_width(value):
        bbox = draw.textbbox((0, 0), value, font=font, anchor="lt")
        return bbox[2] - bbox[0]

    lines = []
    paragraphs = text.splitlines() or [""]
    for paragraph in paragraphs:
        if paragraph == "":
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if text_width(candidate) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
            if text_width(word) <= max_width:
                current = word
            else:
                chunk = ""
                for ch in word:
                    chunk_candidate = chunk + ch
                    if text_width(chunk_candidate) <= max_width or not chunk:
                        chunk = chunk_candidate
                    else:
                        lines.append(chunk)
                        chunk = ch
                current = chunk

        lines.append(current)
    return lines


def render_memos_on_image(img, memos):
    if not memos:
        return img

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for memo in memos:
        if not isinstance(memo, dict):
            continue
        text = str(memo.get("text", "") or "").strip()
        if not text:
            continue

        style = memo.get("style") or {}
        font_size = safe_int(style.get("fontSize"), 14)
        font = load_memo_font(font_size, bold=to_bool(style.get("bold"), False))
        padding = max(0, safe_int(style.get("padding"), 6))
        target_width = max(80, safe_int(style.get("width"), 220))
        target_width = min(target_width, base.size[0])
        opacity = safe_float(style.get("opacity"), 1.0)
        if opacity > 1:
            opacity = opacity / 100.0
        opacity = clamp(opacity, 0.0, 1.0)

        background_enabled = to_bool(style.get("backgroundEnabled"), True)
        bg_color = parse_hex_color(style.get("backgroundColor"), (255, 255, 255))
        text_color = parse_hex_color(style.get("color"), (17, 17, 17))
        border_color = parse_hex_color(style.get("borderColor"), (0, 0, 0))
        border_width = max(0, safe_int(style.get("borderWidth"), 1))

        text_area_width = max(1, target_width - padding * 2)
        lines = wrap_text_to_lines(draw, text, font, text_area_width)
        if not lines:
            lines = [""]

        sample_bbox = draw.textbbox((0, 0), "Ag", font=font, anchor="lt")
        line_height = max(1, sample_bbox[3] - sample_bbox[1])
        max_line_width = 0
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font, anchor="lt")
            max_line_width = max(max_line_width, line_bbox[2] - line_bbox[0])

        box_width = max(target_width, max_line_width + padding * 2)
        box_width = min(box_width, base.size[0])
        box_height = line_height * len(lines) + padding * 2
        box_height = min(box_height, base.size[1])

        x_ratio = clamp(safe_float(memo.get("x"), 0.0), 0.0, 1.0)
        y_ratio = clamp(safe_float(memo.get("y"), 0.0), 0.0, 1.0)
        x = int(x_ratio * base.size[0])
        y = int(y_ratio * base.size[1])

        if x + box_width > base.size[0]:
            x = max(0, base.size[0] - box_width)
        if y + box_height > base.size[1]:
            y = max(0, base.size[1] - box_height)

        alpha = int(255 * opacity)
        if background_enabled:
            draw.rectangle(
                (x, y, x + box_width, y + box_height),
                fill=(bg_color[0], bg_color[1], bg_color[2], alpha)
            )

        if border_width > 0:
            for i in range(border_width):
                draw.rectangle(
                    (x + i, y + i, x + box_width - i, y + box_height - i),
                    outline=(border_color[0], border_color[1], border_color[2], alpha)
                )

        text_alpha = int(255 * opacity)
        text_fill = (text_color[0], text_color[1], text_color[2], text_alpha)
        cursor_y = y + padding
        for line in lines:
            draw.text((x + padding, cursor_y), line, font=font, fill=text_fill, anchor="lt")
            cursor_y += line_height

    return Image.alpha_composite(base, overlay)


def replace_text_in_image(image_path, translations, output_path, target_lang="english", memos=None):
    """이미지에서 한글 영역을 지우고 번역된 텍스트로 교체 - v1.8.2 (영어 텍스트 유지, 겹침 감지용 포함)"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # ★ 테이블 영역 감지 (중앙 정렬 적용 여부 판단용)
    table_regions = detect_table_regions(image_path)

    # 1단계: 한글 텍스트 영역만 배경색으로 지우기 (영어는 원본 유지)
    bg_colors = {}
    for i, item in enumerate(translations):
        if item.get("has_korean", True):  # 한글 텍스트만 erase
            bbox = item["bbox"]
            img, bg_color = erase_text_region(img, bbox)
            bg_colors[i] = bg_color
        else:
            bg_colors[i] = (255, 255, 255)  # 영어 텍스트는 erase 안 함

    # 2단계: 텍스트 정보 사전 계산 (겹침 감지용)
    font_sizes = [13, 12, 11, 10, 9, 8, 7]  # 폰트 크기 약간 증가
    text_render_info = []

    img_pil_temp = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw_temp = ImageDraw.Draw(img_pil_temp)

    for i, item in enumerate(translations):
        bbox = item["bbox"]
        translated_text = item["translated"]

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)

        x = int(min(xs))
        y = int(min(ys))

        font = None
        text_width = 0
        for size in font_sizes:
            try:
                font = ImageFont.truetype("malgun.ttf", size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
                except:
                    font = ImageFont.load_default()
                    break

            text_bbox_size = draw_temp.textbbox((0, 0), translated_text, font=font, anchor="lt")
            text_width = text_bbox_size[2] - text_bbox_size[0]
            selected_text_height = text_bbox_size[3] - text_bbox_size[1]

            if selected_text_height <= box_height:  # 셀 높이에 맞춤 (클리핑으로 경계 처리)
                break

        text_bbox_actual = draw_temp.textbbox((0, 0), translated_text, font=font, anchor="lt")
        actual_text_width = text_bbox_actual[2] - text_bbox_actual[0]
        actual_text_height = text_bbox_actual[3] - text_bbox_actual[1]
        text_top_offset = text_bbox_actual[1]  # textbbox의 top offset (글리프 상단까지 거리)

        # ★ 스케일링 반영: 텍스트가 셀보다 크면 리사이즈된 폭/높이 계산
        if actual_text_height > box_height:
            ratio = box_height / actual_text_height
            text_width = max(1, int(actual_text_width * ratio))  # 스케일링된 폭
            render_height = box_height
            actual_text_height = box_height # 정보 업데이트
        else:
            text_width = actual_text_width # 원본 폭
            render_height = actual_text_height

        # Y축 중앙 정렬: 셀 중앙에 텍스트 중앙을 맞춤
        cell_top = int(min(ys))
        cell_center = cell_top + box_height // 2
        y_adjusted = cell_center - render_height // 2 - text_top_offset + 1  # +1: 텍스트를 약간 아래로

        bg_color = bg_colors.get(i, (255, 255, 255))
        is_vertical = is_vertical_text(bbox)
        # 겹침 감지용: OCR bbox 사용 (같은 행 판단을 위해 원본 좌표 사용)
        cell_bbox = (x, int(min(ys)), box_width, box_height)
        # ★ 테이블 안에 있는지 확인
        in_table = is_inside_table(bbox, table_regions)

        text_render_info.append({
            'x': x, 'y': y, 'y_adjusted': y_adjusted,
            'text': translated_text, 'font': font,
            'text_width': text_width, 'text_height': actual_text_height,
            'cell_bbox': cell_bbox, 'bg_color': bg_color,
            'is_vertical': is_vertical, 'bbox': bbox,
            'has_korean': item.get("has_korean", True),  # 한글 포함 여부 플래그
            'is_in_table': in_table  # ★ 테이블 내 여부
        })

    # 3단계: 겹침 감지 - 왼쪽 텍스트가 오른쪽 셀을 침범하는지 체크
    needs_abbreviation = set()
    logger.info(f"\n{'='*60}")
    logger.info(f"[Overlap Detection - replace] Total texts: {len(text_render_info)}")
    logger.info(f"{'='*60}")
    
    # 3-1: 셀 경계 초과 체크 (OCR 미인식 텍스트 대응)
    OVERFLOW_THRESHOLD = 30  # 30px 이상 초과시 무조건 축약
    for i, info in enumerate(text_render_info):
        text_right_edge = info['x'] + info['text_width']
        cell_x, cell_y, cell_w, cell_h = info['cell_bbox']
        cell_right = cell_x + cell_w
        overflow = text_right_edge - cell_right
        if overflow > OVERFLOW_THRESHOLD:
            needs_abbreviation.add(i)
            logger.info(f"  ★ OVERFLOW ABBREVIATE #{i} '{info['text'][:20]}' | overflow={overflow}px > {OVERFLOW_THRESHOLD}px")
    
    # 3-2: 인접 텍스트 침범 체크
    for i, info in enumerate(text_render_info):
        text_right_edge = info['x'] + info['text_width']
        cell_x, cell_y, cell_w, cell_h = info['cell_bbox']
        logger.debug(f"[#{i}] '{info['text'][:30]}' | x={info['x']}, w={info['text_width']}, right={text_right_edge} | cell=({cell_x},{cell_y},{cell_w},{cell_h})")

        for j, other_info in enumerate(text_render_info):
            if i == j:
                continue
            other_cell_left = other_info['cell_bbox'][0]

            # 현재 텍스트가 오른쪽 셀의 시작점을 침범했는지
            if text_right_edge > other_cell_left and info['x'] < other_cell_left:
                # Y축 겹침 체크 (같은 행인지)
                my_y = info['cell_bbox'][1]
                my_h = info['cell_bbox'][3]
                other_y = other_info['cell_bbox'][1]
                other_h = other_info['cell_bbox'][3]

                y_overlap = not (my_y + my_h <= other_y or other_y + other_h <= my_y)
                logger.info(f"  → #{i} INVADES #{j} '{other_info['text'][:20]}' | other_left={other_cell_left}")
                logger.info(f"     my_y={my_y}, my_h={my_h} (range: {my_y}~{my_y+my_h})")
                logger.info(f"     other_y={other_y}, other_h={other_h} (range: {other_y}~{other_y+other_h})")
                logger.info(f"     y_overlap={y_overlap}")

                if y_overlap:
                    needs_abbreviation.add(i)  # 침범한 쪽(왼쪽)을 약어로
                    logger.info(f"  ★ ABBREVIATE #{i}")
                    break
    logger.info(f"[Overlap Result] needs_abbreviation: {needs_abbreviation}")
    logger.info(f"{'='*60}\n")

    # 4단계: 실제 렌더링
    img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    used_abbreviations = set()
    all_text_bboxes = []

    for i, info in enumerate(text_render_info):
        # ★ 영어 텍스트는 렌더링 건너뛰기 (원본 유지)
        if not info.get('has_korean', True):
            continue
            
        display_text = info['text']

        if i in needs_abbreviation:
            display_text = abbreviate_text(info['text'], used_abbreviations, target_lang)

        text_color = get_text_color_for_background(info['bg_color'])
        text_color_rgb = (text_color[2], text_color[1], text_color[0]) if text_color == (255, 255, 255) else text_color

        if info['is_vertical']:
            draw_vertical_text(draw, display_text, info['x'], info['y'], info['font'],
                             text_color_rgb, info['cell_bbox'][2], info['cell_bbox'][3])
        else:
            # 클리핑: 텍스트를 임시 이미지에 그린 후 셀 높이만큼만 잘라서 붙임
            cell_left = info['cell_bbox'][0]
            cell_top = info['cell_bbox'][1]
            cell_width = info['cell_bbox'][2]
            cell_height = info['cell_bbox'][3]

            # 텍스트 bbox 계산 (충분한 여백에서)
            margin = 50
            text_bbox_temp = draw.textbbox((margin, margin), display_text, font=info['font'], anchor="lt")
            text_width_temp = text_bbox_temp[2] - text_bbox_temp[0]
            text_height_temp = text_bbox_temp[3] - text_bbox_temp[1]
            text_left = text_bbox_temp[0]
            text_top = text_bbox_temp[1]

            # 임시 이미지 생성 (충분히 크게)
            temp_img = Image.new('RGBA', (text_width_temp + margin * 2, text_height_temp + margin * 2), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.text((margin, margin), display_text, fill=text_color_rgb, font=info['font'], anchor="lt")

            # 실제 텍스트 영역만 crop
            temp_img = temp_img.crop((text_left, text_top, text_bbox_temp[2], text_bbox_temp[3]))

            # 셀 높이에 맞춰 추가 crop 및 위치 계산
            y_offset = 0  # ★ 중앙 정렬 (오프셋 제거 - BBox와 일치)
            if text_height_temp > cell_height:
                # 텍스트가 셀보다 큼 → LANCZOS 리사이즈 (잘림 방지)
                ratio = cell_height / text_height_temp
                new_width = max(1, int(text_width_temp * ratio))
                new_height = cell_height
                
                # 리사이즈
                try:
                    resample_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_filter = Image.LANCZOS
                
                temp_img = temp_img.resize((new_width, new_height), resample=resample_filter)
                
                # 붙여넣기 위치 (resize 했으므로 crop 불필요)
                paste_y = cell_top + y_offset
                
                # 리사이즈된 크기로 업데이트 (정렬용)
                text_width_temp = new_width
                text_height_temp = new_height
            else:
                # 텍스트가 셀보다 작음 → 셀 중앙에 배치
                paste_y = cell_top + (cell_height - text_height_temp) // 2 + y_offset

            # ★ X축 정렬: 테이블 안이면 중앙, 아니면 왼쪽
            if info.get('is_in_table', False):
                # 테이블 내 텍스트 → 중앙 정렬
                original_center_x = cell_left + cell_width // 2
                paste_x = original_center_x - text_width_temp // 2
                # 왼쪽 경계 제한
                if paste_x < cell_left:
                    paste_x = cell_left
            else:
                # 테이블 밖 텍스트 → 왼쪽 정렬
                paste_x = info['x']

            # 원본 이미지에 붙여넣기
            img_result.paste(temp_img, (paste_x, paste_y), temp_img)

        text_bbox_new = draw.textbbox((0, 0), display_text, font=info['font'])
        new_width = text_bbox_new[2] - text_bbox_new[0]
        all_text_bboxes.append((info['x'], info['y_adjusted'], new_width, info['text_height']))

    # ★ 범례 렌더링 (약어 사용 시)
    if used_abbreviations:
        legend_y = find_bottom_empty_area(height, all_text_bboxes)
        if legend_y is not None:
            render_legend(draw, used_abbreviations, width, legend_y)

    img_result = render_memos_on_image(img_result, memos or [])
    img_result.save(output_path)
    return output_path


def generate_preview_image(image_base64, translations, target_lang='english'):
    """미리보기 이미지 생성 (메모리에서 처리) - v1.8.0 (겹침 감지 + 약어)"""
    # base64 이미지를 numpy 배열로 변환
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    
    # ★ 테이블 영역 감지 (중앙 정렬 적용 여부 판단용)
    temp_img_path = os.path.join(UPLOAD_FOLDER, f"temp_table_detect_{id(image_base64)}.png")
    cv2.imwrite(temp_img_path, img)
    table_regions = detect_table_regions(temp_img_path)
    try:
        os.remove(temp_img_path)
    except:
        pass

    # 1단계: 한글 텍스트 영역만 배경색으로 지우기 (영어는 원본 유지)
    bg_colors = {}
    for i, item in enumerate(translations):
        if item.get("has_korean", True):  # 한글 텍스트만 erase
            bbox = item["bbox"]
            img, bg_color = erase_text_region(img, bbox)
            bg_colors[i] = bg_color
        else:
            bg_colors[i] = (255, 255, 255)  # 영어 텍스트는 erase 안 함

    # 2단계: 텍스트 정보 사전 계산 (겹침 감지용)
    font_sizes = [13, 12, 11, 10, 9, 8, 7]  # 폰트 크기 약간 증가
    text_render_info = []  # [(x, y_adjusted, text, font, text_width, height, cell_bbox, bg_color, is_vertical)]

    img_pil_temp = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw_temp = ImageDraw.Draw(img_pil_temp)

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
        text_width = 0
        for size in font_sizes:
            try:
                font = ImageFont.truetype("malgun.ttf", size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
                except:
                    font = ImageFont.load_default()
                    break

            text_bbox_size = draw_temp.textbbox((0, 0), translated_text, font=font, anchor="lt")
            text_width = text_bbox_size[2] - text_bbox_size[0]
            selected_text_height = text_bbox_size[3] - text_bbox_size[1]

            if selected_text_height <= box_height:  # 셀 높이에 맞춤 (클리핑으로 경계 처리)
                break

        text_bbox_actual = draw_temp.textbbox((0, 0), translated_text, font=font, anchor="lt")
        actual_text_width = text_bbox_actual[2] - text_bbox_actual[0]
        actual_text_height = text_bbox_actual[3] - text_bbox_actual[1]
        text_top_offset = text_bbox_actual[1]  # textbbox의 top offset (글리프 상단까지 거리)

        # ★ 스케일링 반영: 텍스트가 셀보다 크면 리사이즈된 폭/높이 계산
        if actual_text_height > box_height:
            ratio = box_height / actual_text_height
            text_width = max(1, int(actual_text_width * ratio))  # 스케일링된 폭
            render_height = box_height
            actual_text_height = box_height # 정보 업데이트
        else:
            text_width = actual_text_width # 원본 폭
            render_height = actual_text_height

        # Y축 중앙 정렬: 셀 중앙에 텍스트 중앙을 맞춤
        cell_top = int(min(ys))
        cell_center = cell_top + box_height // 2
        y_adjusted = cell_center - render_height // 2 - text_top_offset + 1  # +1: 텍스트를 약간 아래로

        bg_color = bg_colors.get(i, (255, 255, 255))
        is_vertical = is_vertical_text(bbox)
        # 겹침 감지용: OCR bbox 사용 (같은 행 판단을 위해 원본 좌표 사용)
        cell_bbox = (x, int(min(ys)), box_width, box_height)
        # ★ 테이블 안에 있는지 확인
        in_table = is_inside_table(bbox, table_regions)

        text_render_info.append({
            'x': x, 'y': y, 'y_adjusted': y_adjusted,
            'text': translated_text, 'font': font,
            'text_width': text_width, 'text_height': actual_text_height,
            'cell_bbox': cell_bbox, 'bg_color': bg_color,
            'is_vertical': is_vertical, 'bbox': bbox,
            'has_korean': item.get("has_korean", True),  # 한글 포함 여부 플래그
            'is_in_table': in_table  # ★ 테이블 내 여부
        })

    # 3단계: 겹침 감지 - 왼쪽 텍스트가 오른쪽 셀을 침범하는지 체크
    needs_abbreviation = set()
    logger.info(f"\n{'='*60}")
    logger.info(f"[Overlap Detection - preview] Total texts: {len(text_render_info)}")
    logger.info(f"{'='*60}")
    
    # 3-1: 셀 경계 초과 체크 (OCR 미인식 텍스트 대응)
    OVERFLOW_THRESHOLD = 30  # 30px 이상 초과시 무조건 축약
    for i, info in enumerate(text_render_info):
        text_right_edge = info['x'] + info['text_width']
        cell_x, cell_y, cell_w, cell_h = info['cell_bbox']
        cell_right = cell_x + cell_w
        overflow = text_right_edge - cell_right
        if overflow > OVERFLOW_THRESHOLD:
            needs_abbreviation.add(i)
            logger.info(f"  ★ OVERFLOW ABBREVIATE #{i} '{info['text'][:20]}' | overflow={overflow}px > {OVERFLOW_THRESHOLD}px")
    
    # 3-2: 인접 텍스트 침범 체크
    for i, info in enumerate(text_render_info):
        # 현재 텍스트의 실제 렌더링 영역 (x ~ x+text_width)
        text_right_edge = info['x'] + info['text_width']
        cell_x, cell_y, cell_w, cell_h = info['cell_bbox']
        logger.debug(f"[#{i}] '{info['text'][:25]}' | x={info['x']}, w={info['text_width']}, right={text_right_edge} | cell=({cell_x},{cell_y},{cell_w},{cell_h})")

        # 오른쪽에 있는 모든 셀과 비교
        for j, other_info in enumerate(text_render_info):
            if i == j:
                continue
            other_cell_left = other_info['cell_bbox'][0]

            # 현재 텍스트가 오른쪽 셀의 시작점을 침범했는지
            if text_right_edge > other_cell_left and info['x'] < other_cell_left:
                # Y축도 겹치는지 확인 (같은 행인지)
                my_y = info['cell_bbox'][1]
                my_h = info['cell_bbox'][3]
                other_y = other_info['cell_bbox'][1]
                other_h = other_info['cell_bbox'][3]

                # Y축 겹침 체크
                y_overlap = not (my_y + my_h <= other_y or other_y + other_h <= my_y)
                logger.info(f"  → #{i} INVADES #{j} '{other_info['text'][:15]}' | other_left={other_cell_left}")
                logger.info(f"     my_y={my_y}, my_h={my_h} (range: {my_y}~{my_y+my_h})")
                logger.info(f"     other_y={other_y}, other_h={other_h} (range: {other_y}~{other_y+other_h})")
                logger.info(f"     y_overlap={y_overlap}")
                if y_overlap:
                    needs_abbreviation.add(i)  # 침범한 쪽(왼쪽)을 약어로
                    logger.info(f"  ★ ABBREVIATE #{i}")
                    break
    logger.info(f"[Overlap Result] needs_abbreviation: {needs_abbreviation}")
    logger.info(f"{'='*60}\n")

    # 4단계: 실제 렌더링
    img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    used_abbreviations = set()
    all_text_bboxes = []

    for i, info in enumerate(text_render_info):
        # ★ 영어 텍스트는 렌더링 건너뛰기 (원본 유지)
        if not info.get('has_korean', True):
            continue
            
        display_text = info['text']

        # 침범한 텍스트는 약어로 변환
        if i in needs_abbreviation:
            display_text = abbreviate_text(info['text'], used_abbreviations, target_lang)

        text_color = get_text_color_for_background(info['bg_color'])
        text_color_rgb = (text_color[2], text_color[1], text_color[0]) if text_color == (255, 255, 255) else text_color

        if info['is_vertical']:
            draw_vertical_text(draw, display_text, info['x'], info['y'], info['font'],
                             text_color_rgb, info['cell_bbox'][2], info['cell_bbox'][3])
        else:
            # 클리핑: 텍스트를 임시 이미지에 그린 후 셀 높이만큼만 잘라서 붙임
            cell_left = info['cell_bbox'][0]
            cell_top = info['cell_bbox'][1]
            cell_width = info['cell_bbox'][2]
            cell_height = info['cell_bbox'][3]

            # 텍스트 bbox 계산 (충분한 여백에서)
            margin = 50
            text_bbox_temp = draw.textbbox((margin, margin), display_text, font=info['font'], anchor="lt")
            text_width_temp = text_bbox_temp[2] - text_bbox_temp[0]
            text_height_temp = text_bbox_temp[3] - text_bbox_temp[1]
            text_left = text_bbox_temp[0]
            text_top = text_bbox_temp[1]

            # 임시 이미지 생성 (충분히 크게)
            temp_img = Image.new('RGBA', (text_width_temp + margin * 2, text_height_temp + margin * 2), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.text((margin, margin), display_text, fill=text_color_rgb, font=info['font'], anchor="lt")

            # 실제 텍스트 영역만 crop
            temp_img = temp_img.crop((text_left, text_top, text_bbox_temp[2], text_bbox_temp[3]))

            # 셀 높이에 맞춰 추가 crop 및 위치 계산
            y_offset = 0  # ★ 중앙 정렬 (오프셋 제거 - BBox와 일치)
            if text_height_temp > cell_height:
                # 텍스트가 셀보다 큼 → LANCZOS 리사이즈 (잘림 방지)
                ratio = cell_height / text_height_temp
                new_width = max(1, int(text_width_temp * ratio))
                new_height = cell_height
                
                # 리사이즈
                try:
                    resample_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_filter = Image.LANCZOS
                
                temp_img = temp_img.resize((new_width, new_height), resample=resample_filter)
                
                # 붙여넣기 위치 (resize 했으므로 crop 불필요)
                paste_y = cell_top + y_offset
                
                # 리사이즈된 크기로 업데이트 (정렬용)
                text_width_temp = new_width
                text_height_temp = new_height
            else:
                # 텍스트가 셀보다 작음 → 셀 중앙에 배치
                paste_y = cell_top + (cell_height - text_height_temp) // 2 + y_offset

            # ★ X축 정렬: 테이블 안이면 중앙, 아니면 왼쪽
            if info.get('is_in_table', False):
                # 테이블 내 텍스트 → 중앙 정렬
                original_center_x = cell_left + cell_width // 2
                paste_x = original_center_x - text_width_temp // 2
                # 왼쪽 경계 제한
                if paste_x < cell_left:
                    paste_x = cell_left
            else:
                # 테이블 밖 텍스트 → 왼쪽 정렬
                paste_x = info['x']

            # 원본 이미지에 붙여넣기
            img_result.paste(temp_img, (paste_x, paste_y), temp_img)

        # bbox 기록
        text_bbox_new = draw.textbbox((0, 0), display_text, font=info['font'])
        new_width = text_bbox_new[2] - text_bbox_new[0]
        all_text_bboxes.append((info['x'], info['y_adjusted'], new_width, info['text_height']))

    # ★ 범례 렌더링 (약어 사용 시)
    if used_abbreviations:
        legend_y = find_bottom_empty_area(height, all_text_bboxes)
        if legend_y is not None:
            render_legend(draw, used_abbreviations, width, legend_y)

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
    <title>PDF Translate Note - 의류 기술서 번역</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            padding: 5px;
            overflow: hidden;  /* 전체 페이지 스크롤 방지 */
        }
        .container {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 6px;
            padding: 5px 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            height: calc(100vh - 10px);
            display: flex;
            flex-direction: column;
            overflow: hidden;
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
        .current-ocr-display {
            padding: 3px 10px;
            border: 2px solid #28a745;
            border-radius: 10px;
            background: #e8f5e9;
            color: #28a745;
            font-size: 0.7em;
            font-weight: bold;
        }
        .current-ai-display {
            padding: 3px 12px;
            border: 2px solid #007bff;
            border-radius: 10px;
            background: #e7f1ff;
            color: #007bff;
            font-size: 0.7em;
            font-weight: bold;
            min-width: 180px;
            text-align: center;
            white-space: nowrap;
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
        .dict-btn {
            padding: 3px 8px;
            border: 2px solid #28a745;
            border-radius: 10px;
            background: white;
            color: #28a745;
            cursor: pointer;
            font-size: 0.7em;
            transition: all 0.3s;
        }
        .dict-btn:hover {
            background: #28a745;
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
            /* API 키 복사 방지 */
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
            -webkit-touch-callout: none;
        }
        .api-key-input-wrapper input::selection {
            background: transparent;
        }
        .api-key-input-wrapper input::-moz-selection {
            background: transparent;
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
            flex: 1;           /* 남은 공간 채움 */
            min-height: 0;     /* flex 오버플로우 허용 */
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
            justify-content: flex-start;  /* center→flex-start: 왼쪽 잘림 방지 */
            align-items: flex-start;
        }
        .preview-image img {
            max-width: none;
            max-height: none;
            object-fit: contain;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            pointer-events: none;  /* 이벤트가 previewStage로 전달되도록 */
            transform-origin: top left;  /* 좌상단 기준 확대 - 스크롤 정상 동작 */
            transition: transform 0.1s ease;
        }
        .preview-stage {
            position: relative;
            display: inline-block;
            cursor: crosshair;  /* 미리보기 영역임을 표시 */
        }
        .arrow-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }
        .memo-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: none;
            z-index: 2;
        }
        .memo-item {
            position: absolute;
            cursor: move;
            font-family: Arial, sans-serif;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.2;
            user-select: none;
            resize: both;
            overflow: hidden;
            min-width: 80px;
            min-height: 30px;
        }
        .memo-item:hover {
            outline: 1px solid rgba(102, 126, 234, 0.5);
        }
        .memo-item.selected {
            outline: 2px dashed #667eea;
            outline-offset: 2px;
        }
        .memo-context-menu {
            position: fixed;
            z-index: 3000;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 6px;
            display: none;
            min-width: 140px;
        }
        .memo-context-menu.active {
            display: block;
        }
        .memo-context-menu button {
            width: 100%;
            padding: 8px 10px;
            border: none;
            background: none;
            text-align: left;
            cursor: pointer;
            border-radius: 6px;
            font-size: 0.9em;
        }
        .memo-context-menu button:hover {
            background: #f0f3ff;
        }
        .memo-editor {
            position: fixed;
            z-index: 3001;
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            padding: 10px;
            display: none;
            width: 280px;
        }
        .memo-editor.active {
            display: block;
        }
        .memo-editor-title {
            font-weight: bold;
            margin-bottom: 6px;
            color: #333;
        }
        .memo-editor textarea {
            width: 100%;
            height: 100px;
            resize: vertical;
            padding: 8px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 0.9em;
            font-family: Arial, sans-serif;
        }
        .memo-editor-actions {
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            margin-top: 8px;
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
        /* 메모 스타일 패널 - 2줄 컴팩트 툴바 */
        .memo-panel {
            border-top: 1px solid #eee;
            background: #fafbff;
            padding: 6px 10px;
            flex-shrink: 0;  /* 항상 표시 - 잘림 방지 */
        }
        .memo-toolbar {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .memo-toolbar-row {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .memo-toolbar-group {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 0 6px;
            border-right: 1px solid #ddd;
        }
        .memo-toolbar-group:last-child {
            border-right: none;
        }
        .memo-toolbar-row.row2 {
            display: none;
        }
        .memo-panel.expanded .memo-toolbar-row.row2 {
            display: flex;
        }
        .memo-toolbar label {
            color: #555;
            font-size: 0.75em;
            white-space: nowrap;
        }
        .memo-toolbar input[type="number"] {
            width: 45px;
            padding: 2px 4px;
            border: 1px solid #ccc;
            border-radius: 3px;
            font-size: 0.8em;
        }
        .memo-toolbar input[type="color"] {
            width: 22px;
            height: 22px;
            padding: 0;
            border: 1px solid #ccc;
            border-radius: 3px;
            cursor: pointer;
        }
        .memo-toolbar input[type="range"] {
            width: 70px;
            height: 4px;
        }
        .memo-toolbar input[type="checkbox"] {
            margin: 0;
            width: 14px;
            height: 14px;
        }
        .memo-toolbar-btn {
            padding: 3px 6px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            cursor: pointer;
            font-size: 0.8em;
            font-weight: 600;
            min-width: 24px;
            text-align: center;
        }
        .memo-toolbar-btn:hover {
            background: #f0f0f0;
        }
        .memo-toolbar-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .memo-expand-btn {
            padding: 2px 8px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            cursor: pointer;
            font-size: 0.8em;
            margin-left: auto;
        }
        .memo-expand-btn:hover {
            background: #f0f0f0;
        }
        .memo-toolbar-title {
            font-weight: bold;
            color: #333;
            font-size: 0.85em;
        }
        .memo-default-btn {
            padding: 2px 6px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background: #fff;
            cursor: pointer;
            font-size: 0.7em;
            color: #666;
            margin-left: auto;
        }
        .memo-default-btn:hover {
            background: #f5f5f5;
        }

        /* 용어 사전 모달 스타일 */
        .dict-modal {
            width: 890px;
            max-width: 95vw;
            max-height: 99vh;
        }
        .dict-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .dict-tab {
            padding: 8px 12px;
            border: 2px solid #ddd;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .dict-tab:hover {
            border-color: #667eea;
        }
        .dict-tab.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .dict-add-form {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: nowrap;
        }
        .dict-add-form input {
            flex: 1;
            min-width: 120px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 0.95em;
        }
        .dict-add-form input:focus {
            outline: none;
            border-color: #667eea;
        }
        .dict-add-form button {
            flex-shrink: 0;
            white-space: nowrap;
        }
        .dict-search {
            margin-bottom: 10px;
        }
        .dict-search input {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 0.95em;
        }
        .dict-search input:focus {
            outline: none;
            border-color: #667eea;
        }
        .dict-table-wrapper {
            max-height: 350px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .dict-table {
            width: 100%;
            border-collapse: collapse;
        }
        .dict-table th, .dict-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .dict-table th {
            background: #f8f9fa;
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        .dict-table th:nth-child(1) { width: 25%; }  /* 한글 */
        .dict-table th:nth-child(2) { width: 35%; }  /* 번역 */
        .dict-table th:nth-child(3) { width: 20%; }  /* 약어 */
        .dict-table th:nth-child(4) { width: 20%; }  /* 작업 */
        .dict-table .abbr-cell {
            color: #666;
            font-style: italic;
        }
        .dict-table tr:hover {
            background: #f8f9fa;
        }
        .dict-table .actions {
            display: flex;
            gap: 5px;
        }
        .dict-table .edit-btn, .dict-table .delete-btn, .dict-table .save-btn, .dict-table .cancel-btn {
            padding: 4px 8px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }
        .dict-table .edit-btn {
            background: #ffc107;
            color: #333;
        }
        .dict-table .delete-btn {
            background: #dc3545;
            color: white;
        }
        .dict-table .save-btn {
            background: #28a745;
            color: white;
        }
        .dict-table .cancel-btn {
            background: #6c757d;
            color: white;
        }
        .dict-table .edit-input {
            width: 100%;
            padding: 5px;
            border: 2px solid #667eea;
            border-radius: 4px;
        }
        .dict-count {
            color: #666;
            font-size: 0.9em;
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
            <h1>📝 PDF Translate Note</h1>
            <span class="version-badge">v{{ version }}</span>
            <span class="subtitle">한글→다국어</span>
            <button type="button" class="lang-btn active" data-lang="english">🇺🇸EN</button>
            <button type="button" class="lang-btn" data-lang="vietnamese">🇻🇳VI</button>
            <button type="button" class="lang-btn" data-lang="chinese">🇨🇳中</button>
            <button type="button" class="lang-btn" data-lang="indonesian">🇮🇩ID</button>
            <button type="button" class="lang-btn" data-lang="bengali">🇧🇩BN</button>
            <button type="button" class="lang-btn" data-lang="myanmar">🇲🇲MY</button>
            <button type="button" class="file-select-btn" id="fileSelectBtn">📁 파일선택</button>
            <button type="button" class="translate-btn" id="translateBtn" disabled>🚀 번역</button>
            <button type="button" class="dict-btn" id="dictBtn" title="용어 사전 관리">📖</button>
            <button type="button" class="settings-btn" id="settingsBtn">⚙️</button>
            <span id="currentOcrDisplay" class="current-ocr-display">PaddleOCR v5</span>
            <span id="currentAiDisplay" class="current-ai-display">GPT-4o-mini</span>
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
                        <label>OCR 엔진 선택</label>
                        <select id="ocrEngineSelect">
                            <option value="paddleocr" selected>🖥️ PaddleOCR v5 (로컬, 무료)</option>
                            <option value="google_vision">☁️ Google Vision API (클라우드, 빠름)</option>
                        </select>
                        <p class="setting-hint">PaddleOCR v5: 무료, 로컬 처리 | Google Vision API: 월 1,000장 무료, 이후 $1.50/1,000장</p>
                    </div>

                    <div class="setting-group">
                        <label>AI 엔진 선택</label>
                        <select id="aiEngineSelect">
                            <option value="openai" selected>🟢 OpenAI GPT</option>
                            <option value="claude_sije">🟣 Claude (Sije)</option>
                            <option value="claude_seam">🟣 Claude (SEAM)</option>
                            <option value="gemini">🔵 Google Gemini</option>
                            <option value="ollama">🖥️ Ollama (로컬)</option>
                        </select>
                        <p class="setting-hint">API 키가 서버에 설정되어 있습니다.</p>
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

                    <div class="setting-group">
                        <label>번역 처리 방식</label>
                        <div style="display: flex; gap: 20px; margin-top: 8px;">
                            <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                <input type="radio" name="translateMode" id="translateModeSequential" value="sequential" checked>
                                <span>📄 순차 (페이지별)</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                <input type="radio" name="translateMode" id="translateModeBatch" value="batch">
                                <span>📦 배치 (전체 일괄)</span>
                            </label>
                        </div>
                        <p class="setting-hint">순차: 안정적, 토큰 한도 내 처리 | 배치: 빠름, 긴 문서 시 토큰 초과 위험</p>
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

        <!-- 비밀번호 확인 모달 -->
        <div class="modal-overlay" id="passwordModal">
            <div class="modal-content" style="max-width: 350px;">
                <div class="modal-header">
                    <h2>🔐 설정 접근</h2>
                    <button class="modal-close" id="closePassword">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="setting-group">
                        <label>비밀번호를 입력하세요</label>
                        <input type="password" id="settingsPassword" placeholder="비밀번호" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;">
                        <p class="setting-hint" id="passwordError" style="color: #e74c3c; display: none;">비밀번호가 틀렸습니다.</p>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-secondary" id="cancelPassword">취소</button>
                    <button type="button" class="btn-primary" id="confirmPassword">확인</button>
                </div>
            </div>
        </div>

        <!-- 용어 사전 모달 -->
        <div class="modal-overlay" id="dictModal">
            <div class="modal-content dict-modal">
                <div class="modal-header">
                    <h2>📖 용어 사전 관리</h2>
                    <button class="modal-close" id="closeDict">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="dict-tabs">
                        <button class="dict-tab active" data-lang="english">🇺🇸 영어</button>
                        <button class="dict-tab" data-lang="vietnamese">🇻🇳 베트남어</button>
                        <button class="dict-tab" data-lang="chinese">🇨🇳 중국어</button>
                        <button class="dict-tab" data-lang="indonesian">🇮🇩 인도네시아어</button>
                        <button class="dict-tab" data-lang="bengali">🇧🇩 벵골어</button>
                        <button class="dict-tab" data-lang="myanmar">🇲🇲 미얀마어</button>
                    </div>
                    <div class="dict-add-form">
                        <input type="text" id="dictKorean" placeholder="한글 용어">
                        <input type="text" id="dictTranslation" placeholder="번역">
                        <input type="text" id="dictAbbr" placeholder="약어 (선택)">
                        <button type="button" class="btn-primary" id="addTermBtn">➕ 추가</button>
                    </div>
                    <div class="dict-search">
                        <input type="text" id="dictSearch" placeholder="🔍 검색...">
                    </div>
                    <div class="dict-table-wrapper">
                        <table class="dict-table">
                            <thead>
                                <tr>
                                    <th>한글</th>
                                    <th>번역</th>
                                    <th>약어</th>
                                    <th>작업</th>
                                </tr>
                            </thead>
                            <tbody id="dictBody">
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="modal-footer">
                    <span class="dict-count" id="dictCount">총 0개 용어</span>
                    <button type="button" class="btn-secondary" id="closeDictBtn">닫기</button>
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
                    <div class="preview-stage" id="previewStage">
                        <img id="previewImg" src="" alt="페이지 프리뷰">
                        <svg class="arrow-layer" id="arrowLayer"></svg>
                        <div class="memo-layer" id="memoLayer"></div>
                    </div>
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
                <div class="memo-panel" id="memoPanel">
                    <div class="memo-toolbar">
                        <!-- 1줄: 핵심 옵션 -->
                        <div class="memo-toolbar-row">
                            <span class="memo-toolbar-title">메모</span>
                            <div class="memo-toolbar-group">
                                <input type="number" id="memoFontSize" min="6" max="72" value="14" title="글자 크기">
                                <input type="color" id="memoTextColor" value="#111111" title="글자 색">
                                <button type="button" class="memo-toolbar-btn" id="memoBoldBtn" title="굵게">B</button>
                            </div>
                            <div class="memo-toolbar-group">
                                <label>너비</label>
                                <input type="number" id="memoWidth" min="80" max="800" value="220" title="메모 너비">
                            </div>
                            <div class="memo-toolbar-group">
                                <label>여백</label>
                                <input type="number" id="memoPadding" min="0" max="40" value="6" title="내부 여백">
                            </div>
                            <div class="memo-toolbar-group">
                                <input type="checkbox" id="memoBgEnabled" checked title="배경 표시">
                                <input type="color" id="memoBgColor" value="#ffffff" title="배경 색">
                            </div>
                            <button type="button" class="memo-expand-btn" id="memoExpandBtn" title="더보기">▼</button>
                        </div>
                        <!-- 2줄: 추가 옵션 (접힘) -->
                        <div class="memo-toolbar-row row2">
                            <div class="memo-toolbar-group">
                                <label>테두리</label>
                                <input type="number" id="memoBorderWidth" min="0" max="10" value="0" title="테두리 두께">
                                <input type="color" id="memoBorderColor" value="#333333" title="테두리 색">
                            </div>
                            <div class="memo-toolbar-group">
                                <label>투명도</label>
                                <input type="range" id="memoOpacity" min="0" max="100" value="90" title="투명도">
                            </div>
                            <button type="button" class="memo-default-btn" id="memoDefaultBtn">기본값</button>
                        </div>
                    </div>
                    <input type="checkbox" id="memoBold" style="display:none">
                </div>
            </div>
        </div>

        <div class="memo-context-menu" id="memoContextMenu">
            <button type="button" data-action="add">✍️ 메모 추가</button>
            <button type="button" data-action="edit">📝 메모 편집</button>
            <button type="button" data-action="delete">🗑️ 메모 삭제</button>
        </div>

        <div class="memo-editor" id="memoEditor">
            <div class="memo-editor-title" id="memoEditorTitle">메모 입력</div>
            <textarea id="memoEditorText" placeholder="메모를 입력하세요"></textarea>
            <div class="memo-editor-actions">
                <button type="button" class="btn-secondary" id="memoEditorCancel">취소</button>
                <button type="button" class="btn-primary" id="memoEditorSave">저장</button>
            </div>
        </div>

        <!-- 결과 표시 -->
        <div class="results" id="results"></div>
    </div>

    <script>
        // 상태 변수
        let currentPage = 0;
        let totalPages = 0;
        let pagesData = [];  // [{image: base64, texts: [...], translations: [...], confirmed: bool, memos: [...]}]

        // DOM 요소
        const fileInput = document.getElementById('fileInput');
        const fileSelectBtn = document.getElementById('fileSelectBtn');
        const translateBtn = document.getElementById('translateBtn');
        const langBtns = document.querySelectorAll('.lang-btn');
        const targetLang = document.getElementById('targetLang');
        const status = document.getElementById('status');
        const initialGuide = document.getElementById('initialGuide');
        const editorContainer = document.getElementById('editorContainer');
        const previewStage = document.getElementById('previewStage');
        const previewImg = document.getElementById('previewImg');
        const memoLayer = document.getElementById('memoLayer');
        const arrowLayer = document.getElementById('arrowLayer');
        const pageInfo = document.getElementById('pageInfo');
        const prevPageBtn = document.getElementById('prevPageBtn');
        const nextPageBtn = document.getElementById('nextPageBtn');
        const translationBody = document.getElementById('translationBody');
        const confirmBtn = document.getElementById('confirmBtn');
        const results = document.getElementById('results');
        const memoContextMenu = document.getElementById('memoContextMenu');
        const memoEditor = document.getElementById('memoEditor');
        const memoEditorTitle = document.getElementById('memoEditorTitle');
        const memoEditorText = document.getElementById('memoEditorText');
        const memoEditorSave = document.getElementById('memoEditorSave');
        const memoEditorCancel = document.getElementById('memoEditorCancel');
        const memoBoldBtn = document.getElementById('memoBoldBtn');
        const memoFontSize = document.getElementById('memoFontSize');
        const memoTextColor = document.getElementById('memoTextColor');
        const memoBold = document.getElementById('memoBold');
        const memoWidth = document.getElementById('memoWidth');
        const memoPadding = document.getElementById('memoPadding');
        const memoOpacity = document.getElementById('memoOpacity');
        const memoBgEnabled = document.getElementById('memoBgEnabled');
        const memoBgColor = document.getElementById('memoBgColor');
        const memoBorderWidth = document.getElementById('memoBorderWidth');
        const memoBorderColor = document.getElementById('memoBorderColor');

        // 미리보기 토글 버튼
        const showOriginalBtn = document.getElementById('showOriginal');
        const showPreviewBtn = document.getElementById('showPreview');

        // 미리보기 상태
        let isPreviewMode = false;
        let previewCache = {};  // 페이지별 미리보기 캐시

        // 줌 상태
        let currentZoom = 100;
        const ZOOM_MIN = 50;
        const ZOOM_MAX = 200;
        const ZOOM_STEP = 10;

        // 줌 적용 함수
        function applyZoom(zoom) {
            currentZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom));
            previewImg.style.transform = `scale(${currentZoom / 100})`;
            console.log(`Zoom: ${currentZoom}%`);
        }

        // Ctrl+휠 줌 이벤트 (PDF 미리보기 영역에서만)
        const previewImageContainer = document.querySelector('.preview-image');
        if (previewImageContainer) {
            previewImageContainer.addEventListener('wheel', function(e) {
                if (e.ctrlKey) {
                    e.preventDefault();
                    e.stopPropagation();
                    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
                    applyZoom(currentZoom + delta);
                }
            }, { passive: false });
        }

        // 메모 상태
        let selectedMemoId = null;
        let memoIdSeq = 1;
        let memoEditorState = null;
        let memoContextState = null;
        let memoDragState = null;
        let arrowDragState = null;  // 화살표 드래그 상태
        const memoDefaults = {
            fontSize: 14,
            color: '#111111',
            bold: false,
            width: 220,
            padding: 6,
            opacity: 0.9,
            backgroundEnabled: true,
            backgroundColor: '#ffffff',
            borderWidth: 0,
            borderColor: '#333333'
        };

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

        // 비밀번호 모달 관련 요소
        const passwordModal = document.getElementById('passwordModal');
        const settingsPassword = document.getElementById('settingsPassword');
        const confirmPassword = document.getElementById('confirmPassword');
        const cancelPassword = document.getElementById('cancelPassword');
        const closePassword = document.getElementById('closePassword');
        const passwordError = document.getElementById('passwordError');

        // 설정 비밀번호
        const SETTINGS_PASSWORD_HASH = 'suk1892!';
        // 세션 기반 인증 (페이지 새로고침하면 다시 비밀번호 필요)
        let isSettingsAuthenticated = false;

        // API 키 복사 방지
        apiKeyInput.addEventListener('copy', (e) => e.preventDefault());
        apiKeyInput.addEventListener('cut', (e) => e.preventDefault());
        apiKeyInput.addEventListener('contextmenu', (e) => e.preventDefault());

        // AI 설정 상태 (localStorage에서 로드)
        let currentAiEngine = localStorage.getItem('pdf_translator_ai_engine') || 'openai';
        let currentModel = localStorage.getItem('pdf_translator_model') || '';
        let apiKeys = JSON.parse(localStorage.getItem('pdf_translator_api_keys') || '{}');

        // OCR 엔진 설정
        let currentOcrEngine = localStorage.getItem('pdf_translator_ocr_engine') || 'paddleocr';

        // 번역 처리 방식 설정
        let currentTranslateMode = localStorage.getItem('pdf_translator_translate_mode') || 'sequential';

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
            claude_sije: {
                models: ['claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
                default: 'claude-sonnet-4-20250514',
                hints: {
                    'claude-opus-4-20250514': 'Opus 4.5 - 최고 성능 (비용 높음)',
                    'claude-sonnet-4-20250514': 'Sonnet 4 - 고성능 (권장)',
                    'claude-3-5-sonnet-20241022': '안정적인 성능',
                    'claude-3-haiku-20240307': '빠르고 저렴'
                }
            },
            claude_seam: {
                models: ['claude-haiku-4-20250514', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514', 'claude-opus-4-20250514'],
                default: 'claude-haiku-4-20250514',
                hints: {
                    'claude-haiku-4-20250514': 'Haiku 4.5 - 최신, 빠르고 저렴 (권장)',
                    'claude-3-5-haiku-20241022': 'Haiku 3.5 - 빠르고 저렴',
                    'claude-sonnet-4-20250514': 'Sonnet 4 - 고성능',
                    'claude-opus-4-20250514': 'Opus 4.5 - 최고 성능 (비용 높음)'
                }
            },
            openai: {
                models: ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
                default: 'gpt-4o-mini',
                hints: {
                    'gpt-4o-mini': '저렴하고 빠름 (권장)',
                    'gpt-4o': '최신 멀티모달, 고성능',
                    'gpt-4-turbo': '안정적, Vision 지원'
                }
            },
            gemini: {
                models: ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro'],
                default: 'gemini-2.0-flash',
                hints: {
                    'gemini-2.0-flash': '빠르고 저렴 (권장)',
                    'gemini-2.5-flash': '65K 출력 토큰, 긴 문서용',
                    'gemini-2.5-pro': '고성능, 복잡한 작업용'
                }
            }
        };

        // API 키 힌트 정보 (하드코딩으로 인해 사용되지 않음)
        const apiKeyInfo = {
            claude_sije: {
                label: 'Claude API Key (Sije)',
                hint: '서버에 설정되어 있습니다.',
                placeholder: 'sk-ant-...'
            },
            claude_seam: {
                label: 'Claude API Key (SEAM)',
                hint: '서버에 설정되어 있습니다.',
                placeholder: 'sk-ant-...'
            },
            openai: {
                label: 'OpenAI API Key',
                hint: '서버에 설정되어 있습니다.',
                placeholder: 'sk-...'
            },
            gemini: {
                label: 'Google Gemini API Key',
                hint: '서버에 설정되어 있습니다.',
                placeholder: 'AIza...'
            }
        };

        // 초기 설정 UI 업데이트
        function initSettings() {
            aiEngineSelect.value = currentAiEngine;
            document.getElementById('ocrEngineSelect').value = currentOcrEngine;
            // 번역 처리 방식 라디오 버튼 설정
            const translateModeRadio = document.querySelector(`input[name="translateMode"][value="${currentTranslateMode}"]`);
            if (translateModeRadio) translateModeRadio.checked = true;
            updateApiKeyVisibility();
            updateModelOptions();
        }

        // 툴바 요소
        const currentOcrDisplay = document.getElementById('currentOcrDisplay');
        const currentAiDisplay = document.getElementById('currentAiDisplay');

        // AI 모델 표시명 매핑
        const aiDisplayNames = {
            openai: 'GPT-4o-mini',
            claude_sije: 'Claude-Sije',
            claude_seam: 'Claude-SEAM',
            gemini: 'Gemini',
            ollama: 'Ollama'
        };

        // 페이지 로드 시 툴바 초기화 (DOMContentLoaded 후 실행)
        document.addEventListener('DOMContentLoaded', function initToolbar() {
            // OCR 표시 초기화
            updateOcrDisplay();

            // AI 표시 초기화 (모델명 포함)
            updateAiDisplay();
        });

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

        // 모델 ID → 표시명 매핑
        const modelDisplayNames = {
            'claude-haiku-4-20250514': 'Claude Haiku 4.5',
            'claude-3-5-haiku-20241022': 'Claude Haiku 3.5',
            'claude-sonnet-4-20250514': 'Claude Sonnet 4',
            'claude-3-5-sonnet-20241022': 'Claude Sonnet 3.5',
            'claude-opus-4-20250514': 'Claude Opus 4.5',
            'claude-3-haiku-20240307': 'Claude Haiku 3',
            'gpt-4o-mini': 'GPT-4o-mini',
            'gpt-4o': 'GPT-4o',
            'gpt-4-turbo': 'GPT-4 Turbo',
            'gemini-2.0-flash': 'Gemini 2.0 Flash',
            'gemini-1.5-flash': 'Gemini 1.5 Flash',
            'gemini-1.5-pro': 'Gemini 1.5 Pro'
        };

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
                option.textContent = modelDisplayNames[model] || model;
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

        // 설정 모달 열기 (비밀번호 확인 필요)
        settingsBtn.addEventListener('click', () => {
            if (isSettingsAuthenticated) {
                // 이미 인증됨 - 바로 설정 모달 열기
                initSettings();
                settingsModal.classList.add('active');
            } else {
                // 비밀번호 확인 필요
                settingsPassword.value = '';
                passwordError.style.display = 'none';
                passwordModal.classList.add('active');
                settingsPassword.focus();
            }
        });

        // 비밀번호 확인
        confirmPassword.addEventListener('click', () => {
            if (settingsPassword.value === SETTINGS_PASSWORD_HASH) {
                // 인증 성공
                isSettingsAuthenticated = true;
                localStorage.setItem('pdf_translator_auth', 'true');
                passwordModal.classList.remove('active');
                initSettings();
                settingsModal.classList.add('active');
            } else {
                // 인증 실패
                passwordError.style.display = 'block';
                settingsPassword.value = '';
                settingsPassword.focus();
            }
        });

        // 비밀번호 모달에서 Enter 키 처리
        settingsPassword.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                confirmPassword.click();
            }
        });

        // 비밀번호 모달 닫기
        function closePasswordModal() {
            passwordModal.classList.remove('active');
            settingsPassword.value = '';
            passwordError.style.display = 'none';
        }
        closePassword.addEventListener('click', closePasswordModal);
        cancelPassword.addEventListener('click', closePasswordModal);
        passwordModal.addEventListener('click', (e) => {
            if (e.target === passwordModal) closePasswordModal();
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
            const ocrEngine = document.getElementById('ocrEngineSelect').value;
            const translateMode = document.querySelector('input[name="translateMode"]:checked').value;
            currentAiEngine = engine;
            currentModel = model;
            currentOcrEngine = ocrEngine;
            currentTranslateMode = translateMode;

            // localStorage에 저장
            localStorage.setItem('pdf_translator_ai_engine', engine);
            localStorage.setItem(`pdf_translator_model_${engine}`, model);
            localStorage.setItem('pdf_translator_ocr_engine', ocrEngine);
            localStorage.setItem('pdf_translator_translate_mode', translateMode);

            // API 키 저장 (Ollama 제외)
            if (engine !== 'ollama' && apiKeyInput.value) {
                apiKeys[engine] = apiKeyInput.value;
                localStorage.setItem('pdf_translator_api_keys', JSON.stringify(apiKeys));
            }

            // AI 및 OCR 표시 업데이트
            updateAiDisplay();
            updateOcrDisplay();

            closeModal();
            const ocrName = ocrEngine === 'google_vision' ? 'Google Vision API' : 'PaddleOCR v5';
            status.className = 'status success';
            status.innerHTML = `✅ OCR: ${ocrName} | AI: ${getEngineName(engine)} - ${model} 설정 완료`;
        });

        // AI 표시 업데이트 함수
        function updateAiDisplay() {
            const engineName = aiDisplayNames[currentAiEngine] || 'GPT-4o-mini';
            // 저장된 모델 또는 기본 모델 가져오기
            const savedModel = localStorage.getItem(`pdf_translator_model_${currentAiEngine}`);
            const modelName = savedModel || aiModels[currentAiEngine]?.default || '';
            // 모델명에서 핵심 부분만 추출 - 버전 명확히 표시
            let shortModel = modelName;
            if (modelName.includes('claude-haiku-4')) shortModel = 'Haiku 4.5';
            else if (modelName.includes('claude-3-5-haiku')) shortModel = 'Haiku 3.5';
            else if (modelName.includes('claude-sonnet-4')) shortModel = 'Sonnet 4';
            else if (modelName.includes('claude-3-5-sonnet')) shortModel = 'Sonnet 3.5';
            else if (modelName.includes('claude-opus-4')) shortModel = 'Opus 4.5';
            else if (modelName.includes('gpt-4o-mini')) shortModel = 'GPT-4o-mini';
            else if (modelName.includes('gpt-4o')) shortModel = 'GPT-4o';
            else if (modelName.includes('gemini-2.0')) shortModel = 'Gemini 2.0 Flash';
            else if (modelName.includes('gemini-1.5')) shortModel = 'Gemini 1.5 Flash';

            currentAiDisplay.textContent = `${engineName} (${shortModel})`;
        }

        // OCR 표시 업데이트 함수
        function updateOcrDisplay() {
            const ocrName = currentOcrEngine === 'google_vision' ? 'Google Vision API' : 'PaddleOCR v5';
            currentOcrDisplay.textContent = ocrName;
        }

        // 엔진 이름 반환
        function getEngineName(engine) {
            const names = {
                ollama: 'Ollama (로컬)',
                claude_sije: 'Claude (Sije)',
                claude_seam: 'Claude (SEAM)',
                openai: 'OpenAI GPT',
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
                        // "완료" 상태면 스피너 없이 표시
                        if (prog.stage === '완료') {
                            status.innerHTML = `✅ ${prog.stage} (${prog.current}/${prog.total}) - ${prog.detail}`;
                        } else {
                            status.innerHTML = `<span class="spinner"></span>${prog.stage} (${prog.current}/${prog.total}) - ${prog.detail} [${prog.elapsed}]`;
                        }
                    }
                } catch (e) {}
            }, 500);

            const formData = new FormData();
            formData.append('file', file);
            formData.append('target_lang', targetLang.value);
            formData.append('ai_engine', currentAiEngine);
            formData.append('model', getCurrentModel());
            formData.append('ocr_engine', currentOcrEngine);
            formData.append('translate_mode', currentTranslateMode);
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
                    pagesData = data.pages.map(page => ({
                        ...page,
                        memos: []
                    }));
                    totalPages = pagesData.length;
                    currentPage = 0;
                    selectedMemoId = null;
                    syncMemoControls();

                    status.className = 'status success';
                    status.innerHTML = `✅ ${totalPages}페이지 분석 완료! 번역을 편집하세요.`;

                    // 에디터 표시
                    editorContainer.classList.add('active');
                    translateBtn.disabled = false;

                    // 첫 페이지 표시
                    showPage(0);
                } else {
                    status.className = 'status error';
                    status.innerHTML = `❌ 오류: ${data.error}`;
                    initialGuide.style.display = 'block';
                }
            } catch (err) {
                // ★ 에러 시에도 폴링 중지
                if (progressInterval) {
                    clearInterval(progressInterval);
                    progressInterval = null;
                }
                status.className = 'status error';
                status.innerHTML = `❌ 오류: ${err.message}`;
                initialGuide.style.display = 'block';
            }
        }

        // 페이지 표시
        function showPage(pageIdx) {
            if (pageIdx < 0 || pageIdx >= totalPages) return;

            currentPage = pageIdx;
            const page = pagesData[pageIdx];

            // 페이지 전환 시 줌 리셋
            applyZoom(100);

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

            selectedMemoId = null;
            closeMemoContextMenu();
            closeMemoEditor();
            syncMemoControls();
            renderMemos();
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
                console.log('[Preview Debug] Sending request to /generate_preview...');
                console.log('[Preview Debug] image length:', page.image ? page.image.length : 'null');
                console.log('[Preview Debug] translations count:', page.translations ? page.translations.length : 0);
                
                const response = await fetch('/generate_preview', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image: page.image,
                        translations: page.translations,
                        target_lang: targetLang.value
                    })
                });

                console.log('[Preview Debug] Response status:', response.status);
                const data = await response.json();
                console.log('[Preview Debug] Response data:', data.success, data.error || 'OK');

                if (data.success) {
                    previewCache[pageIdx] = data.preview;
                    previewImg.src = 'data:image/png;base64,' + data.preview;
                    console.log('[Preview Debug] Preview loaded successfully');
                } else {
                    console.error('Preview generation failed:', data.error);
                    alert('미리보기 생성 실패: ' + data.error);
                    previewImg.src = 'data:image/png;base64,' + page.image;
                }
            } catch (error) {
                console.error('Preview error:', error);
                alert('미리보기 오류: ' + error.message);
                previewImg.src = 'data:image/png;base64,' + page.image;
            } finally {
                console.log('[Preview Debug] Finally block executed');
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
            closeMemoContextMenu();
            closeMemoEditor();
            renderMemos();
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
            renderMemos();
        });

        // ============================================================================
        // 메모 기능
        // ============================================================================
        function clampValue(value, min, max) {
            return Math.max(min, Math.min(max, value));
        }

        function ensureMemoArray(page) {
            if (!page.memos) {
                page.memos = [];
            }
            return page.memos;
        }

        function getMemoById(memoId) {
            const page = pagesData[currentPage];
            if (!page || !memoId) return null;
            return ensureMemoArray(page).find(memo => memo.id === memoId) || null;
        }

        function getMemoStyle(memo) {
            return {
                ...memoDefaults,
                ...(memo && memo.style ? memo.style : {})
            };
        }

        function createMemoId() {
            memoIdSeq += 1;
            return `memo_${Date.now()}_${memoIdSeq}`;
        }

        function getStageRect() {
            return previewStage.getBoundingClientRect();
        }

        function clientToStageRatio(clientX, clientY) {
            const rect = getStageRect();
            if (!rect.width || !rect.height) {
                return { x: 0, y: 0 };
            }
            const x = (clientX - rect.left) / rect.width;
            const y = (clientY - rect.top) / rect.height;
            return {
                x: clampValue(x, 0, 1),
                y: clampValue(y, 0, 1)
            };
        }

        function positionFloating(element, clientX, clientY) {
            element.style.left = `${clientX}px`;
            element.style.top = `${clientY}px`;

            requestAnimationFrame(() => {
                const rect = element.getBoundingClientRect();
                let left = clientX;
                let top = clientY;

                if (left + rect.width > window.innerWidth - 8) {
                    left = window.innerWidth - rect.width - 8;
                }
                if (top + rect.height > window.innerHeight - 8) {
                    top = window.innerHeight - rect.height - 8;
                }
                if (left < 8) left = 8;
                if (top < 8) top = 8;

                element.style.left = `${left}px`;
                element.style.top = `${top}px`;
            });
        }

        function closeMemoContextMenu() {
            memoContextMenu.classList.remove('active');
            memoContextState = null;
        }

        function openMemoContextMenu({ clientX, clientY, memoId, position }) {
            memoContextState = { memoId, position, clientX, clientY };
            memoContextMenu.querySelector('[data-action="add"]').style.display = memoId ? 'none' : 'block';
            memoContextMenu.querySelector('[data-action="edit"]').style.display = memoId ? 'block' : 'none';
            memoContextMenu.querySelector('[data-action="delete"]').style.display = memoId ? 'block' : 'none';
            memoContextMenu.classList.add('active');
            positionFloating(memoContextMenu, clientX, clientY);
        }

        function closeMemoEditor() {
            memoEditor.classList.remove('active');
            memoEditorState = null;
            memoEditorText.value = '';
        }

        function openMemoEditor({ mode, memoId, position, clientX, clientY, arrow = null }) {
            memoEditorState = { mode, memoId, position, arrow };
            memoEditorTitle.textContent = mode === 'edit' ? '메모 편집' : (arrow ? '메모 추가 (화살표)' : '메모 추가');
            if (mode === 'edit' && memoId) {
                const memo = getMemoById(memoId);
                memoEditorText.value = memo ? memo.text : '';
            } else {
                memoEditorText.value = '';
            }
            memoEditor.classList.add('active');
            positionFloating(memoEditor, clientX, clientY);
            memoEditorText.focus();
        }

        function setSelectedMemo(memoId, skipRender = false) {
            selectedMemoId = memoId;
            syncMemoControls();
            if (!skipRender) {
                renderMemos();
            }
        }

        let memoControlSync = false;
        function syncMemoControls() {
            const memo = getMemoById(selectedMemoId);
            const style = getMemoStyle(memo);
            memoControlSync = true;
            memoFontSize.value = style.fontSize;
            memoTextColor.value = style.color;
            memoBold.checked = !!style.bold;
            memoWidth.value = style.width;
            memoPadding.value = style.padding;
            memoOpacity.value = Math.round(style.opacity * 100);
            memoBgEnabled.checked = !!style.backgroundEnabled;
            memoBgColor.value = style.backgroundColor;
            memoBorderWidth.value = style.borderWidth;
            memoBorderColor.value = style.borderColor;
            memoBgColor.disabled = !memoBgEnabled.checked;
            // 굵게 버튼 상태 동기화
            memoBoldBtn.classList.toggle('active', style.bold);
            memoControlSync = false;
        }

        function applyMemoStyle(key, value) {
            if (memoControlSync) return;
            const memo = getMemoById(selectedMemoId);
            if (memo) {
                memo.style = memo.style || {};
                memo.style[key] = value;
                renderMemos();
            } else {
                memoDefaults[key] = value;
            }
            syncMemoControls();
        }

        function renderMemos() {
            memoLayer.innerHTML = '';
            if (!isPreviewMode) {
                memoLayer.style.display = 'none';
                return;
            }
            const page = pagesData[currentPage];
            if (!page) {
                memoLayer.style.display = 'none';
                return;
            }
            const memos = ensureMemoArray(page);
            memoLayer.style.display = 'block';

            const rect = getStageRect();
            const stageWidth = rect.width || 1;
            const stageHeight = rect.height || 1;

            memos.forEach(memo => {
                const style = getMemoStyle(memo);
                const memoEl = document.createElement('div');
                memoEl.className = 'memo-item' + (memo.id === selectedMemoId ? ' selected' : '');
                memoEl.dataset.memoId = memo.id;
                memoEl.textContent = memo.text;
                memoEl.style.left = `${memo.x * stageWidth}px`;
                memoEl.style.top = `${memo.y * stageHeight}px`;
                memoEl.style.width = `${style.width}px`;
                memoEl.style.padding = `${style.padding}px`;
                memoEl.style.fontSize = `${style.fontSize}px`;
                memoEl.style.color = style.color;
                memoEl.style.opacity = style.opacity;
                memoEl.style.fontWeight = style.bold ? 'bold' : 'normal';
                memoEl.style.backgroundColor = style.backgroundEnabled ? style.backgroundColor : 'transparent';
                memoEl.style.border = style.borderWidth > 0
                    ? `${style.borderWidth}px solid ${style.borderColor}`
                    : 'none';

                memoEl.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return;
                    e.preventDefault();
                    setSelectedMemo(memo.id, true);
                    memoLayer.querySelectorAll('.memo-item.selected').forEach(item => item.classList.remove('selected'));
                    memoEl.classList.add('selected');
                    const startRect = getStageRect();
                    memoDragState = {
                        memoId: memo.id,
                        element: memoEl,
                        startClientX: e.clientX,
                        startClientY: e.clientY,
                        startLeft: memo.x * (startRect.width || 1),
                        startTop: memo.y * (startRect.height || 1),
                        stageWidth: startRect.width || 1,
                        stageHeight: startRect.height || 1
                    };
                    document.addEventListener('mousemove', onMemoDrag);
                    document.addEventListener('mouseup', onMemoDragEnd);
                });

                memoEl.addEventListener('click', (e) => {
                    e.stopPropagation();
                    setSelectedMemo(memo.id);
                });

                // 메모 박스 리사이즈 감지 (드래그로 크기 조절)
                const resizeObserver = new ResizeObserver((entries) => {
                    for (const entry of entries) {
                        const newWidth = Math.round(entry.contentRect.width + (style.padding * 2));
                        if (newWidth !== style.width && newWidth >= 80) {
                            memo.style = memo.style || {};
                            memo.style.width = newWidth;
                            memoWidth.value = newWidth;
                        }
                    }
                });
                resizeObserver.observe(memoEl);

                memoLayer.appendChild(memoEl);
            });

            // 화살표 렌더링
            renderArrows(memos, stageWidth, stageHeight);
        }

        // ★ object-fit: contain 이미지의 실제 렌더링 크기 계산
        // 주의: max-width/max-height: none이면 이미지는 자연 크기로 표시됨
        function getActualImageSize(imgElement) {
            // getBoundingClientRect()는 transform(줌) 적용 후 크기 반환
            const rect = imgElement.getBoundingClientRect();
            const zoomScale = currentZoom / 100;
            
            // 줌 적용 전 원본 렌더링 크기 (줌 제거)
            const baseWidth = rect.width / zoomScale;
            const baseHeight = rect.height / zoomScale;
            
            const naturalWidth = imgElement.naturalWidth || 1;
            const naturalHeight = imgElement.naturalHeight || 1;
            
            // object-fit: contain이 실제로 적용되는 경우를 위한 계산
            // (현재 CSS에서는 max-width/max-height: none이라 적용 안됨)
            // 이미지가 자연 크기로 표시되므로 offset은 0
            return {
                width: baseWidth,
                height: baseHeight,
                offsetX: 0,
                offsetY: 0,
                containerWidth: baseWidth,
                containerHeight: baseHeight
            };
        }

        function renderArrows(memos, stageWidth, stageHeight) {
            // ★ 줌을 고려한 이미지 위치 계산
            const imgRect = previewImg.getBoundingClientRect();
            const stageRect = previewStage.getBoundingClientRect();
            const imgOffsetX = imgRect.left - stageRect.left;
            const imgOffsetY = imgRect.top - stageRect.top;
            const imgWidth = imgRect.width || 1;
            const imgHeight = imgRect.height || 1;
            
            let arrowsHtml = `
                <defs>
                    <marker id="arrowhead-saved" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#ff6b6b"/>
                    </marker>
                </defs>
            `;
            memos.forEach(memo => {
                if (memo.arrow) {
                    const memoX = memo.x * stageWidth;
                    const memoY = memo.y * stageHeight;
                    // ★ 화살표 타겟은 이미지 기준 비율이므로 이미지 렌더링 크기로 변환
                    const targetX = imgOffsetX + memo.arrow.targetX * imgWidth;
                    const targetY = imgOffsetY + memo.arrow.targetY * imgHeight;
                    arrowsHtml += `
                        <line x1="${memoX}" y1="${memoY}" x2="${targetX}" y2="${targetY}"
                              stroke="#ff6b6b" stroke-width="2" marker-end="url(#arrowhead-saved)"
                              data-memo-id="${memo.id}"/>
                    `;
                }
            });
            arrowLayer.innerHTML = arrowsHtml;
        }

        function onMemoDrag(e) {
            if (!memoDragState) return;
            const memo = getMemoById(memoDragState.memoId);
            if (!memo) return;

            const dx = e.clientX - memoDragState.startClientX;
            const dy = e.clientY - memoDragState.startClientY;
            let newLeft = memoDragState.startLeft + dx;
            let newTop = memoDragState.startTop + dy;

            newLeft = clampValue(newLeft, 0, memoDragState.stageWidth);
            newTop = clampValue(newTop, 0, memoDragState.stageHeight);

            memo.x = memoDragState.stageWidth ? newLeft / memoDragState.stageWidth : memo.x;
            memo.y = memoDragState.stageHeight ? newTop / memoDragState.stageHeight : memo.y;

            if (memoDragState.element) {
                memoDragState.element.style.left = `${newLeft}px`;
                memoDragState.element.style.top = `${newTop}px`;
            }
        }

        function onMemoDragEnd() {
            if (!memoDragState) return;
            memoDragState = null;
            document.removeEventListener('mousemove', onMemoDrag);
            document.removeEventListener('mouseup', onMemoDragEnd);
            renderMemos();
        }

        // ★ 클릭 위치에서 가장 가까운 bbox 찾아 중앙 좌표 반환
        function findNearestBboxCenter(clickX, clickY) {
            const page = pagesData[currentPage];
            if (!page || !page.translations || page.translations.length === 0) {
                console.log('[Arrow] No translations data');
                return null;
            }

            // ★ 줌을 고려한 이미지 렌더링 크기 사용
            const imgRect = previewImg.getBoundingClientRect();
            const stageRect = previewStage.getBoundingClientRect();
            
            // 스테이지 내에서의 상대 클릭 좌표
            const relClickX = clickX;
            const relClickY = clickY;
            
            // 이미지 실제 크기 (bbox는 이미지 픽셀 좌표 기준)
            const imgWidth = previewImg.naturalWidth || 1;
            const imgHeight = previewImg.naturalHeight || 1;
            
            // ★ 줌 적용된 이미지 렌더링 크기
            const renderedWidth = imgRect.width || 1;
            const renderedHeight = imgRect.height || 1;
            
            // ★ 이미지가 스테이지 내에서 어디에 위치하는지 계산
            const imgOffsetX = imgRect.left - stageRect.left;
            const imgOffsetY = imgRect.top - stageRect.top;
            
            // ★ 클릭이 이미지 영역 내인지 확인하고 이미지 내 좌표로 변환
            const imgClickX = ((relClickX - imgOffsetX) / renderedWidth) * imgWidth;
            const imgClickY = ((relClickY - imgOffsetY) / renderedHeight) * imgHeight;

            console.log('[Arrow] Zoom level:', currentZoom + '%');
            console.log('[Arrow] Click pos (stage):', clickX, clickY);
            console.log('[Arrow] Image offset in stage:', imgOffsetX, imgOffsetY);
            console.log('[Arrow] Rendered size:', renderedWidth, 'x', renderedHeight);
            console.log('[Arrow] Natural size:', imgWidth, 'x', imgHeight);
            console.log('[Arrow] Click pos (image px):', imgClickX, imgClickY);

            let nearestBbox = null;
            let minDistance = Infinity;
            let nearestText = '';

            page.translations.forEach((item, idx) => {
                if (!item.bbox || item.bbox.length < 4) return;

                // bbox 형식 확인: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 또는 다른 형식
                let bboxX1, bboxY1, bboxX2, bboxY2;
                
                // bbox[0]이 배열인지 확인
                if (Array.isArray(item.bbox[0])) {
                    // [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] 형식
                    const xs = item.bbox.map(p => p[0]);
                    const ys = item.bbox.map(p => p[1]);
                    bboxX1 = Math.min(...xs);
                    bboxY1 = Math.min(...ys);
                    bboxX2 = Math.max(...xs);
                    bboxY2 = Math.max(...ys);
                } else {
                    // [x1, y1, x2, y2] 형식
                    bboxX1 = item.bbox[0];
                    bboxY1 = item.bbox[1];
                    bboxX2 = item.bbox[2];
                    bboxY2 = item.bbox[3];
                }

                // bbox 중앙 좌표
                const centerX = (bboxX1 + bboxX2) / 2;
                const centerY = (bboxY1 + bboxY2) / 2;

                // 클릭이 bbox 내부인지 확인
                const isInside = imgClickX >= bboxX1 && imgClickX <= bboxX2 &&
                                 imgClickY >= bboxY1 && imgClickY <= bboxY2;

                // 거리 계산 (내부면 0, 외부면 중앙까지 거리)
                let distance;
                if (isInside) {
                    distance = 0;
                } else {
                    distance = Math.sqrt(
                        Math.pow(imgClickX - centerX, 2) +
                        Math.pow(imgClickY - centerY, 2)
                    );
                }

                if (distance < minDistance) {
                    minDistance = distance;
                    nearestText = item.text || item.translated || '';
                    nearestBbox = {
                        centerX: centerX,
                        centerY: centerY,
                        imgWidth: imgWidth,
                        imgHeight: imgHeight,
                        bbox: [bboxX1, bboxY1, bboxX2, bboxY2]
                    };
                }
            });

            // 스냅 거리 제한 (이미지 픽셀 기준 150px 이내)
            if (nearestBbox && minDistance <= 150) {
                console.log('[Arrow] Nearest bbox:', nearestBbox.bbox, 'text:', nearestText);
                console.log('[Arrow] Center (image px):', nearestBbox.centerX, nearestBbox.centerY);
                // 이미지 좌표를 비율(0~1)로 변환
                const result = {
                    x: nearestBbox.centerX / nearestBbox.imgWidth,
                    y: nearestBbox.centerY / nearestBbox.imgHeight
                };
                console.log('[Arrow] Center (ratio):', result.x, result.y);
                return result;
            }

            console.log('[Arrow] No snap target (distance:', minDistance, ')');
            return null;  // 스냅 대상 없음
        }

        // 화살표 드래그 헬퍼 함수
        function drawTempArrow(x1, y1, x2, y2) {
            arrowLayer.innerHTML = `
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#667eea"/>
                    </marker>
                </defs>
                <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
                      stroke="#667eea" stroke-width="2" marker-end="url(#arrowhead)"/>
            `;
        }
        function clearTempArrow() {
            arrowLayer.innerHTML = '';
        }

        // 우클릭 + 드래그로 화살표 생성
        previewStage.addEventListener('mousedown', (e) => {
            if (e.button !== 2) return;  // 우클릭만
            if (!isPreviewMode) return;

            const memoEl = e.target.closest('.memo-item');
            if (memoEl) return;  // 메모 위에서는 기존 동작

            const rect = getStageRect();
            const startX = e.clientX - rect.left;
            const startY = e.clientY - rect.top;
            
            // ★ bbox 중앙으로 스냅 시도 (에러 방지)
            let snappedPosition = null;
            try {
                snappedPosition = findNearestBboxCenter(startX, startY);
            } catch (err) {
                console.error('[Arrow] Snap error:', err);
            }
            
            let startPosition;
            let snappedStartX = startX;
            let snappedStartY = startY;
            
            if (snappedPosition) {
                // ★ 줌을 고려하여 비율 → 스테이지 픽셀 변환
                const imgRect = previewImg.getBoundingClientRect();
                const imgOffsetX = imgRect.left - rect.left;
                const imgOffsetY = imgRect.top - rect.top;
                
                startPosition = snappedPosition;
                snappedStartX = imgOffsetX + snappedPosition.x * imgRect.width;
                snappedStartY = imgOffsetY + snappedPosition.y * imgRect.height;
                console.log('[Arrow] Snapped to bbox center:', snappedPosition);
                console.log('[Arrow] Image offset:', imgOffsetX, imgOffsetY);
                console.log('[Arrow] Snapped stage coords:', snappedStartX, snappedStartY);
            } else {
                // 스냅 대상 없으면 원래 클릭 위치 사용
                startPosition = clientToStageRatio(e.clientX, e.clientY);
            }

            arrowDragState = {
                startX: snappedStartX,
                startY: snappedStartY,
                startPosition,
                startClientX: e.clientX,
                startClientY: e.clientY,
                hasDragged: false
            };
        });

        document.addEventListener('mousemove', (e) => {
            if (!arrowDragState) return;

            const dx = e.clientX - arrowDragState.startClientX;
            const dy = e.clientY - arrowDragState.startClientY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance > 10) {
                arrowDragState.hasDragged = true;
                const rect = getStageRect();
                const currentX = e.clientX - rect.left;
                const currentY = e.clientY - rect.top;
                // 화살표: 현재 위치(메모박스) → 시작 위치(화살촉)
                drawTempArrow(currentX, currentY, arrowDragState.startX, arrowDragState.startY);
            }
        });

        document.addEventListener('mouseup', (e) => {
            if (!arrowDragState) return;

            if (arrowDragState.hasDragged) {
                // ★ 화살표 유지 (clearTempArrow 제거) - 메모 에디터에서 처리
                const endPosition = clientToStageRatio(e.clientX, e.clientY);
                // 메모 에디터 열기 (화살표 정보 포함)
                openMemoEditor({
                    mode: 'add',
                    memoId: null,
                    position: endPosition,  // 메모박스 위치
                    clientX: e.clientX,
                    clientY: e.clientY,
                    arrow: {
                        targetX: arrowDragState.startPosition.x,  // 화살촉 위치
                        targetY: arrowDragState.startPosition.y
                    }
                });
            }
            arrowDragState = null;
        });

        previewStage.addEventListener('contextmenu', (e) => {
            if (!isPreviewMode) return;
            e.preventDefault();

            // 드래그 중이면 컨텍스트 메뉴 무시
            if (arrowDragState && arrowDragState.hasDragged) return;

            const memoEl = e.target.closest('.memo-item');
            const position = clientToStageRatio(e.clientX, e.clientY);

            if (memoEl) {
                const memoId = memoEl.dataset.memoId;
                setSelectedMemo(memoId);
                openMemoContextMenu({
                    clientX: e.clientX,
                    clientY: e.clientY,
                    memoId,
                    position
                });
            } else {
                openMemoContextMenu({
                    clientX: e.clientX,
                    clientY: e.clientY,
                    memoId: null,
                    position
                });
            }
        });

        previewStage.addEventListener('click', (e) => {
            // memoContextMenu 내부 클릭은 무시
            if (memoContextMenu.contains(e.target)) return;
            closeMemoContextMenu();
            if (selectedMemoId && !e.target.closest('.memo-item')) {
                setSelectedMemo(null);
            }
        });

        memoContextMenu.addEventListener('click', (e) => {
            const action = e.target.dataset.action;
            if (!action || !memoContextState) return;

            if (action === 'add') {
                openMemoEditor({
                    mode: 'add',
                    memoId: null,
                    position: memoContextState.position,
                    clientX: memoContextState.clientX,
                    clientY: memoContextState.clientY
                });
            } else if (action === 'edit' && memoContextState.memoId) {
                openMemoEditor({
                    mode: 'edit',
                    memoId: memoContextState.memoId,
                    position: memoContextState.position,
                    clientX: memoContextState.clientX,
                    clientY: memoContextState.clientY
                });
            } else if (action === 'delete' && memoContextState.memoId) {
                const page = pagesData[currentPage];
                if (page) {
                    page.memos = ensureMemoArray(page).filter(memo => memo.id !== memoContextState.memoId);
                    if (selectedMemoId === memoContextState.memoId) {
                        selectedMemoId = null;
                    }
                    renderMemos();
                    syncMemoControls();
                }
            }
            closeMemoContextMenu();
        });

        memoEditorSave.addEventListener('click', () => {
            if (!memoEditorState) return;
            const text = memoEditorText.value.trim();
            const page = pagesData[currentPage];
            if (!page) return;
            const memos = ensureMemoArray(page);

            if (memoEditorState.mode === 'add') {
                if (text) {
                    const memo = {
                        id: createMemoId(),
                        text,
                        x: memoEditorState.position.x,
                        y: memoEditorState.position.y,
                        style: { ...memoDefaults }
                    };
                    // 화살표 정보 추가
                    if (memoEditorState.arrow) {
                        memo.arrow = {
                            targetX: memoEditorState.arrow.targetX,
                            targetY: memoEditorState.arrow.targetY
                        };
                    }
                    memos.push(memo);
                    setSelectedMemo(memo.id);
                }
            } else if (memoEditorState.mode === 'edit') {
                const memo = getMemoById(memoEditorState.memoId);
                if (memo) {
                    if (text) {
                        memo.text = text;
                    } else {
                        page.memos = memos.filter(item => item.id !== memo.id);
                        if (selectedMemoId === memo.id) {
                            selectedMemoId = null;
                        }
                    }
                }
            }

            renderMemos();
            closeMemoEditor();
        });

        memoEditorCancel.addEventListener('click', closeMemoEditor);

        document.addEventListener('click', (e) => {
            if (!memoContextMenu.contains(e.target)) {
                closeMemoContextMenu();
            }
        });

        window.addEventListener('resize', renderMemos);
        previewImg.addEventListener('load', () => {
            renderMemos();
        });

        // 메모 패널 펼침/접힘 토글
        const memoExpandBtn = document.getElementById('memoExpandBtn');
        const memoPanel = document.getElementById('memoPanel');
        memoExpandBtn.addEventListener('click', () => {
            const isExpanded = memoPanel.classList.toggle('expanded');
            memoExpandBtn.textContent = isExpanded ? '▲' : '▼';
        });

        // 굵게 버튼 (B) 토글
        memoBoldBtn.addEventListener('click', () => {
            memoBold.checked = !memoBold.checked;
            memoBoldBtn.classList.toggle('active', memoBold.checked);
            applyMemoStyle('bold', memoBold.checked);
        });

        // 기본값 버튼
        const memoDefaultBtn = document.getElementById('memoDefaultBtn');
        memoDefaultBtn.addEventListener('click', () => {
            memoFontSize.value = memoDefaults.fontSize;
            memoTextColor.value = '#111111';
            memoBold.checked = false;
            memoBoldBtn.classList.remove('active');
            memoWidth.value = memoDefaults.width;
            memoPadding.value = memoDefaults.padding;
            memoOpacity.value = memoDefaults.opacity * 100;
            memoBgEnabled.checked = true;
            memoBgColor.value = '#ffffff';
            memoBorderWidth.value = memoDefaults.borderWidth;
            memoBorderColor.value = '#333333';
        });

        memoFontSize.addEventListener('input', () => {
            const value = parseInt(memoFontSize.value, 10);
            applyMemoStyle('fontSize', Number.isNaN(value) ? memoDefaults.fontSize : value);
        });
        memoTextColor.addEventListener('input', () => applyMemoStyle('color', memoTextColor.value));
        memoBold.addEventListener('change', () => applyMemoStyle('bold', memoBold.checked));
        memoWidth.addEventListener('input', () => {
            const value = parseInt(memoWidth.value, 10);
            applyMemoStyle('width', Number.isNaN(value) ? memoDefaults.width : value);
        });
        memoPadding.addEventListener('input', () => {
            const value = parseInt(memoPadding.value, 10);
            applyMemoStyle('padding', Number.isNaN(value) ? memoDefaults.padding : value);
        });
        memoOpacity.addEventListener('input', () => {
            const value = parseInt(memoOpacity.value, 10);
            const resolved = Number.isNaN(value) ? memoDefaults.opacity : value / 100;
            applyMemoStyle('opacity', resolved);
        });
        memoBgEnabled.addEventListener('change', () => applyMemoStyle('backgroundEnabled', memoBgEnabled.checked));
        memoBgColor.addEventListener('input', () => applyMemoStyle('backgroundColor', memoBgColor.value));
        memoBorderWidth.addEventListener('input', () => {
            const value = parseInt(memoBorderWidth.value, 10);
            applyMemoStyle('borderWidth', Number.isNaN(value) ? memoDefaults.borderWidth : value);
        });
        memoBorderColor.addEventListener('input', () => applyMemoStyle('borderColor', memoBorderColor.value));

        // 번역 테이블 갱신
        function updateTranslationTable(page) {
            translationBody.innerHTML = '';

            if (!page.translations || page.translations.length === 0) {
                translationBody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#999;padding:30px;">한글 텍스트가 없습니다</td></tr>';
                return;
            }

            // 한글이 포함된 텍스트만 필터링 (영어/숫자만 있는 항목 제외)
            const koreanRegex = /[\uAC00-\uD7AF]/;
            let displayIdx = 0;

            page.translations.forEach((item, idx) => {
                // 한글이 없으면 제외
                if (!koreanRegex.test(item.text)) return;

                displayIdx++;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="idx">${displayIdx}</td>
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
            status.innerHTML = `✅ 페이지 ${currentPage + 1} 번역 확정됨`;
            
            // 미리보기 모드면 즉시 갱신 (캐시 무효화 후)
            if (isPreviewMode) {
                delete previewCache[currentPage];  // 캐시 삭제
                showPreviewImage(currentPage, true);  // 강제 새로고침
            }
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
                status.innerHTML = `✅ 전체 ${pagesData.length}페이지 재번역 완료`;
            } catch (err) {
                status.className = 'status error';
                status.innerHTML = `❌ 재번역 오류: ${err.message}`;
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
                    status.innerHTML = `✅ 번역 완료! ${data.files.length}개 파일 생성됨`;

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
                    status.innerHTML = `❌ 오류: ${data.error}`;
                }
            } catch (err) {
                status.className = 'status error';
                status.innerHTML = `❌ 오류: ${err.message}`;
            }

            translateBtn.disabled = false;
        });

        // ============================================================================
        // 용어 사전 관리
        // ============================================================================
        const dictBtn = document.getElementById('dictBtn');
        const dictModal = document.getElementById('dictModal');
        const closeDict = document.getElementById('closeDict');
        const closeDictBtn = document.getElementById('closeDictBtn');
        const dictTabs = document.querySelectorAll('.dict-tab');
        const dictBody = document.getElementById('dictBody');
        const dictKorean = document.getElementById('dictKorean');
        const dictTranslation = document.getElementById('dictTranslation');
        const addTermBtn = document.getElementById('addTermBtn');
        const dictSearch = document.getElementById('dictSearch');
        const dictCount = document.getElementById('dictCount');

        let currentDictLang = 'english';
        let dictData = {};

        // 모달 열기
        dictBtn.addEventListener('click', async () => {
            dictModal.classList.add('active');
            await loadDictionary();
        });

        // 모달 닫기
        closeDict.addEventListener('click', () => dictModal.classList.remove('active'));
        closeDictBtn.addEventListener('click', () => dictModal.classList.remove('active'));
        dictModal.addEventListener('click', (e) => {
            if (e.target === dictModal) dictModal.classList.remove('active');
        });

        // 탭 전환
        dictTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                dictTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentDictLang = tab.dataset.lang;
                renderDictTable();
            });
        });

        // 사전 로드
        async function loadDictionary() {
            try {
                const res = await fetch('/api/dictionary');
                dictData = await res.json();
                renderDictTable();
            } catch (err) {
                console.error('Failed to load dictionary:', err);
            }
        }

        // 테이블 렌더링
        function renderDictTable() {
            const langDict = dictData[currentDictLang] || {};
            const searchTerm = dictSearch.value.toLowerCase();

            const entries = Object.entries(langDict)
                .filter(([kr, termData]) => {
                    const full = typeof termData === 'object' ? termData.full : termData;
                    const abbr = typeof termData === 'object' ? (termData.abbr || '') : '';
                    return kr.toLowerCase().includes(searchTerm) ||
                           full.toLowerCase().includes(searchTerm) ||
                           abbr.toLowerCase().includes(searchTerm);
                })
                .sort((a, b) => a[0].localeCompare(b[0], 'ko'));

            dictBody.innerHTML = entries.map(([korean, termData]) => {
                const full = typeof termData === 'object' ? termData.full : termData;
                const abbr = typeof termData === 'object' ? (termData.abbr || '') : '';
                return `
                <tr data-korean="${korean}">
                    <td class="korean-cell">${korean}</td>
                    <td class="trans-cell">${full}</td>
                    <td class="abbr-cell">${abbr}</td>
                    <td class="actions">
                        <button class="edit-btn" onclick="editTerm('${korean}')">✏️</button>
                        <button class="delete-btn" onclick="deleteTerm('${korean}')">🗑️</button>
                    </td>
                </tr>
            `}).join('');

            dictCount.textContent = `총 ${entries.length}개 용어`;
        }

        // 검색
        dictSearch.addEventListener('input', renderDictTable);

        // 용어 추가
        addTermBtn.addEventListener('click', async () => {
            const korean = dictKorean.value.trim();
            const translation = dictTranslation.value.trim();
            const abbr = document.getElementById('dictAbbr').value.trim();

            if (!korean || !translation) {
                alert('한글 용어와 번역을 모두 입력하세요.');
                return;
            }

            try {
                const res = await fetch(`/api/dictionary/${currentDictLang}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ korean, translation, abbr })
                });
                const data = await res.json();

                if (data.success) {
                    dictKorean.value = '';
                    dictTranslation.value = '';
                    document.getElementById('dictAbbr').value = '';
                    await loadDictionary();
                } else {
                    alert('추가 실패: ' + data.error);
                }
            } catch (err) {
                alert('오류: ' + err.message);
            }
        });

        // 용어 수정
        window.editTerm = function(korean) {
            const row = document.querySelector(`tr[data-korean="${korean}"]`);
            const transCell = row.querySelector('.trans-cell');
            const abbrCell = row.querySelector('.abbr-cell');
            const actionsCell = row.querySelector('.actions');
            const currentTrans = transCell.textContent;
            const currentAbbr = abbrCell.textContent;

            transCell.innerHTML = `<input type="text" class="edit-input edit-trans" value="${currentTrans}">`;
            abbrCell.innerHTML = `<input type="text" class="edit-input edit-abbr" value="${currentAbbr}">`;
            actionsCell.innerHTML = `
                <button class="save-btn" onclick="saveTerm('${korean}')">💾</button>
                <button class="cancel-btn" onclick="renderDictTable()">✖️</button>
            `;
            transCell.querySelector('input').focus();
        };

        // 용어 저장
        window.saveTerm = async function(korean) {
            const row = document.querySelector(`tr[data-korean="${korean}"]`);
            const transInput = row.querySelector('.edit-trans');
            const abbrInput = row.querySelector('.edit-abbr');
            const translation = transInput.value.trim();
            const abbr = abbrInput ? abbrInput.value.trim() : '';

            if (!translation) {
                alert('번역을 입력하세요.');
                return;
            }

            try {
                const res = await fetch(`/api/dictionary/${currentDictLang}/${encodeURIComponent(korean)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ translation, abbr })
                });
                const data = await res.json();

                if (data.success) {
                    await loadDictionary();
                } else {
                    alert('수정 실패: ' + data.error);
                }
            } catch (err) {
                alert('오류: ' + err.message);
            }
        };

        // 용어 삭제
        window.deleteTerm = async function(korean) {
            if (!confirm(`"${korean}" 용어를 삭제하시겠습니까?`)) return;

            try {
                const res = await fetch(`/api/dictionary/${currentDictLang}/${encodeURIComponent(korean)}`, {
                    method: 'DELETE'
                });
                const data = await res.json();

                if (data.success) {
                    await loadDictionary();
                } else {
                    alert('삭제 실패: ' + data.error);
                }
            } catch (err) {
                alert('오류: ' + err.message);
            }
        };
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
    """파일 업로드 + OCR + 초기 번역 (배치 OCR + 병렬 번역 최적화)"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "파일이 없습니다"})

        file = request.files['file']
        target_lang = request.form.get('target_lang', 'english')
        ai_engine = request.form.get('ai_engine', 'openai')
        api_key = request.form.get('api_key', None)
        model = request.form.get('model', None)
        ocr_engine = request.form.get('ocr_engine', 'paddleocr')
        translate_mode = request.form.get('translate_mode', 'sequential')  # 'batch' 또는 'sequential'

        # ★ 하드코딩된 API 키 우선 사용
        if ai_engine in HARDCODED_API_KEYS:
            api_key = HARDCODED_API_KEYS[ai_engine]

        print(f"[AI Engine] {ai_engine}, [Model] {model}, [OCR Engine] {ocr_engine}, [Translate Mode] {translate_mode}", flush=True)
        logger.info(f"[AI Engine] {ai_engine}, [Model] {model}, [OCR Engine] {ocr_engine}, [Translate Mode] {translate_mode}")
        print(f"[Debug] ai_engine raw: '{request.form.get('ai_engine')}' -> parsed: '{ai_engine}'", flush=True)
        print(f"[Debug] api_key present: {bool(api_key)}, length: {len(api_key) if api_key else 0}", flush=True)
        logger.info(f"[Debug] api_key present: {bool(api_key)}, length: {len(api_key) if api_key else 0}")

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

        pages = []
        total_pages = len(image_paths)
        all_pages_data = []
        
        import time  # 성능 측정용

        # ===== 1단계: 배치 OCR (모든 페이지 한번에) =====
        ocr_name = 'Vision' if ocr_engine == 'google_vision' else 'PaddleOCR'
        update_progress("OCR", 1, total_pages, f"전체 {total_pages}개 페이지 일괄 OCR 처리 중... ({ocr_name})")
        print(f"[{ocr_name}] Processing {total_pages} pages at once...", flush=True)

        ocr_start = time.time()
        all_ocr_results = get_ocr_results_with_engine(image_paths, ocr_engine)
        ocr_time = time.time() - ocr_start
        print(f"[TIMING] OCR ({ocr_name}) took {ocr_time:.2f}s for {total_pages} pages", flush=True)
        
        # OCR 결과와 이미지 정보 결합
        for i, (img_path, texts) in enumerate(zip(image_paths, all_ocr_results)):
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            all_pages_data.append({
                "page_idx": i,
                "img_path": img_path,
                "image_base64": image_base64,
                "texts": texts
            })
        
        print(f"[Batch OCR] Complete - {sum(len(p['texts']) for p in all_pages_data)} total texts", flush=True)

        # ===== 2단계: 번역 (엔진별 최적화) =====
        total_texts = sum(len(p["texts"]) for p in all_pages_data)
        translate_start = time.time()

        logger.info(f"[Translation Start] ai_engine='{ai_engine}', total_texts={total_texts}, api_key_present={bool(api_key)}")
        print(f"[Translation Start] ai_engine='{ai_engine}', total_texts={total_texts}, api_key_present={bool(api_key)}", flush=True)

        if ai_engine == "gemini" and api_key and total_texts > 0:
            logger.info(f"[Translation] Using GEMINI branch, translate_mode={translate_mode}")

            if translate_mode == "batch":
                # Gemini 배치 모드: 청크 단위로 API 호출 (100개씩)
                num_chunks = (total_texts + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE
                update_progress("번역", 0, total_texts, f"전체 {total_texts}개 텍스트 일괄 번역 중... (Gemini 배치)")
                print(f"[Gemini Batch] Total {total_texts} texts from {total_pages} pages", flush=True)

                batch_input = [{"page_idx": p["page_idx"], "texts": p["texts"]} for p in all_pages_data]
                translations_by_page = translate_batch_with_gemini(batch_input, target_lang, api_key, model)

                translate_time = time.time() - translate_start
                print(f"[TIMING] Gemini Batch Translation took {translate_time:.2f}s for {total_texts} texts", flush=True)

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
                # Gemini 순차 모드: 페이지별 API 호출 (안정적, 토큰 한도 내 처리)
                print(f"[Gemini Sequential] Processing {total_pages} pages one by one", flush=True)

                for page_data in all_pages_data:
                    page_idx = page_data["page_idx"]
                    page_texts = page_data["texts"]
                    update_progress("번역", page_idx + 1, total_pages,
                                   f"페이지 {page_idx + 1}/{total_pages} - {len(page_texts)}개 텍스트 번역 중... (Gemini 순차)")

                    if page_texts:
                        # 단일 페이지를 배치로 전송
                        batch_input = [{"page_idx": page_idx, "texts": page_texts}]
                        translations_by_page = translate_batch_with_gemini(batch_input, target_lang, api_key, model)
                        translations = translations_by_page.get(page_idx, [])
                    else:
                        translations = []

                    pages.append({
                        "image": page_data["image_base64"],
                        "image_path": page_data["img_path"],
                        "translations": translations,
                        "confirmed": False
                    })

                translate_time = time.time() - translate_start
                print(f"[TIMING] Gemini Sequential Translation took {translate_time:.2f}s for {total_texts} texts ({total_pages} pages)", flush=True)
                
        elif ai_engine in ("claude_sije", "claude_seam", "openai") and api_key and total_texts > 0:
            # Claude/OpenAI: 병렬 번역 (동시 API 호출)
            logger.info(f"[Translation] Using CLAUDE/OPENAI branch - ai_engine='{ai_engine}'")
            update_progress("번역", 1, 1, f"전체 {total_texts}개 텍스트 병렬 번역 중... ({ai_engine.upper()} 병렬)")
            print(f"[Parallel Translation] {ai_engine.upper()} - {total_pages} pages", flush=True)
            
            translations_by_page = translate_pages_parallel(
                all_pages_data, target_lang, ai_engine, api_key, model, max_workers=3
            )
            
            translate_time = time.time() - translate_start
            print(f"[TIMING] {ai_engine.upper()} Parallel Translation took {translate_time:.2f}s for {total_texts} texts ({total_pages} pages)", flush=True)
            
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
            # Ollama 등: 순차 번역 (로컬 모델은 병렬화 이점 적음)
            logger.info(f"[Translation] Fallback to VLM - ai_engine='{ai_engine}', api_key={bool(api_key)}, total_texts={total_texts}")
            print(f"[Translation] Fallback to VLM - ai_engine='{ai_engine}', api_key={bool(api_key)}, total_texts={total_texts}", flush=True)
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
        ai_engine = data.get('ai_engine', 'openai')
        api_key = data.get('api_key', None)
        model = data.get('model', None)
        image_base64 = data.get('image', None)
        texts = data.get('texts', [])

        # ★ 하드코딩된 API 키 우선 사용
        if ai_engine in HARDCODED_API_KEYS:
            api_key = HARDCODED_API_KEYS[ai_engine]

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
        target_lang = data.get('target_lang', 'english')

        print(f"[generate_preview] Received {len(translations)} translations, target_lang={target_lang}")
        for i, t in enumerate(translations[:5]):  # 처음 5개만 출력
            text_val = t.get('text', 'N/A')
            trans_val = t.get('translated', 'N/A')
            print(f"  [{i}] text: '{text_val[:30] if text_val else 'N/A'}' -> translated: '{trans_val[:30] if trans_val else 'N/A'}'")

        if not image_base64:
            return jsonify({"success": False, "error": "이미지가 없습니다"})

        if not translations:
            print("[generate_preview] ERROR: No translations provided!")
            return jsonify({"success": False, "error": "번역 데이터가 없습니다"})

        # 미리보기 이미지 생성
        print("[generate_preview] Calling generate_preview_image...")
        preview_base64 = generate_preview_image(image_base64, translations, target_lang)
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
            memos = page.get('memos', [])

            if translations or memos:
                # 이미지에 번역 적용
                output_filename = f"translated_{timestamp}_page{i+1}_{target_lang}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                replace_text_in_image(temp_img_path, translations, output_path, target_lang, memos)
                output_files.append(output_filename)
            else:
                print(f"  No translations or memos, skipping...")

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
                replace_text_in_image(img_path, translations, output_path, target_lang)
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


# ============================================================================
# 용어 사전 API
# ============================================================================

@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    """전체 용어 사전 조회"""
    global GARMENT_DICT
    GARMENT_DICT = load_garment_dict()  # 최신 데이터 로드
    return jsonify(GARMENT_DICT)

@app.route('/api/dictionary/<language>', methods=['GET'])
def get_dictionary_by_language(language):
    """특정 언어의 용어 사전 조회"""
    global GARMENT_DICT
    GARMENT_DICT = load_garment_dict()
    if language in GARMENT_DICT:
        return jsonify(GARMENT_DICT[language])
    return jsonify({"error": f"Language '{language}' not found"}), 404

@app.route('/api/dictionary/<language>', methods=['POST'])
def add_term(language):
    """용어 추가 (한글: {full: 번역, abbr: 약어})"""
    global GARMENT_DICT
    GARMENT_DICT = load_garment_dict()

    data = request.json
    korean = data.get('korean', '').strip()
    translation = data.get('translation', '').strip()
    abbr = data.get('abbr', '').strip()

    if not korean or not translation:
        return jsonify({"error": "korean and translation are required"}), 400

    if language not in GARMENT_DICT:
        return jsonify({"error": f"Language '{language}' not found"}), 404

    GARMENT_DICT[language][korean] = {"full": translation, "abbr": abbr}

    if save_garment_dict(GARMENT_DICT):
        return jsonify({"success": True, "korean": korean, "translation": translation, "abbr": abbr})
    return jsonify({"error": "Failed to save dictionary"}), 500

@app.route('/api/dictionary/<language>/<korean>', methods=['PUT'])
def update_term(language, korean):
    """용어 수정"""
    global GARMENT_DICT
    GARMENT_DICT = load_garment_dict()

    data = request.json
    translation = data.get('translation', '').strip()
    abbr = data.get('abbr', '').strip()

    if not translation:
        return jsonify({"error": "translation is required"}), 400

    if language not in GARMENT_DICT:
        return jsonify({"error": f"Language '{language}' not found"}), 404

    if korean not in GARMENT_DICT[language]:
        return jsonify({"error": f"Term '{korean}' not found"}), 404

    GARMENT_DICT[language][korean] = {"full": translation, "abbr": abbr}

    if save_garment_dict(GARMENT_DICT):
        return jsonify({"success": True, "korean": korean, "translation": translation, "abbr": abbr})
    return jsonify({"error": "Failed to save dictionary"}), 500

@app.route('/api/dictionary/<language>/<korean>', methods=['DELETE'])
def delete_term(language, korean):
    """용어 삭제"""
    global GARMENT_DICT
    GARMENT_DICT = load_garment_dict()

    if language not in GARMENT_DICT:
        return jsonify({"error": f"Language '{language}' not found"}), 404

    if korean not in GARMENT_DICT[language]:
        return jsonify({"error": f"Term '{korean}' not found"}), 404

    del GARMENT_DICT[language][korean]

    if save_garment_dict(GARMENT_DICT):
        return jsonify({"success": True, "deleted": korean})
    return jsonify({"error": "Failed to save dictionary"}), 500

@app.route('/api/dictionary/bulk', methods=['POST'])
def bulk_add_terms():
    """여러 언어에 동시에 용어 추가"""
    global GARMENT_DICT
    GARMENT_DICT = load_garment_dict()

    data = request.json
    korean = data.get('korean', '').strip()
    translations = data.get('translations', {})  # {language: translation}

    if not korean:
        return jsonify({"error": "korean is required"}), 400

    updated = []
    for lang, trans in translations.items():
        if lang in GARMENT_DICT and trans.strip():
            GARMENT_DICT[lang][korean] = trans.strip()
            updated.append(lang)

    if updated and save_garment_dict(GARMENT_DICT):
        return jsonify({"success": True, "korean": korean, "updated_languages": updated})
    return jsonify({"error": "No valid translations provided or save failed"}), 400


if __name__ == '__main__':
    print("=" * 60)
    print(f"PDF Translate Note v{VERSION} - 의류 기술서 번역 앱")
    print("=" * 60)
    print(f"Version: {VERSION} ({VERSION_DATE})")
    print("Engine: PaddleOCR + VLM (qwen2.5vl)")
    print("Languages: English, Vietnamese, Chinese, Indonesian, Bengali")
    print("Port: 6009")
    print("=" * 60)

    # OCR 엔진 미리 로드
    get_ocr_engine()

    app.run(host='0.0.0.0', port=7000, debug=True)
