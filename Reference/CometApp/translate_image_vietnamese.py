# -*- coding: utf-8 -*-
"""
한글 텍스트를 베트남어로 번역하고 이미지에서 교체하는 스크립트
- VLM (qwen2.5vl) 사용하여 의류 전문 용어 번역
- 폰트 크기/위치 자동 조정
"""

import os
import sys
import io

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR
import cv2

# 설정
OLLAMA_URL = "http://localhost:11434/api/generate"

# 의류 전문 용어 사전 (한글 → 베트남어)
GARMENT_DICT = {
    "남성": "Nam",
    "여성": "Nữ",
    "자켓": "Áo khoác",
    "다운자켓": "Áo phao",
    "후드": "Mũ trùm",
    "에리": "Cổ áo",
    "봉제": "May",
    "작업": "Công việc",
    "원단": "Vải",
    "안감": "Lót",
    "겉감": "Vỏ ngoài",
    "소매": "Tay áo",
    "밑단": "Gấu áo",
    "어깨": "Vai",
    "가슴": "Ngực",
    "허리": "Eo",
    "지퍼": "Khóa kéo",
    "스토퍼": "Nút chặn",
    "고리": "Vòng",
    "테이프": "Băng dính",
    "앞판": "Thân trước",
    "뒷판": "Thân sau",
    "로고": "Logo",
    "벨크로": "Velcro",
    "밴드": "Dây đai",
    "조절장치": "Bộ điều chỉnh",
    "아일렛": "Lỗ xỏ dây",
    "스트링": "Dây rút",
    "주머니": "Túi",
    "포켓": "Túi",
    "비드": "Hạt",
    "메인": "Chính",
    "라벨": "Nhãn",
}


def get_ocr_results(image_path):
    """PaddleOCR로 텍스트와 위치 추출"""
    print("[1/4] OCR로 텍스트 추출 중...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="korean")
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
                        print(f"  발견: {text_str}")

    print(f"  총 {len(texts)}개의 한글 텍스트 발견")
    return texts


def translate_with_dict_first(korean_text):
    """사전 기반 번역 시도"""
    result = korean_text
    for kor, viet in GARMENT_DICT.items():
        result = result.replace(kor, viet)
    return result


def translate_with_vlm(image_path, texts):
    """VLM으로 이미지 컨텍스트와 함께 베트남어로 번역"""
    print("\n[2/4] VLM으로 베트남어 번역 중...")

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    translations = []

    # 모든 한글 텍스트를 한 번에 번역 요청
    korean_list = [item["text"] for item in texts]
    korean_joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(korean_list)])

    prompt = f"""This is a garment/clothing technical specification image (tech pack).
Translate the following Korean texts to VIETNAMESE. These are garment industry terms.
Keep translations SHORT and professional. Only respond with numbered translations in Vietnamese.

Korean texts:
{korean_joined}

Vietnamese translations (same numbering, SHORT answers only):"""

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
            print(f"  VLM 응답:\n{response_text}")

            # 응답 파싱
            lines = response_text.split("\n")
            viet_translations = {}
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    parts = line.split(".", 1)
                    if len(parts) == 2:
                        idx = int(parts[0]) - 1
                        viet = parts[1].strip()
                        if idx < len(korean_list):
                            viet_translations[idx] = viet

            # 결과 매핑
            for i, item in enumerate(texts):
                if i in viet_translations:
                    viet = viet_translations[i]
                else:
                    # 사전 기반 fallback
                    viet = translate_with_dict_first(item["text"])

                translations.append({
                    **item,
                    "vietnamese": viet
                })
                print(f"  {item['text']} → {viet}")

        else:
            print(f"  VLM 오류: {response.status_code}")
            # fallback: 사전 번역
            for item in texts:
                viet = translate_with_dict_first(item["text"])
                translations.append({**item, "vietnamese": viet})

    except Exception as e:
        print(f"  VLM 연결 오류: {e}")
        # fallback: 사전 번역
        for item in texts:
            viet = translate_with_dict_first(item["text"])
            translations.append({**item, "vietnamese": viet})

    return translations


def inpaint_and_replace(image_path, translations, output_path):
    """이미지에서 한글 영역을 지우고 베트남어로 교체"""
    print("\n[3/4] 이미지에서 한글 영역 제거 중...")

    # 이미지 로드
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 먼저 제목 영역 (상단 30픽셀) 특별 처리
    title_items = [item for item in translations if min(p[1] for p in item["bbox"]) < 25]
    if title_items:
        title_y_max = max(max(p[1] for p in item["bbox"]) for item in title_items) + 5
        cv2.rectangle(img, (0, 0), (width, int(title_y_max)), (255, 255, 255), -1)

    # 한글 영역을 배경색으로 직접 덮기
    for item in translations:
        bbox = item["bbox"]
        pts = np.array(bbox, dtype=np.int32)

        # bbox 영역의 배경색 샘플링
        x_min = max(0, int(min(p[0] for p in bbox)) - 5)
        y_min = max(0, int(min(p[1] for p in bbox)) - 5)
        x_max = min(width, int(max(p[0] for p in bbox)) + 5)
        y_max = min(height, int(max(p[1] for p in bbox)) + 5)

        # 주변 픽셀의 평균 색상 (배경색 추정)
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

        # 영역을 배경색으로 채우기 (확장된 영역)
        expanded_pts = pts.copy().astype(np.float64)
        center = np.mean(pts, axis=0)
        for i in range(len(expanded_pts)):
            direction = expanded_pts[i] - center
            expanded_pts[i] = expanded_pts[i] + direction * 0.35

        cv2.fillPoly(img, [expanded_pts.astype(np.int32)], bg_color.tolist())

        # 추가로 사각형 영역도 덮기
        x1 = max(0, int(min(p[0] for p in bbox)) - 5)
        y1 = max(0, int(min(p[1] for p in bbox)) - 3)
        x2 = min(width, int(max(p[0] for p in bbox)) + 5)
        y2 = min(height, int(max(p[1] for p in bbox)) + 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color.tolist(), -1)
        cv2.rectangle(img, (x1-2, y1-2), (x2+2, y2+2), bg_color.tolist(), -1)

    inpainted = img

    print("\n[4/4] 베트남어 텍스트 삽입 중...")

    # PIL로 변환하여 텍스트 삽입
    img_result = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    # 폰트 설정 - 베트남어 지원 폰트 사용
    font_sizes = [11, 10, 9, 8, 7]

    for item in translations:
        bbox = item["bbox"]
        vietnamese_text = item["vietnamese"]

        # bbox 크기 계산
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)

        # bbox의 좌상단 좌표
        x = int(min(xs))
        y = int(min(ys))

        # 적절한 폰트 크기 찾기
        font = None
        for size in font_sizes:
            try:
                # 베트남어 지원 폰트 시도
                font = ImageFont.truetype("arial.ttf", size)
            except:
                font = ImageFont.load_default()
                break

            # 텍스트 크기 측정
            text_bbox = draw.textbbox((0, 0), vietnamese_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]

            if text_width <= box_width * 1.5:
                break

        # 텍스트가 너무 길면 줄임
        if text_width > box_width * 2:
            words = vietnamese_text.split()
            if len(words) > 3:
                vietnamese_text = " ".join(words[:3]) + "..."

        # 텍스트 삽입 (검정색)
        draw.text((x, y), vietnamese_text, fill=(0, 0, 0), font=font)
        print(f"  삽입: {vietnamese_text} at ({x}, {y})")

    # 결과 저장
    img_result.save(output_path)
    print(f"\n완료! 결과 저장: {output_path}")
    return output_path


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = r"E:\Antigravity\Black_Yak\Reference\CometApp\L_block_jacket_page2.png"

    if not os.path.exists(image_path):
        print(f"이미지 파일이 없습니다: {image_path}")
        return

    # 1. OCR로 한글 텍스트 추출
    texts = get_ocr_results(image_path)

    if not texts:
        print("한글 텍스트가 발견되지 않았습니다.")
        return

    # 2. VLM으로 베트남어 번역
    translations = translate_with_vlm(image_path, texts)

    # 3. 이미지 처리 (인페인팅 + 텍스트 삽입)
    output_path = image_path.replace(".png", "_vietnamese.png")
    if output_path == image_path:
        output_path = image_path + "_vietnamese.png"

    inpaint_and_replace(image_path, translations, output_path)


if __name__ == "__main__":
    main()
