"""
한글 텍스트를 영어로 번역하고 이미지에서 교체하는 스크립트
로컬 AI (Ollama) 사용
"""

import os
import sys
import json
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from paddleocr import PaddleOCR
import cv2

# 설정
OLLAMA_URL = "http://localhost:11434/api/generate"
IMAGE_PATH = r"E:\Antigravity\Black_Yak\Reference\CometApp\input_image.png"
OUTPUT_PATH = r"E:\Antigravity\Black_Yak\Reference\CometApp\output_translated.png"

def get_ocr_results(image_path):
    """PaddleOCR로 텍스트와 위치 추출"""
    print("[1/4] OCR로 텍스트 추출 중...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="korean")

    # 새 API (predict) 사용
    result = ocr.predict(image_path)

    texts = []

    # 결과 구조 디버깅
    print(f"  [DEBUG] Result type: {type(result)}")

    # PaddleX 새 결과 구조 파싱
    if result:
        # result가 dict 형태일 수 있음
        if isinstance(result, dict):
            rec_texts = result.get('rec_text', [])
            rec_scores = result.get('rec_score', [])
            dt_polys = result.get('dt_polys', [])

            for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, dt_polys)):
                if any('\uac00' <= c <= '\ud7a3' for c in text):
                    texts.append({
                        "bbox": poly.tolist() if hasattr(poly, 'tolist') else poly,
                        "text": text,
                        "confidence": float(score)
                    })
                    print(f"  발견: {text}")

        # result가 list 형태일 수 있음
        elif isinstance(result, list):
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
                        if any('\uac00' <= c <= '\ud7a3' for c in str(text)):
                            bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
                            texts.append({
                                "bbox": bbox,
                                "text": str(text),
                                "confidence": float(score) if score else 1.0
                            })
                            print(f"  발견: {text}")

    print(f"  총 {len(texts)}개의 한글 텍스트 발견")
    return texts


def translate_with_ollama(texts):
    """Ollama로 한글을 영어로 번역"""
    print("\n[2/4] 로컬 AI로 번역 중...")

    translations = []
    for item in texts:
        korean_text = item["text"]

        prompt = f"""Translate the following Korean text to English. Only respond with the English translation, nothing else.

Korean: {korean_text}
English:"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.2:latest",  # 빠른 번역용
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                english_text = result.get("response", "").strip()
                print(f"  {korean_text} → {english_text}")
            else:
                english_text = korean_text  # 실패 시 원본 유지
                print(f"  번역 실패: {korean_text}")

        except Exception as e:
            english_text = korean_text
            print(f"  오류: {e}")

        translations.append({
            **item,
            "english": english_text
        })

    return translations


def inpaint_and_replace(image_path, translations, output_path):
    """이미지에서 한글 영역을 지우고 영어로 교체"""
    print("\n[3/4] 이미지에서 한글 영역 제거 중...")

    # 이미지 로드
    img = cv2.imread(image_path)
    img_pil = Image.open(image_path).convert("RGBA")

    # 마스크 생성 (한글 영역)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for item in translations:
        bbox = item["bbox"]
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # 인페인팅 (한글 영역을 배경색으로 채우기)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    print("\n[4/4] 영어 텍스트 삽입 중...")

    # PIL로 변환하여 텍스트 삽입
    img_result = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_result)

    # 폰트 설정 (시스템 폰트 사용)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for item in translations:
        bbox = item["bbox"]
        english_text = item["english"]

        # bbox의 좌상단 좌표
        x = int(min(p[0] for p in bbox))
        y = int(min(p[1] for p in bbox))

        # 텍스트 삽입 (검정색)
        draw.text((x, y), english_text, fill=(0, 0, 0), font=font)
        print(f"  삽입: {english_text} at ({x}, {y})")

    # 결과 저장
    img_result.save(output_path)
    print(f"\n완료! 결과 저장: {output_path}")
    return output_path


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = IMAGE_PATH

    if not os.path.exists(image_path):
        print(f"이미지 파일이 없습니다: {image_path}")
        return

    # 1. OCR로 한글 텍스트 추출
    texts = get_ocr_results(image_path)

    if not texts:
        print("한글 텍스트가 발견되지 않았습니다.")
        return

    # 2. Ollama로 번역
    translations = translate_with_ollama(texts)

    # 3. 이미지 처리 (인페인팅 + 텍스트 삽입)
    output_path = image_path.replace(".png", "_translated.png")
    if output_path == image_path:
        output_path = image_path.replace(".jpg", "_translated.jpg")
    if output_path == image_path:
        output_path = image_path + "_translated.png"

    inpaint_and_replace(image_path, translations, output_path)


if __name__ == "__main__":
    main()
