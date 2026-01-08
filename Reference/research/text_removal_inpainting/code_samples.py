# -*- coding: utf-8 -*-
"""
Text Removal & Inpainting 코드 샘플 모음
검색일: 2026-01-08
출처: Brave Search, Exa Search, GitHub
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

# =============================================================================
# 방법 1: OpenCV Inpainting (가장 간단, 권장)
# 출처: https://opencv.org/blog/text-detection-and-removal-using-opencv/
# =============================================================================

def erase_text_opencv_inpaint(img, bbox, inpaint_radius=5, method='telea'):
    """
    OpenCV inpainting으로 텍스트 제거

    Args:
        img: OpenCV 이미지 (BGR)
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태의 bounding box
        inpaint_radius: inpainting 반경 (기본값: 5)
        method: 'telea' 또는 'ns'

    Returns:
        텍스트가 제거된 이미지
    """
    height, width = img.shape[:2]

    # 마스크 생성
    mask = np.zeros((height, width), dtype=np.uint8)

    # bbox를 polygon으로 변환
    pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # 마스크 확장 (텍스트 경계까지 확실히 커버)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Inpainting 적용
    if method == 'ns':
        flags = cv2.INPAINT_NS
    else:
        flags = cv2.INPAINT_TELEA

    result = cv2.inpaint(img, mask, inpaint_radius, flags)

    return result


# =============================================================================
# 방법 2: Keras-OCR 스타일 Inpainting
# 출처: https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4/
# =============================================================================

def midpoint(x1, y1, x2, y2):
    """두 점의 중점 계산"""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def erase_text_keras_style(img, bbox, inpaint_radius=7):
    """
    Keras-OCR 스타일 line mask + inpainting

    Args:
        img: OpenCV 이미지
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        텍스트가 제거된 이미지
    """
    import math

    mask = np.zeros(img.shape[:2], dtype="uint8")

    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    x2, y2 = bbox[2]
    x3, y3 = bbox[3]

    # 중점 계산
    x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
    x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

    # 두께 계산
    thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

    # 마스크에 선 그리기
    cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)

    # Inpainting
    result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_NS)

    return result


# =============================================================================
# 방법 3: LaMa Inpainting (고품질)
# 출처: https://github.com/enesmsahin/simple-lama-inpainting
# 설치: pip install simple-lama-inpainting
# =============================================================================

def erase_text_lama(img_pil, bbox):
    """
    LaMa 모델로 고품질 텍스트 제거

    Args:
        img_pil: PIL Image
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        텍스트가 제거된 PIL Image
    """
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        print("pip install simple-lama-inpainting 필요")
        return img_pil

    simple_lama = SimpleLama()

    # 마스크 생성
    mask = Image.new('L', img_pil.size, 0)
    draw = ImageDraw.Draw(mask)
    pts = [(int(p[0]), int(p[1])) for p in bbox]
    draw.polygon(pts, fill=255)

    # LaMa inpainting
    result = simple_lama(img_pil, mask)
    return result


# =============================================================================
# 방법 4: 배경색 샘플링 (단순 배경용)
# 출처: https://github.com/boysugi20/python-image-translator
# =============================================================================

def get_background_color(image_pil, bbox, margin=10):
    """
    bbox 주변에서 배경색 샘플링

    Args:
        image_pil: PIL Image
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        margin: 샘플링 마진

    Returns:
        (R, G, B) 튜플
    """
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    edge_pixels = []

    # 상단 가장자리
    for x in range(max(0, x_min - margin), min(image_pil.width, x_max + margin)):
        if y_min - margin >= 0:
            edge_pixels.append(image_pil.getpixel((x, y_min - margin)))

    # 하단 가장자리
    for x in range(max(0, x_min - margin), min(image_pil.width, x_max + margin)):
        if y_max + margin < image_pil.height:
            edge_pixels.append(image_pil.getpixel((x, y_max + margin)))

    # 좌측 가장자리
    for y in range(max(0, y_min), min(image_pil.height, y_max)):
        if x_min - margin >= 0:
            edge_pixels.append(image_pil.getpixel((x_min - margin, y)))

    # 우측 가장자리
    for y in range(max(0, y_min), min(image_pil.height, y_max)):
        if x_max + margin < image_pil.width:
            edge_pixels.append(image_pil.getpixel((x_max + margin, y)))

    if edge_pixels:
        # RGB만 추출 (RGBA인 경우)
        rgb_pixels = [p[:3] if len(p) > 3 else p for p in edge_pixels]
        most_common = Counter(rgb_pixels).most_common(1)[0][0]
        return most_common

    return (255, 255, 255)  # 기본값: 흰색


def erase_text_fill_color(img_pil, bbox):
    """
    배경색으로 텍스트 영역 채우기

    Args:
        img_pil: PIL Image
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        텍스트가 지워진 PIL Image
    """
    draw = ImageDraw.Draw(img_pil)

    # 배경색 샘플링
    bg_color = get_background_color(img_pil, bbox)

    # bbox 좌표
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    # 사각형으로 채우기
    draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=bg_color)

    return img_pil


# =============================================================================
# 방법 5: HSV 기반 텍스트 마스크 생성
# 출처: https://stackoverflow.com/questions/63986588/removing-white-text-with-black-borders-from-image
# =============================================================================

def create_text_mask_hsv(img, thresh_value=200):
    """
    HSV 변환으로 텍스트 마스크 생성 (흰색 텍스트용)

    Args:
        img: OpenCV 이미지
        thresh_value: 임계값

    Returns:
        마스크 이미지
    """
    # HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # 이진화 및 반전
    _, thresh = cv2.threshold(sat, 10, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    # Morphology로 노이즈 제거 및 확장
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    return mask


def erase_white_text(img):
    """
    흰색 테두리 텍스트 제거

    Args:
        img: OpenCV 이미지

    Returns:
        텍스트가 제거된 이미지
    """
    mask = create_text_mask_hsv(img)
    result = cv2.inpaint(img, mask, 11, cv2.INPAINT_TELEA)
    return result


# =============================================================================
# 유틸리티 함수
# =============================================================================

def cv2_to_pil(img_cv2):
    """OpenCV 이미지를 PIL로 변환"""
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img_pil):
    """PIL 이미지를 OpenCV로 변환"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    # 테스트 이미지 로드
    image_path = "test_image.png"
    img = cv2.imread(image_path)

    # 테스트 bbox (예시)
    test_bbox = [[100, 50], [200, 50], [200, 80], [100, 80]]

    # 방법 1: OpenCV Inpainting
    result1 = erase_text_opencv_inpaint(img.copy(), test_bbox)
    cv2.imwrite("result_opencv_inpaint.png", result1)

    # 방법 2: Keras-OCR 스타일
    result2 = erase_text_keras_style(img.copy(), test_bbox)
    cv2.imwrite("result_keras_style.png", result2)

    # 방법 4: 배경색 채우기
    img_pil = cv2_to_pil(img)
    result4 = erase_text_fill_color(img_pil.copy(), test_bbox)
    result4.save("result_fill_color.png")

    print("테스트 완료!")
