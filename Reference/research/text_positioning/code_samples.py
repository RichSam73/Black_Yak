# -*- coding: utf-8 -*-
"""
Text Positioning 코드 샘플 모음
검색일: 2026-01-08
출처: GitHub, WebSearch
"""

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# =============================================================================
# 1. Bounding Box 좌표 추출
# =============================================================================

def get_bbox_rect(bbox):
    """
    4점 bbox에서 사각형 좌표 추출

    Args:
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        (x_min, y_min, x_max, y_max, width, height)
    """
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    width = x_max - x_min
    height = y_max - y_min

    return x_min, y_min, x_max, y_max, width, height


# =============================================================================
# 2. 폰트 크기 자동 조절
# 출처: https://github.com/boysugi20/python-image-translator
# =============================================================================

def get_font_to_fit_box(text, max_width, max_height, font_path=None, start_size=1, max_size=500):
    """
    박스에 맞는 최대 폰트 크기 찾기

    Args:
        text: 텍스트
        max_width: 최대 너비
        max_height: 최대 높이
        font_path: 폰트 파일 경로 (None이면 기본 폰트)
        start_size: 시작 크기
        max_size: 최대 크기

    Returns:
        (font, font_size, text_width, text_height)
    """
    # 임시 이미지로 텍스트 크기 측정
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    best_font = None
    best_size = start_size
    best_width = 0
    best_height = 0

    for size in range(start_size, max_size):
        if font_path:
            font = ImageFont.truetype(font_path, size)
        else:
            font = ImageFont.load_default(size=size)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 박스를 넘어가면 이전 크기 사용
        if text_width > max_width or text_height > max_height:
            break

        best_font = font
        best_size = size
        best_width = text_width
        best_height = text_height

    return best_font, best_size, best_width, best_height


def get_font_fixed_sizes(text, max_width, max_height, font_path=None, sizes=[12, 11, 10, 9, 8, 7]):
    """
    고정 크기 목록에서 맞는 폰트 찾기

    Args:
        text: 텍스트
        max_width: 최대 너비
        max_height: 최대 높이
        font_path: 폰트 파일 경로
        sizes: 시도할 크기 목록 (큰 것부터)

    Returns:
        (font, font_size) or (None, None)
    """
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    for size in sizes:
        if font_path:
            font = ImageFont.truetype(font_path, size)
        else:
            font = ImageFont.load_default(size=size)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width <= max_width and text_height <= max_height:
            return font, size

    # 가장 작은 크기로 반환
    if font_path:
        return ImageFont.truetype(font_path, sizes[-1]), sizes[-1]
    else:
        return ImageFont.load_default(size=sizes[-1]), sizes[-1]


# =============================================================================
# 3. 텍스트 정렬
# =============================================================================

def calculate_text_position(text, bbox, font, draw, align="left", valign="top"):
    """
    정렬 방식에 따른 텍스트 위치 계산

    Args:
        text: 텍스트
        bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        font: PIL 폰트
        draw: ImageDraw 객체
        align: 'left', 'center', 'right'
        valign: 'top', 'middle', 'bottom'

    Returns:
        (x, y) 텍스트 시작 좌표
    """
    x_min, y_min, x_max, y_max, box_width, box_height = get_bbox_rect(bbox)

    # 텍스트 크기 계산
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 수평 정렬
    if align == "center":
        x = x_min + (box_width - text_width) // 2
    elif align == "right":
        x = x_max - text_width
    else:  # left
        x = x_min

    # 수직 정렬
    if valign == "middle":
        y = y_min + (box_height - text_height) // 2
    elif valign == "bottom":
        y = y_max - text_height
    else:  # top
        y = y_min

    # textbbox 오프셋 보정
    x -= text_bbox[0]
    y -= text_bbox[1]

    return x, y


# =============================================================================
# 4. 텍스트 줄바꿈
# =============================================================================

def wrap_text_to_width(text, font, max_width, draw):
    """
    텍스트를 max_width에 맞게 줄바꿈

    Args:
        text: 텍스트
        font: PIL 폰트
        max_width: 최대 너비
        draw: ImageDraw 객체

    Returns:
        줄바꿈된 텍스트 (\\n으로 구분)
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def wrap_text_cjk(text, font, max_width, draw):
    """
    CJK (한중일) 텍스트 줄바꿈 (글자 단위)

    Args:
        text: 텍스트
        font: PIL 폰트
        max_width: 최대 너비
        draw: ImageDraw 객체

    Returns:
        줄바꿈된 텍스트
    """
    lines = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = char

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)


# =============================================================================
# 5. 텍스트 색상 결정
# =============================================================================

def get_text_color_for_background(bg_color):
    """
    배경색에 따른 텍스트 색상 결정

    Args:
        bg_color: (R, G, B) 튜플

    Returns:
        'black' 또는 'white'
    """
    r, g, b = bg_color[:3]

    # ITU-R BT.601 휘도 공식
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    if luminance > 0.5:
        return "black"
    else:
        return "white"


def get_contrasting_color(bg_color, light_color=(0, 0, 0), dark_color=(255, 255, 255)):
    """
    배경색과 대비되는 색상 반환

    Args:
        bg_color: (R, G, B) 배경색
        light_color: 밝은 배경에 사용할 색
        dark_color: 어두운 배경에 사용할 색

    Returns:
        (R, G, B) 텍스트 색상
    """
    r, g, b = bg_color[:3]
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    if luminance > 0.5:
        return light_color
    else:
        return dark_color


# =============================================================================
# 6. 완전한 텍스트 교체 함수
# =============================================================================

def replace_text_on_image(
    img_pil,
    translations,
    font_path=None,
    align="left",
    valign="top",
    fixed_font_sizes=[12, 11, 10, 9, 8, 7]
):
    """
    이미지에서 텍스트 영역에 번역 텍스트 삽입

    Args:
        img_pil: PIL Image (텍스트가 이미 지워진 상태)
        translations: [{"bbox": [...], "translated": "..."}, ...]
        font_path: 폰트 파일 경로
        align: 수평 정렬
        valign: 수직 정렬
        fixed_font_sizes: 시도할 폰트 크기 목록

    Returns:
        텍스트가 삽입된 PIL Image
    """
    img_result = img_pil.copy()
    draw = ImageDraw.Draw(img_result)

    for item in translations:
        bbox = item["bbox"]
        translated_text = item.get("translated", "")

        if not translated_text:
            continue

        # bbox 정보 추출
        x_min, y_min, x_max, y_max, box_width, box_height = get_bbox_rect(bbox)

        # 폰트 크기 선택
        font, font_size = get_font_fixed_sizes(
            translated_text, box_width, box_height,
            font_path, fixed_font_sizes
        )

        if font is None:
            continue

        # 텍스트 위치 계산
        x, y = calculate_text_position(
            translated_text, bbox, font, draw, align, valign
        )

        # 텍스트 그리기 (검은색)
        draw.text((x, y), translated_text, fill="black", font=font)

    return img_result


def replace_text_with_wrap(
    img_pil,
    translations,
    font_path=None,
    font_size=10,
    align="left"
):
    """
    긴 텍스트는 줄바꿈하여 삽입

    Args:
        img_pil: PIL Image
        translations: [{"bbox": [...], "translated": "..."}, ...]
        font_path: 폰트 파일 경로
        font_size: 고정 폰트 크기
        align: 정렬

    Returns:
        텍스트가 삽입된 PIL Image
    """
    img_result = img_pil.copy()
    draw = ImageDraw.Draw(img_result)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default(size=font_size)

    for item in translations:
        bbox = item["bbox"]
        translated_text = item.get("translated", "")

        if not translated_text:
            continue

        x_min, y_min, x_max, y_max, box_width, box_height = get_bbox_rect(bbox)

        # 텍스트 줄바꿈
        wrapped_text = wrap_text_to_width(translated_text, font, box_width, draw)

        # 위치 계산 (왼쪽 상단)
        x = x_min
        y = y_min

        # 텍스트 그리기
        draw.multiline_text((x, y), wrapped_text, fill="black", font=font, align=align)

    return img_result


# =============================================================================
# 7. OpenCV + PIL 통합 파이프라인
# =============================================================================

def complete_text_replacement_pipeline(
    image_path,
    translations,
    output_path,
    erase_method="inpaint",
    font_path=None
):
    """
    완전한 텍스트 교체 파이프라인

    Args:
        image_path: 입력 이미지 경로
        translations: [{"bbox": [...], "translated": "..."}, ...]
        output_path: 출력 이미지 경로
        erase_method: "inpaint" 또는 "fill"
        font_path: 폰트 파일 경로

    Returns:
        출력 이미지 경로
    """
    # 1단계: 이미지 로드
    img_cv2 = cv2.imread(image_path)

    # 2단계: 텍스트 영역 지우기
    if erase_method == "inpaint":
        for item in translations:
            bbox = item["bbox"]
            # OpenCV inpainting
            height, width = img_cv2.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
            img_cv2 = cv2.inpaint(img_cv2, mask, 5, cv2.INPAINT_TELEA)

    # 3단계: PIL로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    # 4단계: 번역 텍스트 삽입
    img_result = replace_text_on_image(img_pil, translations, font_path)

    # 5단계: 저장
    img_result.save(output_path)

    return output_path


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    # 테스트 데이터
    test_translations = [
        {
            "bbox": [[100, 50], [200, 50], [200, 80], [100, 80]],
            "original": "안녕하세요",
            "translated": "Hello"
        },
        {
            "bbox": [[100, 100], [300, 100], [300, 130], [100, 130]],
            "original": "번역 테스트입니다",
            "translated": "This is a translation test"
        }
    ]

    # 파이프라인 실행
    # complete_text_replacement_pipeline(
    #     "input.png",
    #     test_translations,
    #     "output.png"
    # )

    print("코드 샘플 로드 완료!")
