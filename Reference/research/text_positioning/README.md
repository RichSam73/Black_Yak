# Text Positioning 연구 자료

**검색일**: 2026-01-08
**검색 목적**: 번역된 텍스트를 원본 위치에 정확하게 배치하는 방법

---

## 검색 도구별 결과

### 1. WebSearch (Claude 내장)

| 제목 | URL | 핵심 내용 |
|------|-----|----------|
| python-image-translator | https://github.com/boysugi20/python-image-translator | OCR bbox 기반 텍스트 교체 |
| ImageTrans Tool | https://www.basiccat.org/details-about-image-text-removal-using-imagetrans/ | 전문 이미지 번역 도구 |

### 2. GitHub Code Search (`mcp__github__search_code`)

| 프로젝트 | URL | 핵심 내용 |
|----------|-----|----------|
| Glossarion | https://github.com/Shirochi-stack/Glossarion | AI 기반 소설/만화 번역 |
| Arabic-Translation | https://github.com/akhilesh-av/Arabic-Translation | 아랍어 이미지 번역 |
| translatify | https://github.com/stephen-ics/translatify | 이미지 번역 앱 |

### 3. GitHub File Contents (`mcp__github__get_file_contents`)

**python-image-translator/main.py** 전체 코드 분석:

---

## 핵심 기술 요약

### 1. Bounding Box에서 정확한 좌표 추출

```python
def get_text_position(bbox):
    """OCR bbox에서 텍스트 위치 추출"""
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    box_width = x_max - x_min
    box_height = y_max - y_min

    return x_min, y_min, box_width, box_height
```

### 2. 폰트 크기 자동 조절 (Fit to Box)

```python
from PIL import Image, ImageDraw, ImageFont

def get_font_to_fit(image, text, width, height):
    """박스에 맞는 최대 폰트 크기 찾기"""
    draw = ImageDraw.Draw(image)

    font = None
    font_size = 1

    # 점진적으로 폰트 크기 증가
    for size in range(1, 500):
        new_font = ImageFont.truetype("arial.ttf", size)  # 또는 load_default(size=size)
        bbox = draw.textbbox((0, 0), text, font=new_font)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 박스를 넘어가면 이전 크기 사용
        if text_width > width or text_height > height:
            break

        font = new_font
        font_size = size

    return font, font_size
```

### 3. 텍스트 정렬 (왼쪽/중앙)

```python
def draw_text_aligned(draw, text, bbox, font, align="left"):
    """정렬 방식에 따라 텍스트 배치"""
    x_min, y_min, box_width, box_height = get_text_position(bbox)

    # 텍스트 크기 계산
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 수직 중앙 정렬
    y = y_min + (box_height - text_height) // 2

    if align == "center":
        x = x_min + (box_width - text_width) // 2
    elif align == "right":
        x = x_min + box_width - text_width
    else:  # left
        x = x_min

    return x, y
```

### 4. 배경색 기반 텍스트 색상 결정

```python
def get_text_color(background_color):
    """배경색 밝기에 따라 텍스트 색상 결정"""
    r, g, b = background_color[:3]

    # 휘도 계산 (ITU-R BT.601)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    if luminance > 0.5:
        return "black"  # 밝은 배경 → 검은 텍스트
    else:
        return "white"  # 어두운 배경 → 흰 텍스트
```

### 5. 완전한 텍스트 교체 함수

```python
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def replace_text_complete(image_path, translations, output_path):
    """텍스트 지우고 번역 텍스트로 교체"""

    # OpenCV로 이미지 로드
    img = cv2.imread(image_path)

    # 1단계: 모든 텍스트 영역 Inpainting
    for item in translations:
        bbox = item["bbox"]
        img = erase_text_inpaint(img, bbox)

    # 2단계: PIL로 변환하여 텍스트 삽입
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for item in translations:
        bbox = item["bbox"]
        translated_text = item["translated"]

        # 위치 및 크기 계산
        x_min, y_min, box_width, box_height = get_text_position(bbox)

        # 폰트 크기 자동 조절
        font, _ = get_font_to_fit(img_pil, translated_text, box_width, box_height)

        # 텍스트 위치 계산 (왼쪽 정렬)
        x, y = draw_text_aligned(draw, translated_text, bbox, font, align="left")

        # 텍스트 그리기
        draw.text((x, y), translated_text, fill="black", font=font)

    # 저장
    img_pil.save(output_path)
    return img_pil
```

---

## 문제 해결 팁

### 문제 1: 텍스트가 박스를 벗어남

**원인**: 번역 텍스트가 원본보다 길 때
**해결**:
- 폰트 크기 자동 축소
- 긴 텍스트는 줄바꿈 처리

```python
def wrap_text(text, font, max_width, draw):
    """텍스트를 max_width에 맞게 줄바꿈"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)
```

### 문제 2: 원본 텍스트가 완전히 지워지지 않음

**원인**: Inpainting 마스크가 텍스트 경계를 정확히 커버하지 못함
**해결**:
- 마스크 dilate iterations 증가 (3→5)
- inpaintRadius 증가 (5→7)

### 문제 3: 번역 텍스트 위치가 어긋남

**원인**: bbox 좌표 계산 오류
**해결**:
- `min(xs)`, `min(ys)`로 정확한 시작점 계산
- PIL의 textbbox 오프셋 보정

---

## 적용 권장사항

| 상황 | 권장 방법 |
|------|----------|
| 짧은 텍스트 (1-2 단어) | 폰트 크기 자동 조절 + 중앙 정렬 |
| 긴 텍스트 (문장) | 줄바꿈 처리 + 왼쪽 정렬 |
| 테이블 셀 | 고정 폰트 크기 + 왼쪽 상단 정렬 |
| 제목 | 중앙 정렬 + 큰 폰트 |

---

## 참고 프로젝트

1. **boysugi20/python-image-translator** - EasyOCR + PIL 기반 번역
2. **Shirochi-stack/Glossarion** - AI 기반 만화 번역
3. **ImageTrans (BasicCAT)** - 전문 이미지 번역 도구
