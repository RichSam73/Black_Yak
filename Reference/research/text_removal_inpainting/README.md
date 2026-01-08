# Text Removal & Inpainting 연구 자료

**검색일**: 2026-01-08
**검색 목적**: 이미지에서 텍스트를 깨끗하게 지우고 배경을 복원하는 방법

---

## 검색 도구별 결과

### 1. Brave Search (`mcp__brave-search__brave_web_search`)

| 제목 | URL | 핵심 내용 |
|------|-----|----------|
| OpenCV Text Detection and Removal | https://opencv.org/blog/text-detection-and-removal-using-opencv/ | EAST, DB50, DB18 모델로 텍스트 감지 후 cv2.inpaint로 제거 |
| Remove Text from Images (TowardsDataScience) | https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4/ | Keras-OCR + cv2.inpaint 조합 |
| Stack Overflow - Remove text | https://stackoverflow.com/questions/53592055/opencv-remove-text-from-image | thresholding + inpainting 기법 |
| PyImageSearch - Image Inpainting | https://pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/ | TELEA vs NS 알고리즘 비교 |
| OpenCV Inpainting Docs | https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html | 공식 문서 |
| LaMa GitHub | https://github.com/advimman/lama | SOTA AI 기반 inpainting |
| simple-lama-inpainting (PyPI) | https://pypi.org/project/simple-lama-inpainting/ | LaMa 간단 pip 패키지 |
| lama-cleaner (PyPI) | https://pypi.org/project/lama-cleaner/0.31.0/ | GUI 포함 inpainting 도구 |

### 2. Exa Search (`mcp__exa__web_search_exa`)

| 제목 | URL | 핵심 내용 |
|------|-----|----------|
| Remove Text (Medium) | https://medium.com/data-science/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4 | 상세 튜토리얼 |
| DigitalSreeni YouTube | https://www.youtube.com/watch?v=3RNPJbUHZKs | 영상 튜토리얼 (14K views) |
| bnsreenu GitHub | https://github.com/bnsreenu/python_for_microscopists | Tips_Tricks_42 코드 |
| OCR-SAM | https://github.com/yeungchenwa/OCR-SAM | MMOCR + SAM + Stable Diffusion 조합 |
| unscribe | https://github.com/manbehindthemadness/unscribe | LaMa + CRAFT 현대적 구현 |

### 3. Exa Code Context (`mcp__exa__get_code_context_exa`)

| 제목 | URL | 핵심 내용 |
|------|-----|----------|
| Stack Overflow - White Text Removal | https://stackoverflow.com/questions/63986588/removing-white-text-with-black-borders-from-image | HSV + morphology + inpaint |
| Stack Overflow - Replace cropped rect | https://stackoverflow.com/questions/73084045/how-to-replace-cropped-rectangle-in-opencv | `im2[y:y+h, x:x+w] = cropped` |
| PaddleX LayoutOCR | https://github.com/PaddlePaddle/PaddleX | 레이아웃 분석 + OCR 파이프라인 |

---

## 핵심 기술 요약

### 방법 1: OpenCV Inpainting (권장 - 간단)

```python
import cv2
import numpy as np

def erase_text_inpaint(img, bbox):
    """Inpainting으로 텍스트 영역 지우기"""
    height, width = img.shape[:2]

    # 마스크 생성 (텍스트 영역을 흰색으로)
    mask = np.zeros((height, width), dtype=np.uint8)

    # bbox를 polygon으로 변환
    pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # 마스크 확장 (텍스트 가장자리까지 커버)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Inpainting 적용 (TELEA 또는 NS 알고리즘)
    result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result
```

**알고리즘 비교:**
- `cv2.INPAINT_TELEA`: Fast Marching Method (빠름, 일반적으로 좋은 결과)
- `cv2.INPAINT_NS`: Navier-Stokes 기반 (느리지만 부드러운 결과)

### 방법 2: LaMa Inpainting (고품질)

```bash
pip install simple-lama-inpainting
```

```python
from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageDraw

simple_lama = SimpleLama()

def erase_text_lama(img_pil, bbox):
    """LaMa로 고품질 텍스트 제거"""
    # 마스크 생성
    mask = Image.new('L', img_pil.size, 0)
    draw = ImageDraw.Draw(mask)
    pts = [(int(p[0]), int(p[1])) for p in bbox]
    draw.polygon(pts, fill=255)

    # LaMa inpainting
    result = simple_lama(img_pil, mask)
    return result
```

### 방법 3: 배경색 샘플링 (단순 배경용)

```python
from collections import Counter

def get_background_color(image, bbox):
    """bbox 주변에서 배경색 샘플링"""
    x_min = int(min(p[0] for p in bbox))
    y_min = int(min(p[1] for p in bbox))
    x_max = int(max(p[0] for p in bbox))
    y_max = int(max(p[1] for p in bbox))

    margin = 10
    edge_pixels = []

    # 상단/하단/좌측/우측 가장자리에서 샘플링
    for x in range(max(0, x_min-margin), min(image.width, x_max+margin)):
        if y_min - margin >= 0:
            edge_pixels.append(image.getpixel((x, y_min - margin)))
        if y_max + margin < image.height:
            edge_pixels.append(image.getpixel((x, y_max + margin)))

    for y in range(max(0, y_min), min(image.height, y_max)):
        if x_min - margin >= 0:
            edge_pixels.append(image.getpixel((x_min - margin, y)))
        if x_max + margin < image.width:
            edge_pixels.append(image.getpixel((x_max + margin, y)))

    # 가장 많이 나온 색상 사용
    if edge_pixels:
        most_common = Counter(edge_pixels).most_common(1)[0][0]
        return most_common
    return (255, 255, 255)  # 기본값: 흰색
```

---

## 적용 권장사항

| 상황 | 권장 방법 |
|------|----------|
| 기술서 (흰색/단색 배경) | OpenCV Inpainting (TELEA) |
| 복잡한 배경 이미지 | LaMa Inpainting |
| 단순한 단색 배경 | 배경색 샘플링 + rectangle fill |
| 만화/웹툰 | OCR-SAM 또는 BubbleBlaster |

---

## 참고 GitHub 프로젝트

1. **advimman/lama** - SOTA inpainting 모델
2. **yeungchenwa/OCR-SAM** - OCR + SAM + Stable Diffusion
3. **manbehindthemadness/unscribe** - LaMa + CRAFT 조합
4. **a-milenkin/lama-cleaner** - GUI 포함 도구
5. **bnsreenu/python_for_microscopists** - 실용적인 예제 코드
