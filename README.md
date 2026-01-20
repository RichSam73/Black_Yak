# 📊 PDF 요소별 추출기 (AI & OCR 기반)

블랙야크 의류 워크시트 및 일반 PDF 파일에서 텍스트, 도면(벡터), 차트, 이미지를 지능적으로 추출하고, ERP 연동 데이터를 생성하는 도구입니다.

## 🚀 주요 기능

### 1. CometApp - ERP 테이블 추출 (최신)

#### 1-1. Qwen2.5-VL 기반 (권장)

**위치**: `Reference/CometApp/app_qwen.py`

**기술 스택**:
- **PaddleOCR v5** (PP-OCRv5_server_det + korean_PP-OCRv5_mobile_rec)
- **qwen2.5vl** (AI Vision 메인 모델)
- **AI 모델 폴백 체인**: qwen2.5vl → gemma3:4b → llama3.2-vision
- **Flask** 웹 서버 (포트 6002)

**핵심 기능**:
- 하이브리드 OCR: PaddleOCR + AI Vision 조합으로 누락 텍스트 자동 보정
- X-좌표 클러스터링 (threshold=30) 기반 컬럼 감지
- **Round 17**: 헤더 기반 컬럼 위치 감지 (SIZE 숫자 095-130 패턴)
- **Round 18**: 타이틀 헤더("SUB MATERIAL INFORMATION") 병합 스킵
- COLOR/SIZE QTY 테이블 자동 감지 (Method 2 전용)
- 수직선 기반 강제 컬럼 분리 (OpenCV)

**실행 방법**:
```bash
cd Reference/CometApp
python app_qwen.py
# http://localhost:6002 접속
```

#### 1-2. Llama3.2-vision 기반 (레거시)

**위치**: `Reference/CometApp/app_ai.py`

**기술 스택**:
- **PaddleOCR v5** + **llama3.2-vision**
- **Flask** 웹 서버 (포트 6001)

**실행 방법**:
```bash
cd Reference/CometApp
python app_ai.py
# http://localhost:6001 접속
```

### 2. AI 스마트 테이블 추출 (Comet Legacy)

**파일**: `smart_table_extractor.py`

**기술**: PaddleOCR + Table Transformer (MS/HuggingFace)

**기능**: 이미지나 벡터로 된 PDF 페이지를 AI가 분석하여 텍스트를 인식하고, 원본 이미지 위에 선택 가능한 투명 텍스트 레이어를 생성합니다.

### 3. 벡터 도면 자동 분류

**파일**: `app.py`

**기술**: PyMuPDF (`fitz`) Clustering

**기능**: PDF 내부의 수천 개의 벡터(선, 도형) 데이터를 분석하여 '자켓 도면'과 같은 의미 있는 덩어리를 자동으로 감지하고 이미지로 추출합니다.

### 4. VLM (Vision Language Model) 지원

**기술**: Ollama (`llama3.2-vision`, `granite3.2-vision`)

**기능**: 문서의 시각적 맥락을 이해하여 의미 기반으로 데이터를 추출합니다.

---

## 🛠 기술 스택 (Tech Stack)

| 구성 요소 | 기술 |
|-----------|------|
| Frontend/UI | Streamlit, Flask |
| PDF Processing | PyMuPDF (fitz) |
| OCR Engine | PaddleOCR v5 (PP-OCRv5_server) |
| AI Vision | Qwen2.5-VL / Gemma 3 / Llama 3.2 Vision (Ollama) |
| Table Detection | Table Transformer (HuggingFace) |
| Language | Python 3.9+ |

---

## 🔬 AI Vision 모델 성능 비교

OCR/문서 파싱 용도 벤치마크 결과 (Clarifai, NVIDIA L40S 기준):

| 모델 | 개발사 | 강점 | 처리량 (tokens/sec) | 권장 용도 |
|------|--------|------|---------------------|-----------|
| **Qwen2.5-VL** | Alibaba | OCR/문서 파싱 특화 | 1,017 | 🥇 ERP 테이블 추출 |
| **MiniCPM-o 2.6** | OpenBMB | 전체 성능 최고 | 1,075 | 범용 Vision |
| **Gemma 3** | Google | 텍스트 작업 우수 | 943 | 한글 문서 |
| **Llama 3.2 Vision** | Meta | 범용 | - | 일반 이미지 |

**현재 설치된 모델** (`ollama list`):
- `qwen2.5vl` (6GB) - OCR 최적화
- `gemma3:27b` (17GB) - 고성능 텍스트
- `llama3.2-vision` - 범용

---

## 💻 설치 및 실행 (Installation)

### 1. 필수 프로그램 설치
```bash
# Python 3.9 이상
# Ollama (AI Vision용)
```

### 2. 라이브러리 설치
```bash
pip install -r requirements.txt
pip install paddleocr paddlex protobuf==3.20.0
pip install flask ollama

# 선택 사항 (GPU 사용 시)
# pip install paddlepaddle-gpu
```

### 3. 프로그램 실행

**CometApp (ERP 테이블 추출)**:
```bash
cd Reference/CometApp
python app_ai.py
# http://localhost:6001
```

**Streamlit 앱 (PDF 추출)**:
```bash
streamlit run app.py
```

---

## 📝 최근 업데이트

### Round 18 (2025-12-31) - app_qwen.py
- 타이틀 헤더("SUB MATERIAL INFORMATION") 병합 스킵 로직 추가
- DIV, CODE, NAME 등 실제 데이터 컬럼만 병합 대상

### Round 17 (2025-12-31) - app_qwen.py
- 헤더 기반 컬럼 위치 감지 (SIZE 숫자 095-130 패턴)
- 125/130 사이즈 컬럼 누락 문제 해결

### Round 16 (2025-12-31) - app_qwen.py
- AI 모델 폴백 체인: qwen2.5vl → gemma3:4b → llama3.2-vision
- 모델 장애 시 자동 대체

### Round 15 (2025-12-31) - app_qwen.py
- COLOR/SIZE QTY 테이블 패턴 자동 감지
- Method 2(헤더 기반 병합)만 허용

### Round 9 (2025-12-30) - app_ai.py
- SUP CD / SUP NM 컬럼 병합 방지
- Method 3 임계값 150px → 40px 축소

### Round 8 (2025-12-29) - app_ai.py
- COLOR/SIZE QTY 테이블 Method 2만 허용
- 빈 컬럼 오류 해결

### Round 7 (2025-12-28) - app_ai.py
- 하이브리드 OCR + 테이블 구조 분석 기반 누락 텍스트 자동 삽입

---

## 🧪 테스트 결과 (app_qwen.py)

| 테이블 | 컬럼 수 | AI 검증 | 비고 |
|--------|---------|---------|------|
| BY_Original_Table.png | 9개 | ✅ 통과 | COLOR/SIZE QTY (095-120) |
| 005M_Table.png | 11개 | ✅ 통과 | COLOR/SIZE QTY (095-130, 125/130 포함) |
| Submaterial_information.png | 13개 | ✅ 통과 | SUP CD/SUP NM 분리 |

**테스트 실행**:
```bash
cd Reference/CometApp
python test_all_tables.py
```

---

## ☁️ 서버 배포 시 주의사항 (Deployment)

1. **컴퓨팅 자원 (Compute Resources)**
   - 서버의 CPU/GPU를 사용하여 AI 모델을 구동합니다.
   - 최소 8GB RAM, 멀티코어 CPU 권장

2. **필수 시스템 패키지** (Linux 서버 기준)
   ```bash
   apt-get install libgl1-mesa-glx libgomp1
   ```

3. **모델 데이터**
   - 최초 실행 시 PaddleOCR 학습 모델(약 500MB)이 자동 다운로드됩니다.
   - Ollama 모델: `ollama pull llama3.2-vision`

---

## Research 자료 모음

이 섹션은 `Reference/research/README.md`와 `Reference/research/text_positioning/README.md`의 내용을 합친 것입니다.

이 폴더는 웹 검색을 통해 수집한 연구 자료를 주제별로 정리한 것입니다.

---

### 폴더 구조

```
research/
├── README.md                          # 이 파일
├── text_removal_inpainting/           # 텍스트 제거 및 Inpainting
│   ├── README.md                      # 검색 결과 요약 및 기술 정리
│   └── code_samples.py                # 코드 샘플 모음
└── text_positioning/                  # 텍스트 위치 배치
    ├── README.md                      # 검색 결과 요약 및 기술 정리
    └── code_samples.py                # 코드 샘플 모음
```

---

### 검색 도구 목록

| 도구 | MCP 이름 | 용도 |
|------|----------|------|
| Brave Search | `mcp__brave-search__brave_web_search` | 일반 웹 검색 |
| Exa Search | `mcp__exa__web_search_exa` | Semantic 웹 검색 |
| Exa Code Context | `mcp__exa__get_code_context_exa` | 코드/라이브러리 검색 |
| WebSearch | Claude 내장 | 일반 웹 검색 |
| GitHub Code Search | `mcp__github__search_code` | GitHub 코드 검색 |
| GitHub File Contents | `mcp__github__get_file_contents` | GitHub 파일 내용 조회 |

---

### 주제별 요약

#### 1. Text Removal & Inpainting (2026-01-08)

**목적**: 이미지에서 텍스트를 깨끗하게 지우고 배경을 복원

**핵심 방법**:
1. **OpenCV Inpainting** - `cv2.inpaint()` (TELEA/NS 알고리즘)
2. **LaMa Inpainting** - AI 기반 고품질 복원 (`pip install simple-lama-inpainting`)
3. **배경색 샘플링** - 단순 배경에서 주변 색상으로 채우기

**권장**: 기술서 문서는 대부분 흰색 배경이므로 OpenCV Inpainting으로 충분

#### 2. Text Positioning (2026-01-08)

**목적**: 번역된 텍스트를 원본 위치에 정확하게 배치

**핵심 방법**:
1. **Bounding Box 좌표 추출** - OCR 결과에서 min/max 좌표 계산
2. **폰트 크기 자동 조절** - 박스에 맞는 최대 크기 탐색
3. **텍스트 정렬** - 왼쪽/중앙/오른쪽 + 상단/중앙/하단
4. **텍스트 줄바꿈** - 긴 텍스트 처리

**권장**: 고정 폰트 크기 목록에서 맞는 크기 선택 + 왼쪽 정렬

---

### 사용 라이브러리 목록

#### Python 패키지 (pip)

| 라이브러리 | 설치 명령 | 용도 |
|-----------|----------|------|
| **OpenCV** | `pip install opencv-python` | 이미지 처리, Inpainting (`cv2.inpaint`) |
| **Pillow (PIL)** | `pip install Pillow` | 이미지/텍스트 렌더링 (`ImageDraw`, `ImageFont`) |
| **NumPy** | `pip install numpy` | 배열/마스크 처리 |
| **simple-lama-inpainting** | `pip install simple-lama-inpainting` | AI 기반 고품질 Inpainting |
| **lama-cleaner** | `pip install lama-cleaner` | GUI 포함 Inpainting 도구 |
| **keras-ocr** | `pip install keras-ocr` | OCR + 텍스트 감지 |
| **EasyOCR** | `pip install easyocr` | 다국어 OCR |
| **PaddleOCR** | `pip install paddleocr paddlepaddle` | 고성능 OCR + 레이아웃 분석 |

#### 핵심 함수/API

| 함수 | 라이브러리 | 용도 |
|------|----------|------|
| `cv2.inpaint()` | OpenCV | 텍스트 영역 복원 (TELEA/NS) |
| `cv2.fillPoly()` | OpenCV | 마스크 폴리곤 채우기 |
| `cv2.dilate()` | OpenCV | 마스크 확장 |
| `ImageDraw.text()` | Pillow | 텍스트 렌더링 |
| `ImageDraw.textbbox()` | Pillow | 텍스트 바운딩 박스 계산 |
| `ImageFont.truetype()` | Pillow | 폰트 로드 |
| `SimpleLama()` | simple-lama | AI Inpainting |

#### GitHub 참고 프로젝트

| 프로젝트 | URL | 설명 |
|----------|-----|------|
| advimman/lama | https://github.com/advimman/lama | SOTA AI Inpainting 모델 |
| yeungchenwa/OCR-SAM | https://github.com/yeungchenwa/OCR-SAM | OCR + SAM + Stable Diffusion |
| manbehindthemadness/unscribe | https://github.com/manbehindthemadness/unscribe | LaMa + CRAFT 조합 |
| boysugi20/python-image-translator | https://github.com/boysugi20/python-image-translator | EasyOCR + PIL 번역 |
| bnsreenu/python_for_microscopists | https://github.com/bnsreenu/python_for_microscopists | 실용적 예제 코드 |

---

### 추가 예정 주제

- [ ] OCR 정확도 향상
- [ ] 테이블 구조 인식
- [ ] 다국어 폰트 렌더링
- [ ] PDF 처리 최적화

---

## Text Positioning 연구 자료

**검색일**: 2026-01-08
**검색 목적**: 번역된 텍스트를 원본 위치에 정확하게 배치하는 방법

---

### 검색 도구별 결과

#### 1. WebSearch (Claude 내장)

| 제목 | URL | 핵심 내용 |
|------|-----|----------|
| python-image-translator | https://github.com/boysugi20/python-image-translator | OCR bbox 기반 텍스트 교체 |
| ImageTrans Tool | https://www.basiccat.org/details-about-image-text-removal-using-imagetrans/ | 전문 이미지 번역 도구 |

#### 2. GitHub Code Search (`mcp__github__search_code`)

| 프로젝트 | URL | 핵심 내용 |
|----------|-----|----------|
| Glossarion | https://github.com/Shirochi-stack/Glossarion | AI 기반 소설/만화 번역 |
| Arabic-Translation | https://github.com/akhilesh-av/Arabic-Translation | 아랍어 이미지 번역 |
| translatify | https://github.com/stephen-ics/translatify | 이미지 번역 앱 |

#### 3. GitHub File Contents (`mcp__github__get_file_contents`)

**python-image-translator/main.py** 전체 코드 분석:

---

### 핵심 기술 요약

#### 1. Bounding Box에서 정확한 좌표 추출

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

#### 2. 폰트 크기 자동 조절 (Fit to Box)

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

#### 3. 텍스트 정렬 (왼쪽/중앙)

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

#### 4. 배경색 기반 텍스트 색상 결정

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

#### 5. 완전한 텍스트 교체 함수

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

### 문제 해결 팁

#### 문제 1: 텍스트가 박스를 벗어남

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

#### 문제 2: 원본 텍스트가 완전히 지워지지 않음

**원인**: Inpainting 마스크가 텍스트 경계를 정확히 커버하지 못함
**해결**:
- 마스크 dilate iterations 증가 (3→5)
- inpaintRadius 증가 (5→7)

#### 문제 3: 번역 텍스트 위치가 어긋남

**원인**: bbox 좌표 계산 오류
**해결**:
- `min(xs)`, `min(ys)`로 정확한 시작점 계산
- PIL의 textbbox 오프셋 보정

---

### 적용 권장사항

| 상황 | 권장 방법 |
|------|----------|
| 짧은 텍스트 (1-2 단어) | 폰트 크기 자동 조절 + 중앙 정렬 |
| 긴 텍스트 (문장) | 줄바꿈 처리 + 왼쪽 정렬 |
| 테이블 셀 | 고정 폰트 크기 + 왼쪽 상단 정렬 |
| 제목 | 중앙 정렬 + 큰 폰트 |

---

### 참고 프로젝트

1. **boysugi20/python-image-translator** - EasyOCR + PIL 기반 번역
2. **Shirochi-stack/Glossarion** - AI 기반 만화 번역
3. **ImageTrans (BasicCAT)** - 전문 이미지 번역 도구


## PDF 작업 가이드

PDF 관련 작업 시 `~/.claude/skills/pdf/` 폴더의 문서 참조:
- SKILL.md: 기본 가이드 (병합, 분할, 텍스트/테이블 추출, 생성)
- FORMS.md: PDF 폼 작성 (fillable fields, annotation 방식)
- REFERENCE.md: 고급 기능, 성능 최적화

### 권장 라이브러리 (작업별)
| 작업 | 라이브러리 |
|------|-----------|
| 텍스트 추출 | pdfplumber |
| 테이블 추출 | pdfplumber + pandas |
| PDF 생성 | reportlab |
| 병합/분할/회전 | pypdf |
| 빠른 렌더링 | pypdfium2 |
| 스캔 PDF OCR | pytesseract + pdf2image |
| 폼 작성 | pypdf 또는 pdf-lib (JS) |

### 성능 팁
- 대용량 텍스트 추출: `pdftotext` CLI가 가장 빠름
- 이미지 추출: `pdfimages` CLI 사용
- 대용량 PDF: chunk 단위 처리


C:\Users\suksu\.claude\
├── CLAUDE.md          ← 전역 설정 (여기에 PDF 가이드 추가)
└── skills\
    └── pdf\
        ├── SKILL.md
        ├── FORMS.md
        └── REFERENCE.md