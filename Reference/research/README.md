# Research 자료 모음

이 폴더는 웹 검색을 통해 수집한 연구 자료를 주제별로 정리한 것입니다.

---

## 폴더 구조

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

## 검색 도구 목록

| 도구 | MCP 이름 | 용도 |
|------|----------|------|
| Brave Search | `mcp__brave-search__brave_web_search` | 일반 웹 검색 |
| Exa Search | `mcp__exa__web_search_exa` | Semantic 웹 검색 |
| Exa Code Context | `mcp__exa__get_code_context_exa` | 코드/라이브러리 검색 |
| WebSearch | Claude 내장 | 일반 웹 검색 |
| GitHub Code Search | `mcp__github__search_code` | GitHub 코드 검색 |
| GitHub File Contents | `mcp__github__get_file_contents` | GitHub 파일 내용 조회 |

---

## 주제별 요약

### 1. Text Removal & Inpainting (2026-01-08)

**목적**: 이미지에서 텍스트를 깨끗하게 지우고 배경을 복원

**핵심 방법**:
1. **OpenCV Inpainting** - `cv2.inpaint()` (TELEA/NS 알고리즘)
2. **LaMa Inpainting** - AI 기반 고품질 복원 (`pip install simple-lama-inpainting`)
3. **배경색 샘플링** - 단순 배경에서 주변 색상으로 채우기

**권장**: 기술서 문서는 대부분 흰색 배경이므로 OpenCV Inpainting으로 충분

### 2. Text Positioning (2026-01-08)

**목적**: 번역된 텍스트를 원본 위치에 정확하게 배치

**핵심 방법**:
1. **Bounding Box 좌표 추출** - OCR 결과에서 min/max 좌표 계산
2. **폰트 크기 자동 조절** - 박스에 맞는 최대 크기 탐색
3. **텍스트 정렬** - 왼쪽/중앙/오른쪽 + 상단/중앙/하단
4. **텍스트 줄바꿈** - 긴 텍스트 처리

**권장**: 고정 폰트 크기 목록에서 맞는 크기 선택 + 왼쪽 정렬

---

## 사용 라이브러리 목록

### Python 패키지 (pip)

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

### 핵심 함수/API

| 함수 | 라이브러리 | 용도 |
|------|-----------|------|
| `cv2.inpaint()` | OpenCV | 텍스트 영역 복원 (TELEA/NS) |
| `cv2.fillPoly()` | OpenCV | 마스크 폴리곤 채우기 |
| `cv2.dilate()` | OpenCV | 마스크 확장 |
| `ImageDraw.text()` | Pillow | 텍스트 렌더링 |
| `ImageDraw.textbbox()` | Pillow | 텍스트 바운딩 박스 계산 |
| `ImageFont.truetype()` | Pillow | 폰트 로드 |
| `SimpleLama()` | simple-lama | AI Inpainting |

### GitHub 참고 프로젝트

| 프로젝트 | URL | 설명 |
|----------|-----|------|
| advimman/lama | https://github.com/advimman/lama | SOTA AI Inpainting 모델 |
| yeungchenwa/OCR-SAM | https://github.com/yeungchenwa/OCR-SAM | OCR + SAM + Stable Diffusion |
| manbehindthemadness/unscribe | https://github.com/manbehindthemadness/unscribe | LaMa + CRAFT 조합 |
| boysugi20/python-image-translator | https://github.com/boysugi20/python-image-translator | EasyOCR + PIL 번역 |
| bnsreenu/python_for_microscopists | https://github.com/bnsreenu/python_for_microscopists | 실용적 예제 코드 |

---

## 추가 예정 주제

- [ ] OCR 정확도 향상
- [ ] 테이블 구조 인식
- [ ] 다국어 폰트 렌더링
- [ ] PDF 처리 최적화
