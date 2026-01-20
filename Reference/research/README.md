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

## AI API 가격 비교 (2025년 1월 기준)

### 1M 토큰당 가격 (공식 가격)

| 모델 | Input ($/1M) | Output ($/1M) | 비고 |
|------|-------------|---------------|------|
| **Ollama (로컬)** | $0 | $0 | 무료, GPU 전기세만 |
| **Gemini 2.0 Flash** | $0.10 | $0.40 | 무료 티어 있음 |
| **Gemini 1.5 Flash** | $0.075 | $0.30 | 가장 저렴 |
| **Gemini 1.5 Pro** | $1.25 | $5.00 | ≤200K 기준 |
| **GPT-4o-mini** | $0.15 | $0.60 | 가성비 최고 |
| **GPT-4o** | $2.50 | $10.00 | 고품질 |
| **GPT-4-turbo** | $10.00 | $30.00 | 레거시 |
| **Claude Haiku 3** | $0.25 | $1.25 | 구버전 |
| **Claude Haiku 3.5** | $0.80 | $4.00 | 빠름 |
| **Claude Haiku 4.5** | $1.00 | $5.00 | 최신 |
| **Claude Sonnet 3.5** | $3.00 | $15.00 | 균형 |
| **Claude Sonnet 4** | $3.00 | $15.00 | 균형 |
| **Claude Opus 4.1** | $15.00 | $75.00 | 프리미엄 |
| **Claude Opus 4.5** | $5.00 | $25.00 | 최신 플래그십 |

### 10페이지 PDF 번역 비용 (Input 6K + Output 6K 토큰 기준)

| 모델 | Input 비용 | Output 비용 | 총 비용 | 원화 환산 |
|------|-----------|-------------|---------|-----------|
| **Ollama (로컬)** | $0 | $0 | $0 | 0원 |
| **Gemini 2.0 Flash** | $0.0006 | $0.0024 | $0.003 | ~4원 |
| **GPT-4o-mini** | $0.0009 | $0.0036 | $0.005 | ~7원 |
| **Claude Haiku 3** | $0.0015 | $0.0075 | $0.009 | ~12원 |
| **Gemini 2.5 Flash** | $0.0018 | $0.015 | $0.017 | ~23원 |
| **Claude Haiku 3.5** | $0.0048 | $0.024 | $0.029 | ~40원 |
| **Claude Haiku 4.5** | $0.006 | $0.03 | $0.036 | ~50원 |
| **GPT-4o** | $0.015 | $0.06 | $0.075 | ~100원 |
| **Claude Sonnet 3.5/4** | $0.018 | $0.09 | $0.108 | ~150원 |
| **Gemini 2.5 Pro** | $0.0075 | $0.06 | $0.068 | ~93원 |
| **Claude Opus 4.5** | $0.03 | $0.15 | $0.18 | ~250원 |
| **GPT-4-turbo** | $0.06 | $0.18 | $0.24 | ~330원 |
| **Claude Opus 4.1** | $0.09 | $0.45 | $0.54 | ~740원 |

### 100페이지 PDF 번역 비용 (Input 60K + Output 60K 토큰 기준)

| 모델 | 총 비용 | 원화 환산 |
|------|---------|-----------|
| **Ollama (로컬)** | $0 | 0원 |
| **Gemini 2.0 Flash** | $0.03 | ~40원 |
| **GPT-4o-mini** | $0.05 | ~70원 |
| **Claude Haiku 3** | $0.09 | ~120원 |
| **Gemini 2.5 Flash** | $0.17 | ~230원 |
| **Claude Haiku 3.5** | $0.29 | ~400원 |
| **Claude Haiku 4.5** | $0.36 | ~500원 |
| **Gemini 2.5 Pro** | $0.68 | ~930원 |
| **GPT-4o** | $0.75 | ~1,000원 |
| **Claude Sonnet 3.5/4** | $1.08 | ~1,500원 |
| **Claude Opus 4.5** | $1.80 | ~2,500원 |
| **GPT-4-turbo** | $2.40 | ~3,300원 |
| **Claude Opus 4.1** | $5.40 | ~7,400원 |

### PDF Translator 앱 추천 모델

| 순위 | 모델 | 추천 이유 |
|-----|------|----------|
| 1️⃣ | **Gemini 2.0 Flash** | 최저가 (~4원/10p), 빠름 |
| 2️⃣ | **GPT-4o-mini** | 품질↑ 가격↓ 가성비 최고 |
| 3️⃣ | **Gemini 2.5 Flash** | 출력 토큰 65K, 긴 문서용 |
| 4️⃣ | **Claude Haiku 3.5** | 빠르고 안정적 |
| 5️⃣ | **Ollama (로컬)** | 완전 무료, 오프라인 |

### Gemini 무료 티어 한도 (2025년 12월 변경)

| 모델 | RPM (분당) | TPM (분당 토큰) | RPD (일일) |
|------|-----------|----------------|-----------|
| **Gemini 2.5 Pro** | 5 | 250,000 | 25~100 |
| **Gemini 2.5 Flash** | 10 | 250,000 | 250 |
| **Gemini 2.5 Flash-Lite** | 15~30 | 250,000 | 1,000~1,500 |
| **Gemini 2.0 Flash** | 10 | 250,000 | 250 |

> ⚠️ **주의**: 2025년 12월 7일에 무료 티어가 50~80% 축소됨. 일일 250회 제한으로 대량 번역 시 한도 초과 가능.

---

## OCR API 가격 비교 (2025년 1월 기준)

### Google Cloud Vision API

| 월간 사용량 | 가격 ($/1,000건) | 원화 환산 |
|------------|-----------------|-----------|
| 1 ~ 1,000건 | **무료** | 0원 |
| 1,001 ~ 5,000,000건 | $1.50 | ~2,050원 |
| 5,000,001건 이상 | $0.60 | ~820원 |

**기능별 가격** (1,001 ~ 5,000,000건 기준):

| 기능 | 가격 ($/1,000건) | 용도 |
|------|-----------------|------|
| **TEXT_DETECTION** | $1.50 | 일반 텍스트 OCR |
| **DOCUMENT_TEXT_DETECTION** | $1.50 | 문서/PDF OCR (고밀도 텍스트) |
| **Label Detection** | $1.50 | 이미지 라벨링 |
| **Face Detection** | $1.50 | 얼굴 감지 |
| **Logo Detection** | $1.50 | 로고 감지 |
| **Object Localization** | $2.25 | 객체 위치 감지 |
| **Web Detection** | $3.50 | 웹 이미지 검색 |

### PDF 번역 시 OCR 비용 계산

| 페이지 수 | Google Vision | PaddleOCR (로컬) | 비고 |
|----------|---------------|------------------|------|
| 10페이지 | $0.015 (~20원) | $0 (무료) | Vision: 무료 한도 내 |
| 100페이지 | $0.15 (~200원) | $0 (무료) | Vision: 무료 한도 내 |
| 1,000페이지 | $1.50 (~2,050원) | $0 (무료) | Vision: 유료 구간 |
| 10,000페이지 | $15.00 (~20,500원) | $0 (무료) | Vision: 유료 구간 |

> 💡 **참고**: Google Vision은 PDF의 각 페이지를 개별 이미지로 처리하므로 10페이지 PDF = 10건 요청

### OCR 서비스 비교

| 서비스 | 무료 한도 | 유료 가격 | 장점 | 단점 |
|--------|----------|----------|------|------|
| **Google Vision** | 월 1,000건 | $1.50/1K건 | 정확도↑, 60+언어 | 클라우드 의존 |
| **PaddleOCR** | 무제한 | 무료 | 로컬, 빠름 | GPU 필요 |
| **EasyOCR** | 무제한 | 무료 | 설치 쉬움 | 속도↓ |
| **Azure Read API** | 없음 | $1.00/1K건 | 대용량 할인 | 무료 없음 |
| **AWS Textract** | 월 1,000건 | $1.50/1K건 | AWS 통합 | 복잡한 설정 |

### Google Vision API 사용 시 주의사항

1. **과금 방식**: 이미지당 + 기능당 과금 (TEXT_DETECTION + LABEL_DETECTION = 2건)
2. **신규 고객 크레딧**: $300 무료 크레딧 제공 (90일 유효)
3. **PDF 처리**: PDF/TIFF 파일은 페이지별로 개별 요청
4. **언어 힌트**: `languageHints` 파라미터로 OCR 정확도 향상 가능
5. **API 엔드포인트**: `vision.googleapis.com` (글로벌), `eu-vision.googleapis.com` (EU)

---

## 추가 예정 주제

- [ ] OCR 정확도 향상
- [ ] 테이블 구조 인식
- [ ] 다국어 폰트 렌더링
- [ ] PDF 처리 최적화
