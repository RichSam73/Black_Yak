# 📊 PDF 요소별 추출기 (AI & OCR 기반)

블랙야크 의류 워크시트 및 일반 PDF 파일에서 텍스트, 도면(벡터), 차트, 이미지를 지능적으로 추출하고, ERP 연동 데이터를 생성하는 도구입니다.

## 🚀 주요 기능

### 1. CometApp - ERP 테이블 추출 (최신)

**위치**: `Reference/CometApp/app_ai.py`

**기술 스택**:
- **PaddleOCR v5** (PP-OCRv5_server_det + korean_PP-OCRv5_mobile_rec)
- **llama3.2-vision** (AI Vision 보정)
- **Flask** 웹 서버 (포트 6001)

**핵심 기능**:
- 하이브리드 OCR: PaddleOCR + AI Vision 조합으로 누락 텍스트 자동 보정
- X-좌표 클러스터링 (threshold=30) 기반 컬럼 감지
- 3-Method 병합 헤더 컬럼 감지:
  - Method 2: 헤더 범위 기반 병합
  - Method 3: 빈 헤더 컬럼 병합 (임계값 40px)
- COLOR/SIZE QTY 테이블 특수 처리

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

### Round 9 (2025-12-30)
- SUP CD / SUP NM 컬럼 병합 방지
- Method 3 임계값 150px → 40px 축소

### Round 8 (2025-12-29)
- COLOR/SIZE QTY 테이블 Method 2만 허용
- 빈 컬럼 오류 해결

### Round 7 (2025-12-28)
- 하이브리드 OCR + 테이블 구조 분석 기반 누락 텍스트 자동 삽입

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
