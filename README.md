# 📊 PDF 요소별 추출기 (AI & OCR 기반)

블랙야크 의류 워크시트 및 일반 PDF 파일에서 텍스트, 도면(벡터), 차트, 이미지를 지능적으로 추출하고, ERP 연동 데이터를 생성하는 도구입니다.

## 🚀 주요 기능

1.  **AI 스마트 테이블 추출 (Comet Legacy)** (`smart_table_extractor.py`)
    *   **기술**: **PaddleOCR** (Deep Learning OCR) + **Table Transformer** (MS/HuggingFace)
    *   **기능**: 이미지나 벡터로 된 PDF 페이지를 AI가 분석하여 텍스트를 인식하고, 원본 이미지 위에 **선택 가능한 투명 텍스트 레이어**를 생성합니다.
    *   **정확도**: 딥러닝 기반 OCR을 사용하여 비정형 텍스트 인식률이 매우 높습니다.

2.  **벡터 도면 자동 분류** (`app.py`)
    *   **기술**: PyMuPDF (`fitz`) Clustering
    *   **기능**: PDF 내부의 수천 개의 벡터(선, 도형) 데이터를 분석하여 '자켓 도면'과 같은 의미 있는 덩어리를 자동으로 감지하고 이미지로 추출합니다.

3.  **VLM (Vision Language Model) 실험적 지원**
    *   **기술**: Ollama (`granite3.2-vision`)
    *   **기능**: 문서의 시각적 맥락을 이해하여 의미 기반으로 데이터를 추출합니다.

## 🛠 기술 스택 (Tech Stack)

이 프로젝트는 다음과 같은 핵심 기술을 사용합니다:

*   **Frontend/UI**: [Streamlit](https://streamlit.io/) (웹 인터페이스)
*   **PDF Processing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
*   **AI/OCR Engine**:
    *   **PaddleOCR**: 오픈소스 딥러닝 OCR 엔진 (로컬 설치)
    *   **Table Transformer**: 마이크로소프트의 테이블 구조 인식 모델
*   **Language**: Python 3.9+

## 💻 설치 및 실행 (Installation)

### 1. 필수 프로그램 설치
이 프로그램은 로컬 컴퓨터의 자원(CPU/GPU)을 사용합니다.
*   Python 3.9 이상 설치

### 2. 라이브러리 설치
```bash
pip install -r requirements.txt
pip install paddleocr protobuf==3.20.0
# 선택 사항 (GPU 사용 시)
# pip install paddlepaddle-gpu
```

### 3. 프로그램 실행
```bash
streamlit run app.py
```
실행 후 브라우저가 자동으로 열리며 프로그램이 시작됩니다.

## ☁️ 서버 배포 시 주의사항 (Deployment)

이 서비스를 웹 서버에 배포하여 운영하려면 다음 사항을 고려해야 합니다:

1.  **컴퓨팅 자원 (Compute Resources)**
    *   이 '웹 서비스'는 클라이언트(사용자)가 아닌 **서버의 CPU/GPU**를 사용하여 AI 모델을 구동합니다.
    *   최소 4GB 이상의 RAM과 멀티코어 CPU가 권장됩니다.

2.  **필수 시스템 패키지** (Linux 서버 기준)
    *   PaddleOCR 및 OpenCV 구동을 위한 라이브러리가 필요합니다.
    *   `apt-get install libgl1-mesa-glx libgomp1`

3.  **모델 데이터**
    *   최초 실행 시 PaddleOCR 학습 모델(약 100MB)이 서버에 자동으로 다운로드됩니다. 내부망 서버의 경우 사전에 모델 파일을 옮겨두어야 합니다.
