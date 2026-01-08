# PaddleOCR Bbox 좌표 불일치 문제 조사 결과

**조사 일시**: 2026-01-08
**조사자**: Claude

---

## 사용한 MCP 도구

| MCP 도구 | 용도 | 쿼리 |
|----------|------|------|
| `WebSearch` | Claude 내장 웹 검색 | "PaddleOCR bbox coordinates wrong position misaligned 2024 2025" |
| `mcp__brave-search__brave_web_search` | Brave Search | "PaddleOCR dt_polys rec_polys coordinates mismatch image" |
| `mcp__exa__web_search_exa` | Exa AI 시맨틱 검색 | "PaddleOCR bounding box coordinates not matching text position BGR RGB image format" |
| `mcp__context7__resolve-library-id` | Context7 라이브러리 ID 조회 | "PaddleOCR bbox coordinates dt_polys doc_preprocessor_res" |
| `mcp__context7__query-docs` | Context7 최신 문서 조회 | "doc_preprocessor_res output_img bbox coordinates image preprocessing" |
| `WebFetch` | GitHub Discussion 상세 조회 | https://github.com/PaddlePaddle/PaddleOCR/discussions/15957 |

---

## 🔍 문제 원인 (핵심)

PaddleOCR의 `predict()` 함수는 **기본적으로 이미지를 전처리**합니다:

1. **Document Unwarping** (`use_doc_unwarping=True` 기본값) - 문서 왜곡 보정
2. **Orientation Classification** (`use_doc_orientation_classify`) - 문서 방향 분류
3. **Textline Orientation** (`use_textline_orientation=True` 기본값) - 텍스트라인 방향 분류
4. **이미지 스케일링/크롭** - 내부적으로 이미지 크기 변환

**반환된 bbox 좌표는 "전처리된 이미지" 기준**이므로, 원본 이미지에 적용하면 좌표가 맞지 않습니다.

### GitHub Discussion #15957 핵심 내용

> "When running layout detection directly on the original PDF or on an image rendered from a specific PDF page, the predicted bounding boxes look correct when visualized immediately after inference. However, when trying to process these coordinates programmatically... they don't align correctly with the actual content anymore. The boxes appear offset or scaled incorrectly."

**원인**: PaddleOCR은 `doc_preprocessor_res`에 전처리된 이미지를 저장하고, bbox 좌표는 이 전처리된 이미지 기준입니다.

---

## ✅ 해결 방법 3가지

### 방법 1: 전처리 비활성화 (권장) ⭐

```python
ocr = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,  # 문서 방향 분류 끄기
    use_doc_unwarping=False,             # 문서 왜곡 보정 끄기
    use_textline_orientation=False       # 텍스트라인 방향 분류 끄기
)
```

**장점**: 가장 간단, 원본 이미지 기준 좌표 반환
**단점**: 왜곡된 문서나 기울어진 이미지에서 정확도 감소 가능

### 방법 2: doc_preprocessor_res 이미지 사용

```python
result = ocr.predict(img)
for item in result:
    # 전처리된 이미지 추출
    if hasattr(item, 'doc_preprocessor_res'):
        preprocessed_img = item.doc_preprocessor_res.get('output_img')
        # 이 이미지에 bbox를 적용하면 정확히 맞음
```

**장점**: 전처리 혜택 유지, 좌표 정확
**단점**: 전처리된 이미지를 별도 관리해야 함

### 방법 3: 스케일 비율 계산

```python
original_size = original_img.shape[:2]  # (height, width)
processed_size = preprocessed_img.shape[:2]
scale_x = original_size[1] / processed_size[1]
scale_y = original_size[0] / processed_size[0]

# 좌표 변환
adjusted_bbox = [[p[0] * scale_x, p[1] * scale_y] for p in bbox]
```

**장점**: 원본 이미지에 적용 가능
**단점**: 크롭이 발생한 경우 오프셋 계산도 필요

---

## 📊 PaddleOCR 출력 구조

```python
result[0].keys() = [
    'input_path',
    'page_index',
    'doc_preprocessor_res',  # ← 전처리 결과
    'dt_polys',              # ← detection polygons (원본 감지 좌표)
    'model_settings',
    'text_det_params',
    'text_type',
    'text_rec_score_thresh',
    'return_word_box',
    'rec_texts',             # ← 인식된 텍스트
    'rec_scores',            # ← 신뢰도
    'rec_polys',             # ← recognition polygons (필터링된 좌표)
    'vis_fonts',
    'textline_orientation_angles',
    'rec_boxes'              # ← [x_min, y_min, x_max, y_max] 형식
]
```

### doc_preprocessor_res 구조

```python
doc_preprocessor_res = {
    'input_path': None,
    'model_settings': {
        'use_doc_orientation_classify': True/False,
        'use_doc_unwarping': True/False
    },
    'angle': -1,  # 또는 [0,1,2,3] → [0°,90°,180°,270°]
    'output_img': <전처리된 이미지>  # ← 이 이미지 기준으로 bbox 좌표가 계산됨
}
```

---

## 📚 참고 자료 (Sources)

### GitHub Discussions
- [PaddleOCR Layout Coordinate Mismatch - #15957](https://github.com/PaddlePaddle/PaddleOCR/discussions/15957) ⭐ 핵심
- [How to get pixel perfect location of text? - #14769](https://github.com/PaddlePaddle/PaddleOCR/discussions/14769)
- [Bug Report: Incorrect Character Box Order for Inverted Text - #14570](https://github.com/PaddlePaddle/PaddleOCR/discussions/14570)

### 공식 문서
- [PaddleOCR Documentation - OCR Pipeline](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html)
- [PaddleX Documentation - OCR](https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html)
- [PaddleOCR Quick Start](https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html)

### Stack Overflow
- [Paddle OCR BoundingBox Format](https://stackoverflow.com/questions/72893442/paddle-ocr-boundingbox-format)
- [Extract bounding Boxes from an Image Paddleocr](https://stackoverflow.com/questions/72840785/extract-bounding-boxes-from-an-image-paddleocr)

### 기타
- [Medium - How To Use OCR Bounding Boxes](https://medium.com/@michael71314/how-to-use-ocr-bounding-boxes-c00303bc11c4)
- [HuggingFace - PP-OCRv5_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec)

---

## 🛠️ 적용할 코드 수정

### test_erase.py 수정

```python
# 기존
ocr = PaddleOCR(lang='korean', use_textline_orientation=True)

# 수정
ocr = PaddleOCR(
    lang='korean',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
```

### app.py 수정

```python
# 기존 (get_ocr_engine 함수)
ocr_engine = PaddleOCR(use_textline_orientation=True, lang="korean")

# 수정
ocr_engine = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
```

---

## 결론

PaddleOCR의 기본 전처리 기능(문서 왜곡 보정, 방향 분류)이 활성화되면 내부적으로 이미지가 변환되고, 반환되는 bbox 좌표는 변환된 이미지 기준입니다. 원본 이미지에 bbox를 적용하려면 **전처리를 비활성화**하거나, **전처리된 이미지를 사용**해야 합니다.
