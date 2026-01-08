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

## 추가 예정 주제

- [ ] OCR 정확도 향상
- [ ] 테이블 구조 인식
- [ ] 다국어 폰트 렌더링
- [ ] PDF 처리 최적화
