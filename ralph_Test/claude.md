# PDF Translator - Claude 참고사항

## 🚨 중요: 폰트 설정

### 문제
- `arial.ttf` 폰트는 **한글 글리프가 없음**
- 한글 텍스트가 □□□ (tofu)로 깨져서 표시됨

### 해결
- 반드시 `malgun.ttf` (맑은 고딕) 사용
- 맑은 고딕은 한글, 영어, 중국어, 일본어 등 다국어 지원

### 코드 예시
```python
# ❌ 잘못된 코드
font = ImageFont.truetype("arial.ttf", font_size)
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)

# ✅ 올바른 코드
font = ImageFont.truetype("malgun.ttf", font_size)
font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
```

### Windows 한글 폰트 목록
| 폰트 파일 | 폰트 이름 | 용도 |
|-----------|-----------|------|
| `malgun.ttf` | 맑은 고딕 | 기본 UI용 (권장) |
| `malgunbd.ttf` | 맑은 고딕 Bold | 강조용 |
| `gulim.ttc` | 굴림 | 레거시 |
| `batang.ttc` | 바탕 | 명조체 |
| `NanumGothic.ttf` | 나눔고딕 | 무료 폰트 |

---

## 📁 프로젝트 구조

```
E:\Antigravity\Black_Yak\
├── PDF_Translator/      # 메인 PDF 번역 앱
│   ├── app.py           # Flask 앱 (포트 6009)
│   ├── garment_dict.json # 의류 용어 사전
│   └── output/          # 번역 결과물
├── PDF_Translate_Note/  # 메모 기능 추가 버전 (v1.9.0)
└── Reference/           # 참고 자료
```

---

## 🔧 자주 발생하는 문제

### 1. 한글 깨짐 (□□□)
- **원인**: arial.ttf 등 한글 미지원 폰트 사용
- **해결**: malgun.ttf로 변경

### 2. 텍스트 겹침
- **원인**: OCR bbox와 렌더링 영역 불일치
- **해결**: 겹침 감지 로직 확인 (overlap_debug.log)

### 3. API 키 오류
- **위치**: HARDCODED_API_KEYS 딕셔너리
- **키 종류**: openai, claude_sije, claude_seam, gemini

---

## 📝 버전 관리

- Git repo: `RichSam73/Black_Yak`
- 브랜치: main
- 커밋 메시지 형식: `v{버전}-{변경내용}`

---

*최종 업데이트: 2026-01-20 (v1.8.3)*
