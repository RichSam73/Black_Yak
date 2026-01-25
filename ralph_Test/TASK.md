# Ralph Loop Task: 숲프린트/가슴로고 겹침 해결

## ⛔ 테스트 격리 규칙
- **수정 대상**: `ralph_Test/` 폴더 내 파일만
- **원본 금지**: `PDF_Translate_Note/` 등 원본 폴더 절대 수정 금지
- 테스트 성공 후 사용자가 직접 원본 반영 여부 결정

## 문제 정의
- **현상**: "숲프린트" 번역 텍스트가 "가슴로고" 영역을 12px 침범
- **원인**: 번역된 영어 텍스트가 원본 한글보다 길어서 셀 경계 초과

## 성공 기준
```
숲프린트 → 가슴로고 침범 횟수 = 0
```

## 테스트 절차

### 1단계: 앱 실행
```bash
python "E:/Antigravity/Black_Yak/ralph_Test/app.py"
# 포트 7000에서 실행됨
```

### 2단계: 번역 수행 (수동)
1. http://localhost:7000 접속
2. test_input.pdf 업로드
3. 번역 실행
4. 로그 생성 대기

### 3단계: 테스트 실행
```bash
cd E:/Antigravity/Black_Yak/ralph_Test && python test_overlap.py
```

### 성공 조건
- `TEST PASSED` 출력
- `target_overlap_count: 0`
- `chest_logo_invaded: false`
- 종료 코드: 0

## 수정 대상

### 파일: `app.py`

#### 1. 겹침 감지 로직 (line 2338-2385)
```python
# 현재: OVERFLOW_THRESHOLD = 30
# 검토: 임계값 조정 또는 조기 감지
```

#### 2. 약어 처리 (line 1933-1957)
```python
# 숲프린트/가슴로고 전용 약어 추가 검토
# garment_dict.json에 약어 정의
```

#### 3. 폰트 크기 동적 축소 (신규)
```python
# 겹침 발생 시 폰트 크기 자동 축소
# 최소 폰트 크기: 8pt
```

## 참고 데이터

### 문제 발생 위치
```
숲프린트: x=924, w=52, right=976, cell=(924,481,55,20)
가슴로고: x=964 시작
침범량: 976 - 964 = 12px
```

### 해결 방향
1. **약어 적용**: "Forest Print" → "F.Print" 또는 "숲프린트" 유지
2. **폰트 축소**: 12pt → 10pt로 줄여서 너비 감소
3. **셀 너비 확인**: OCR bbox vs 렌더링 영역 일치 확인

## 제약 조건
- 기존 번역 품질 유지
- 가독성 유지 (최소 폰트 8pt)
- 원본 레이아웃 보존
- 다른 텍스트에 부작용 없음
