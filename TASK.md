# 작업 목표

Ctrl+휠 줌 기능이 동작하는 독립 테스트 파일 작성 및 검증

## 상세 내용

- `PDF_Translator/app.py`는 그대로 유지
- **`ralph_Test/` 폴더에서 테스트 진행**
- 테스트 HTML 파일(`ralph_Test/test_wheel_zoom.html`)로 Ctrl+휠 줌 구현
- 테스트 성공 시 해당 코드를 `PDF_Translator/app.py`에 적용

### 테스트 파일 요구사항
1. 이미지가 있는 컨테이너 (overflow: auto)
2. Ctrl+휠로 확대/축소
3. 일반 휠은 스크롤 유지
4. 줌 레벨 표시 (50%-200%)

## 성공 조건

- [ ] `ralph_Test/` 폴더 생성
- [ ] `ralph_Test/test_wheel_zoom.html` 파일 생성
- [ ] Ctrl+휠 확대/축소 동작 확인 (Playwright 자동 테스트)
- [ ] 일반 휠 스크롤 정상 동작
- [ ] 줌 범위 제한 동작

## 테스트 방법

1. Playwright로 `ralph_Test/test_wheel_zoom.html` 열기
2. Ctrl+휠 이벤트 발생시켜 줌 변화 확인
3. DOM 상태(줌 레벨 텍스트)로 검증

## 참고 사항

- 테스트 폴더: `ralph_Test/`
- 테스트 파일: `ralph_Test/test_wheel_zoom.html`
- 성공 후 코드를 `PDF_Translator/app.py`에 이식
