# -*- coding: utf-8 -*-
"""VLM OCR 테스트"""
import requests
import base64
import json

# 테스트 이미지 (간단한 테이블)
# 실제로는 업로드된 이미지 사용

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

prompt = """이 테이블 이미지의 모든 텍스트를 JSON 2D 배열로 추출하세요.

규칙:
1. 각 행은 배열의 요소
2. 각 셀은 행 배열의 요소
3. 빈 셀은 빈 문자열 ""
4. 병합된 셀은 첫 번째 위치에만 값, 나머지는 ""
5. JSON만 출력, 다른 설명 없음

출력 형식:
[["헤더1","헤더2","헤더3"],["값1","값2","값3"],["값4","값5","값6"]]"""

# 이미지 없이 테스트 (텍스트만)
print("=== VLM 연결 테스트 ===")
try:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": "간단히 '테스트 성공'이라고 답하세요.",
            "stream": False,
            "options": {"num_predict": 100, "temperature": 0}
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result.get('response', 'N/A')[:200]}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== VLM 모델 정보 ===")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=10)
    models = response.json().get('models', [])
    for m in models:
        if 'qwen' in m['name'].lower() or 'vision' in m['name'].lower():
            print(f"  {m['name']} - {m.get('size', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")
