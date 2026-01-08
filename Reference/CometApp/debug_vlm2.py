# -*- coding: utf-8 -*-
import requests
import base64
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

img_path = r"E:\Antigravity\Black_Yak\.playwright-mcp\ai_vision_erp_table.png"

print(f"테스트 이미지: {img_path}")

# base64 인코딩
with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(f"Base64 길이: {len(img_base64)}")

# 프롬프트
prompt = """이 테이블 이미지의 모든 텍스트를 JSON 2D 배열로 추출하세요.

규칙:
1. 각 행은 배열의 요소
2. 각 셀은 행 배열의 요소  
3. 빈 셀은 빈 문자열 ""
4. JSON만 출력, 다른 설명 없음

출력 형식:
[["헤더1","헤더2"],["값1","값2"]]"""

print("\n=== VLM 호출 중... ===")
response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "images": [img_base64],
        "stream": False,
        "options": {"num_predict": 4096, "temperature": 0}
    },
    timeout=180
)

print(f"Status: {response.status_code}")
result = response.json()
raw = result.get('response', '').strip()

print(f"\n=== 원본 응답 (길이: {len(raw)}) ===")
print(raw[:2500])
