# -*- coding: utf-8 -*-
import requests
import base64
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

img_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"

with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(f"Base64: {len(img_base64)}")

# 한글 간단 프롬프트
prompt = "이 테이블의 첫 번째 행(헤더)만 JSON 배열로 추출하세요. 예: [\"DIV\",\"CODE\",\"NAME\"]"

print("\n=== 테스트 1: 헤더만 ===")
response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "images": [img_base64],
        "stream": False
    },
    timeout=180
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    raw = result.get('response', '').strip()
    print(f"응답: {raw[:500]}")
else:
    print(f"에러: {response.text[:500]}")
