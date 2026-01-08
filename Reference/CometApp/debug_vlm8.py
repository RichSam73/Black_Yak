# -*- coding: utf-8 -*-
import requests
import base64
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

img_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"

with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

prompt = """테이블의 모든 행을 JSON 2D 배열로 추출하세요.
빈 셀은 ""로 표시.
JSON만 출력."""

# 여러 모델 테스트
models = ["llama3.2-vision:latest", "granite3.2-vision:latest"]

for model in models:
    print(f"\n{'='*50}")
    print(f"=== 모델: {model} ===")
    print(f"{'='*50}")
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=300
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            raw = result.get('response', '').strip()
            print(f"응답 길이: {len(raw)}")
            print(f"응답:\n{raw[:1200]}")
        else:
            print(f"에러: {response.text[:300]}")
    except Exception as e:
        print(f"예외: {e}")
