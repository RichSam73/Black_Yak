# -*- coding: utf-8 -*-
import requests
import base64

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

img_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"

print(f"이미지: {img_path}")

with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(f"Base64 길이: {len(img_base64)}")

prompt = "이 이미지에 무엇이 보이나요? 간단히 설명해주세요."

print("\n=== VLM 호출 ===")
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
print(f"Response: {response.text[:1500]}")
