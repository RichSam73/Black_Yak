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

# 전체 테이블 추출 프롬프트
prompt = """테이블의 모든 행을 JSON 2D 배열로 추출하세요.
빈 셀은 ""로 표시.
JSON만 출력."""

print("\n=== 전체 테이블 추출 ===")
response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
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
    print(f"응답 앞부분:\n{raw[:1500]}")
    
    # JSON 파싱
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1:
        try:
            t = json.loads(raw[start:end+1])
            print(f"\n=== 파싱 성공! ===")
            print(f"행 수: {len(t)}")
            for i, row in enumerate(t[:5]):
                print(f"행{i}: {str(row)[:100]}...")
        except Exception as e:
            print(f"파싱 실패: {e}")
else:
    print(f"에러: {response.text[:500]}")
