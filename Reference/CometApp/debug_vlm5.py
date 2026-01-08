# -*- coding: utf-8 -*-
import requests
import base64
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

img_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"

with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(f"Base64 길이: {len(img_base64)}")

# 간단한 JSON 프롬프트
prompt = """Extract all text from this table as JSON 2D array.
Format: [["header1","header2"],["value1","value2"]]
Empty cells: ""
Output JSON only, no explanation."""

print("\n=== VLM 호출 (options 없이) ===")
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
result = response.json()
raw = result.get('response', '').strip()

print(f"\n=== 응답 (길이: {len(raw)}) ===")
print(raw[:2000])

# JSON 파싱
print("\n=== JSON 파싱 ===")
start = raw.find('[')
end = raw.rfind(']')
if start != -1 and end != -1:
    try:
        t = json.loads(raw[start:end+1])
        print(f"성공! 행수: {len(t)}")
        for i, row in enumerate(t[:5]):
            print(f"  행{i}: {str(row)[:80]}...")
    except Exception as e:
        print(f"실패: {e}")
