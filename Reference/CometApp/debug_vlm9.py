# -*- coding: utf-8 -*-
import requests
import base64
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2-vision:latest"

img_path = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"

with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

# 명확한 2D 배열 프롬프트
prompt = """Extract this table as a JSON 2D array.
Rules:
- First row is headers
- Each row is an array element
- Each cell is a string in the row array
- Empty cells: ""
- Output ONLY the JSON array, nothing else

Example format:
[["Header1","Header2"],["Value1","Value2"],["Value3","Value4"]]"""

print(f"=== llama3.2-vision 2D 배열 테스트 ===")
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
result = response.json()
raw = result.get('response', '').strip()
print(f"응답 길이: {len(raw)}")

# JSON 파싱
start = raw.find('[')
end = raw.rfind(']')
if start != -1 and end != -1:
    json_str = raw[start:end+1]
    try:
        t = json.loads(json_str)
        print(f"\n=== 파싱 성공! ===")
        print(f"행 수: {len(t)}")
        print(f"타입: {type(t[0]) if t else 'N/A'}")
        
        # 2D 배열인지 확인
        if t and isinstance(t[0], list):
            print("형식: 2D 배열 ✓")
            for i, row in enumerate(t[:5]):
                print(f"행{i}: {row}")
        else:
            print("형식: 객체 배열 (변환 필요)")
            print(f"첫 요소: {t[0]}")
    except Exception as e:
        print(f"파싱 실패: {e}")
        print(f"JSON 시작: {json_str[:300]}...")
else:
    print("JSON 없음")
    print(f"응답:\n{raw[:1000]}")
