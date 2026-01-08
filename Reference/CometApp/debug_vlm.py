# -*- coding: utf-8 -*-
import requests
import base64
import json
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

# 테스트 이미지 찾기
img_path = None
search_paths = [
    r"E:\Antigravity\Black_Yak\Reference\CometApp\uploads",
    r"E:\Antigravity\Black_Yak\Sample",
]

for search_dir in search_paths:
    if os.path.exists(search_dir):
        for f in os.listdir(search_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(search_dir, f)
                break
    if img_path:
        break

if not img_path:
    print("이미지 없음!")
    exit(1)

print(f"테스트 이미지: {img_path}")
print(f"파일 크기: {os.path.getsize(img_path)/1024:.1f} KB")

# base64 인코딩
with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

# 프롬프트
prompt = """이 테이블 이미지의 모든 텍스트를 JSON 2D 배열로 추출하세요.

규칙:
1. 각 행은 배열의 요소
2. 각 셀은 행 배열의 요소
3. 빈 셀은 빈 문자열 ""
4. 병합된 셀은 첫 번째 위치에만 값, 나머지는 ""
5. JSON만 출력, 다른 설명 없음

출력 형식:
[["헤더1","헤더2","헤더3"],["값1","값2","값3"],["값4","값5","값6"]]"""

print("\n=== VLM 호출 중 (최대 180초) ===")
try:
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
    print(raw[:3000])
    
    print("\n=== JSON 파싱 시도 ===")
    # 방법 1
    try:
        t = json.loads(raw)
        print(f"직접 파싱 성공! 행수: {len(t)}, 첫행: {t[0][:3] if t else 'N/A'}...")
    except:
        # 방법 2
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1:
            try:
                t = json.loads(raw[start:end+1])
                print(f"추출 파싱 성공! 행수: {len(t)}")
            except Exception as e:
                print(f"파싱 실패: {e}")
        else:
            print("JSON 구조 없음")

except Exception as e:
    print(f"오류: {e}")
