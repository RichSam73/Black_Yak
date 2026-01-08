# -*- coding: utf-8 -*-
"""실제 테이블 이미지로 VLM OCR 테스트"""
import requests
import base64
import json
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:latest"

# 테스트할 이미지 경로 (CometApp 폴더의 테스트 이미지)
test_images = [
    r"E:\Antigravity\Black_Yak\Sample\005M_Table.png",
    r"E:\Antigravity\Black_Yak\Sample\Submaterial_information.png",
]

# 존재하는 첫 번째 이미지 사용
img_path = None
for p in test_images:
    if os.path.exists(p):
        img_path = p
        break

if not img_path:
    # Sample 폴더에서 png 파일 찾기
    sample_dir = r"E:\Antigravity\Black_Yak\Sample"
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith('.png') or f.endswith('.jpg'):
                img_path = os.path.join(sample_dir, f)
                break

if not img_path:
    print("테스트 이미지를 찾을 수 없습니다.")
    exit(1)

print(f"=== 테스트 이미지: {img_path} ===")
print(f"파일 크기: {os.path.getsize(img_path) / 1024:.1f} KB")

# 이미지를 base64로 인코딩
with open(img_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(f"Base64 길이: {len(img_base64)}")

# JSON 프롬프트
prompt = """이 테이블 이미지의 모든 텍스트를 JSON 2D 배열로 추출하세요.

규칙:
1. 각 행은 배열의 요소
2. 각 셀은 행 배열의 요소
3. 빈 셀은 빈 문자열 ""
4. JSON만 출력, 다른 설명 없음

출력 형식 예시:
[["헤더1","헤더2"],["값1","값2"]]"""

print("\n=== VLM OCR 요청 중... ===")
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
    
    if response.status_code == 200:
        result = response.json()
        raw_text = result.get('response', '').strip()
        
        print(f"\n=== VLM 응답 (길이: {len(raw_text)}) ===")
        print(raw_text[:2000])
        if len(raw_text) > 2000:
            print(f"\n... (총 {len(raw_text)} 문자)")
        
        # JSON 파싱 시도
        print("\n=== JSON 파싱 시도 ===")
        try:
            # 방법 1: 직접 파싱
            table = json.loads(raw_text)
            print(f"직접 파싱 성공! 행 수: {len(table)}")
            if table:
                print(f"첫 행: {table[0][:5]}..." if len(table[0]) > 5 else f"첫 행: {table[0]}")
        except:
            print("직접 파싱 실패")
            
            # 방법 2: [ ] 추출
            start = raw_text.find('[')
            end = raw_text.rfind(']')
            if start != -1 and end != -1:
                json_str = raw_text[start:end+1]
                try:
                    table = json.loads(json_str)
                    print(f"추출 파싱 성공! 행 수: {len(table)}")
                except Exception as e:
                    print(f"추출 파싱 실패: {e}")
                    print(f"추출된 문자열: {json_str[:500]}...")
    else:
        print(f"API 오류: {response.text}")
        
except Exception as e:
    print(f"오류: {e}")
