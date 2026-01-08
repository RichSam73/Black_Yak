# -*- coding: utf-8 -*-
"""
Vision 모델 비교 테스트
"""
import requests
import base64
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

# 테스트할 모델들
MODELS = [
    "deepseek-ocr:latest",
    "qwen2.5vl:latest",
    "llama3.2-vision:latest",
    "granite3.2-vision:2b",
    "moondream:latest",
]

# 이미지 경로
IMG_PATH = r"E:\Antigravity\Black_Yak\.playwright-mcp\ai_vision_erp_table.png"

def test_model(model_name, img_base64):
    prompt = """Extract ALL text from this table as JSON 2D array.
Output ONLY JSON, no explanation.
Format: [["header1","header2"],["value1","value2"]]
Empty cells: ""
"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    start = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"num_predict": 4096, "temperature": 0}
            },
            timeout=300
        )
        elapsed = time.time() - start
        
        result = response.json()
        raw = result.get('response', '').strip()
        
        print(f"Time: {elapsed:.1f}s")
        print(f"Response length: {len(raw)} chars")
        print(f"\n--- Response (first 1500 chars) ---")
        print(raw[:1500])
        
        return {
            "model": model_name,
            "time": elapsed,
            "length": len(raw),
            "response": raw
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"model": model_name, "error": str(e)}


if __name__ == "__main__":
    # 이미지 로드
    print("Loading image...")
    with open(IMG_PATH, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()
    print(f"Image size: {len(img_base64)} bytes (base64)")
    
    results = []
    for model in MODELS:
        result = test_model(model, img_base64)
        results.append(result)
        print("\n")
    
    # 결과 요약
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        if 'error' in r:
            print(f"{r['model']}: ERROR - {r['error']}")
        else:
            print(f"{r['model']}: {r['time']:.1f}s, {r['length']} chars")
