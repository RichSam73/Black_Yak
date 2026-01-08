# -*- coding: utf-8 -*-
"""
PaddleX TableRecognitionV2 간단 테스트
단일 이미지로 테스트
"""

import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

print("1. PaddleX import 시작...")
from paddlex import create_pipeline
print("2. PaddleX import 완료")

print("3. 파이프라인 생성 시작...")
pipeline = create_pipeline(pipeline="table_recognition_v2")
print("4. 파이프라인 생성 완료")

# 테스트 이미지
img_path = r"e:\Antigravity\Black_Yak\Reference\BY_Original_Table.png"
print(f"5. 이미지 처리 시작: {img_path}")

if os.path.exists(img_path):
    output = pipeline.predict(img_path)
    print("6. 예측 완료")

    # Generator를 리스트로 변환
    results = list(output)
    print(f"7. 결과 개수: {len(results)}")

    for i, res in enumerate(results):
        print(f"\n=== 결과 {i+1} ===")
        print(f"타입: {type(res)}")

        # 주요 속성 확인
        if hasattr(res, 'html'):
            html = res.html
            print(f"html 길이: {len(html)}")
            print(f"colspan 포함: {'colspan' in html}")
            print(f"rowspan 포함: {'rowspan' in html}")
            print(f"\n미리보기:\n{html[:1500]}")

            # 저장
            with open("poc_result.html", "w", encoding="utf-8") as f:
                f.write(html)
            print("\n저장됨: poc_result.html")
        else:
            # 가능한 속성 탐색
            attrs = [a for a in dir(res) if not a.startswith('_')]
            print(f"속성들: {attrs}")

            for attr in ['table_html', 'pred_html', 'tables', 'result']:
                if hasattr(res, attr):
                    val = getattr(res, attr)
                    print(f"{attr}: {type(val)} - {str(val)[:200]}")
else:
    print(f"파일 없음: {img_path}")
