# -*- coding: utf-8 -*-
"""
PaddleX TableRecognitionV2 디버그 테스트
결과 객체 구조 상세 분석
"""

import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

print("1. PaddleX import...")
from paddlex import create_pipeline
print("2. import 완료")

print("3. 파이프라인 생성...")
pipeline = create_pipeline(pipeline="table_recognition_v2")
print("4. 파이프라인 생성 완료")

# 테스트 이미지
img_path = r"e:\Antigravity\Black_Yak\Reference\BY_Original_Table.png"
print(f"5. 이미지: {img_path}")

if os.path.exists(img_path):
    output = pipeline.predict(img_path)
    print("6. 예측 완료")

    results = list(output)
    print(f"7. 결과 개수: {len(results)}")

    for i, res in enumerate(results):
        print(f"\n{'='*60}")
        print(f"결과 {i+1}")
        print("="*60)
        print(f"타입: {type(res)}")

        # 모든 public 속성 확인
        attrs = [a for a in dir(res) if not a.startswith('_')]
        print(f"\n속성 목록 ({len(attrs)}개):")
        for attr in attrs:
            try:
                val = getattr(res, attr)
                if callable(val):
                    continue  # 메서드 스킵
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  {attr}: {type(val).__name__} = {val_str}")
            except Exception as e:
                print(f"  {attr}: ERROR - {e}")

        # 특별 속성들 상세 확인
        print("\n--- 상세 확인 ---")

        # html
        if hasattr(res, 'html'):
            html = res.html
            print(f"html: 타입={type(html)}, 길이/값={len(html) if hasattr(html, '__len__') else html}")
            if html:
                print(f"  내용: {html[:500] if len(html) > 500 else html}")

        # table_res (테이블 결과)
        if hasattr(res, 'table_res'):
            print(f"\ntable_res: {type(res.table_res)}")
            if res.table_res:
                print(f"  길이: {len(res.table_res)}")
                for j, tbl in enumerate(res.table_res):
                    print(f"  테이블 {j}: {type(tbl)}")
                    if hasattr(tbl, 'html'):
                        print(f"    html: {tbl.html[:300] if tbl.html else 'None'}...")

        # layout (레이아웃 분석 결과)
        if hasattr(res, 'layout'):
            print(f"\nlayout: {type(res.layout)}")
            if res.layout:
                print(f"  내용: {str(res.layout)[:500]}")

        # boxes
        if hasattr(res, 'boxes'):
            print(f"\nboxes: {type(res.boxes)}")
            if res.boxes is not None and len(res.boxes) > 0:
                print(f"  개수: {len(res.boxes)}")
                print(f"  첫번째: {res.boxes[0] if len(res.boxes) > 0 else 'N/A'}")

        # img
        if hasattr(res, 'img'):
            print(f"\nimg: {type(res.img)}")
            if res.img is not None:
                print(f"  shape: {res.img.shape if hasattr(res.img, 'shape') else 'N/A'}")

else:
    print(f"파일 없음: {img_path}")
