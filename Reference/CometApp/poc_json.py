# -*- coding: utf-8 -*-
"""
PaddleX TableRecognitionV2 JSON 결과 분석
"""

import os
import json as json_lib
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

    res = results[0]

    # JSON 결과 상세 분석
    print("\n" + "="*60)
    print("JSON 결과 분석")
    print("="*60)

    if hasattr(res, 'json'):
        json_data = res.json
        print(f"json 타입: {type(json_data)}")

        if isinstance(json_data, dict):
            # 최상위 키
            print(f"\n최상위 키: {list(json_data.keys())}")

            # res 키 분석
            if 'res' in json_data:
                res_data = json_data['res']
                print(f"\nres 키: {list(res_data.keys()) if isinstance(res_data, dict) else type(res_data)}")

                # 테이블 관련 키 찾기
                for key in res_data:
                    val = res_data[key]
                    if 'table' in key.lower() or 'html' in key.lower():
                        print(f"\n[테이블 관련] {key}:")
                        print(f"  타입: {type(val)}")
                        if isinstance(val, str):
                            print(f"  길이: {len(val)}")
                            print(f"  내용: {val[:500]}...")
                        elif isinstance(val, list):
                            print(f"  길이: {len(val)}")
                            if len(val) > 0:
                                print(f"  첫번째: {val[0]}")
                        elif isinstance(val, dict):
                            print(f"  키: {list(val.keys())}")

                # layout_parsing_result 확인
                if 'layout_parsing_result' in res_data:
                    layout = res_data['layout_parsing_result']
                    print(f"\n[layout_parsing_result]:")
                    print(f"  타입: {type(layout)}")
                    if isinstance(layout, list):
                        print(f"  길이: {len(layout)}")
                        for i, item in enumerate(layout[:3]):  # 처음 3개만
                            print(f"\n  항목 {i}:")
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    v_str = str(v)
                                    if len(v_str) > 100:
                                        v_str = v_str[:100] + "..."
                                    print(f"    {k}: {v_str}")

        # 전체 JSON 파일로 저장
        with open("poc_result.json", "w", encoding="utf-8") as f:
            json_lib.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        print("\n저장됨: poc_result.json")

else:
    print(f"파일 없음: {img_path}")
