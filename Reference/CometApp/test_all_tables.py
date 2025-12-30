#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3개 테이블 자동 테스트 스크립트
- BY_Original_Table.png
- 005M_Table.png (BK/CM/TE)
- Submaterial_information.png
"""
import requests
import os
import json
import sys

BASE_DIR = r"E:\Antigravity\Black_Yak\Reference"
SERVER_URL = "http://localhost:6002/upload"

TEST_FILES = [
    ("BY_Original_Table.png", "원본 테이블 (125/130 컬럼 포함)"),
    ("005M_Table.png", "BK/CM/TE 테이블 (115/120 컬럼)"),
    ("Submaterial_information.png", "부자재 정보 테이블"),
]

def test_table(filename, description):
    """테이블 이미지를 서버에 업로드하고 결과 확인"""
    filepath = os.path.join(BASE_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[ERROR] 파일 없음: {filepath}")
        return None

    print(f"\n{'='*60}")
    print(f"테스트: {description}")
    print(f"파일: {filename}")
    print(f"{'='*60}")

    try:
        with open(filepath, 'rb') as f:
            files = {'image': (filename, f, 'image/png')}
            response = requests.post(SERVER_URL, files=files, timeout=120)

        print(f"[DEBUG] HTTP Status: {response.status_code}")
        print(f"[DEBUG] Response Keys: {list(response.json().keys()) if response.status_code == 200 else 'N/A'}")

        if response.status_code == 200:
            result = response.json()

            # 전체 응답 구조 출력
            print(f"[DEBUG] 응답 키: {list(result.keys())}")
            for key in result.keys():
                val = result[key]
                if isinstance(val, str):
                    print(f"[DEBUG] {key}: {len(val)} chars")
                else:
                    print(f"[DEBUG] {key}: {type(val)}")

            # HTML 결과에서 테이블 추출하여 분석
            html = result.get('html', '')

            # erp_table_html 확인 (실제 키 이름)
            erp_table = result.get('erp_table_html', '')

            # 간단한 분석: 행과 열 수 확인
            row_count = html.count('<tr')

            print(f"[결과] 상태: 성공")
            print(f"[결과] HTML 행 수: {row_count}")
            print(f"[결과] ERP 테이블 길이: {len(erp_table)} chars")

            # 헤더 행 분석
            if '<thead>' in html:
                thead_start = html.find('<thead>')
                thead_end = html.find('</thead>')
                thead = html[thead_start:thead_end]
                th_count = thead.count('<th')
                print(f"[결과] 헤더 열 수: {th_count}")

            # 테이블 내용 출력 (간략히)
            print(f"\n[HTML 미리보기]")
            # HTML을 파일로 저장
            result_file = os.path.join(BASE_DIR, "CometApp", f"test_result_{filename.replace('.png', '.html')}")

            # erp_table이 있으면 그것을 저장
            content_to_save = erp_table if erp_table else html
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"<html><head><meta charset='utf-8'><style>table{{border-collapse:collapse;}} td,th{{border:1px solid #ccc;padding:5px;}}</style></head><body>{content_to_save}</body></html>")
            print(f"[저장됨] {result_file}")

            # ERP 테이블 일부 출력
            if erp_table:
                print(f"\n[ERP 테이블 미리보기]")
                print(erp_table[:500] + "..." if len(erp_table) > 500 else erp_table)

            return result
        else:
            print(f"[ERROR] HTTP {response.status_code}: {response.text}")
            return None

    except Exception as e:
        import traceback
        print(f"[ERROR] 요청 실패: {e}")
        traceback.print_exc()
        return None

def main():
    print("="*60)
    print("3개 테이블 자동 테스트 시작")
    print("="*60)

    results = {}
    for filename, description in TEST_FILES:
        result = test_table(filename, description)
        results[filename] = result

    print("\n" + "="*60)
    print("테스트 요약")
    print("="*60)

    for filename, result in results.items():
        status = "성공" if result else "실패"
        print(f"  - {filename}: {status}")

    return results

if __name__ == "__main__":
    main()
