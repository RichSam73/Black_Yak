# -*- coding: utf-8 -*-
"""
PaddleX TableRecognitionV2 PoC 테스트
Round 22: colspan/rowspan 출력 확인

목표: PaddleX V2가 테이블 구조(셀 병합)를 정확히 추출하는지 검증
"""

import os
import sys

# 테스트 이미지 경로 (Reference 폴더에 위치)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES = [
    os.path.join(BASE_DIR, "BY_Original_Table.png"),
    os.path.join(BASE_DIR, "005M_Table.png"),
    os.path.join(BASE_DIR, "Submaterial_information.png")
]


def test_paddlex_v2():
    """PaddleX TableRecognitionV2 파이프라인 테스트"""
    print("=" * 60)
    print("PaddleX TableRecognitionV2 PoC 테스트")
    print("=" * 60)

    # PaddleX import 시도
    try:
        from paddlex import create_pipeline
        print("[OK] paddlex 모듈 import 성공")
    except ImportError as e:
        print(f"[ERROR] paddlex import 실패: {e}")
        print("\n설치 방법:")
        print("  pip install paddlex")
        print("  또는")
        print("  pip install paddlepaddle paddlex")
        return False

    # 파이프라인 생성
    print("\n[INFO] table_recognition_v2 파이프라인 생성 중...")
    try:
        pipeline = create_pipeline(pipeline="table_recognition_v2")
        print("[OK] 파이프라인 생성 성공")
    except Exception as e:
        print(f"[ERROR] 파이프라인 생성 실패: {e}")

        # 대안: table_recognition (V1) 시도
        print("\n[INFO] table_recognition (V1) 파이프라인 시도...")
        try:
            pipeline = create_pipeline(pipeline="table_recognition")
            print("[OK] V1 파이프라인 생성 성공")
        except Exception as e2:
            print(f"[ERROR] V1 파이프라인도 실패: {e2}")
            return False

    # 테스트 이미지 처리
    results = {}
    for img_path in TEST_IMAGES:
        img_name = os.path.basename(img_path)
        print(f"\n{'=' * 60}")
        print(f"테스트: {img_name}")
        print("=" * 60)

        if not os.path.exists(img_path):
            print(f"[SKIP] 파일 없음: {img_path}")
            results[img_name] = {"status": "file_not_found"}
            continue

        try:
            # 예측 실행
            print(f"[INFO] 이미지 처리 중...")
            output = pipeline.predict(img_path)

            # 출력 구조 분석
            print(f"[DEBUG] output 타입: {type(output)}")

            # Generator인 경우 리스트로 변환
            if hasattr(output, '__iter__') and not isinstance(output, (str, dict)):
                output_list = list(output)
                print(f"[DEBUG] 결과 개수: {len(output_list)}")

                for i, item in enumerate(output_list):
                    print(f"\n[결과 {i+1}]")
                    analyze_output(item, img_name, results)
            else:
                analyze_output(output, img_name, results)

        except Exception as e:
            print(f"[ERROR] 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            results[img_name] = {"status": "error", "message": str(e)}

    # 결과 요약
    print_summary(results)
    return results


def analyze_output(output, img_name, results):
    """출력 결과 분석"""
    result = {"status": "processed"}

    # 속성 목록 확인
    attrs = [a for a in dir(output) if not a.startswith('_')]
    print(f"[DEBUG] 속성 목록: {attrs[:20]}...")  # 처음 20개만 표시

    # pred_html 확인
    if hasattr(output, 'pred_html'):
        html = output.pred_html
        print(f"\n[pred_html 발견]")
        print(f"  - 길이: {len(html)} 문자")

        # colspan/rowspan 포함 여부
        has_colspan = 'colspan' in html.lower()
        has_rowspan = 'rowspan' in html.lower()
        print(f"  - colspan 포함: {has_colspan}")
        print(f"  - rowspan 포함: {has_rowspan}")

        # 처음 1000자 미리보기
        print(f"\n[HTML 미리보기 (처음 1000자)]:")
        print("-" * 40)
        print(html[:1000])
        print("-" * 40)

        result["has_colspan"] = has_colspan
        result["has_rowspan"] = has_rowspan
        result["html_length"] = len(html)

        # HTML 파일로 저장
        output_path = os.path.join(os.path.dirname(__file__), f"poc_result_{img_name}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"<html><head><meta charset='utf-8'><title>{img_name}</title></head><body>")
            f.write(html)
            f.write("</body></html>")
        print(f"\n[저장됨] {output_path}")

    elif hasattr(output, 'html'):
        # 다른 속성명으로 HTML 반환하는 경우
        html = output.html
        print(f"\n[html 속성 발견]")
        print(f"  - 길이: {len(html)} 문자")
        result["html_attr"] = "html"
        result["html_length"] = len(html)

    else:
        print("\n[WARNING] pred_html 속성 없음")

        # 대안 속성 탐색
        for attr in ['tables', 'table_html', 'result', 'output', 'boxes']:
            if hasattr(output, attr):
                val = getattr(output, attr)
                print(f"  - {attr}: {type(val)}")
                if isinstance(val, str) and len(val) < 500:
                    print(f"    값: {val}")
                elif isinstance(val, (list, dict)):
                    print(f"    길이/키: {len(val) if isinstance(val, list) else list(val.keys())}")

    results[img_name] = result


def print_summary(results):
    """결과 요약 출력"""
    print("\n")
    print("=" * 60)
    print("PoC 테스트 결과 요약")
    print("=" * 60)

    success_count = 0
    colspan_count = 0
    rowspan_count = 0

    for img_name, result in results.items():
        status = result.get("status", "unknown")
        has_colspan = result.get("has_colspan", False)
        has_rowspan = result.get("has_rowspan", False)

        print(f"\n{img_name}:")
        print(f"  - 상태: {status}")

        if status == "processed":
            success_count += 1
            if has_colspan:
                colspan_count += 1
                print(f"  - colspan: ✅ 포함")
            else:
                print(f"  - colspan: ❌ 없음")

            if has_rowspan:
                rowspan_count += 1
                print(f"  - rowspan: ✅ 포함")
            else:
                print(f"  - rowspan: ❌ 없음")

    print("\n" + "-" * 40)
    print(f"총 테스트: {len(results)}개")
    print(f"성공: {success_count}개")
    print(f"colspan 지원: {colspan_count}개")
    print(f"rowspan 지원: {rowspan_count}개")

    if colspan_count > 0 or rowspan_count > 0:
        print("\n🎉 PoC 성공: PaddleX V2가 셀 병합을 지원합니다!")
        print("   → Round 23에서 Adapter 함수 구현 진행")
    else:
        print("\n⚠️ PoC 결과: 셀 병합 정보 없음")
        print("   → 대안 검토 필요 (Option A 또는 현행 유지)")


if __name__ == "__main__":
    test_paddlex_v2()
