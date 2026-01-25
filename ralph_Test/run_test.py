# -*- coding: utf-8 -*-
"""
자동화 테스트 스크립트 - ralph-loop용
1. API로 번역 실행
2. 로그 확인
3. 겹침 테스트 실행
"""

import os
import sys
import time
import json
import requests

BASE_DIR = os.path.dirname(__file__)
APP_URL = "http://localhost:7000"
TEST_PDF = os.path.join(BASE_DIR, "test_input.pdf")
LOG_FILE = os.path.join(BASE_DIR, "overlap_debug.log")


def clear_log():
    """기존 로그 삭제 (실패해도 계속 진행)"""
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            print("[1] 기존 로그 삭제됨")
        else:
            print("[1] 기존 로그 없음")
    except PermissionError:
        print("[1] 로그 파일 사용 중 - 삭제 건너뜀 (기존 로그 사용)")


def run_translation():
    """API로 번역 실행"""
    print("[2] 번역 API 호출 중...")

    try:
        with open(TEST_PDF, 'rb') as f:
            files = {'file': ('test_input.pdf', f, 'application/pdf')}
            data = {'target_lang': 'english'}

            response = requests.post(
                f"{APP_URL}/translate",
                files=files,
                data=data,
                timeout=300  # 5분 타임아웃
            )

        result = response.json()

        if result.get('success'):
            print(f"    번역 성공: {result.get('files', [])}")
            return True
        else:
            print(f"    번역 실패: {result.get('error', 'Unknown error')}")
            return False

    except requests.exceptions.ConnectionError:
        print("    연결 실패: 앱이 실행 중인지 확인하세요")
        return False
    except Exception as e:
        print(f"    오류: {e}")
        return False


def wait_for_log(timeout=60):
    """로그 파일 생성 대기"""
    print(f"[3] 로그 파일 확인 중...")

    # 이미 존재하면 바로 진행
    if os.path.exists(LOG_FILE):
        print(f"    로그 파일 존재: {LOG_FILE}")
        time.sleep(2)  # 쓰기 완료 대기
        return True

    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(LOG_FILE):
            time.sleep(2)
            print(f"    로그 파일 생성됨: {LOG_FILE}")
            return True
        time.sleep(1)

    print("    로그 파일 생성 타임아웃")
    return False


def run_overlap_test():
    """겹침 테스트 실행"""
    print("[4] 겹침 테스트 실행 중...")

    # test_overlap.py import
    sys.path.insert(0, BASE_DIR)
    from test_overlap import run_test

    result = run_test()
    return result


def main():
    print("=" * 60)
    print("자동화 테스트 시작")
    print("=" * 60)

    # 1. 로그 초기화
    clear_log()

    # 2. 번역 실행
    if not run_translation():
        print("\n❌ 번역 실패 - 테스트 중단")
        return {'status': 'ERROR', 'message': 'Translation failed'}

    # 3. 로그 대기
    if not wait_for_log():
        print("\n❌ 로그 생성 실패 - 테스트 중단")
        return {'status': 'ERROR', 'message': 'Log not generated'}

    # 4. 겹침 테스트
    result = run_overlap_test()

    print("\n" + "=" * 60)
    print("[TEST_RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get('status') == 'PASS' else 1)
