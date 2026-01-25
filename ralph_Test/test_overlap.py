# -*- coding: utf-8 -*-
"""
글자 겹침 테스트 스크립트 - 숲프린트/가슴로고 전용
- 성공 기준: 숲프린트와 가슴로고 번역 텍스트가 겹치지 않음
"""

import os
import sys
import json
import re
import logging

# 테스트 설정
LOG_FILE = os.path.join(os.path.dirname(__file__), 'overlap_debug.log')
TARGET_TEXTS = ['숲프린트', '가슴로고', '숲 프린트', '가슴 로고']

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('test_overlap')


def parse_overlap_log(log_path):
    """
    overlap_debug.log 파싱하여 겹침 정보 추출

    Returns:
        dict: {
            'total_overlaps': int,
            'target_overlaps': list,  # 숲프린트/가슴로고 관련 겹침
            'all_overlaps': list
        }
    """
    if not os.path.exists(log_path):
        return {'error': f'Log file not found: {log_path}'}

    target_overlaps = []
    all_overlaps = []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # INVADES 패턴 찾기
        if 'INVADES' in line and 'y_overlap' not in line:
            # 다음 줄들에서 y_overlap=True 확인
            for j in range(i+1, min(i+5, len(lines))):
                if 'y_overlap=True' in lines[j]:
                    # 실제 겹침 발생
                    match = re.search(r"#(\d+) INVADES #(\d+) '([^']+)'", line)
                    if match:
                        invader_idx = match.group(1)
                        victim_idx = match.group(2)
                        victim_text = match.group(3)

                        overlap_info = {
                            'invader_idx': invader_idx,
                            'victim_idx': victim_idx,
                            'victim_text': victim_text,
                            'line': line.strip()
                        }
                        all_overlaps.append(overlap_info)

                        # 타겟 텍스트 관련 겹침인지 확인
                        for target in TARGET_TEXTS:
                            if target in line:
                                target_overlaps.append(overlap_info)
                                break
                    break
        i += 1

    return {
        'total_overlaps': len(all_overlaps),
        'target_overlaps': target_overlaps,
        'target_overlap_count': len(target_overlaps),
        'all_overlaps': all_overlaps[:20]  # 처음 20개만
    }


def check_specific_overlap(log_path):
    """
    숲프린트 → 가슴로고 겹침 여부 확인

    Returns:
        dict: 겹침 상세 정보
    """
    if not os.path.exists(log_path):
        return {'error': f'Log file not found: {log_path}'}

    overlaps = []

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 숲프린트가 가슴로고를 침범하는 패턴 찾기
    pattern = r"→ #\d+ INVADES #\d+ '가슴로고'"
    matches = re.findall(pattern, content)

    # 상세 정보 추출
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '가슴로고' in line and 'INVADES' in line:
            # 다음 줄에서 y_overlap 확인
            for j in range(i+1, min(i+5, len(lines))):
                if 'y_overlap=True' in lines[j]:
                    overlaps.append({
                        'line_num': i,
                        'content': line.strip(),
                        'y_overlap': True
                    })
                    break

    return {
        'chest_logo_invaded': len(overlaps) > 0,
        'invasion_count': len(overlaps),
        'details': overlaps[:10]
    }


def run_test():
    """
    테스트 실행

    성공 기준:
    - 숲프린트와 가슴로고 사이에 y_overlap=True가 0개

    Returns:
        dict: 테스트 결과
    """
    logger.info("=" * 60)
    logger.info("글자 겹침 테스트 - 숲프린트/가슴로고")
    logger.info("=" * 60)

    # 1. 로그 파일 확인
    if not os.path.exists(LOG_FILE):
        logger.error(f"로그 파일 없음: {LOG_FILE}")
        logger.info("앱 실행 후 번역을 수행해야 로그가 생성됩니다.")
        return {
            'status': 'ERROR',
            'message': 'Log file not found. Run translation first.',
            'log_file': LOG_FILE
        }

    logger.info(f"로그 파일: {LOG_FILE}")

    # 2. 전체 겹침 분석
    logger.info("\n[1단계] 전체 겹침 분석...")
    result = parse_overlap_log(LOG_FILE)

    if 'error' in result:
        return {'status': 'ERROR', 'message': result['error']}

    logger.info(f"  - 전체 겹침: {result['total_overlaps']}개")
    logger.info(f"  - 타겟 관련 겹침: {result['target_overlap_count']}개")

    # 3. 숲프린트→가슴로고 겹침 확인
    logger.info("\n[2단계] 숲프린트→가슴로고 겹침 확인...")
    specific = check_specific_overlap(LOG_FILE)

    if 'error' in specific:
        return {'status': 'ERROR', 'message': specific['error']}

    logger.info(f"  - 가슴로고 침범 여부: {specific['chest_logo_invaded']}")
    logger.info(f"  - 침범 횟수: {specific['invasion_count']}")

    if specific['details']:
        logger.info("\n[상세 겹침 정보]")
        for d in specific['details'][:5]:
            logger.info(f"  {d['content'][:80]}")

    # 4. 판정
    logger.info("\n" + "-" * 60)

    if specific['invasion_count'] == 0:
        logger.info("✅ TEST PASSED - 숲프린트/가슴로고 겹침 없음")
        status = 'PASS'
    else:
        logger.info(f"❌ TEST FAILED - {specific['invasion_count']}개 겹침 발생")
        status = 'FAIL'

    logger.info("=" * 60)

    return {
        'status': status,
        'target_overlap_count': specific['invasion_count'],
        'total_overlap_count': result['total_overlaps'],
        'chest_logo_invaded': specific['chest_logo_invaded'],
        'details': specific['details'][:5]
    }


if __name__ == "__main__":
    result = run_test()

    # JSON 결과 출력 (ralph-loop 연동용)
    print("\n[TEST_RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 종료 코드
    sys.exit(0 if result['status'] == 'PASS' else 1)
