"""격자선 기반 테이블 추출 상세 테스트"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'e:/Antigravity/Black_Yak')

from smart_table_extractor import extract_comet_with_table_detection

# PDF 로드
pdf_path = 'e:/Antigravity/Black_Yak/제로스팟 다운자켓#1 오더 등록 작지 1BYPAWU005-M-1.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

def progress(page, total, msg):
    pass  # 진행상황 출력 생략

# 추출 실행
result = extract_comet_with_table_detection(pdf_bytes, progress_callback=progress)

print('=' * 80)
print('페이지 1 - COLOR/SIZE QTY 테이블 상세 분석')
print('=' * 80)

# 테이블 2 찾기 (COLOR/SIZE QTY)
for table in result['tables']:
    if table['page'] == 1 and 'COLOR' in table.get('table_name', ''):
        print(f'\n테이블명: {table.get("table_name", "")}')
        print(f'추출 방식: {table.get("grid_detection", "N/A")}')
        grid_info = table.get('grid_info', {})
        print(f'감지된 격자: {grid_info.get("horizontal_lines", 0)}개 수평선, {grid_info.get("vertical_lines", 0)}개 수직선')
        print(f'테이블 크기: {table["row_count"]} 행 x {table["col_count"]} 열')

        print('\n전체 데이터:')
        for i, row in enumerate(table['data']):
            # 빈 셀은 [빈]으로 표시
            formatted_row = [cell if cell.strip() else '[빈]' for cell in row]
            print(f'  행{i+1}: {formatted_row}')
        break

# 페이지 1의 모든 테이블 요약
print('\n' + '=' * 80)
print('페이지 1 전체 테이블 요약')
print('=' * 80)

for table in result['tables']:
    if table['page'] == 1:
        name = table.get('table_name', '(이름없음)')[:30]
        method = table.get('grid_detection', 'N/A')
        grid = table.get('grid_info', {})
        h_lines = grid.get('horizontal_lines', 0)
        v_lines = grid.get('vertical_lines', 0)
        rows = table['row_count']
        cols = table['col_count']

        print(f'  테이블 {table["table_index"]}: {name}')
        print(f'    방식: {method}, 격자: {h_lines}H x {v_lines}V, 크기: {rows}행 x {cols}열')
