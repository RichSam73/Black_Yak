"""격자선 기반 테이블 추출 테스트"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'e:/Antigravity/Black_Yak')

from smart_table_extractor import (
    detect_grid_lines, map_ocr_to_grid_cells,
    extract_comet_with_table_detection,
    PADDLEOCR_AVAILABLE, TABLE_TRANSFORMER_AVAILABLE
)
from PIL import Image
import fitz

print('=' * 60)
print('격자선 기반 테이블 추출 테스트')
print('=' * 60)
print(f'PaddleOCR: {PADDLEOCR_AVAILABLE}')
print(f'Table Transformer: {TABLE_TRANSFORMER_AVAILABLE}')

# PDF 로드
pdf_path = 'e:/Antigravity/Black_Yak/제로스팟 다운자켓#1 오더 등록 작지 1BYPAWU005-M-1.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

def progress(page, total, msg):
    print(f'  [{page}/{total}] {msg}')

# 추출 실행
print('\n테이블 추출 중...')
result = extract_comet_with_table_detection(pdf_bytes, progress_callback=progress)

print(f'\n결과:')
print(f'  총 테이블 수: {result["total_tables"]}')

for table in result['tables'][:3]:  # 처음 3개만
    print(f'\n  페이지 {table["page"]}, 테이블 {table["table_index"]}:')
    print(f'    테이블명: {table.get("table_name", "")}')
    print(f'    추출 방식: {table.get("grid_detection", "N/A")}')
    grid_info = table.get('grid_info', {})
    print(f'    격자선 감지: {grid_info.get("horizontal_lines", 0)}개 수평선, {grid_info.get("vertical_lines", 0)}개 수직선')
    print(f'    크기: {table["row_count"]} 행 x {table["col_count"]} 열')
    print(f'    샘플 데이터 (처음 5행):')
    for row in table['data'][:5]:
        print(f'      {row}')
