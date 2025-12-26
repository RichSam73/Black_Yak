"""
테이블 이미지를 HTML로 변환하는 스크립트
원본과 완전히 동일한 테이블 생성
"""

def generate_color_size_qty_table():
    """COLOR/SIZE QTY 테이블을 HTML로 생성"""

    # 데이터 정의
    headers = ["095", "100", "105", "110", "115", "120", "125", "130"]

    data = [
        {"code": "BK", "name": "BLACK", "values": ["2,200", "5,000", "5,000", "2,200", "800", "500", "150", "150"], "total": "16,000"},
        {"code": "NA", "name": "NAVY", "values": ["1,100", "2,300", "1,650", "500", "200", "150", "50", "50"], "total": "6,000"},
        {"code": "D3", "name": "D/TAUPE GRAY", "values": ["650", "1,200", "800", "400", "150", "100", "50", "50"], "total": "3,400"},
        {"code": "SV", "name": "SILVER BEIGE", "values": ["900", "1,550", "950", "300", "150", "50", "50", "50"], "total": "4,000"},
    ]

    totals = ["4,850", "10,050", "8,400", "3,400", "1,300", "800", "300", "300"]
    grand_total = "29,400"

    html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>COLOR/SIZE QTY</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background: #fff;
        }
        table {
            border-collapse: collapse;
            font-size: 13px;
        }
        td, th {
            border: 1px solid #000;
            padding: 6px 12px;
            text-align: center;
            height: 28px;
        }
        /* 1행: COLOR/SIZE QTY 헤더 */
        .header-black {
            background: #000;
            color: white;
            font-weight: bold;
            font-size: 13px;
        }
        .header-gray-dark {
            background: #666;
            border: 1px solid #000;
        }
        .header-gray-light {
            background: #999;
            border: 1px solid #000;
        }
        /* 2행: 서브 헤더 */
        .sub-header {
            background: #fff;
            font-weight: bold;
        }
        /* 컬러 코드/이름 (회색 배경) */
        .color-code {
            background: #e0e0e0;
            font-weight: bold;
        }
        .color-name {
            background: #e0e0e0;
            font-weight: bold;
        }
        /* 데이터 셀 */
        .data-cell {
            background: #fff;
            text-align: right;
        }
        /* 빈 열 - 테두리 없음 */
        .empty-col {
            background: #fff;
            border-left: none;
            border-right: none;
            width: 40px;
        }
        .empty-col-header {
            background: #fff;
            border-left: none;
            border-right: none;
            width: 40px;
        }
        /* TOTAL 열 */
        .total-col {
            background: #fff;
            font-weight: bold;
            text-align: right;
        }
        /* 빈 행 */
        .empty-row td {
            background: #fff;
            height: 28px;
        }
        /* TOTAL 행 */
        .total-row td {
            background: #c0c0c0;
            font-weight: bold;
        }
        .total-row .data-cell {
            background: #c0c0c0;
            text-align: right;
        }
        .total-row .empty-col {
            background: #c0c0c0;
        }
        .total-row .total-col {
            background: #c0c0c0;
        }
        /* 두꺼운 테두리 */
        .border-top-thick {
            border-top: 2px solid #000;
        }
        .border-bottom-thick {
            border-bottom: 2px solid #000;
        }
        .border-left-thick {
            border-left: 2px solid #000;
        }
        .border-right-thick {
            border-right: 2px solid #000;
        }
    </style>
</head>
<body>
    <table>
        <!-- 1행: COLOR/SIZE QTY 헤더 -->
        <tr>
            <td class="header-black border-left-thick border-top-thick" colspan="2"></td>
            <td class="header-black border-top-thick" colspan="8">COLOR/SIZE QTY</td>
            <td class="header-gray-dark border-top-thick"></td>
            <td class="header-gray-light border-top-thick border-right-thick"></td>
        </tr>
        <!-- 2행: 서브 헤더 (COLOR / SIZE, 095~130, 빈열, TOTAL) -->
        <tr>
            <td class="sub-header border-left-thick border-bottom-thick" colspan="2">COLOR / SIZE</td>
"""

    # 헤더 추가 (095 ~ 130)
    for h in headers:
        html += f'            <td class="sub-header border-bottom-thick">{h}</td>\n'

    # 빈 열 + TOTAL
    html += '            <td class="empty-col-header border-bottom-thick"></td>\n'
    html += '            <td class="sub-header border-bottom-thick border-right-thick">TOTAL</td>\n'
    html += '        </tr>\n'

    # 데이터 행 (BK, NA, D3, SV)
    for row in data:
        html += '        <tr>\n'
        html += f'            <td class="color-code border-left-thick">{row["code"]}</td>\n'
        html += f'            <td class="color-name">{row["name"]}</td>\n'
        for val in row["values"]:
            html += f'            <td class="data-cell">{val}</td>\n'
        html += '            <td class="empty-col"></td>\n'
        html += f'            <td class="total-col border-right-thick">{row["total"]}</td>\n'
        html += '        </tr>\n'

    # 빈 행
    html += '        <tr class="empty-row">\n'
    html += '            <td class="border-left-thick"></td>\n'
    html += '            <td></td>\n'
    for _ in range(8):
        html += '            <td></td>\n'
    html += '            <td class="empty-col"></td>\n'
    html += '            <td class="border-right-thick"></td>\n'
    html += '        </tr>\n'

    # TOTAL 행
    html += '        <tr class="total-row">\n'
    html += '            <td class="border-left-thick border-top-thick border-bottom-thick" colspan="2">TOTAL</td>\n'
    for val in totals:
        html += f'            <td class="data-cell border-top-thick border-bottom-thick">{val}</td>\n'
    html += '            <td class="empty-col border-top-thick border-bottom-thick"></td>\n'
    html += f'            <td class="total-col border-right-thick border-top-thick border-bottom-thick">{grand_total}</td>\n'
    html += '        </tr>\n'

    html += """    </table>
</body>
</html>
"""

    return html


def save_html(html_content, filename):
    """HTML 파일 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"파일 저장 완료: {filename}")


if __name__ == "__main__":
    # 테이블 생성
    html = generate_color_size_qty_table()

    # 파일 저장
    output_file = "color_size_qty_final.html"
    save_html(html, output_file)

    # 브라우저에서 열기
    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath(output_file))
