"""
테이블 이미지를 HTML로 변환하는 스크립트
Table 2: 3색상 (BK, CM, TE), 빈열 4개, 빈행 3개
"""

def generate_color_size_qty_table2():
    """COLOR/SIZE QTY 테이블 2를 HTML로 생성"""

    # 테이블 구조 정의
    config = {
        "title": "COLOR/SIZE QTY",
        "headers": ["095", "100", "105", "110", "115", "120"],  # 사이즈 헤더
        "empty_cols": 4,  # 빈 열 개수
        "empty_rows": 3,  # 빈 행 개수
    }

    # 데이터 정의 (각 색상별)
    data = [
        {"code": "BK", "name": "BLACK", "values": ["140", "350", "250", "160", "", ""], "total": "900"},
        {"code": "CM", "name": "CREAM", "values": ["370", "490", "250", "90", "", ""], "total": "1,200"},
        {"code": "TE", "name": "TEAL", "values": ["410", "720", "470", "190", "60", "50"], "total": "1,900"},
    ]

    # TOTAL 행 데이터
    totals = ["920", "1,560", "970", "440", "60", "50"]
    grand_total = "4,000"

    # 전체 열 수 계산: 2(코드+이름) + 헤더수 + 빈열수 + 1(TOTAL)
    total_cols = 2 + len(config["headers"]) + config["empty_cols"] + 1

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{config["title"]} - Table 2</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
            background: #fff;
        }}
        table {{
            border-collapse: collapse;
            font-size: 12px;
            border: 2px solid #000;
        }}
        td {{
            border: 1px solid #000;
            padding: 5px 10px;
            text-align: center;
            height: 24px;
            background: #fff;
            color: #000;
        }}
        .header {{
            font-weight: bold;
        }}
        .sub-header {{
            font-weight: bold;
        }}
        .color-code {{
            font-weight: bold;
        }}
        .color-name {{
            font-weight: bold;
        }}
        .data-cell {{
            text-align: right;
        }}
        .total-row td {{
            font-weight: bold;
        }}
        .total-col {{
            font-weight: bold;
            text-align: right;
        }}
    </style>
</head>
<body>
    <table>
        <!-- 1행: 헤더 (전체 병합) -->
        <tr>
            <td colspan="{total_cols}" class="header">{config["title"]}</td>
        </tr>
        <!-- 2행: 서브 헤더 -->
        <tr>
            <td class="sub-header" colspan="2">COLOR / SIZE</td>
"""

    # 사이즈 헤더 추가
    for h in config["headers"]:
        html += f'            <td class="sub-header">{h}</td>\n'

    # 빈 열 추가
    for _ in range(config["empty_cols"]):
        html += '            <td class="empty-cell"></td>\n'

    # TOTAL 헤더
    html += '            <td class="sub-header">TOTAL</td>\n'
    html += '        </tr>\n'

    # 데이터 행
    for row in data:
        html += '        <tr>\n'
        html += f'            <td class="color-code">{row["code"]}</td>\n'
        html += f'            <td class="color-name">{row["name"]}</td>\n'

        # 값 추가
        for val in row["values"]:
            if val:
                html += f'            <td class="data-cell">{val}</td>\n'
            else:
                html += '            <td class="empty-cell"></td>\n'

        # 빈 열 추가 (사이즈 헤더 수 - 값 수 + 빈열수)
        empty_data_cols = len(config["headers"]) - len(row["values"]) + config["empty_cols"]
        for _ in range(empty_data_cols):
            html += '            <td class="empty-cell"></td>\n'

        # TOTAL
        html += f'            <td class="total-col">{row["total"]}</td>\n'
        html += '        </tr>\n'

    # 빈 행
    for i in range(config["empty_rows"]):
        html += f'        <!-- 빈 행 {i+1} -->\n'
        html += '        <tr>\n'
        for _ in range(total_cols):
            html += '            <td class="empty-cell"></td>\n'
        html += '        </tr>\n'

    # TOTAL 행
    html += '        <!-- TOTAL 행 -->\n'
    html += '        <tr class="total-row">\n'
    html += '            <td colspan="2">TOTAL</td>\n'

    for val in totals:
        if val:
            html += f'            <td>{val}</td>\n'
        else:
            html += '            <td></td>\n'

    # 빈 열
    for _ in range(config["empty_cols"]):
        html += '            <td></td>\n'

    html += f'            <td class="total-col">{grand_total}</td>\n'
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
    html = generate_color_size_qty_table2()

    # 파일 저장
    output_file = "table2_generated.html"
    save_html(html, output_file)

    # 브라우저에서 열기
    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath(output_file))
