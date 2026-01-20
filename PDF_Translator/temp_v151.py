"""
PDF 요소별 추출기
PDF 파일에서 이미지, 도면, 테이블 등을 요소별로 구분하여 추출
ERP 연동용 데이터 추출 기능 포함

v2: AI 테이블 추출 기능 추가 (Table Transformer + PaddleOCR)
"""

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import zipfile
import json
import re
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Tesseract OCR 설정
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSDATA_DIR = str(Path(__file__).parent / "tessdata")
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# AI 테이블 추출 (smart_table_extractor) 설정
AI_TABLE_AVAILABLE = False
VLM_AVAILABLE = False
COMET_AVAILABLE = False
try:
    from smart_table_extractor import (
        extract_smart_tables,
        extract_vlm_tables,
        extract_tables_auto,
        extract_comet_tables,
        extract_comet_with_table_detection,
        extract_ocr_with_coordinates,
        generate_comet_overlay_html,
        generate_comet_full_html,
        is_scanned_pdf,
        check_ollama_model,
        extract_smart_unified,  # 통합 스마트 추출 함수
        extract_text_with_coordinates,  # 텍스트 PDF 직접 추출
        extract_tables_grid_first,  # Grid-First 테이블 추출
        PADDLEOCR_AVAILABLE,
        TABLE_TRANSFORMER_AVAILABLE,
        OLLAMA_AVAILABLE
    )
    AI_TABLE_AVAILABLE = TABLE_TRANSFORMER_AVAILABLE and (PADDLEOCR_AVAILABLE or OCR_AVAILABLE)
    VLM_AVAILABLE = OLLAMA_AVAILABLE and check_ollama_model("granite3.2-vision")
    COMET_AVAILABLE = PADDLEOCR_AVAILABLE
    SMART_UNIFIED_AVAILABLE = PADDLEOCR_AVAILABLE or True  # PyMuPDF는 항상 사용 가능
    GRID_FIRST_AVAILABLE = PADDLEOCR_AVAILABLE  # Grid-First는 PaddleOCR 필요
except ImportError:
    extract_smart_tables = None
    extract_vlm_tables = None
    extract_tables_auto = None
    extract_comet_tables = None
    extract_comet_with_table_detection = None
    extract_ocr_with_coordinates = None
    generate_comet_overlay_html = None
    generate_comet_full_html = None
    is_scanned_pdf = None
    check_ollama_model = None
    extract_smart_unified = None
    extract_text_with_coordinates = None
    extract_tables_grid_first = None
    PADDLEOCR_AVAILABLE = False
    TABLE_TRANSFORMER_AVAILABLE = False
    OLLAMA_AVAILABLE = False
    SMART_UNIFIED_AVAILABLE = False
    GRID_FIRST_AVAILABLE = False

st.set_page_config(
    page_title="PDF 요소별 추출기",
    page_icon="📊",
    layout="wide"
)

st.title("📊 PDF 요소별 추출기")
st.markdown("PDF 파일을 업로드하면 요소별로 구분하여 추출합니다.")


def classify_image(img: Image.Image, width: int, height: int) -> str:
    """이미지 유형 분류"""
    aspect_ratio = width / height if height > 0 else 1

    is_small = width < 100 or height < 100
    is_icon_size = width < 64 and height < 64
    is_square = 0.8 <= aspect_ratio <= 1.2

    try:
        img_rgb = img.convert('RGB')
        colors = img_rgb.getcolors(maxcolors=10000)

        if colors:
            unique_colors = len(colors)

            if is_icon_size or (is_small and unique_colors < 50):
                return "아이콘"

            if is_square and unique_colors < 100 and width < 300:
                return "로고"

            if unique_colors < 500 and (width > 200 or height > 200):
                return "차트/도면"
    except:
        pass

    if width > 300 and height > 300:
        return "사진/이미지"

    return "기타 이미지"


def render_page_to_image(page, zoom: float = 2.0) -> tuple[Image.Image, bytes]:
    """페이지 전체를 이미지로 렌더링"""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return img, img_bytes


def extract_text_with_ocr(page, zoom: float = 2.0) -> str:
    """페이지를 이미지로 변환하고 OCR로 텍스트 추출"""
    if not OCR_AVAILABLE:
        return ""

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    # OCR 실행 (한글+영어)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    return text


def extract_erp_data(pdf_bytes: bytes, use_ocr: bool = False) -> dict:
    """PDF에서 모든 테이블을 있는 그대로 추출. 테이블이 없으면 OCR 사용"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    erp_data = {
        "tables": [],  # 모든 테이블을 페이지별로 저장
        "ocr_text": [],  # OCR로 추출한 텍스트 (이미지 PDF용)
        "is_scanned": False  # 스캔 PDF 여부
    }

    total_tables = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()

        for table_idx, table in enumerate(tables.tables):
            raw_data = table.extract()

            # 테이블 데이터를 그대로 저장 (None을 빈 문자열로만 변환)
            clean_data = []
            for row in raw_data:
                clean_row = [cell.strip() if cell else '' for cell in row]
                # 완전히 빈 행이 아니면 저장
                if any(clean_row):
                    clean_data.append(clean_row)

            if clean_data:
                erp_data["tables"].append({
                    "page": page_num + 1,
                    "table_index": table_idx + 1,
                    "data": clean_data
                })
                total_tables += 1

    # 테이블이 없고 OCR 사용이 활성화된 경우, 스캔 PDF로 간주하고 OCR 실행
    if total_tables == 0 and use_ocr and OCR_AVAILABLE:
        erp_data["is_scanned"] = True
        for page_num in range(len(doc)):
            page = doc[page_num]
            ocr_text = extract_text_with_ocr(page)
            if ocr_text.strip():
                erp_data["ocr_text"].append({
                    "page": page_num + 1,
                    "text": ocr_text
                })

    doc.close()
    return erp_data


def extract_elements_from_pdf(pdf_bytes: bytes, render_pages: bool = True) -> dict:
    """PDF에서 모든 요소 추출"""
    elements = defaultdict(list)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 1. 벡터 도면 추출 (cluster_drawings 활용)
        if render_pages:
            drawings = page.get_drawings()

            if len(drawings) > 50:
                # 벡터 클러스터링으로 개별 도면 영역 감지
                clusters = page.cluster_drawings()
                captured_clusters = []

                for bbox in clusters:
                    rect = fitz.Rect(bbox)
                    area_ratio = rect.get_area() / page.rect.get_area()

                    # 면적 비율 0.9 미만인 클러스터만 개별 추출
                    if area_ratio < 0.9:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        img_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_bytes))

                        elements["차트/도면"].append({
                            "page": page_num + 1,
                            "width": img.width,
                            "height": img.height,
                            "ext": "png",
                            "bytes": img_bytes,
                            "image": img,
                            "vector_count": len(drawings),
                            "area_ratio": round(area_ratio, 3),
                            "type": "차트/도면"
                        })
                        captured_clusters.append(bbox)

                # 개별 클러스터가 없거나 모두 전체 페이지인 경우 fallback
                if not captured_clusters:
                    img, img_bytes = render_page_to_image(page, zoom=2.0)
                    elements["페이지 렌더링"].append({
                        "page": page_num + 1,
                        "width": img.width,
                        "height": img.height,
                        "ext": "png",
                        "bytes": img_bytes,
                        "image": img,
                        "vector_count": len(drawings),
                        "type": "페이지 렌더링"
                    })

        # 2. 임베드된 이미지 추출
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]

                if width < 20 or height < 20:
                    continue

                image = Image.open(io.BytesIO(image_bytes))
                img_type = classify_image(image, width, height)

                metadata = {
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "width": width,
                    "height": height,
                    "ext": image_ext,
                    "bytes": image_bytes,
                    "image": image,
                    "type": img_type
                }

                elements[img_type].append(metadata)

            except Exception:
                continue

        # 3. 텍스트 블록 추출
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            if block[6] == 0:
                text = block[4].strip()
                if text:
                    elements["텍스트"].append({
                        "page": page_num + 1,
                        "content": text,
                        "bbox": block[:4]
                    })

        # 4. 링크 추출
        links = page.get_links()
        for link in links:
            if "uri" in link:
                elements["링크"].append({
                    "page": page_num + 1,
                    "url": link["uri"]
                })

    doc.close()
    return dict(elements)


def create_zip_by_category(elements: dict, pdf_name: str) -> bytes:
    """카테고리별로 ZIP 파일 생성"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for category, items in elements.items():
            if category in ["텍스트", "링크"]:
                continue

            for idx, item in enumerate(items):
                if "bytes" in item:
                    folder = category.replace("/", "_")
                    filename = f"{folder}/{pdf_name}_page{item['page']}_{idx+1}.{item['ext']}"
                    zf.writestr(filename, item["bytes"])

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# 파일 업로드
uploaded_file = st.file_uploader(
    "PDF 파일을 선택하세요",
    type=["pdf"],
    help="PDF 파일을 드래그 앤 드롭하거나 클릭하여 선택하세요"
)

if uploaded_file is not None:
    pdf_name = Path(uploaded_file.name).stem

    # 옵션
    with st.expander("⚙️ 추출 옵션"):
        render_pages = st.checkbox("벡터 도면이 있는 페이지 전체 렌더링", value=True,
                                   help="자켓 도면, 차트 등 벡터 그래픽이 많은 페이지를 이미지로 렌더링합니다")

    with st.spinner("PDF 요소를 분석하는 중..."):
        pdf_bytes = uploaded_file.read()
        elements = extract_elements_from_pdf(pdf_bytes, render_pages)

    if elements:
        st.success("✅ PDF 분석 완료!")

        # 요소별 개수 표시
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📄 페이지 렌더링", len(elements.get("페이지 렌더링", [])))
        with col2:
            st.metric("📊 차트/도면", len(elements.get("차트/도면", [])))
        with col3:
            st.metric("🖼️ 사진/이미지", len(elements.get("사진/이미지", [])))
        with col4:
            st.metric("🏷️ 로고/아이콘", len(elements.get("로고", [])) + len(elements.get("아이콘", [])))

        # ZIP 다운로드
        image_categories = ["페이지 렌더링", "차트/도면", "사진/이미지", "로고", "아이콘", "기타 이미지"]
        total_images = sum(len(elements.get(cat, [])) for cat in image_categories)

        if total_images > 0:
            zip_data = create_zip_by_category(elements, pdf_name)
            st.download_button(
                label="📥 모든 이미지 다운로드 (ZIP)",
                data=zip_data,
                file_name=f"{pdf_name}_elements.zip",
                mime="application/zip"
            )

        st.divider()

        # AI 테이블 추출 옵션 (통합)
        st.subheader("🤖 스마트 테이블 추출")

        col_opt1, col_opt2, col_opt3 = st.columns(3)

        with col_opt1:
            # 통합 스마트 추출 (권장)
            ai_method = "none"
            if SMART_UNIFIED_AVAILABLE:
                use_smart = st.checkbox(
                    "✨ 스마트 추출 (권장)",
                    value=True,
                    help="자동 PDF 타입 감지: 텍스트 PDF→직접 추출(100% 정확), 스캔 PDF→OCR"
                )
                if use_smart:
                    ai_method = "smart_unified"
            else:
                st.warning("⚠️ 스마트 추출 불가")

        with col_opt2:
            # Grid-First 옵션 (격자선 기반 추출)
            use_grid_first = False
            if GRID_FIRST_AVAILABLE:
                use_grid_first = st.checkbox(
                    "📐 Grid-First 추출",
                    value=False,
                    help="격자선 구조 먼저 감지 후 OCR 매핑 (COLOR/SIZE 테이블에 최적)"
                )
                if use_grid_first:
                    ai_method = "grid_first"

            # 테이블 분리 옵션 (smart_unified 선택 시)
            separate_tables = True
            if ai_method == "smart_unified" and TABLE_TRANSFORMER_AVAILABLE:
                separate_tables = st.checkbox(
                    "📊 테이블 개별 분리",
                    value=True,
                    help="Table Transformer + 격자선 감지로 여러 테이블을 개별 추출"
                )

        with col_opt3:
            if VLM_AVAILABLE and ai_method == "none":
                use_vlm = st.checkbox(
                    "🧠 VLM 추출",
                    value=False,
                    help="Granite3.2-vision VLM으로 문서를 이해하여 추출 (AI 추론 기반)"
                )
                if use_vlm:
                    ai_method = "vlm"
            elif not VLM_AVAILABLE and ai_method == "none":
                st.info("💡 VLM: `ollama pull granite3.2-vision`")

        # 추가 옵션 (필요 시)
        use_ocr = False
        if ai_method == "none" and OCR_AVAILABLE:
            use_ocr = st.checkbox(
                "🔍 기본 OCR",
                value=False,
                help="테이블이 감지되지 않는 경우 OCR로 텍스트만 추출합니다."
            )

        # ERP 데이터 추출
        if ai_method == "smart_unified" and SMART_UNIFIED_AVAILABLE:
            # 통합 스마트 추출 모드 (자동 PDF 타입 감지)
            progress_placeholder = st.empty()

            def progress_callback(page, total, msg):
                progress_placeholder.progress(page / total if total > 0 else 0, text=f"✨ 스마트 추출 중... {msg}")

            with st.spinner("✨ 스마트 추출 중... (PDF 타입 자동 감지)"):
                ai_result = extract_smart_unified(pdf_bytes,
                                                   progress_callback=progress_callback,
                                                   separate_tables=separate_tables)

            progress_placeholder.empty()

            # 결과를 ERP 데이터 형식으로 변환
            is_scanned = ai_result.get("is_scanned", True)
            erp_data = {
                "tables": [],
                "ocr_text": [],
                "is_scanned": is_scanned,
                "is_ai_extracted": True,
                "is_smart_unified": True,
                "is_hybrid": ai_result.get("is_hybrid", False),
                "ocr_engine": ai_result.get("ocr_engine", "Unknown"),
                "comet_html": ai_result.get("comet_html", []),
                "pages": ai_result.get("pages", [])
            }

            for table in ai_result.get("tables", []):
                erp_data["tables"].append({
                    "page": table["page"],
                    "table_index": table["table_index"],
                    "table_name": table.get("table_name", f"테이블 {table['table_index']}"),
                    "data": table["data"],
                    "confidence": table.get("confidence", 0.99 if not is_scanned else 0.95),
                    "row_count": table.get("row_count", 0),
                    "col_count": table.get("col_count", 0),
                    "bbox": table.get("bbox", []),
                    "extraction_method": table.get("extraction_method", ai_result.get("extraction_method", "smart_unified"))
                })

        elif ai_method == "vlm" and VLM_AVAILABLE:
            # VLM 추출 모드
            progress_placeholder = st.empty()

            def progress_callback(page, total, msg):
                progress_placeholder.progress(page / total, text=f"🧠 VLM 분석 중... {msg}")

            with st.spinner("🧠 VLM 테이블 추출 중... (Granite3.2-vision으로 문서 분석)"):
                ai_result = extract_vlm_tables(pdf_bytes, progress_callback=progress_callback)

            progress_placeholder.empty()

            # AI 결과를 ERP 데이터 형식으로 변환
            erp_data = {
                "tables": [],
                "ocr_text": [],
                "is_scanned": True,
                "is_ai_extracted": True,
                "ocr_engine": f"VLM ({ai_result.get('model', 'granite3.2-vision')})"
            }

            for table in ai_result.get("tables", []):
                erp_data["tables"].append({
                    "page": table["page"],
                    "table_index": table["table_index"],
                    "data": table["data"],
                    "confidence": table.get("confidence", 0.95),
                    "row_count": table.get("row_count", 0),
                    "col_count": table.get("col_count", 0),
                    "title": table.get("title", "")
                })

            # 페이지별 필드 정보도 추가
            for page_data in ai_result.get("pages", []):
                fields = page_data.get("fields", {})
                if fields:
                    erp_data["fields"] = fields

        elif ai_method == "grid_first" and GRID_FIRST_AVAILABLE:
            # Grid-First 추출 모드 (격자선 구조 먼저 감지)
            progress_placeholder = st.empty()

            def progress_callback(page, total, msg):
                progress_placeholder.progress(page / total if total > 0 else 0, text=f"📐 Grid-First 분석 중... {msg}")

            with st.spinner("📐 Grid-First 추출 중... (격자선 감지 + PaddleOCR)"):
                ai_result = extract_tables_grid_first(pdf_bytes, progress_callback=progress_callback, min_cells=10)

            progress_placeholder.empty()

            # 결과를 ERP 데이터 형식으로 변환
            erp_data = {
                "tables": [],
                "ocr_text": [],
                "is_scanned": True,
                "is_ai_extracted": True,
                "is_grid_first": True,
                "ocr_engine": ai_result.get("ocr_engine", "PaddleOCR")
            }

            for table in ai_result.get("tables", []):
                erp_data["tables"].append({
                    "page": table["page"],
                    "table_index": table["table_index"],
                    "table_name": f"Grid Table {table['table_index']}",
                    "data": table["data"],
                    "confidence": table.get("confidence", 1.0),
                    "row_count": table.get("row_count", 0),
                    "col_count": table.get("col_count", 0),
                    "bbox": table.get("box", []),
                    "extraction_method": "grid_first"
                })

            if ai_result.get("tables"):
                st.success(f"📐 Grid-First 추출 완료 - {len(ai_result['tables'])}개 테이블 발견")

        elif ai_method == "table_transformer" and AI_TABLE_AVAILABLE:
            # Table Transformer 추출 모드
            progress_placeholder = st.empty()

            def progress_callback(page, total, msg):
                progress_placeholder.progress(page / total, text=f"📊 AI 분석 중... {msg}")

            with st.spinner("📊 AI 테이블 추출 중... (Table Transformer + PaddleOCR)"):
                ai_result = extract_smart_tables(pdf_bytes, progress_callback=progress_callback, use_paddle=PADDLEOCR_AVAILABLE)

            progress_placeholder.empty()

            # AI 결과를 ERP 데이터 형식으로 변환
            erp_data = {
                "tables": [],
                "ocr_text": [],
                "is_scanned": True,
                "is_ai_extracted": True,
                "ocr_engine": ai_result.get("ocr_engine", "Unknown")
            }

            for table in ai_result.get("tables", []):
                erp_data["tables"].append({
                    "page": table["page"],
                    "table_index": table["table_index"],
                    "data": table["data"],
                    "confidence": table.get("confidence", 0),
                    "row_count": table.get("row_count", 0),
                    "col_count": table.get("col_count", 0)
                })
        else:
            # 기존 방식
            with st.spinner("데이터 추출 중..." + (" (OCR 처리 시 시간이 걸릴 수 있습니다)" if use_ocr else "")):
                erp_data = extract_erp_data(pdf_bytes, use_ocr=use_ocr)

        # 탭으로 요소별 표시
        tabs = st.tabs(["📦 ERP 데이터", "📄 페이지 렌더링", "📊 차트/도면", "🖼️ 사진/이미지", "🏷️ 로고/아이콘", "📝 텍스트"])

        # ERP 데이터 탭
        with tabs[0]:
            st.subheader("📦 ERP 연동용 데이터")

            tables = erp_data.get("tables", [])
            ocr_texts = erp_data.get("ocr_text", [])
            is_scanned = erp_data.get("is_scanned", False)
            is_ai_extracted = erp_data.get("is_ai_extracted", False)
            ocr_engine = erp_data.get("ocr_engine", "")

            # 스마트 추출 결과 표시
            is_smart_unified = erp_data.get("is_smart_unified", False)
            is_hybrid = erp_data.get("is_hybrid", False)

            if is_smart_unified:
                # PDF 타입에 따른 메시지
                if is_scanned:
                    st.success(f"✨ 스마트 추출 완료 - 스캔 PDF → OCR 사용 (엔진: {ocr_engine})")
                else:
                    st.success(f"✨ 스마트 추출 완료 - 텍스트 PDF → 직접 추출 (100% 정확도)")

                if is_hybrid:
                    st.caption("📊 Table Transformer + 격자선 감지로 테이블이 개별 분리되었습니다.")

                # 페이지 오버레이 다운로드/미리보기
                comet_pages = erp_data.get("pages", [])
                if comet_pages:
                    # 전체 HTML 생성
                    comet_data = {
                        "pages": comet_pages,
                        "total_pages": len(comet_pages)
                    }
                    full_html = generate_comet_full_html(comet_data, scale=0.7)

                    st.download_button(
                        "📥 HTML 다운로드 (선택 가능 텍스트)",
                        data=full_html,
                        file_name=f"{pdf_name}_smart_overlay.html",
                        mime="text/html",
                        help="HTML 파일을 브라우저에서 열면 텍스트를 선택/복사할 수 있습니다"
                    )

                    # 미리보기
                    with st.expander("📄 페이지 미리보기", expanded=False):
                        for page_data in comet_pages[:3]:
                            page_num = page_data["page"]
                            img_base64 = page_data.get("image_base64", "")

                            if img_base64:
                                st.image(
                                    f"data:image/png;base64,{img_base64}",
                                    caption=f"페이지 {page_num}",
                                    use_container_width=True
                                )

                        if len(comet_pages) > 3:
                            st.info(f"... 외 {len(comet_pages) - 3}페이지 더 있음")

                st.divider()

            # VLM 필드 정보 표시
            fields = erp_data.get("fields", {})
            if fields:
                st.success("🧠 VLM 문서 필드 추출 완료")
                st.markdown("### 📋 추출된 필드 정보")

                field_cols = st.columns(2)
                field_items = list(fields.items())

                for i, (key, value) in enumerate(field_items):
                    col = field_cols[i % 2]
                    with col:
                        if isinstance(value, list):
                            # COLOR_WAY 등 리스트 형태
                            st.markdown(f"**{key}:**")
                            for item in value:
                                if isinstance(item, dict):
                                    st.write(f"  - {item.get('code', '')} : {item.get('name', '')}")
                                else:
                                    st.write(f"  - {item}")
                        else:
                            st.markdown(f"**{key}:** {value}")

                st.divider()

            if tables:
                if is_ai_extracted:
                    st.success(f"🤖 AI 테이블 추출 완료: {len(tables)}개 테이블 (엔진: {ocr_engine})")
                else:
                    st.success(f"총 {len(tables)}개 테이블 추출됨")

                # 각 테이블을 그대로 표시
                for table_info in tables:
                    page_num = table_info["page"]
                    table_idx = table_info["table_index"]
                    table_name = table_info.get("table_name", "")
                    data = table_info["data"]
                    confidence = table_info.get("confidence", 0)
                    row_count = table_info.get("row_count", len(data) if data else 0)
                    col_count = table_info.get("col_count", max(len(row) for row in data) if data else 0)

                    # 헤더 표시
                    if table_name:
                        header_text = f"### 📄 페이지 {page_num} - {table_name}"
                    else:
                        header_text = f"### 📄 페이지 {page_num} - 테이블 {table_idx}"
                    if is_ai_extracted and confidence:
                        header_text += f" (신뢰도: {confidence:.1%})"
                    st.markdown(header_text)

                    if is_ai_extracted:
                        st.caption(f"크기: {row_count}행 × {col_count}열")

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("빈 테이블")

                    st.divider()

            elif is_scanned and ocr_texts:
                st.warning("⚠️ 스캔 PDF 감지됨 - OCR로 텍스트 추출")
                st.info("이 PDF는 이미지 기반입니다. OCR로 텍스트를 추출했습니다.")

                for ocr_item in ocr_texts:
                    page_num = ocr_item["page"]
                    text = ocr_item["text"]

                    st.markdown(f"### 📄 페이지 {page_num} - OCR 텍스트")
                    st.text_area(
                        f"페이지 {page_num}",
                        value=text,
                        height=400,
                        key=f"ocr_text_{page_num}"
                    )
                    st.divider()

            else:
                st.warning("추출된 테이블이 없습니다.")
                if AI_TABLE_AVAILABLE:
                    st.info("💡 스캔 PDF인 경우 위의 '🤖 AI 테이블 추출' 체크박스를 활성화해보세요.")
                elif not use_ocr and OCR_AVAILABLE:
                    st.info("💡 스캔 PDF인 경우 위의 'OCR 사용' 체크박스를 활성화해보세요.")

            # 다운로드 버튼
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                # JSON 다운로드
                json_data = json.dumps(erp_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 JSON 다운로드 (ERP 연동용)",
                    data=json_data,
                    file_name=f"{pdf_name}_erp_data.json",
                    mime="application/json"
                )

            with col2:
                # 전체 CSV 다운로드 (모든 테이블 합쳐서)
                if tables:
                    all_rows = []
                    for table_info in tables:
                        page_num = table_info["page"]
                        table_idx = table_info["table_index"]
                        for row in table_info["data"]:
                            all_rows.append([f"P{page_num}-T{table_idx}"] + row)

                    csv_df = pd.DataFrame(all_rows)
                    csv_data = csv_df.to_csv(index=False, header=False)
                    st.download_button(
                        "📥 CSV 다운로드 (전체 테이블)",
                        data=csv_data,
                        file_name=f"{pdf_name}_all_tables.csv",
                        mime="text/csv"
                    )

        # 페이지 렌더링
        with tabs[1]:
            pages = elements.get("페이지 렌더링", [])
            if pages:
                st.subheader(f"페이지 렌더링 ({len(pages)}개)")
                st.caption("벡터 도면이 포함된 페이지를 고해상도로 렌더링했습니다.")

                for idx, item in enumerate(pages):
                    st.markdown(f"### 페이지 {item['page']} (벡터 요소: {item['vector_count']}개)")
                    st.image(item["image"], width=800)
                    st.download_button(
                        "💾 다운로드 (PNG)",
                        data=item["bytes"],
                        file_name=f"{pdf_name}_page{item['page']}_render.png",
                        key=f"render_{idx}"
                    )
                    st.divider()
            else:
                st.info("벡터 도면이 포함된 페이지가 없습니다.")

        # 차트/도면
        with tabs[2]:
            charts = elements.get("차트/도면", [])
            if charts:
                st.subheader(f"차트/도면 ({len(charts)}개)")
                cols = st.columns(3)
                for idx, item in enumerate(charts):
                    with cols[idx % 3]:
                        st.image(item["image"], caption=f"페이지 {item['page']} ({item['width']}x{item['height']})", width="stretch")
                        st.download_button(
                            "💾 다운로드",
                            data=item["bytes"],
                            file_name=f"{pdf_name}_chart_{idx+1}.{item['ext']}",
                            key=f"chart_{idx}"
                        )
            else:
                st.info("차트/도면이 없습니다. '페이지 렌더링' 탭을 확인하세요.")

        # 사진/이미지
        with tabs[3]:
            photos = elements.get("사진/이미지", [])
            if photos:
                st.subheader(f"사진/이미지 ({len(photos)}개)")
                cols = st.columns(3)
                for idx, item in enumerate(photos):
                    with cols[idx % 3]:
                        st.image(item["image"], caption=f"페이지 {item['page']} ({item['width']}x{item['height']})", width="stretch")
                        st.download_button(
                            "💾 다운로드",
                            data=item["bytes"],
                            file_name=f"{pdf_name}_photo_{idx+1}.{item['ext']}",
                            key=f"photo_{idx}"
                        )
            else:
                st.info("사진/이미지가 없습니다.")

        # 로고/아이콘
        with tabs[4]:
            logos = elements.get("로고", []) + elements.get("아이콘", [])
            if logos:
                st.subheader(f"로고/아이콘 ({len(logos)}개)")
                cols = st.columns(6)
                for idx, item in enumerate(logos):
                    with cols[idx % 6]:
                        st.image(item["image"], caption=f"p.{item['page']}", width="stretch")
                        st.download_button(
                            "💾",
                            data=item["bytes"],
                            file_name=f"{pdf_name}_logo_{idx+1}.{item['ext']}",
                            key=f"logo_{idx}"
                        )
            else:
                st.info("로고/아이콘이 없습니다.")

        # 텍스트
        with tabs[5]:
            texts = elements.get("텍스트", [])
            if texts:
                st.subheader(f"텍스트 블록 ({len(texts)}개)")

                # 텍스트 전체 다운로드
                all_text = "\n\n".join([f"[페이지 {t['page']}]\n{t['content']}" for t in texts])
                st.download_button(
                    "📥 텍스트 전체 다운로드 (TXT)",
                    data=all_text,
                    file_name=f"{pdf_name}_text.txt",
                    mime="text/plain"
                )

                for item in texts[:30]:
                    with st.expander(f"페이지 {item['page']}: {item['content'][:60]}..."):
                        st.text(item["content"])
                if len(texts) > 30:
                    st.info(f"... 외 {len(texts) - 30}개 더 있음")
            else:
                st.info("텍스트가 없습니다.")
    else:
        st.warning("⚠️ PDF에서 요소를 찾을 수 없습니다.")

# 사이드바
with st.sidebar:
    st.header("📌 추출 방식")
    st.markdown("""
    **페이지 렌더링**
    - 벡터 도면(자켓 도면 등)이 50개 이상인 페이지를 고해상도 이미지로 렌더링

    **차트/도면**
    - PDF에 임베드된 이미지 중 색상 수가 적고 크기가 큰 것

    **사진/이미지**
    - 크기가 크고 색상이 다양한 이미지

    **로고/아이콘**
    - 작은 크기의 이미지
    """)

    st.divider()
    st.caption("💡 자켓 도면 등 벡터 그래픽은 '페이지 렌더링' 탭에서 확인하세요!")
