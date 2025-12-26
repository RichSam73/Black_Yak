"""
PDF 요소별 추출기
PDF 파일에서 이미지, 도면, 테이블 등을 요소별로 구분하여 추출
ERP 연동용 데이터 추출 기능 포함
- Table Transformer를 사용한 스마트 테이블 인식 (스캔 PDF 지원)
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
import warnings
warnings.filterwarnings("ignore")

# Tesseract OCR 설정
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR	esseract.exe"
    TESSDATA_DIR = str(Path(__file__).parent / "tessdata")
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Table Transformer 설정 (스캔 PDF용 AI 테이블 인식)
TABLE_TRANSFORMER_AVAILABLE = False
try:
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TABLE_TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

# 스마트 테이블 추출 모듈
try:
    from smart_table_extractor import extract_smart_tables, is_scanned_pdf
    SMART_TABLE_AVAILABLE = TABLE_TRANSFORMER_AVAILABLE and OCR_AVAILABLE
except ImportError:
    SMART_TABLE_AVAILABLE = False
    extract_smart_tables = None
    is_scanned_pdf = None
