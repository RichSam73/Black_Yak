import os
import requests
from dotenv import load_dotenv

os.chdir(r'E:\Antigravity\Black_Yak\PDF_Translate_Note')
load_dotenv()

# Gemini API Test - 올바른 모델명 사용
gemini_key = os.environ.get('GEMINI_API_KEY')
print(f'Gemini Key: {gemini_key[:10]}...{gemini_key[-5:]}')

# 모델명: gemini-2.0-flash
resp = requests.post(
    f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}',
    headers={'Content-Type': 'application/json'},
    json={'contents': [{'parts': [{'text': '안녕하세요를 영어로 번역해주세요'}]}]}
)
print(f'Gemini Status: {resp.status_code}')
print(resp.text[:1000])
