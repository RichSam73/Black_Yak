import requests
import json

api_key = 'AIzaSyChVCPKxMpOSdFLaxl95grdOoVlVRZgCm4'
model = 'gemini-2.5-flash'
url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'

prompt = """Extract Korean term and translate to 5 languages. Respond with JSON ONLY.

Input: "결재진행중이 영어로 뭐야?"

Output format (JSON only, no other text, start with { end with }):
{
  "korean": "한글용어",
  "english": "English",
  "vietnamese": "Tieng Viet",
  "chinese": "中文",
  "indonesian": "Indonesia",
  "bengali": "বাংলা"
}

Rules:
- Extract the Korean term from input
- If English is given in input, use it
- Translate to garment/sewing industry terms
- JSON ONLY. No explanations."""

payload = {
    'contents': [{'parts': [{'text': prompt}]}],
    'generationConfig': {
        'temperature': 0.1,
        'maxOutputTokens': 500,
        'responseMimeType': 'application/json'
    }
}

response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
print('Status:', response.status_code)
result = response.json()
print('Full response:', json.dumps(result, indent=2, ensure_ascii=False))

# Extract text
if 'candidates' in result:
    text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
    print('\n--- Extracted text ---')
    print(text)
