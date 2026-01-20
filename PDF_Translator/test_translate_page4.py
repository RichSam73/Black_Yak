# 직접 번역 테스트 스크립트 v2
import requests
import os
import time

pdf_path = r"E:\Antigravity\Black_Yak\Reference\Translator\RBY25-B0035 ORDER 등록용 WORKSHEET.pdf"
output_dir = r"E:\Antigravity\Black_Yak\PDF_Translator\output"

base_url = "http://localhost:6009"

print("="*60)
print("[테스트] PDF 4페이지 번역 (/translate 엔드포인트)")
print("="*60)

# /translate 엔드포인트 사용 (파일 업로드 + 바로 번역)
print("\n[1/2] PDF 업로드 및 번역 중... (시간 소요)")
with open(pdf_path, 'rb') as f:
    files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
    data = {
        'target_lang': 'english',
        'model': 'claude',
        'start_page': '4',
        'end_page': '4'
    }
    
    start_time = time.time()
    response = requests.post(f"{base_url}/translate", files=files, data=data)
    elapsed = time.time() - start_time

print(f"\n응답 코드: {response.status_code}")
print(f"소요 시간: {elapsed:.1f}초")

if response.status_code == 200:
    result = response.json()
    print(f"성공: {result.get('success')}")
    
    if result.get('success'):
        outputs = result.get('outputs', [])
        print(f"출력 파일 수: {len(outputs)}")
        
        for out in outputs:
            print(f"\n  페이지 {out.get('page')}: {out.get('output_file')}")
            
            # 결과 이미지 저장
            output_file = out.get('output_file')
            if output_file:
                img_url = f"{base_url}/output/{output_file}"
                img_response = requests.get(img_url)
                if img_response.status_code == 200:
                    save_path = os.path.join(output_dir, f"overlap_test_{output_file}")
                    with open(save_path, 'wb') as f:
                        f.write(img_response.content)
                    print(f"  → 저장됨: {save_path}")
    else:
        print(f"에러: {result.get('error')}")
else:
    print(f"실패: {response.text[:500]}")

print("\n" + "="*60)
print("[완료]")
print("="*60)
