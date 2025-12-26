"""PaddleOCR 결과 형식 확인"""
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en')
image_path = r"E:\Antigravity\Black_Yak\Reference\BY_Original_Table.png"

result = ocr.predict(image_path)

print("Result type:", type(result))
print("Result keys:", result.keys() if hasattr(result, 'keys') else "N/A")
print("\nFull result:")
print(result)
