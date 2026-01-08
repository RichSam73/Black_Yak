import requests
import base64
import time
import os
import json

# Configuration
IMAGE_PATH = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = [
    "deepseek-ocr:latest",
    "llama3.2-vision:latest",
    "granite3.2-vision:2b",
    "moondream:latest"
]
OUTPUT_FILE = "model_comparison_results.txt"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_model(model_name, image_base64):
    print(f"\n--- Testing {model_name} ---")
    start_time = time.time()
    
    payload = {
        "model": model_name,
        "prompt": "Analyze this image and extract all text from the table. Output in Markdown format.",
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0  # Deterministic output
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result.get("response", "")
        
        elapsed_time = time.time() - start_time
        print(f"Success! Time taken: {elapsed_time:.2f} seconds")
        return content, elapsed_time
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return f"ERROR: {str(e)}", 0

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Loading image from {IMAGE_PATH}...")
    try:
        image_base64 = encode_image(IMAGE_PATH)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    results = []
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Model Comparison Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")

    for model in MODELS:
        content, elapsed = test_model(model, image_base64)
        
        # Save immediate results
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Time: {elapsed:.2f} seconds\n")
            f.write("-" * 20 + " OUTPUT " + "-" * 20 + "\n")
            f.write(content + "\n")
            f.write("="*50 + "\n\n")
            
    print(f"\nAll tests completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
