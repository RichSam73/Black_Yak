import requests
import base64
import time
import os
import json

# Configuration
IMAGE_PATH = r"E:\Antigravity\Black_Yak\Reference\Submaterial_information.png"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:latest"
OUTPUT_FILE = "qwen_test_result.txt"

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
            "temperature": 0
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600) # Increased timeout
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

    print(f"Starting test for {MODEL_NAME}...")
    content, elapsed = test_model(MODEL_NAME, image_base64)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Time: {elapsed:.2f} seconds\n")
        f.write("-" * 20 + " OUTPUT " + "-" * 20 + "\n")
        f.write(content + "\n")
        f.write("="*50 + "\n\n")
            
    print(f"\nTest completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
