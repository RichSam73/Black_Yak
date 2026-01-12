
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import json
import traceback
import os
import renderer
import uvicorn
from contextlib import asynccontextmanager

# App definition
app = FastAPI(
    title="PDF Image Renderer API",
    description="API to render translated text onto images",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Dictionary Management ---
DICT_FILE = "garment_dict.json"

@app.get("/dictionary")
def get_dictionary():
    """Get the current garment dictionary."""
    try:
        if os.path.exists(DICT_FILE):
            with open(DICT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {} # Return empty dict if file doesn't exist
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dictionary: {str(e)}")

@app.post("/dictionary")
async def update_dictionary(new_dict: dict):
    """Update the garment dictionary."""
    try:
        # Validate structure (simple check)
        if not isinstance(new_dict, dict):
             raise HTTPException(status_code=400, detail="Invalid format. Dictionary expected.")
        
        with open(DICT_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_dict, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "message": "Dictionary updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save dictionary: {str(e)}")
# -----------------------------

@app.post("/render")
async def render_image(
    file: UploadFile = File(...),
    data: str = Form(..., description="JSON string containing list of text items with bbox")
):
    """
    Render translated text onto the uploaded image.
    
    - **file**: Image file (PNG/JPG)
    - **data**: JSON string of list of objects:
      [
        {
          "bbox": [x1, y1, x2, y2],
          "text": "Translated Text",
          "bg_color": [r, g, b]  // optional
        },
        ...
      ]
    """
    try:
        # 1. Parse JSON data
        try:
            items_data = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in 'data' field")
        
        if not isinstance(items_data, list):
            raise HTTPException(status_code=400, detail="'data' must be a list of items")

        # 2. Read image file
        image_bytes = await file.read()
        
        # 3. Process image (CPU bound, running in threadpool via FastAPI default behavior for def)
        # However, since process_image is synchronous, calling it directly here blocks the event loop 
        # if defined as 'async def'. But 'process_image' is pure sync IO/CPU.
        # Ideally, we should run it in a thread pool.
        # FastAPI runs 'async def' in the event loop and 'def' in a thread pool.
        # Since this is 'async def' (to await file.read()), we should use run_in_threadpool or 
        # keep it simple. For heavy image processing, running in a separate thread is better.
        
        # Simple sync call for now, assuming moderate load. 
        # If needed, use: from fastapi.concurrency import run_in_threadpool
        # result_image = await run_in_threadpool(renderer.process_image, image_bytes, items_data)
        
        result_image = renderer.process_image(image_bytes, items_data)
        
        # 4. Return result
        return Response(content=result_image, media_type="image/png")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Version for deployment verification
VERSION = "1.0.0"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting PDF Image Renderer API (v{VERSION}) on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
