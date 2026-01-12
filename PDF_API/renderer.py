
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def is_vertical_text(bbox):
    """
    Check if text is vertical based on bbox aspect ratio.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    # Simple heuristic: if height is significantly larger than width (e.g., 2x), treat as vertical
    return height > width * 2

def get_text_color_for_background(bg_color):
    """
    Determine text color (black or white) based on background brightness.
    bg_color: (B, G, R) or (R, G, B) depending on context. Assuming RGB here for PIL.
    """
    # Calculate luminance
    # Y = 0.299*R + 0.587*G + 0.114*B
    if len(bg_color) == 3:
        r, g, b = bg_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)
    return (0, 0, 0)  # Default black

def process_image(image_bytes, data):
    """
    Process image: erase original text regions and render new text.
    
    Args:
        image_bytes: Raw bytes of the image
        data: List of dicts, each containing:
              - 'bbox': [x1, y1, x2, y2]
              - 'text': str (translated text)
              - 'bg_color': [r, g, b] (optional, override auto-detection)
    
    Returns:
        Bytes of the processed image (PNG format)
    """
    # 1. Load image with OpenCV for inpainting/drawing backgrounds
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    height, width = img.shape[:2]

    # Pre-calculate background colors and eraser regions
    for item in data:
        bbox = item.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clip coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Sampling background color logic (from app.py)
        # Check border pixels to determine background color
        border_pixels = []
        margin = 5
        
        # Sample pixels around the bbox
        sample_x_min = max(0, x1 - margin)
        sample_y_min = max(0, y1 - margin)
        sample_x_max = min(width, x2 + margin)
        sample_y_max = min(height, y2 + margin)

        # Top & Bottom borders
        # We can implement a more robust sampling if needed, but mean of the area is a good start
        # Or specifically sample just outside the box as in app.py
        
        # Simple approach: crop the expanded area and take the median/mean of the boundary
        # For speed/simplicity here, we'll try to use the provided bg_color or sample locally
        
        bg_color_bgr = None
        if 'bg_color' in item:
            r, g, b = item['bg_color']
            bg_color_bgr = (b, g, r)
        else:
            # Auto-detect
            # Sample surrounding area (excluding the text box itself if possible, but simpler to just use corners)
            corners = [
                img[y1, x1], img[y1, x2-1],
                img[y2-1, x1], img[y2-1, x2-1]
            ]
            # Average of corners
            avg_bgr = np.mean(corners, axis=0)
            bg_color_bgr = tuple(map(int, avg_bgr))
        
        # Fill the region with background color (Erasure)
        # Using a slightly larger box to clean up artifacts
        erase_margin = 2
        ex1 = max(0, x1 - erase_margin)
        ey1 = max(0, y1 - erase_margin)
        ex2 = min(width, x2 + erase_margin)
        ey2 = min(height, y2 + erase_margin)
        
        cv2.rectangle(img, (ex1, ey1), (ex2, ey2), bg_color_bgr, -1)
        
        # Store bg_color for text rendering phase (convert BGR to RGB)
        item['final_bg_color_rgb'] = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0])

    # 2. Render Text with PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load Font
    try:
        # Try to load Arial or similar sans-serif
        font_name = "arial.ttf"
        base_font = ImageFont.truetype(font_name, 10)
    except IOError:
        base_font = ImageFont.load_default()
        font_name = None

    for item in data:
        bbox = item.get('bbox')
        text = item.get('text')
        if not bbox or not text:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Skip invalid boxes
        if box_width <= 0 or box_height <= 0:
            continue
            
        bg_color_rgb = item.get('final_bg_color_rgb', (255, 255, 255))
        text_color_rgb = get_text_color_for_background(bg_color_rgb)
        
        # Auto-sizing font logic
        # Start with a heuristic size based on box height
        target_font_size = int(box_height * 0.8) # 80% of height
        if target_font_size < 8: target_font_size = 8
        
        font = base_font
        if font_name:
            font = ImageFont.truetype(font_name, target_font_size)
        
        # Check text dimensions
        # If it's too wide, shrink font
        if font_name:
            for size in range(target_font_size, 5, -1):
                font = ImageFont.truetype(font_name, size)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                
                if text_w <= box_width * 1.5: # Allow some overflow width wise? 
                    # app.py allows 1.5x width or shrinks. 
                    # Let's try to fit width.
                    if text_w <= box_width + 10: # Strict width fit
                        break
                if size == 6: # Minimum size reached
                    pass
        
        # Calculate centering
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        center_x = x1 + box_width // 2
        center_y = y1 + box_height // 2
        
        # Draw position (centered)
        draw_x = center_x - text_w // 2
        draw_y = center_y - text_h // 2
        
        # Optional: Left align if not in table (simplified here to always center or follow basic logic)
        # Simply using Center for now as it looks better for replaced labels
        
        draw.text((draw_x, draw_y), text, fill=text_color_rgb, font=font)

    # 3. Save to bytes
    output_buffer = io.BytesIO()
    img_pil.save(output_buffer, format='PNG')
    return output_buffer.getvalue()
