"""OCR service using EasyOCR for text extraction."""
import httpx
import easyocr
import numpy as np
from PIL import Image
from io import BytesIO

# Initialized lazily
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        # Initialize for English. add other languages if needed
        _reader = easyocr.Reader(['en'])
    return _reader

def extract_text(image_url: str) -> list[dict]:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        image_url: URL or path to the image
    
    Returns:
        List of extracted text items with text, confidence, and position
    """
    image_input = image_url
    
    # Download image if URL
    if image_url.startswith(('http://', 'https://')):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(image_url)
                response.raise_for_status()
                # Pass bytes directly to EasyOCR
                image_input = response.content
        except Exception as e:
            print(f"Failed to download image: {e}")
            return []
    
    return _run_ocr(image_input)


def _run_ocr(image_input) -> list[dict]:
    """
    Run EasyOCR on the input (bytes, path, or array).
    """
    try:
        reader = get_reader()
        # detail=1 returns [bbox, text, conf]
        results = reader.readtext(image_input, detail=1)
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return []
    
    extracted_data = []
    
    for (bbox, text, conf) in results:
        text = text.strip()
        if not text or conf < 0.5:  # 50% confidence threshold
            continue
            
        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        # We'll use top-left (x1, y1) for position
        top_left = bbox[0]
        
        extracted_data.append({
            "text": text,
            "confidence": float(conf),
            "position_x": float(top_left[0]),
            "position_y": float(top_left[1]),
        })
        
    return extracted_data
