"""Object detection service using YOLOv8 for weapon/threat detection.

This module uses a fine-tuned YOLOv8 model to detect weapons and threats in images.
Detected classes: Gun, explosion, grenade, knife
"""
import os
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

# Path to the threat detection model (relative to this file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "Suspicious_Activities_nano.pt")

# Load the model once when the module is imported
_model: YOLO | None = None


def get_model() -> YOLO:
    """Lazy-load the YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect_objects(image_url: str, confidence_threshold: float = 0.6) -> list[dict]:
    """
    Run object detection on an image URL.
    
    Args:
        image_url: URL of the image to analyze
        confidence_threshold: Minimum confidence score for detections (default: 0.25)
        
    Returns:
        List of detected objects with format:
        [
            {
                "label": str,
                "confidence": float,
                "bbox": {
                    "x": float,
                    "y": float,
                    "width": float,
                    "height": float
                }
            },
            ...
        ]
    """
    # Download image from URL
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    
    # Run detection
    model = get_model()
    results = model(image, conf=confidence_threshold, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for i in range(len(boxes)):
            # Get bounding box coordinates (xyxy format)
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = xyxy
            
            # Convert to x, y, width, height format
            width = x2 - x1
            height = y2 - y1
            
            # Get confidence and class
            confidence = float(boxes.conf[i])
            class_id = int(boxes.cls[i])
            label = model.names[class_id]
            
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height
                }
            })
    
    return detections
