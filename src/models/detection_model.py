"""
Object detection model using YOLOv8.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_size="nano", conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
        """Initialize YOLO model."""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        
        # Determine best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("CUDA is available! Using GPU.")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("CUDA is not available. Using CPU.")
        else:
            self.device = device
        
        # Load model
        try:
            print(f"\nLoading YOLOv8 model with settings:")
            print(f"- Model size: {model_size}")
            print(f"- Confidence threshold: {conf_thres}")
            print(f"- IoU threshold: {iou_thres}")
            print(f"- Device: {self.device}")
            
            # Determine model path or size
            if Path(model_size).exists():  # Custom model path
                model_path = model_size
            else:  # Standard model size
                size_map = {
                    'nano': 'yolov8n.pt',
                    'small': 'yolov8s.pt',
                    'medium': 'yolov8m.pt',
                    'large': 'yolov8l.pt',
                    'xlarge': 'yolov8x.pt',
                    # Also support direct size indicators
                    'n': 'yolov8n.pt',
                    's': 'yolov8s.pt',
                    'm': 'yolov8m.pt',
                    'l': 'yolov8l.pt',
                    'x': 'yolov8x.pt'
                }
                model_path = size_map.get(model_size.lower(), 'yolov8n.pt')
            
            # Load the model and move to device
            print(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Warm up the model habe three models 
            print("Warming up model...")
            dummy_input = torch.zeros((1, 3, 640, 640)).to(self.device)
            for _ in range(2):  # Run twice for warm-up
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                _ = self.model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            print("Falling back to CPU and nano model")
            self.device = 'cpu'
            self.model = YOLO('yolov8n.pt')
            self.model.to('cpu')
    
    def detect(self, frame, track=False):
        """Detect objects in frame."""
        try:
            # Ensure frame is on the correct device
            if isinstance(frame, torch.Tensor):
                frame = frame.to(self.device)
            
            # Run inference
            if track:
                results = self.model.track(
                    source=frame,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    persist=True,
                    verbose=False,
                    device=self.device
                )
            else:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    verbose=False,
                    device=self.device
                )
            
            if results is None or len(results) == 0:
                return frame, []
            
            # Process results
            detections = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                
                boxes = r.boxes
                for box in boxes:
                    try:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Get confidence
                        conf = float(box.conf[0])
                        # Get class
                        cls = int(box.cls[0])
                        # Get tracking ID if available
                        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                        
                        detections.append(([x1, y1, x2, y2], conf, cls, track_id))
                        
                    except Exception as e:
                        print(f"Error processing detection box: {e}")
                        continue
            
            return frame, detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return frame, []
    
    def get_class_names(self):
        """Get list of class names."""
        return self.model.names 