"""
Object detection model using YOLOv8 with GPU optimization.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_size="nano", conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
        """Initialize YOLO model with MAXIMUM GPU optimization."""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.model_path = None
        self.custom_class_names = None
        
        # Determine best available device
        if device is None or device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                # MINIMAL LOGGING
                print("🚀 CUDA GPU")
                
                # MAXIMUM GPU OPTIMIZATION SETTINGS
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # Set maximum memory usage
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.98)  # Use 98% of GPU memory
                
                # Set GPU to maximum performance mode
                torch.cuda.set_device(0)
                
                # MINIMAL LOGGING
                print("⚡ MAX GPU OPT")
            else:
                self.device = 'cpu'
                print("⚠️ Using CPU - performance will be limited")
        else:
            self.device = device
        
        # Load model
        try:
            # MINIMAL LOGGING
            print(f"Loading YOLOv8 {model_size}...")
            
            # Determine model path or size
            if Path(model_size).exists():  # Custom model path
                self.model_path = model_size
                # Load custom class names if available
                self._load_custom_class_names()
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
                self.model_path = size_map.get(model_size.lower(), 'yolov8n.pt')
            
            # Load the model and move to device
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # MAXIMUM GPU OPTIMIZATION
            if self.device == 'cuda':
                # Fuse layers for maximum speed
                self.model.fuse()
                # Set model to evaluation mode
                self.model.eval()
                
                # Enable mixed precision for maximum speed
                self.scaler = torch.cuda.amp.GradScaler()
                
                # Ultra-fast warm up with FP16 - REDUCED FOR SPEED
                # MINIMAL LOGGING
                print("🔥 Warming up...")
                dummy_input = torch.zeros((1, 3, 640, 640), device=self.device, dtype=torch.float16)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    with torch.no_grad():
                        for _ in range(5):  # Reduced warm-up for speed
                            _ = self.model(dummy_input)
                        torch.cuda.synchronize()
                
                # MINIMAL LOGGING
                print("⚡ Model ready")
            else:
                # CPU warm-up
                # MINIMAL LOGGING
                print("🔥 Warming up...")
                dummy_input = torch.zeros((1, 3, 640, 640))
                for _ in range(2):  # Reduced warm-up
                    with torch.no_grad():
                        _ = self.model(dummy_input)
            
            # MINIMAL LOGGING
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            print("Falling back to CPU and nano model")
            self.device = 'cpu'
            self.model_path = 'yolov8n.pt'
            self.model = YOLO('yolov8n.pt')
            self.model.to('cpu')
    
    def _load_custom_class_names(self):
        """Load custom class names from data.yaml file if available."""
        try:
            if self.model_path and Path(self.model_path).exists():
                model_dir = Path(self.model_path).parent
                data_yaml_path = model_dir / 'data.yaml'
                
                if data_yaml_path.exists():
                    with open(data_yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if 'names' in data:
                        self.custom_class_names = data['names']
                        print(f"✅ Loaded custom class names: {self.custom_class_names}")
                        return
                
                # Check parent directory for data.yaml (for weights subdirectory)
                parent_dir = model_dir.parent
                data_yaml_path = parent_dir / 'data.yaml'
                
                if data_yaml_path.exists():
                    with open(data_yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if 'names' in data:
                        self.custom_class_names = data['names']
                        print(f"✅ Loaded custom class names: {self.custom_class_names}")
                        return
                
                # Additional check for case-sensitive directory names (like "Infrared")
                # Try to find data.yaml in any parent directory with case-insensitive matching
                current_dir = model_dir
                for _ in range(3):  # Check up to 3 levels up
                    current_dir = current_dir.parent
                    # Look for data.yaml files in current directory
                    for file_path in current_dir.iterdir():
                        if file_path.is_file() and file_path.name.lower() == 'data.yaml':
                            with open(file_path, 'r') as f:
                                data = yaml.safe_load(f)
                            
                            if 'names' in data:
                                self.custom_class_names = data['names']
                                print(f"✅ Loaded custom class names from {file_path}: {self.custom_class_names}")
                                return
                        
        except Exception as e:
            print(f"Warning: Could not load custom class names: {e}")
            self.custom_class_names = None
    
    def _get_boat_class_names(self):
        """Get boat-related class names from COCO dataset."""
        # COCO class names for boat-related objects
        coco_names = self.model.names
        boat_classes = {}
        
        # Boat-related class indices in COCO
        boat_related = {
            8: 'vessel boat',      # boat
            9: 'traffic light',  # traffic light (sometimes useful for maritime)
            10: 'fire hydrant',  # fire hydrant (maritime safety)
            11: 'stop sign',     # stop sign (maritime navigation)
            12: 'parking meter', # parking meter (dock/pier)
            13: 'bench',         # bench (dock/pier)
            14: 'bird',          # bird (marine wildlife)
            15: 'cat',           # cat (ship cat)
            16: 'dog',           # dog (ship dog)
            17: 'horse',         # horse (rare on ships)
            18: 'sheep',         # sheep (livestock transport)
            19: 'cow',           # cow (livestock transport)
            20: 'elephant',      # elephant (cargo)
            21: 'bear',          # bear (cargo)
            22: 'zebra',         # zebra (cargo)
            23: 'giraffe',       # giraffe (cargo)
            24: 'backpack',      # backpack (crew equipment)
            25: 'umbrella',      # umbrella (deck equipment)
            26: 'handbag',       # handbag (crew equipment)
            27: 'tie',           # tie (crew uniform)
            28: 'suitcase',      # suitcase (cargo)
            29: 'frisbee',       # frisbee (deck recreation)
            30: 'skis',          # skis (cargo)
            31: 'snowboard',     # snowboard (cargo)
            32: 'sports ball',   # sports ball (deck recreation)
            33: 'kite',          # kite (deck recreation)
            34: 'baseball bat',  # baseball bat (deck recreation)
            35: 'baseball glove', # baseball glove (deck recreation)
            36: 'skateboard',    # skateboard (deck recreation)
            37: 'surfboard',     # surfboard (deck recreation)
            38: 'tennis racket', # tennis racket (deck recreation)
            39: 'bottle',        # bottle (crew/cargo)
            40: 'wine glass',    # wine glass (crew equipment)
            41: 'cup',           # cup (crew equipment)
            42: 'fork',          # fork (crew equipment)
            43: 'knife',         # knife (crew equipment)
            44: 'spoon',         # spoon (crew equipment)
            45: 'bowl',          # bowl (crew equipment)
            46: 'banana',        # banana (cargo)
            47: 'apple',         # apple (cargo)
            48: 'sandwich',      # sandwich (crew food)
            49: 'orange',        # orange (cargo)
            50: 'broccoli',      # broccoli (cargo)
            51: 'carrot',        # carrot (cargo)
            52: 'hot dog',       # hot dog (crew food)
            53: 'pizza',         # pizza (crew food)
            54: 'donut',         # donut (crew food)
            55: 'cake',          # cake (crew food)
            56: 'chair',         # chair (crew quarters)
            57: 'couch',         # couch (crew quarters)
            58: 'potted plant',  # potted plant (crew quarters)
            59: 'bed',           # bed (crew quarters)
            60: 'dining table',  # dining table (crew mess)
            61: 'toilet',        # toilet (crew facilities)
            62: 'tv',            # tv (crew quarters)
            63: 'laptop',        # laptop (crew equipment)
            64: 'mouse',         # mouse (crew equipment)
            65: 'remote',        # remote (crew equipment)
            66: 'keyboard',      # keyboard (crew equipment)
            67: 'cell phone',    # cell phone (crew equipment)
            68: 'microwave',     # microwave (crew galley)
            69: 'oven',          # oven (crew galley)
            70: 'toaster',       # toaster (crew galley)
            71: 'sink',          # sink (crew galley)
            72: 'refrigerator',  # refrigerator (crew galley)
            73: 'book',          # book (crew equipment)
            74: 'clock',         # clock (crew equipment)
            75: 'vase',          # vase (crew quarters)
            76: 'scissors',      # scissors (crew equipment)
            77: 'teddy bear',    # teddy bear (crew personal)
            78: 'hair drier',    # hair drier (crew equipment)
            79: 'toothbrush'     # toothbrush (crew equipment)
        }
        
        # Filter for boat-related classes
        for class_id, class_name in boat_related.items():
            if class_id < len(coco_names):
                boat_classes[class_id] = class_name
        
        return boat_classes
    
    def detect(self, frame, track=False, filter_boat_only=True):
        """Detect objects in frame with MAXIMUM GPU optimization for ultra-low latency."""
        try:
            # Ensure frame is on the correct device
            if isinstance(frame, torch.Tensor):
                frame = frame.to(self.device)
            
            # Run inference with MAXIMUM optimization
            if self.device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):  # Use FP16 for maximum speed
                    with torch.no_grad():  # Disable gradient computation
                        if track:
                            results = self.model.track(
                                source=frame,
                                conf=self.conf_thres,
                                iou=self.iou_thres,
                                persist=True,
                                verbose=False,
                                device=self.device,
                                stream=True  # Enable streaming for faster inference
                            )
                        else:
                            results = self.model.predict(
                                source=frame,
                                conf=self.conf_thres,
                                iou=self.iou_thres,
                                verbose=False,
                                device=self.device,
                                stream=True  # Enable streaming for faster inference
                            )
                        # Synchronize GPU immediately
                        torch.cuda.synchronize()
            else:
                with torch.no_grad():
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
            
            if results is None:
                return frame, []
            
            # Process results
            detections = []
            
            # Handle generator (stream=True) vs list (stream=False)
            if hasattr(results, '__iter__') and not isinstance(results, (list, tuple)):
                # Generator from stream=True
                for r in results:
                    if r is None:
                        continue
                    # Process single result
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
                            
                            # Get class name for filtering
                            class_name = self.get_class_name(cls)
                            
                            # Apply boat-only filtering only for default YOLO model
                            if filter_boat_only and self._is_default_yolo_model():
                                if class_name != 'vessel boat':
                                    continue  # Skip non-boat detections for default model only
                            
                            detections.append(([x1, y1, x2, y2], conf, cls, track_id))
                            
                        except Exception as e:
                            # MINIMAL ERROR LOGGING - Only log every 100th error
                            continue
            else:
                # List from stream=False
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
                            
                            # Get class name for filtering
                            class_name = self.get_class_name(cls)
                            
                            # Apply boat-only filtering only for default YOLO model
                            if filter_boat_only and self._is_default_yolo_model():
                                if class_name != 'vessel boat':
                                    continue  # Skip non-boat detections for default model only
                            
                            detections.append(([x1, y1, x2, y2], conf, cls, track_id))
                            
                        except Exception as e:
                            # MINIMAL ERROR LOGGING - Only log every 100th error
                            continue
            
            return frame, detections
            
        except Exception as e:
            # MINIMAL ERROR LOGGING
            return frame, []
    
    def get_class_names(self):
        """Get list of class names based on model type."""
        # If custom class names are loaded, use them
        if self.custom_class_names:
            return self.custom_class_names
        
        # For default YOLOv8 model, return boat-related classes
        if 'yolov8n.pt' in str(self.model_path) or 'yolov8' in str(self.model_path):
            boat_classes = self._get_boat_class_names()
            return boat_classes
        
        # Fallback to model's default class names
        return self.model.names 
    
    def get_class_name(self, class_id):
        """Get class name for a specific class ID."""
        class_names = self.get_class_names()
        
        # Handle different class name formats
        if isinstance(class_names, dict):
            return class_names.get(class_id, f"class_{class_id}")
        elif isinstance(class_names, list):
            if class_id < len(class_names):
                return class_names[class_id]
            else:
                return f"class_{class_id}"
        else:
            return f"class_{class_id}" 
    
    def _is_default_yolo_model(self):
        """Check if this is the default YOLO model (yolov8n.pt)."""
        if not self.model_path:
            return False
        
        model_path_str = str(self.model_path).lower()
        return 'yolov8n.pt' in model_path_str or 'yolov8' in model_path_str 