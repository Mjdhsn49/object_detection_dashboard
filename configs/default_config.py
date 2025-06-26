"""
Default configuration for Marine 3D Detection.
"""

class Config:
    # Model settings
    CUSTOM_MODEL_PATH = None  # Path to custom YOLOv8 model, if None will use default YOLOv8n
    CLASSES_FILE = None  # Path to classes.txt file for custom model
    
    # Detection settings
    CONF_THRESHOLD = 0.25  # Confidence threshold for object detection
    IOU_THRESHOLD = 0.45  # IoU threshold for NMS
    CLASSES = None  # Filter by class indices, None for all classes
    
    # Feature toggles
    ENABLE_TRACKING = True  # Enable object tracking
    ENABLE_BEV = True  # Enable Bird's Eye View visualization
    ENABLE_PSEUDO_3D = True  # Enable pseudo-3D visualization
    
    # Visualization settings
    SHOW_CLASS_NAMES = True  # Show class names in visualization
    SHOW_SCORES = True  # Show confidence scores
    SHOW_IDS = True  # Show tracking IDs
    SHOW_DEPTH = True  # Show depth values
    
    # Device settings
    DEVICE = 'cpu'  # Force CPU for stability
    
    # Camera settings
    CAMERA_PARAMS_FILE = None  # Path to camera parameters file
    
    # BEV settings
    BEV_SCALE = 60  # Scale factor for Bird's Eye View
    BEV_SIZE = (300, 300)  # Size of Bird's Eye View visualization 