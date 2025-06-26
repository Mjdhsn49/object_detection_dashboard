"""
Depth estimation model using either MiDaS or Depth Anything v2.
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import pipeline

class DepthEstimator:
    def __init__(self, model_type="midas", model_size="small", device='cpu'):
        """Initialize depth estimation model.
        
        Args:
            model_type (str): Either 'midas' or 'depthanything'
            model_size (str): Model size - varies by type:
                - MiDaS: 'small' or 'large'
                - Depth Anything: 'small' or 'base'
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # For Depth Anything v2, we'll use CPU directly due to MPS compatibility issues
            self.pipe_device = 'cpu'
            print("Forcing CPU for depth estimation pipeline due to MPS compatibility issues")
        else:
            self.pipe_device = self.device
        
        print(f"Using device: {self.device} for depth estimation (pipeline on {self.pipe_device})")
        
        # Model initialization
        try:
            if self.model_type == "midas":
                self._init_midas()
            elif self.model_type == "depthanything":
                self._init_depth_anything()
            else:
                print(f"Unknown model type {model_type}, falling back to MiDaS small")
                self.model_type = "midas"
                self.model_size = "small"
                self._init_midas()
            
            self.initialized = True
            print(f"Initialized {self.model_type} {self.model_size} model on {device}")
            
        except Exception as e:
            print(f"Error initializing depth model: {e}")
            print("Falling back to placeholder depth estimation")
            self.initialized = False
    
    def _init_midas(self):
        """Initialize MiDaS model."""
        if self.model_size == "small":
            model_name = "MiDaS_small"
        elif self.model_size == "large":
            model_name = "DPT_Large"
        else:
            print(f"Unknown MiDaS size {self.model_size}, falling back to small")
            model_name = "MiDaS_small"
            self.model_size = "small"
        
        # Initialize MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_size == "small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform
    
    def _init_depth_anything(self):
        """Initialize Depth Anything v2 model."""
        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        model_name = model_map.get(self.model_size.lower(), model_map['small'])
        
        # Create pipeline
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {self.model_size} model on {self.pipe_device}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {self.model_size} model on CPU (fallback)")
    
    def estimate_depth(self, frame, depth_range=(1.0, 50.0), smooth=True):
        """Estimate depth for the entire frame."""
        if not self.initialized:
            # Fallback to placeholder
            height, width = frame.shape[:2]
            return np.random.uniform(depth_range[0], depth_range[1], (height, width)).astype(np.float32)
        
        try:
            if self.model_type == "midas":
                # MiDaS processing
                input_batch = self.transform(frame).to(self.device)
                with torch.no_grad():
                    prediction = self.model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth_map = prediction.cpu().numpy()
            else:  # depth anything
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(image_rgb)
                
                # Get depth map
                try:
                    depth_result = self.pipe(pil_image)
                    depth_map = depth_result["depth"]
                    
                    # Convert PIL Image to numpy array if needed
                    if isinstance(depth_map, Image.Image):
                        depth_map = np.array(depth_map)
                    elif isinstance(depth_map, torch.Tensor):
                        depth_map = depth_map.cpu().numpy()
                except RuntimeError as e:
                    # Handle potential MPS errors during inference
                    if self.device == 'mps':
                        print(f"MPS error during depth estimation: {e}")
                        print("Temporarily falling back to CPU for this frame")
                        # Create a CPU pipeline for this frame
                        cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                        depth_result = cpu_pipe(pil_image)
                        depth_map = depth_result["depth"]
                        
                        # Convert PIL Image to numpy array if needed
                        if isinstance(depth_map, Image.Image):
                            depth_map = np.array(depth_map)
                        elif isinstance(depth_map, torch.Tensor):
                            depth_map = depth_map.cpu().numpy()
                    else:
                        # Re-raise the error if not MPS
                        raise
            
            # Normalize depth map
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
            # Scale to specified depth range
            depth_range_size = depth_range[1] - depth_range[0]
            depth_map = depth_map * depth_range_size + depth_range[0]
            
            if smooth:
                # Apply bilateral filter to smooth depth while preserving edges
                depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
            
            return depth_map.astype(np.float32)
            
        except Exception as e:
            print(f"Error during depth estimation: {e}")
            # Fallback to placeholder
            height, width = frame.shape[:2]
            return np.random.uniform(depth_range[0], depth_range[1], (height, width)).astype(np.float32)
    
    def colorize_depth(self, depth_map, depth_range=(1.0, 50.0)):
        """Convert depth map to color visualization."""
        try:
            # Normalize depth map to 0-255 with better contrast
            depth_min, depth_max = depth_range
            normalized_depth = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # Create a more visually appealing colormap
            # Use COLORMAP_INFERNO for better depth perception
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
            
            # Enhance contrast
            colored_depth = cv2.convertScaleAbs(colored_depth, alpha=1.2, beta=0)
            
            # Add a color legend
            height, width = colored_depth.shape[:2]
            legend_width = 30
            legend = np.zeros((height, legend_width, 3), dtype=np.uint8)
            for i in range(height):
                color = cv2.applyColorMap(np.array([[int(255 * (1 - i/height))]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0,0]
                legend[i, :] = color
            
            # Add depth values to legend
            for i in range(5):
                y = int(i * height / 4)
                depth_value = depth_min + (depth_max - depth_min) * (1 - i/4)
                cv2.putText(legend, f"{depth_value:.1f}m", (2, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add model type indicator
            cv2.putText(legend, self.model_type[:6], (2, height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Combine depth map with legend
            colored_depth = np.hstack([colored_depth, legend])
            
            return colored_depth
        except Exception as e:
            print(f"Error colorizing depth map: {e}")
            return np.zeros_like(depth_map, dtype=np.uint8)
    
    def get_depth_at_point(self, depth_map, x, y):
        """Get depth value at a specific point."""
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """Get depth value in a region defined by bbox."""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(depth_map.shape[1] - 1, x2)
            y2 = min(depth_map.shape[0] - 1, y2)
            
            # Extract region
            region = depth_map[y1:y2, x1:x2]
            
            if region.size == 0:
                return 0.0
            
            # Compute depth based on method
            if method == 'median':
                return float(np.median(region))
            elif method == 'mean':
                return float(np.mean(region))
            elif method == 'min':
                return float(np.min(region))
            else:
                return float(np.median(region))
        except Exception as e:
            print(f"Error getting depth in region: {e}")
            return 5.0  # Return middle value as fallback 