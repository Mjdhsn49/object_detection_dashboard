"""
Depth estimation model using either MiDaS or Depth Anything v2 with GPU optimization.
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
        """Initialize depth estimation model with GPU optimization.
        
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
            print(f"✅ Initialized {self.model_type} {self.model_size} model on {device}")
            
        except Exception as e:
            print(f"Error initializing depth model: {e}")
            print("Falling back to placeholder depth estimation")
            self.initialized = False
    
    def _init_midas(self):
        """Initialize MiDaS model with GPU optimization."""
        if self.model_size == "small":
            model_name = "MiDaS_small"
        elif self.model_size == "large":
            model_name = "DPT_Large"
        else:
            print(f"Unknown MiDaS size {self.model_size}, falling back to small")
            model_name = "MiDaS_small"
            self.model_size = "small"
        
        # Initialize MiDaS with GPU optimization
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Optimize for inference
        if self.device == 'cuda':
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Clear GPU cache
            torch.cuda.empty_cache()
            print("🚀 MiDaS model optimized for GPU inference")
        
        # Initialize transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_size == "small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform
    
    def _init_depth_anything(self):
        """Initialize Depth Anything v2 model with GPU optimization."""
        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        model_name = model_map.get(self.model_size.lower(), model_map['small'])
        
        # Create pipeline with GPU optimization
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            if self.device == 'cuda':
                print(f"🚀 Loaded Depth Anything v2 {self.model_size} model on GPU")
            else:
                print(f"✅ Loaded Depth Anything v2 {self.model_size} model on {self.pipe_device}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"✅ Loaded Depth Anything v2 {self.model_size} model on CPU (fallback)")
    
    def estimate_depth(self, frame, depth_range=(1.0, 50.0), smooth=True):
        """Estimate depth for the entire frame with GPU optimization and noise reduction."""
        if not self.initialized:
            # Fallback to placeholder with better structure
            height, width = frame.shape[:2]
            # Create a more realistic depth map instead of random noise
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            # Create depth gradient from top to bottom (closer objects at bottom)
            depth_map = depth_range[0] + (depth_range[1] - depth_range[0]) * (y_coords / height)
            return depth_map.astype(np.float32)
        
        try:
            if self.model_type == "midas":
                # MiDaS processing with GPU optimization
                input_batch = self.transform(frame).to(self.device)
                
                with torch.no_grad():
                    if self.device == 'cuda':
                        with torch.cuda.amp.autocast():  # Use mixed precision
                            prediction = self.model(input_batch)
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=frame.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze()
                        torch.cuda.synchronize()  # Ensure GPU operations complete
                    else:
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
            
            # Improved depth map processing to reduce noise
            # Normalize depth map properly
            depth_min = np.percentile(depth_map, 5)  # Use 5th percentile to avoid outliers
            depth_max = np.percentile(depth_map, 95)  # Use 95th percentile to avoid outliers
            
            if depth_max > depth_min:
                # Clip outliers and normalize
                depth_map = np.clip(depth_map, depth_min, depth_max)
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                # Fallback normalization if percentiles are the same
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            # Scale to specified depth range
            depth_range_size = depth_range[1] - depth_range[0]
            depth_map = depth_map * depth_range_size + depth_range[0]
            
            # Apply multiple smoothing techniques to reduce noise
            if smooth:
                # First, apply bilateral filter to preserve edges while smoothing
                depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 15, 50, 50)
                
                # Then apply Gaussian blur for additional smoothing
                depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
                
                # Finally, apply median filter to remove salt-and-pepper noise
                depth_map = cv2.medianBlur(depth_map.astype(np.uint8), 3).astype(np.float32)
                
                # Ensure depth values are within reasonable bounds
                depth_map = np.clip(depth_map, depth_range[0], depth_range[1])
            
            return depth_map.astype(np.float32)
            
        except Exception as e:
            print(f"Error during depth estimation: {e}")
            # Fallback to structured depth map instead of random noise
            height, width = frame.shape[:2]
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            # Create depth gradient from top to bottom
            depth_map = depth_range[0] + (depth_range[1] - depth_range[0]) * (y_coords / height)
            return depth_map.astype(np.float32)
    
    def colorize_depth(self, depth_map, depth_range=(1.0, 50.0)):
        """Convert depth map to color visualization with noise reduction."""
        try:
            # Apply additional smoothing to the depth map before colorization
            smoothed_depth = cv2.GaussianBlur(depth_map, (3, 3), 0)
            
            # Normalize depth map to 0-255 with better contrast
            depth_min, depth_max = depth_range
            normalized_depth = ((smoothed_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # Apply histogram equalization for better contrast
            normalized_depth = cv2.equalizeHist(normalized_depth)
            
            # Create a more visually appealing colormap
            # Use COLORMAP_INFERNO for better depth perception
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
            
            # Enhance contrast and reduce noise
            colored_depth = cv2.convertScaleAbs(colored_depth, alpha=1.3, beta=10)
            
            # Apply bilateral filter to reduce color noise while preserving edges
            colored_depth = cv2.bilateralFilter(colored_depth, 9, 75, 75)
            
            # Add a color legend
            height, width = colored_depth.shape[:2]
            legend_width = 30
            legend_height = height // 2
            
            # Create legend gradient
            legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
            for i in range(legend_height):
                ratio = i / legend_height
                color = cv2.applyColorMap(np.array([[int(ratio * 255)]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
                legend[i, :] = color
            
            # Add legend to the right side
            colored_depth[0:legend_height, width-legend_width:width] = legend
            
            # Add depth labels to legend
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            
            # Add depth values
            for i in range(3):
                y_pos = int(legend_height * (i / 2))
                depth_val = depth_max - (depth_max - depth_min) * (i / 2)
                cv2.putText(colored_depth, f"{depth_val:.1f}m", 
                           (width - legend_width + 2, y_pos + 10), 
                           font, font_scale, (255, 255, 255), thickness)
            
            return colored_depth
            
        except Exception as e:
            print(f"Error colorizing depth map: {e}")
            # Fallback to grayscale with smoothing
            try:
                smoothed_depth = cv2.GaussianBlur(depth_map, (3, 3), 0)
                normalized_depth = ((smoothed_depth - depth_range[0]) / (depth_range[1] - depth_range[0]) * 255).astype(np.uint8)
                return cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)
            except:
                # Final fallback
                return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    
    def get_depth_at_point(self, depth_map, x, y):
        """Get depth value at a specific point."""
        try:
            return float(depth_map[int(y), int(x)])
        except (IndexError, ValueError):
            return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """Get depth value in a region using specified method."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            region = depth_map[y1:y2, x1:x2]
            
            if method == 'median':
                return float(np.median(region))
            elif method == 'mean':
                return float(np.mean(region))
            elif method == 'min':
                return float(np.min(region))
            elif method == 'max':
                return float(np.max(region))
            else:
                return float(np.median(region))
        except Exception as e:
            print(f"Error getting depth in region: {e}")
            return 0.0 