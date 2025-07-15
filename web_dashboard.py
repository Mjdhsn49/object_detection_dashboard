#!/usr/bin/env python3
"""
Web Dashboard for Object Detection on RTMP Streams
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set default stream host environment variable
if 'STREAM_HOST' not in os.environ:
    os.environ['STREAM_HOST'] = 'http://192.168.128.209:5000'

import sys
import time
import cv2
import numpy as np
import threading
import json
import base64
import subprocess
import signal
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
# import queue  # Removed - using real-time processing without queues
import logging
import torch
import requests

# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import detection modules
try:
    from src.models import ObjectDetector  # , DepthEstimator  # COMMENTED OUT FOR SPEED
    # from src.utils.bbox3d_utils import BBox3DEstimator, BirdEyeView  # COMMENTED OUT FOR SPEED
    from configs.default_config import Config
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Detection modules not available: {e}")
    DETECTION_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'object-detection-dashboard-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
stream_processors = {}  # key: stream_path, value: StreamProcessor
detection_threads = {}  # key: stream_path, value: Thread
stop_detection_flags = {}  # key: stream_path, value: bool
current_configs = {}  # key: stream_path, value: config

class StreamProcessor:
    def __init__(self):
        self.cap = None
        self.ffmpeg_process = None  # FFmpeg subprocess for direct RTMP
        self.use_ffmpeg_direct = True  # Use FFmpeg by default
        self.detector = None
        # self.depth_estimator = None  # COMMENTED OUT FOR SPEED
        # self.bbox3d_estimator = None  # COMMENTED OUT FOR SPEED
        # self.bev = None  # COMMENTED OUT FOR SPEED
        self.config = None
        self.is_running = False
        self.raw_stream_mode = False  # Skip detection for ultra-low latency
        
        # Real-time processing components - NO QUEUES, ONLY LATEST FRAMES
        self.frame_reader_thread = None
        self.detection_thread = None
        self.latest_raw_frame = None  # Latest frame from RTMP
        self.latest_processed_frame = None  # Latest frame with detection
        self.frame_lock = threading.Lock()
        self.detection_processing = False  # Flag to prevent overlapping detection
        self.frame_timestamp = 0  # Timestamp of latest frame
        self.detection_skip_count = 0  # Count of skipped detections for stats
        self.frame_width = 1920  # Expected frame dimensions
        self.frame_height = 1080
        self.ffmpeg_fps = 30  # Target FPS for FFmpeg
        
    def initialize_models(self, config):
        """Initialize detection models with GPU optimization."""
        if not DETECTION_AVAILABLE:
            return False
        try:
            self.config = config
            device = config.get('device', 'auto')
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                    print("🚀 CUDA is available! Using GPU for all models.")
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    # Optimize CUDA settings for maximum performance
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.enabled = True
                    # Set memory fraction for better performance
                    torch.cuda.empty_cache()
                else:
                    device = 'cpu'
                    print("⚠️ CUDA is not available. Using CPU for all models.")
            print(f"Initializing models on device: {device}")
            
            # Only allow three local models (relative paths for Docker)
            allowed_models = [
                'data/other_models/default_model/yolov8n.pt',
                'data/other_models/cross_model/weights/best.onnx',
                'data/other_models/Infrared/weights/best.pt'
            ]
            model_path = config.get('model', allowed_models[0])
            if model_path not in allowed_models:
                print(f"⚠️ Model {model_path} not allowed. Using default model.")
                model_path = allowed_models[0]
            
            print(f"Loading model from: {model_path}")
            
            try:
                self.detector = ObjectDetector(
                    model_size=model_path,
                    conf_thres=config.get('confidence', 0.25),
                    iou_thres=config.get('iou', 0.45),
                    classes=None,
                    device=device
                )
                
                # Verify GPU usage
                if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'device'):
                    model_device = str(self.detector.model.device)
                    if 'cuda' in model_device:
                        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                    else:
                        print("⚠️ Model is on CPU")
                else:
                    print("⚠️ Could not verify model device")
            except Exception as e:
                print(f"Error initializing object detector: {e}")
                print("Falling back to CPU for object detection")
                self.detector = ObjectDetector(
                    model_size=allowed_models[0],
                    conf_thres=config.get('confidence', 0.25),
                    iou_thres=config.get('iou', 0.45),
                    classes=None,
                    device='cpu'
                )
            
            # Initialize depth estimator
            # try:
            #     self.depth_estimator = DepthEstimator(
            #         model_type=config.get('depth_model', 'midas'),
            #         model_size=config.get('depth_size', 'small'),
            #         device=device
            #     )
            #     print("✅ Depth estimator initialized successfully")
            # except Exception as e:
            #     print(f"Error initializing depth estimator: {e}")
            #     print("Falling back to CPU for depth estimation")
            #     self.depth_estimator = DepthEstimator(
            #         model_type=config.get('depth_model', 'midas'),
            #         model_size=config.get('depth_size', 'small'),
            #         device='cpu'
            #     )
            
            # Initialize 3D bounding box estimator
            # self.bbox3d_estimator = BBox3DEstimator()
            # print("✅ 3D bounding box estimator initialized")
            
            # Initialize Bird's Eye View
            # if config.get('enable_bev', True):
            #     self.bev = BirdEyeView(scale=60, size=(300, 300))
            #     print("✅ Bird's Eye View initialized")
                
            return True
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def open_stream(self, rtmp_url):
        """Open RTMP stream with proper FFmpeg settings and fallback options."""
        try:
            # Method 1: Try with RTMP-specific settings
            print(f"🔗 Attempting to open RTMP: {rtmp_url}")
            
            # Set OpenCV FFmpeg options for RTMP - ULTRA LOW LATENCY
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtmp_live=1;"
                "rtmp_buffer=0;"
                "fflags=nobuffer+flush_packets;"
                "flags=low_delay;"
                "analyzeduration=0;"
                "probesize=32;"
                "max_delay=0"
            )
            
            # Create VideoCapture with RTMP-specific settings
            self.cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
            
            # Set additional properties for RTMP - REAL-TIME OPTIMIZED
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # NO BUFFER - Real-time mode
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Expected FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Expected width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Expected height
            
            # Try to set timeout for RTMP connection (may not be available in all OpenCV versions)
            try:
                self.cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 second timeout
            except:
                pass  # Timeout property not available
            
            if not self.cap.isOpened():
                print(f"❌ Method 1 failed, trying fallback...")
                return self._open_stream_fallback(rtmp_url)
            
            # Test if we can read at least one frame with retry
            test_frame = None
            for attempt in range(3):
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    break
                time.sleep(0.1)  # Wait 100ms between attempts
            
            if not ret or test_frame is None:
                print(f"❌ Cannot read test frame after 3 attempts, trying fallback...")
                return self._open_stream_fallback(rtmp_url)
            
            print(f"✅ RTMP stream opened successfully: {rtmp_url}")
            print(f"📐 Frame size: {test_frame.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error opening RTMP stream {rtmp_url}: {e}")
            return self._open_stream_fallback(rtmp_url)
    
    def _open_stream_fallback(self, rtmp_url):
        """Fallback method for opening RTMP streams."""
        try:
            print(f"🔄 Trying fallback method for: {rtmp_url}")
            
            # Release previous capture
            if self.cap is not None:
                self.cap.release()
            
            # Method 2: Try with different FFmpeg options for RTMP - ULTRA LOW LATENCY
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtmp_live=1;"
                "fflags=nobuffer+flush_packets;"
                "flags=low_delay;"
                "analyzeduration=0;"
                "probesize=32;"
                "max_delay=0"
            )
            
            self.cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # NO BUFFER - Real-time mode
            
            # Set additional properties for low latency
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            if not self.cap.isOpened():
                print(f"❌ Fallback method also failed")
                return False
            
            # Test frame read with retry
            test_frame = None
            for attempt in range(3):
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    break
                time.sleep(0.1)  # Wait 100ms between attempts
            
            if not ret or test_frame is None:
                print(f"❌ Fallback cannot read frame after 3 attempts")
                return False
            
            print(f"✅ Fallback RTMP stream opened: {rtmp_url}")
            print(f"📐 Frame size: {test_frame.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Fallback method failed: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame with detection."""
        if not DETECTION_AVAILABLE or not self.detector:
            return frame
            
        try:
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            result_frame = frame.copy()
            
            # Object Detection
            detection_frame, detections = self.detector.detect(
                detection_frame, 
                track=self.config.get('enable_tracking', True),
                filter_boat_only=self.config.get('filter_boat_only', True)  # Use configurable filter
            )
            
            # Depth Estimation
            depth_colored = None
            # if self.config.get('show_depth', True):
            #     try:
            #         depth_map = self.depth_estimator.estimate_depth(
            #             original_frame,
            #             depth_range=self.config.get('depth_range', [1.0, 50.0]),
            #             smooth=self.config.get('smooth_depth', True)
            #         )
            #         depth_colored = self.depth_estimator.colorize_depth(
            #             depth_map,
            #             depth_range=self.config.get('depth_range', [1.0, 50.0])
            #         )
            #     except Exception as e:
            #         print(f"Depth estimation error: {e}")
            
            # Process detections
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    class_name = self.detector.get_class_name(class_id)
                    
                    # Get depth in the region
                    # if self.depth_estimator and depth_colored is not None:
                    #     depth_value = self.depth_estimator.get_depth_in_region(
                    #         depth_map, bbox, method='median'
                    #     )
                    # else:
                    #     depth_value = 0.0
                    depth_value = 0.0  # Set to 0 since depth is disabled
                    
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    boxes_3d.append(box_3d)
                    
                    if obj_id is not None:
                        active_ids.append(obj_id)
                        
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers
            # if self.bbox3d_estimator:
            #     self.bbox3d_estimator.cleanup_trackers(active_ids)
            
            # Visualize results
            result_frame = self.visualize_results(result_frame, boxes_3d, depth_colored)
            
            return result_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return frame
    
    def visualize_results(self, frame, boxes_3d, depth_colored):
        """Visualize detection results."""
        height, width = frame.shape[:2]
        
        # Draw detection boxes
        for box_3d in boxes_3d:
            try:
                color = (0, 255, 0)  # Green
                bbox = box_3d['bbox_2d']
                depth = box_3d['depth_value']
                obj_id = box_3d['object_id']
                score = box_3d['score']
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw text with class name
                class_name = box_3d['class_name']
                if obj_id is not None:
                    text = f"{class_name} ID:{int(obj_id)} {score:.2f}"
                else:
                    text = f"{class_name} {score:.2f}"
                
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, 
                            (x1, y1 - text_size[1] - 8), 
                            (x1 + text_size[0] + 4, y1 - 4), 
                            color, -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 6), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw depth value
                # if self.config.get('show_depth', True):
                #     depth_text = f"{depth:.1f}m"
                #     text_size = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                #     cv2.rectangle(frame,
                #                 (x1, y2 + 2),
                #                 (x1 + text_size[0], y2 + text_size[1] + 6),
                #                 color, -1)
                #     cv2.putText(frame, depth_text, (x1, y2 + text_size[1] + 2), 
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
            except Exception as e:
                print(f"Error drawing box: {e}")
                continue
        
        # Draw Bird's Eye View
        # if self.bev is not None and self.config.get('enable_bev', True):
        #     try:
        #         self.bev.reset()
        #         for box_3d in boxes_3d:
        #             self.bev.draw_box(box_3d)
        #         bev_image = self.bev.get_image()
        #         
        #         bev_height = height // 4
        #         bev_width = bev_height
        #         
        #         if bev_height > 0 and bev_width > 0:
        #             bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
        #             frame[height - bev_height:height, 0:bev_width] = bev_resized
        #             cv2.rectangle(frame, 
        #                         (0, height - bev_height), 
        #                         (bev_width, height), 
        #                         (255, 255, 255), 1)
        #         cv2.putText(frame, "Bird's Eye View", 
        #                    (10, height - bev_height + 20), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     except Exception as e:
        #         print(f"Error drawing BEV: {e}")
        
        # Add depth visualization
        # if depth_colored is not None and self.config.get('show_depth', True):
        #     try:
        #         depth_height = height // 4
        #         depth_width = int(depth_height * width / height)
        #         depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
        #         frame[0:depth_height, 0:depth_width] = depth_resized
        #     except Exception as e:
        #         print(f"Error adding depth map: {e}")
        
        return frame
    
    def _cleanup_ffmpeg(self):
        """Clean up FFmpeg process."""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            self.ffmpeg_process = None
    
    def run_detection(self, rtmp_url, config, stop_flag_key):
        """Main detection loop with PARALLEL PROCESSING for maximum real-time performance."""
        global stop_detection_flags
        
        try:
            # Initialize models
            if not self.initialize_models(config):
                socketio.emit('error', {'message': 'Failed to initialize detection models'})
                return
            
            # Open stream with ultra-low latency settings
            if not self.open_stream(rtmp_url):
                socketio.emit('error', {'message': f'Failed to open stream: {rtmp_url}'})
                return
            
            # Verify stream is open and has frames
            if not self.cap.isOpened():
                socketio.emit('error', {'message': f'Stream is not opened: {rtmp_url}'})
                return
            
            # MINIMAL LOGGING - Only essential messages
            print(f"🔗 Stream opened: {rtmp_url}")
            
            # Test if we can read at least one frame
            test_ret, test_frame = self.cap.read()
            if not test_ret:
                print(f"❌ Cannot read test frame - stream may be empty")
            
            self.is_running = True
            
            # REAL-TIME PROCESSING SETUP
            stream_start_time = time.time()
            frame_count = 0
            processed_count = 0
            last_frame_time = 0
            target_fps = 30
            frame_interval = 1.0 / target_fps
            
            # MAXIMUM GPU OPTIMIZATION
            if torch.cuda.is_available():
                # Set maximum GPU performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # Clear GPU cache and set memory fraction
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.98)  # Use 98% of GPU memory
                
                # Set GPU to maximum performance mode
                torch.cuda.set_device(0)
                device = torch.device('cuda:0')
                
                # MINIMAL LOGGING
                print(f"🚀 MAX GPU OPT")
            else:
                device = torch.device('cpu')
                print("⚠️ Using CPU - performance will be limited")
            
            # Performance monitoring - REDUCED FOR SPEED
            processing_times = []
            frame_timing_history = []
            
            # ULTRA-LOW LATENCY PROCESSING MODE - AGGRESSIVE THRESHOLDS
            max_lag_threshold = 0.005 if self.raw_stream_mode else 0.015  # 5ms for raw, 15ms for detection
            
            # MINIMAL LOGGING
            mode_str = "RAW STREAM" if self.raw_stream_mode else "DETECTION"
            method_str = "FFMPEG" if self.use_ffmpeg_direct else "OPENCV"
            print(f"⚡ {mode_str} MODE with {method_str} (max lag: {max_lag_threshold}s)")
            
            # Pre-allocate GPU tensors for faster processing
            if torch.cuda.is_available():
                dummy_input = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float16)
                # Warm up GPU with multiple passes
                with torch.no_grad():
                    for _ in range(3):  # Reduced warm-up for speed
                        _ = self.detector.model(dummy_input)
                    torch.cuda.synchronize()
                # MINIMAL LOGGING
                print("🔥 GPU ready")
            
            # START ULTRA-LOW LATENCY THREADS
            print(f"🚀 Starting ultra-low latency threads for: {stop_flag_key}")
            
            # Start frame reader thread
            self.frame_reader_thread = threading.Thread(
                target=self._frame_reader_worker,
                args=(rtmp_url, stop_flag_key),
                daemon=True
            )
            self.frame_reader_thread.start()
            
            # Start detection worker thread
            self.detection_thread = threading.Thread(
                target=self._detection_worker,
                args=(stop_flag_key,),
                daemon=True
            )
            self.detection_thread.start()
            
            # REDUCED LOGGING FREQUENCY
            log_interval = 30  # Log every 30 frames instead of 10
            
            # MAIN REAL-TIME DISPLAY LOOP - Only handles display and metrics
            while not stop_detection_flags.get(stop_flag_key, False) and self.is_running:
                current_time = time.time()
                
                # Calculate expected frame time for RTMP timing display
                expected_frame_time = stream_start_time + (frame_count * frame_interval)
                current_lag = current_time - expected_frame_time
                
                # Get latest frame from real-time reader
                frame = self.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)  # Minimal wait for frame
                    continue
                
                frame_count += 1
                last_frame_time = current_time
                
                # Detect actual FPS from stream timing - REDUCED FREQUENCY
                if len(frame_timing_history) < 10:  # Reduced for faster adaptation
                    frame_timing_history.append(current_time)
                else:
                    frame_timing_history.pop(0)
                    frame_timing_history.append(current_time)
                    if len(frame_timing_history) >= 2:
                        actual_fps = len(frame_timing_history) / (frame_timing_history[-1] - frame_timing_history[0])
                        if 15 <= actual_fps <= 60:
                            target_fps = actual_fps
                            frame_interval = 1.0 / target_fps
                
                # Get latest detection result from real-time worker
                processed_frame = self.get_latest_result()
                if processed_frame is None:
                    # No detection result yet, use original frame
                    processed_frame = frame
                
                # Calculate frames behind
                frames_behind = int(current_lag / frame_interval) if frame_interval > 0 else 0
                
                processed_count += 1
                
                # Calculate and emit performance metrics - REDUCED FREQUENCY
                if processed_count % log_interval == 0:
                    current_time = time.time()
                    actual_fps = processed_count / (current_time - stream_start_time)
                    
                    # Recalculate final lag
                    final_lag = current_time - expected_frame_time
                    
                    # GPU monitoring - SIMPLIFIED
                    gpu_info = {}
                    if torch.cuda.is_available():
                        try:
                            gpu_info = {
                                'gpu_used': True,
                                'gpu_name': torch.cuda.get_device_name(0),
                                'gpu_memory_used': f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                                'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
                                'gpu_memory_percent': f"{(torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100:.1f}%"
                            }
                            # Try to get GPU utilization if pynvml is available
                            try:
                                import pynvml
                                pynvml.nvmlInit()
                                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_info['gpu_utilization'] = f"{utilization.gpu}%"
                                gpu_info['gpu_memory_utilization'] = f"{utilization.memory}%"
                            except:
                                gpu_info['gpu_utilization'] = "N/A"
                                gpu_info['gpu_memory_utilization'] = "N/A"
                        except Exception as e:
                            gpu_info = {
                                'gpu_used': True,
                                'gpu_name': 'GPU Available',
                                'error': str(e)
                            }
                    else:
                        gpu_info = {
                            'gpu_used': False,
                            'message': 'CUDA not available'
                        }
                    
                    # Determine sync status
                    sync_status = "🟢 SYNC" if final_lag <= max_lag_threshold else "🔴 LAG"
                    
                    # Add ultra-low latency processing info
                    frame_age = self.get_frame_age()
                    mode_str = "RAW" if self.raw_stream_mode else "DET"
                    method_str = "FFM" if self.use_ffmpeg_direct else "OCV"
                    processing_status = "PROCESSING" if self.detection_processing else "READY"
                    parallel_info = f"{mode_str}+{method_str}: {frame_age:.1f}ms | {processing_status}"
                    
                    socketio.emit('fps_update', {
                        'fps': f'{actual_fps:.1f}', 
                        'stream_path': stop_flag_key,
                        'lag': f'{final_lag:.3f}s',  # Show 3 decimal places for precision
                        'avg_process_time': f'Parallel',  # Parallel processing
                        'gpu_info': gpu_info,
                        'rtmp_fps': f'{target_fps:.1f}',
                        'frames_behind': frames_behind,
                        'sync_status': sync_status,
                        'max_lag': f'{max_lag_threshold}s',
                        'buffer_info': parallel_info
                    })
                
                # ULTRA-FAST FRAME ENCODING AND SENDING WITH AGGRESSIVE DROP
                try:
                    # Get current frame age and drop if too old
                    frame_age = self.get_frame_age()
                    if frame_age > 50:  # Drop frames older than 50ms
                        if processed_count % 30 == 0:  # Log drops occasionally
                            print(f"🗑️ DROP: Frame too old ({frame_age:.1f}ms)")
                        continue
                    
                    # Resize frame for web display (optimized)
                    height, width = processed_frame.shape[:2]
                    max_width = 640
                    if width > max_width:
                        scale = max_width / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        processed_frame = cv2.resize(processed_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                    
                    # Encode to JPEG with optimized settings for real-time
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, 70,  # Reduced quality for speed
                        cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # No optimization for speed
                        cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # No progressive for speed
                    ]
                    _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame to web client with server timestamp
                    server_timestamp = time.time() * 1000  # Current time in milliseconds
                    socketio.emit('frame', {
                        'image': frame_data, 
                        'stream_path': stop_flag_key,
                        'timestamp': server_timestamp,
                        'frame_age_ms': frame_age
                    })
                    
                    # Debug: Log successful frame sends - reduced frequency
                    if processed_count % 120 == 0:  # Log every 120th frame
                        print(f"📤 ULTRA-LOW Frame sent: {processed_frame.shape} -> {len(frame_data)} chars, age: {frame_age:.1f}ms")
                    
                except Exception as e:
                    # MINIMAL ERROR LOGGING
                    if processed_count % 300 == 0:  # Log errors every 300 frames
                        print(f"❌ Frame error: {e}")
                
                # NO SLEEP - Maximum real-time speed
                
        except Exception as e:
            print(f"❌ Detection error: {e}")
            socketio.emit('error', {'message': f'Detection error: {str(e)}', 'stream_path': stop_flag_key})
        
        finally:
            self.cleanup()
    
    def _frame_reader_worker(self, rtmp_url, stop_flag_key):
        """Worker thread for reading frames from RTMP stream - REAL-TIME MODE."""
        global stop_detection_flags
        
        print(f"📸 REAL-TIME Frame reader started for: {rtmp_url}")
        frame_count = 0
        
        while not stop_detection_flags.get(stop_flag_key, False) and self.is_running:
            try:
                # Read frame from RTMP stream
                ret, frame = self.cap.read()
                
                if not ret:
                    # Check if stream is still open
                    if not self.cap.isOpened():
                        print(f"🔄 Frame reader reconnecting: {rtmp_url}")
                        if not self.open_stream(rtmp_url):
                            time.sleep(0.1)
                            continue
                    else:
                        time.sleep(0.001)  # Minimal delay if no frame
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # AGGRESSIVE FRAME REPLACEMENT - ALWAYS LATEST ONLY
                with self.frame_lock:
                    # Check if frame is fresh enough to use
                    if self.frame_timestamp > 0:
                        frame_gap = current_time - self.frame_timestamp
                        if frame_gap < 0.01:  # Skip if frames coming too fast (< 10ms gap)
                            continue
                    
                    self.latest_raw_frame = frame  # Direct replacement, no copy for speed
                    self.frame_timestamp = current_time
                
                # Debug logging - reduced frequency
                if frame_count % 60 == 0:  # Every 60 frames instead of 30
                    print(f"📸 REAL-TIME: {frame_count} frames, skipped: {self.detection_skip_count}")
                
            except Exception as e:
                print(f"❌ Frame reader error: {e}")
                time.sleep(0.01)
        
        print(f"📸 REAL-TIME Frame reader stopped for: {rtmp_url}")
    
    def _detection_worker(self, stop_flag_key):
        """Worker thread for processing frames with detection - ULTRA LOW LATENCY MODE."""
        global stop_detection_flags
        
        mode_str = "RAW STREAM" if self.raw_stream_mode else "DETECTION"
        print(f"🚀 {mode_str} worker started for: {stop_flag_key}")
        processed_count = 0
        
        while not stop_detection_flags.get(stop_flag_key, False) and self.is_running:
            try:
                # Check if already processing - if so, skip this cycle (only for detection mode)
                if not self.raw_stream_mode and self.detection_processing:
                    self.detection_skip_count += 1
                    time.sleep(0.001)  # Minimal wait
                    continue
                
                # Get latest frame if available
                frame_to_process = None
                with self.frame_lock:
                    if self.latest_raw_frame is not None:
                        if self.raw_stream_mode:
                            # Raw mode: use frame directly, no copy
                            frame_to_process = self.latest_raw_frame
                        else:
                            # Detection mode: copy frame for processing
                            frame_to_process = self.latest_raw_frame.copy()
                            self.detection_processing = True  # Set processing flag
                
                if frame_to_process is None:
                    time.sleep(0.001)  # Minimal wait if no frame
                    continue
                
                # Process frame based on mode
                if self.raw_stream_mode:
                    # Raw stream mode: no processing, just pass through
                    processed_frame = frame_to_process
                else:
                    # Detection mode: run object detection
                    processed_frame = self.process_frame(frame_to_process)
                
                # Update latest result - ALWAYS replace with newest
                with self.frame_lock:
                    self.latest_processed_frame = processed_frame
                    if not self.raw_stream_mode:
                        self.detection_processing = False  # Clear processing flag
                
                processed_count += 1
                
                # Debug logging - reduced frequency
                if processed_count % 60 == 0:  # Every 60 frames instead of 30
                    skip_info = f", {self.detection_skip_count} skipped" if not self.raw_stream_mode else ""
                    print(f"🚀 {mode_str}: {processed_count} processed{skip_info}")
                
            except Exception as e:
                print(f"❌ {mode_str} worker error: {e}")
                if not self.raw_stream_mode:
                    with self.frame_lock:
                        self.detection_processing = False  # Clear flag on error
                time.sleep(0.01)
        
        print(f"🚀 {mode_str} worker stopped for: {stop_flag_key}")
    
    def get_latest_frame(self):
        """Get the latest raw frame with thread safety."""
        with self.frame_lock:
            return self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None
    
    def get_latest_result(self):
        """Get the latest detection result with thread safety."""
        with self.frame_lock:
            return self.latest_processed_frame.copy() if self.latest_processed_frame is not None else None
    
    def get_frame_age(self):
        """Get age of current frame in milliseconds."""
        with self.frame_lock:
            if self.frame_timestamp > 0:
                return (time.time() - self.frame_timestamp) * 1000
            return 0
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        
        # Stop parallel threads
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            self.frame_reader_thread.join(timeout=1)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1)
        
        # Clear latest frames
        with self.frame_lock:
            self.latest_raw_frame = None
            self.latest_processed_frame = None
            self.detection_processing = False
        
        # Clean up FFmpeg process
        self._cleanup_ffmpeg()
        
        # Release capture
        if self.cap:
            self.cap.release()
        
        socketio.emit('status', {'message': 'Detection stopped'})

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start detection on RTMP stream (multi-stream support)."""
    global stream_processors, detection_threads, stop_detection_flags, current_configs
    try:
        data = request.get_json()
        rtmp_url = data.get('rtmp_url')
        stream_path = data.get('stream_path')  # unique key for the stream
        confidence = float(data.get('confidence', 0.25))
        iou = float(data.get('iou', 0.45))
        model = data.get('model', 'data/other_models/default_model/yolov8n.pt')
        boat_only_filter = data.get('boat_only_filter', True)  # Default to boat-only filtering
        raw_stream_mode = data.get('raw_stream_mode', False)  # Ultra-low latency mode
        use_ffmpeg_direct = data.get('use_ffmpeg_direct', True)  # Use FFmpeg by default
        
        # Debug logging
        mode_str = "RAW STREAM" if raw_stream_mode else "DETECTION"
        method_str = "FFMPEG" if use_ffmpeg_direct else "OPENCV"
        print(f"🚀 Starting {mode_str} mode with {method_str} for: {stream_path}")
        
        if not rtmp_url or not stream_path:
            return jsonify({'error': 'RTMP URL and stream_path are required'}), 400
            
        # Check if this stream is already running
        if stream_path in stream_processors and stream_processors[stream_path].is_running:
            print(f"⚠️ Stream {stream_path} is already running, skipping duplicate start")
            return jsonify({'message': f'Stream {stream_path} is already running'}), 200
            
        # Stop any existing detection for this stream
        stop_detection_flags[stream_path] = True
        if stream_path in detection_threads and detection_threads[stream_path].is_alive():
            detection_threads[stream_path].join(timeout=2)
        # Clean up previous processor for this stream
        if stream_path in stream_processors:
            stream_processors[stream_path].cleanup()
        # Update configuration for this stream
        config = {
            'rtmp_url': rtmp_url,
            'confidence': confidence,
            'iou': iou,
            'model': model,
            'enable_tracking': True,
            'show_depth': True,
            'enable_bev': True,
            'depth_model': 'midas',
            'depth_size': 'small',
            'device': 'auto',
            'depth_range': [1.0, 50.0],
            'smooth_depth': True,
            'filter_boat_only': boat_only_filter,
            'raw_stream_mode': raw_stream_mode,  # Ultra-low latency mode
            'use_ffmpeg_direct': use_ffmpeg_direct  # FFmpeg direct method
        }
        current_configs[stream_path] = config
        # Start new detection thread for this stream
        stop_detection_flags[stream_path] = False
        processor = StreamProcessor()
        processor.raw_stream_mode = raw_stream_mode
        processor.use_ffmpeg_direct = use_ffmpeg_direct
        stream_processors[stream_path] = processor
        detection_threads[stream_path] = threading.Thread(
            target=processor.run_detection,
            args=(rtmp_url, config, stream_path)
        )
        detection_threads[stream_path].daemon = True
        detection_threads[stream_path].start()
        
        mode_msg = "Raw stream" if raw_stream_mode else "Detection"
        return jsonify({'message': f'{mode_msg} started for {stream_path}'})
    except Exception as e:
        return jsonify({'error': f'Failed to start detection: {str(e)}'}), 500

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection_api():
    """Stop detection for a specific stream."""
    global stop_detection_flags, stream_processors, detection_threads
    try:
        data = request.get_json()
        stream_path = data.get('stream_path')
        if not stream_path:
            return jsonify({'error': 'stream_path is required'}), 400
        stop_detection_flags[stream_path] = True
        if stream_path in detection_threads and detection_threads[stream_path].is_alive():
            detection_threads[stream_path].join(timeout=2)
        if stream_path in stream_processors:
            stream_processors[stream_path].cleanup()
        return jsonify({'message': f'Detection stopped for {stream_path}'})
    except Exception as e:
        return jsonify({'error': f'Failed to stop detection: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current detection status."""
    global stream_processors
    
    return jsonify({
        'is_running': [processor.is_running for processor in stream_processors.values()],
        'detection_available': DETECTION_AVAILABLE
    })

@app.route('/api/available_streams', methods=['GET'])
def available_streams():
    """Fetch available RTMP streams from the simulator."""
    try:
        # Get base URL from query parameter or use default
        base_url = request.args.get('base_url', 'http://192.168.128.209:5000')
        
        # Parse the base URL to extract the server domain and port
        if base_url.startswith('http://'):
            server_part = base_url.replace('http://', '')
        elif base_url.startswith('https://'):
            server_part = base_url.replace('https://', '')
        elif base_url.startswith('rtmp://'):
            server_part = base_url.replace('rtmp://', '')
        elif base_url.startswith('rtmps://'):
            server_part = base_url.replace('rtmps://', '')
        else:
            server_part = base_url
        
        # Remove trailing slash and any path
        server_part = server_part.split('/')[0]
        
        # Split server part into domain and port
        if ':' in server_part:
            server_domain, port = server_part.split(':', 1)
        else:
            server_domain = server_part
            port = '5000'  # Default port if not specified
        
        # Construct HTTP API URL for stream parameters (with port)
        stream_params_url = f"http://{server_domain}:{port}/stream_params"
        
        print(f"Fetching streams from: {stream_params_url}")
        
        resp = requests.get(stream_params_url, timeout=10)
        resp.raise_for_status()
        streams = resp.json()
        
        # Add server domain (without port) to each stream for RTMP URL construction
        for stream in streams:
            stream['server_domain'] = server_domain  # RTMP URLs will use domain without port
        
        return jsonify({'streams': streams})
    except Exception as e:
        print(f"Error fetching streams: {e}")
        return jsonify({'error': f'Failed to fetch streams: {str(e)}'}), 500

@app.route('/api/gpu_status', methods=['GET'])
def get_gpu_status():
    """Get GPU availability and status."""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_info = {}
        
        if gpu_available:
            gpu_info = {
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            }
        else:
            gpu_info = {
                'gpu_available': False,
                'message': 'CUDA not available, using CPU'
            }
        
        return jsonify(gpu_info)
    except Exception as e:
        return jsonify({
            'gpu_available': False,
            'error': str(e)
        }), 500

@app.route('/api/start_raw_stream', methods=['POST'])
def start_raw_stream():
    """Start ultra-low latency raw stream (no detection)."""
    try:
        data = request.get_json()
        data['raw_stream_mode'] = True
        data['use_ffmpeg_direct'] = True
        return start_detection()  # Reuse the main detection endpoint
    except Exception as e:
        return jsonify({'error': f'Failed to start raw stream: {str(e)}'}), 500

@app.route('/api/ffmpeg_status', methods=['GET'])
def get_ffmpeg_status():
    """Check if FFmpeg is available on the system."""
    try:
        # Test FFmpeg availability
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            return jsonify({
                'ffmpeg_available': True,
                'version': version_line,
                'recommended': True
            })
        else:
            return jsonify({
                'ffmpeg_available': False,
                'message': 'FFmpeg not working properly'
            })
            
    except FileNotFoundError:
        return jsonify({
            'ffmpeg_available': False,
            'message': 'FFmpeg not installed'
        })
    except Exception as e:
        return jsonify({
            'ffmpeg_available': False,
            'error': str(e)
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    print("Starting Object Detection Dashboard...")
    print(f"Detection modules available: {DETECTION_AVAILABLE}")
    print("Open your browser and go to: http://localhost:7080")
    
    # Use debug=False for production/Docker environment
    socketio.run(app, host='0.0.0.0', port=7080, debug=False, allow_unsafe_werkzeug=True) 