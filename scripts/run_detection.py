#!/usr/bin/env python3
"""
Main script for running 3D object detection and depth estimation.
Supports image folders, video files, and RTMP streams.
"""

import os
# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import argparse
import torch
import re

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import ObjectDetector, DepthEstimator
from src.utils.bbox3d_utils import BBox3DEstimator, BirdEyeView
from src.utils.camera_utils import load_camera_params, apply_camera_params_to_estimator
from configs.default_config import Config

def is_rtmp_url(source):
    """Check if the source is an RTMP URL."""
    rtmp_patterns = [
        r'^rtmp://',
        r'^rtmps://',
        r'^rtsp://',
        r'^http://',
        r'^https://',
        r'^udp://',
        r'^tcp://'
    ]
    return any(re.match(pattern, source) for pattern in rtmp_patterns)

def is_video_file(source):
    """Check if the source is a video file."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    return any(source.lower().endswith(ext) for ext in video_extensions)

def is_image_folder(source):
    """Check if the source is a folder containing images."""
    if os.path.isdir(source):
        image_files = glob(os.path.join(source, "*.[jJ][pP][gG]")) + \
                     glob(os.path.join(source, "*.[pP][nN][gG]")) + \
                     glob(os.path.join(source, "*.[jJ][pP][eE][gG]"))
        return len(image_files) > 0
    return False

def parse_args():
    parser = argparse.ArgumentParser(description='3D Object Detection')
    
    # Input/Output
    parser.add_argument('--source', type=str, default='data/camera_1',
                        help='Path to input folder containing images, video file, or RTMP stream URL')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Path to output video file')
    
    # Model Configuration
    parser.add_argument('--model', type=str, default='yolov8n',
                        help='Path to YOLOv8 model or model size (n/s/m/l/x)')
    parser.add_argument('--depth-model', type=str, default='midas',
                        help='Depth model type (midas/depthanything)')
    parser.add_argument('--depth-size', type=str, default='small',
                        help='Depth model size (small/large for MiDaS, small/base for Depth Anything)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda device=0/1/2/etc. or cpu)')
    
    # Feature Toggles
    parser.add_argument('--no-track', action='store_true',
                        help='Disable object tracking')
    parser.add_argument('--no-bev', action='store_true',
                        help='Disable bird\'s eye view visualization')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth visualization')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable depth map smoothing')
    
    # Visualization
    parser.add_argument('--show-fps', action='store_true',
                        help='Show FPS counter')
    parser.add_argument('--hide-labels', action='store_true',
                        help='Hide object labels')
    parser.add_argument('--hide-conf', action='store_true',
                        help='Hide confidence scores')
    parser.add_argument('--depth-range', nargs=2, type=float, default=[1.0, 50.0],
                        metavar=('MIN', 'MAX'),
                        help='Set depth visualization range in meters')
    
    # Stream-specific options
    parser.add_argument('--buffer-size', type=int, default=1024,
                        help='Buffer size for RTMP streams')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout for stream connection in seconds')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (None for unlimited)')
    
    return parser.parse_args()

def open_video_source(source, buffer_size=1024, timeout=30):
    """Open video source (file or stream)."""
    cap = cv2.VideoCapture()
    
    # Set buffer size for streams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    
    # Set timeout for streams
    if is_rtmp_url(source):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
    
    # Open the source
    if not cap.open(source):
        raise RuntimeError(f"Failed to open video source: {source}")
    
    return cap

def main():
    """Main function."""
    args = parse_args()
    config = Config()
    
    # Update config with command line arguments
    config.ENABLE_TRACKING = not args.no_track
    config.ENABLE_BEV = not args.no_bev
    config.SHOW_DEPTH = not args.no_depth
    config.SHOW_FPS = args.show_fps
    config.SHOW_LABELS = not args.hide_labels
    config.SHOW_CONF = not args.hide_conf
    config.IOU_THRESHOLD = args.iou
    config.DEPTH_RANGE = args.depth_range
    config.SMOOTH_DEPTH = not args.no_smooth
    
    # Determine device from arguments or auto-detect
    if args.device:
        device = args.device
        print(f"\nUsing specified device: {device}")
    else:
        if torch.cuda.is_available():
            device = 'cuda'
            print("\nCUDA is available! Using GPU for all models.")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            # Set CUDA settings for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = 'cpu'
            print("\nCUDA is not available. Using CPU for all models.")
    
    config.DEVICE = device
    
    # Initialize models
    print("\nInitializing models...")
    try:
        detector = ObjectDetector(
            model_size=args.model,
            conf_thres=args.conf,
            iou_thres=config.IOU_THRESHOLD,
            classes=config.CLASSES,
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=args.model,
            conf_thres=args.conf,
            iou_thres=config.IOU_THRESHOLD,
            classes=config.CLASSES,
            device='cpu'
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_type=args.depth_model,
            model_size=args.depth_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_type=args.depth_model,
            model_size=args.depth_size,
            device='cpu'
        )
    
    # Initialize 3D bounding box estimator
    bbox3d_estimator = BBox3DEstimator()
    
    # Initialize Bird's Eye View if enabled
    if config.ENABLE_BEV:
        bev = BirdEyeView(scale=config.BEV_SCALE, size=config.BEV_SIZE)
    
    # Determine source type and open accordingly
    print(f"\nAnalyzing source: {args.source}")
    
    if is_rtmp_url(args.source):
        print("Detected RTMP/stream URL")
        try:
            cap = open_video_source(args.source, args.buffer_size, args.timeout)
            print(f"Successfully connected to stream: {args.source}")
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            return
        source_type = "stream"
        
    elif is_video_file(args.source):
        print("Detected video file")
        try:
            cap = open_video_source(args.source)
            print(f"Successfully opened video file: {args.source}")
        except Exception as e:
            print(f"Error opening video file: {e}")
            return
        source_type = "video"
        
    elif is_image_folder(args.source):
        print("Detected image folder")
        image_files = sorted(glob(os.path.join(args.source, "*.[jJ][pP][gG]")) + 
                            glob(os.path.join(args.source, "*.[pP][nN][gG]")) +
                            glob(os.path.join(args.source, "*.[jJ][pP][eE][gG]")))
        
        if not image_files:
            print(f"Error: No images found in folder {args.source}")
            return
        
        print(f"Found {len(image_files)} images")
        source_type = "images"
        cap = None
        
    else:
        print(f"Error: Unsupported source type: {args.source}")
        print("Supported sources: image folders, video files, RTMP/HTTP streams")
        return
    
    # Get video properties
    if source_type in ["stream", "video"]:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default FPS if not available
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "video" else None
        
        print(f"Video properties: {width}x{height}, {fps:.2f} FPS")
        if total_frames:
            print(f"Total frames: {total_frames}")
    else:
        # For image folders, read first image to get dimensions
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            print(f"Error: Could not read first image {image_files[0]}")
            return
        height, width = first_frame.shape[:2]
        fps = 30
        total_frames = len(image_files)
        print(f"Image properties: {width}x{height}, {len(image_files)} images")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("Starting processing...")
    
    # Main loop
    try:
        while True:
            # Check if we've reached the maximum frame limit
            if args.max_frames and frame_count >= args.max_frames:
                print(f"\nReached maximum frame limit: {args.max_frames}")
                break
            
            # Read frame based on source type
            if source_type in ["stream", "video"]:
                ret, frame = cap.read()
                if not ret:
                    if source_type == "stream":
                        print("\nStream ended or connection lost. Attempting to reconnect...")
                        cap.release()
                        time.sleep(2)
                        try:
                            cap = open_video_source(args.source, args.buffer_size, args.timeout)
                            continue
                        except Exception as e:
                            print(f"Failed to reconnect: {e}")
                            break
                    else:
                        print("\nEnd of video file")
                        break
            else:
                # Image folder processing
                if frame_count >= len(image_files):
                    print("\nProcessed all images")
                    break
                
                image_path = image_files[frame_count]
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Error reading image: {image_path}")
                    frame_count += 1
                    continue
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                fps_display = f"FPS: {current_fps:.1f}"
            
            # Display progress
            if source_type == "images":
                print(f"\rProcessing image {frame_count}/{total_frames}: {os.path.basename(image_files[frame_count-1])}", end="")
            else:
                print(f"\rProcessing frame {frame_count}", end="")
            
            # Process frame
            result_frame = process_frame(frame, detector, depth_estimator, bbox3d_estimator, 
                                      bev if config.ENABLE_BEV else None, config)
            
            # Add FPS display if enabled
            if config.SHOW_FPS:
                cv2.putText(result_frame, fps_display, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(result_frame)
            
            # Display frames
            cv2.imshow("3D Object Detection", result_frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                print("\nExiting program...")
                break
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProcessing complete!")
    
    # Clean up
    print(f"Cleaning up resources...")
    if cap is not None:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {args.output}")

def process_frame(frame, detector, depth_estimator, bbox3d_estimator, bev, config):
    """Process a single frame."""
    # Make copies for different visualizations
    original_frame = frame.copy()
    detection_frame = frame.copy()
    depth_frame = frame.copy()
    result_frame = frame.copy()
    
    print("\nProcessing new frame...")
    print(f"Frame shape: {frame.shape}")
    print(f"Frame type: {frame.dtype}")
    
    # Step 1: Object Detection
    try:
        print("\nRunning object detection...")
        detection_frame, detections = detector.detect(detection_frame, track=config.ENABLE_TRACKING)
        print(f"Detection complete. Found {len(detections)} objects.")
    except Exception as e:
        print(f"Error during object detection: {e}")
        import traceback
        traceback.print_exc()
        detections = []
        cv2.putText(detection_frame, "Detection Error", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Step 2: Depth Estimation
    try:
        print("\nRunning depth estimation...")
        depth_map = depth_estimator.estimate_depth(
            original_frame,
            depth_range=config.DEPTH_RANGE,
            smooth=config.SMOOTH_DEPTH
        )
        depth_colored = depth_estimator.colorize_depth(
            depth_map,
            depth_range=config.DEPTH_RANGE
        )
        print("Depth estimation complete.")
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        import traceback
        traceback.print_exc()
        depth_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        depth_colored = np.zeros_like(frame)
        cv2.putText(depth_colored, "Depth Error", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Step 3: Process detections
    boxes_3d = []
    active_ids = []
    
    for detection in detections:
        try:
            bbox, score, class_id, obj_id = detection
            class_name = detector.get_class_names()[class_id]
            
            # Get depth in the region
            depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
            
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
    bbox3d_estimator.cleanup_trackers(active_ids)
    
    # Step 4: Visualization
    result_frame = visualize_results(result_frame, boxes_3d, depth_colored, bev, config)
    
    return result_frame

def visualize_results(frame, boxes_3d, depth_colored, bev, config):
    """Visualize detection results."""
    height, width = frame.shape[:2]
    
    # Draw boxes
    for box_3d in boxes_3d:
        try:
            # Use green color for all boxes
            color = (0, 255, 0)  # Green
            
            # Get box information
            bbox = box_3d['bbox_2d']
            depth = box_3d['depth_value']
            obj_id = box_3d['object_id']
            score = box_3d['score']
            
            # Draw the bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Format text to show ID and score
            # Always show score, and show ID if available
            if obj_id is not None:
                text = f"ID:{int(obj_id)} {score:.2f}"
            else:
                text = f"{score:.2f}"
            
            # Draw text with background - moved slightly higher for better visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, 
                        (x1, y1 - text_size[1] - 8), 
                        (x1 + text_size[0] + 4, y1 - 4), 
                        color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 6), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw depth value with better visibility
            if config.SHOW_DEPTH:
                depth_text = f"{depth:.1f}m"
                text_size = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame,
                            (x1, y2 + 2),
                            (x1 + text_size[0], y2 + text_size[1] + 6),
                            color, -1)
                cv2.putText(frame, depth_text, (x1, y2 + text_size[1] + 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        except Exception as e:
            print(f"Error drawing box: {e}")
            continue
    
    # Draw Bird's Eye View
    if bev is not None:
        try:
            bev.reset()
            for box_3d in boxes_3d:
                bev.draw_box(box_3d)
            bev_image = bev.get_image()
            
            bev_height = height // 4
            bev_width = bev_height
            
            if bev_height > 0 and bev_width > 0:
                bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                frame[height - bev_height:height, 0:bev_width] = bev_resized
                cv2.rectangle(frame, 
                            (0, height - bev_height), 
                            (bev_width, height), 
                            (255, 255, 255), 1)
                cv2.putText(frame, "Bird's Eye View", 
                           (10, height - bev_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error drawing BEV: {e}")
    
    # Add depth visualization
    try:
        depth_height = height // 4
        depth_width = int(depth_height * width / height)
        depth_resized = cv2.resize(depth_colored, (depth_width + 30, depth_height))  # +30 for the legend
        frame[0:depth_height, 0:depth_width + 30] = depth_resized
    except Exception as e:
        print(f"Error adding depth map: {e}")
    
    return frame

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows() 