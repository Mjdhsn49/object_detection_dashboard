from ultralytics import YOLO
import cv2
import os
import logging
import torch
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_model(weights_path):
    """
    Initialize the YOLO model with specified weights
    """
    try:
        logger.info(f"Loading model from {weights_path}...")
        model = YOLO(weights_path)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def process_image(model, image_path, conf=0.25, save_dir='runs/detect'):
    """
    Process a single image and save results
    """
    try:
        logger.info(f"Processing image: {image_path}")
        # Get predictions
        results = model.predict(
            source=image_path,
            conf=conf,
            save=False
        )
        
        # Process each result
        for r in results:
            # Get the original image
            im_array = cv2.imread(image_path)
            
            # Get boxes
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class ID and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Draw box
                cv2.rectangle(im_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add class ID and confidence
                label = f"ID:{cls_id} {conf:.2f}"
                cv2.putText(im_array, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the image
            save_path = os.path.join(save_dir, 'predict', os.path.basename(image_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, im_array)
            
        logger.info(f"Results saved to {save_dir}")
        return results
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def process_video(model, video_path, conf=0.25, save_dir='runs/detect'):
    """
    Process a video file and save results
    """
    try:
        logger.info(f"Processing video: {video_path}")
        # Get predictions
        results = model.predict(
            source=video_path,
            conf=conf,
            save=False
        )
        
        # Process each result
        for r in results:
            # Get the original frame
            im_array = cv2.imread(video_path)  # For video, you'd need to handle frames differently
            
            # Get boxes
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class ID and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Draw box
                cv2.rectangle(im_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add class ID and confidence
                label = f"ID:{cls_id} {conf:.2f}"
                cv2.putText(im_array, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the frame
            save_path = os.path.join(save_dir, 'predict', os.path.basename(video_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, im_array)
            
        logger.info(f"Results saved to {save_dir}")
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Test YOLO model on images or videos')
    default_weights = str(Path('singapore-maritime-5/runs/train/singapore_yolov8m/weights/best.pt').absolute())
    parser.add_argument('--weights', type=str, default=default_weights,
                        help='path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='path to image or video file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='confidence threshold')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                        help='directory to save results')
    
    args = parser.parse_args()
    
    # Ensure CUDA is properly configured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    model = setup_model(args.weights)
    
    # Process input based on file type
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {args.source}")
    
    # Check if input is video or image
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if source_path.suffix.lower() in video_extensions:
        results = process_video(model, str(source_path), args.conf, args.save_dir)
    else:
        results = process_image(model, str(source_path), args.conf, args.save_dir)
    
    logger.info("Processing completed successfully!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise 