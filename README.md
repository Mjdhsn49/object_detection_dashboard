# YOLO-3D Marine Object Detection System

A comprehensive 3D object detection and depth estimation system optimized for marine applications, combining YOLOv8 for object detection and MiDaS (Intel ISL) for depth estimation.

## Project Structure

```
yolo3d/
├── src/                    # Core source code
│   ├── models/            # Model implementations
│   │   ├── detection_model.py  # YOLOv8 detection model
│   │   ├── depth_model.py      # MiDaS depth estimation
│   │   └── __init__.py
│   ├── utils/             # Utility functions
│   └── __init__.py
├── scripts/               # Running scripts
│   └── run_detection.py   # Main detection script
├── configs/              # Configuration files
├── data/                 # Data directory
│   ├── camera_1/        # Input images directory
│   └── models/          # Custom model weights (optional)
├── output/              # Output directory for results
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Features

- **Object Detection**: Uses YOLOv8 for accurate object detection in marine environments
- **Depth Estimation**: Implements MiDaS (Intel ISL) for precise depth estimation, optimized for marine environments (1-50m range)
- **Real-time Tracking**: Includes object tracking with unique IDs
- **3D Visualization**: Provides bird's eye view visualization
- **GPU Acceleration**: Full CUDA support for faster processing
- **Docker Support**: Easy deployment with Docker containers
- **Model Flexibility**: Support for different YOLOv8 and MiDaS model variants
- **Public Access**: via localtunnel

## Model Options

### YOLOv8 Model Variants
The system supports different YOLOv8 model sizes:
- `yolov8n.pt`: Nano model (default)
- `yolov8s.pt`: Small model
- `yolov8m.pt`: Medium model
- `yolov8l.pt`: Large model
- `yolov8x.pt`: Extra large model

### MiDaS Depth Models
The system supports two MiDaS model variants:
- `small`: MiDaS Small model (faster, lower memory usage)
- `large`: DPT Large model (more accurate, higher memory usage)

### Using Custom Models
You can use your own fine-tuned YOLOv8 model:

1. Place your custom model in `data/models/` directory
2. Run with custom model path:
   ```bash
   # Using Docker
   docker-compose run --rm yolo3d python scripts/run_detection.py --source data/camera_1 --output output.mp4 --model data/models/your_custom_model.pt

   # Without Docker
   python scripts/run_detection.py --source data/camera_1 --output output.mp4 --model data/models/your_custom_model.pt
   ```

### Downloading Models
- Default models are automatically downloaded on first run
- You can pre-download specific models:
  ```python
  from ultralytics import YOLO
  
  # Download specific model
  model = YOLO('yolov8s.pt')  # small model
  model = YOLO('yolov8m.pt')  # medium model
  ```

## Requirements

### Hardware Requirements
- CUDA-capable GPU (recommended) or CPU
- Minimum 8GB RAM (16GB recommended)
- Sufficient storage for models and data

### Software Requirements
- Docker and Docker Compose (recommended)
- NVIDIA Container Toolkit (for GPU support)
- Python 3.8+ (if running without Docker)
- CUDA 11.7+ and cuDNN (if running without Docker with GPU)

## Installation

### Option 1: Using Docker (Recommended)

1. Install Docker and NVIDIA Container Toolkit:
   ```bash
   # For Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yolo3d
   ```

3. Place your input images:
   ```bash
   # Place your images in the data/camera_1 directory
   ```

4. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

### Option 2: Manual Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yolo3d
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Input Format
- Place input images in `data/camera_1/`
- Supported formats: JPG, JPEG, PNG
- Images should be in sequence (processed in alphabetical order)

### Command Line Arguments
```bash
python scripts/run_detection.py [OPTIONS]

Options:
  Input/Output:
    --source PATH          Path to input images directory (default: data/camera_1)
    --output PATH         Path to output video file (default: output.mp4)

  Model Configuration:
    --model PATH          Path to YOLOv8 model or model size (n/s/m/l/x)
    --depth-model SIZE    MiDaS model size (small/large) (default: small)
    --conf FLOAT          Detection confidence threshold (default: 0.25)
    --iou FLOAT          IoU threshold for NMS (default: 0.45)
    --device DEVICE       Device to run on (cuda device=0/1/2/etc. or cpu)

  Feature Toggles:
    --no-track           Disable object tracking
    --no-bev            Disable bird's eye view visualization
    --no-depth          Disable depth visualization
    --no-smooth         Disable depth map smoothing

  Visualization:
    --show-fps          Show FPS counter
    --hide-labels       Hide object labels
    --hide-conf         Hide confidence scores
    --depth-range MIN MAX  Set depth visualization range in meters (default: 1 50)
```

### Example Commands

1. Using default models (YOLOv8n + MiDaS small):
   ```bash
   python scripts/run_detection.py --source data/camera_1 --output output.mp4
   ```

2. Using larger models for better accuracy:
   ```bash
   python scripts/run_detection.py --source data/camera_1 --output output.mp4 --model yolov8l --depth-model large --conf 0.3 --iou 0.5
   ```

3. Using custom YOLOv8 model with depth range optimization:
   ```bash
   python scripts/run_detection.py --source data/camera_1 --output output.mp4 --model data/models/custom_marine.pt --depth-range 1 30
   ```

4. Maximum performance configuration:
   ```bash
   python scripts/run_detection.py --source data/camera_1 --output output.mp4 --model yolov8n --depth-model small --no-smooth --hide-labels --hide-conf --device cuda:0
   ```

5. Maximum accuracy configuration:
   ```bash
   python scripts/run_detection.py --source data/camera_1 --output output.mp4 --model yolov8x --depth-model large --conf 0.3 --iou 0.5 --device cuda:0
   ```

### Output
The system generates:
- Processed video (`output.mp4`)
- Real-time visualization windows:
  - Object detection with tracking IDs
  - Depth map visualization
  - Bird's eye view representation

### Docker Volume Mounts
- Input: `./data/camera_1:/app/data/camera_1`
- Output: `./output:/app/output`
- Models: `./data/models:/app/data/models` (for custom models)

## Troubleshooting

### Common Issues and Solutions

1. CUDA/GPU Issues:
   - Verify NVIDIA drivers are installed
   - Check CUDA version compatibility
   - Try CPU mode if GPU issues persist

2. Image Loading Issues:
   - Verify image format compatibility
   - Check file permissions
   - Confirm correct directory path

3. Memory Issues:
   - Reduce batch size
   - Use smaller model variants
   - Close other GPU-intensive applications

4. Docker Issues:
   - Ensure NVIDIA Container Toolkit is installed
   - Verify Docker daemon is running
   - Check GPU availability with `nvidia-smi`

## Performance Optimization

1. GPU Acceleration:
   - Use CUDA-capable GPU
   - Ensure proper NVIDIA drivers
   - Set appropriate batch sizes

2. Processing Speed:
   - Adjust image resolution
   - Use smaller model variants
   - Enable GPU acceleration

3. Model Selection:
   - Choose model size based on your needs:
     * yolov8n.pt: Fastest but less accurate
     * yolov8s.pt: Good balance for most cases
     * yolov8l.pt/yolov8x.pt: Most accurate but slower

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- MiDaS depth estimation
- NVIDIA for CUDA support
- OpenCV community