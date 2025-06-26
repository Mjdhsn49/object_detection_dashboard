# 🌐 Object Detection Web Dashboard

A modern web-based dashboard for running object detection on **multiple RTMP streams** with a beautiful, user-friendly interface.

## ✨ Features

- **Live Multi-Stream Processing**: Automatically detects and displays all available RTMP streams as separate panels
- **Object Detection**: Uses only local YOLOv8 models with configurable confidence thresholds
- **Depth Estimation**: MiDaS depth estimation with colorized visualization
- **Object Tracking**: Persistent object tracking across frames
- **FPS Monitoring**: Live performance metrics per stream
- **Modern Web Interface**: Responsive design with intuitive controls
- **Configurable Parameters**: Adjust model, confidence, IoU, and features per stream
- **Plug-and-Play**: Just start the backend and open the dashboard to see all available streams

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install web dashboard dependencies
pip install -r web_requirements.txt

# Or install manually
pip install Flask Flask-SocketIO opencv-python numpy
```

### 2. Start the Dashboard

```bash
# Option 1: Use the startup script (recommended)
python start_dashboard.py

# Option 2: Run directly
python web_dashboard.py
```

### 3. Open Your Browser

Navigate to: **http://localhost:7070** (or the port you configured)

## 🐳 Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- Model files in the correct locations (see Model Setup below)

### Docker Options

#### Option 1: Standard Docker Compose
```bash
# Build and run the dashboard
sudo docker-compose up --build

# Or run in background
sudo docker-compose up -d --build
```

#### Option 2: Public Access with Tunnel (Recommended for Remote Access)
```bash
# Run with tunnel for public access
sudo docker-compose -f docker-compose.tunnel.yml up --build
```

#### Option 3: Development Mode
```bash
# Run with source code mounted for development
sudo docker-compose -f docker-compose.dev.yml up --build
```

#### Option 4: Production with Nginx
```bash
# Run with nginx reverse proxy
sudo docker-compose --profile production up --build
```

### Docker Commands

```bash
# View logs
sudo docker-compose logs -f object-detection-dashboard

# Stop services
sudo docker-compose down

# Rebuild after changes
sudo docker-compose up --build

# Check service status
sudo docker-compose ps
```

### Model Setup for Docker

Ensure your model files are in the correct locations before running Docker:

```
yolo3d/
├── data/
│   └── other_models/
│       ├── default_model/
│       │   └── yolov8n.pt
│       ├── cross_model/
│       │   └── weights/
│       │       └── best.pt
│       └── Infrared/
│           └── weights/
│               └── best.onnx
└── yolov8n.pt
```

### Docker Troubleshooting

- **Port conflicts**: Change the port mapping in `docker-compose.yml` if 7070 is already in use
- **Model not found**: Ensure model files exist in the mounted directories
- **Permission issues**: Use `sudo` for Docker commands on Linux
- **Memory issues**: Increase Docker memory limits in Docker Desktop settings

For detailed Docker documentation, see [DOCKER_README.md](DOCKER_README.md).

## 📋 Usage Guide

### Multi-Stream Dashboard

- On page load, the dashboard automatically fetches all available streams from the backend (`/api/available_streams`).
- Each stream is displayed in a large video panel (480x360) with its own controls:
  - **Start Detection**: Begins detection for that stream
  - **Stop Detection**: Stops detection and clears the video panel
  - **Model, Confidence, IoU**: Adjustable per stream (choose from three local models)
  - **FPS**: Live performance for each stream
- No manual RTMP input or scenario selection is required—everything is automatic.

### Example Workflow

1. **Start the backend** (make sure it provides `/api/available_streams` and `/api/start_detection` endpoints)
2. **Open the dashboard in your browser**
3. **See all available streams as separate panels**
4. **Start/stop detection for any stream independently**
5. **When you stop a stream, its video panel is cleared**

## 🔧 Troubleshooting

- **No streams found**: Ensure your backend is running and `/api/available_streams` returns the expected list.
- **Video not clearing on stop**: The dashboard now clears the video panel when detection is stopped for a stream.
- **Streams not showing**: Check browser console and backend logs for errors.

## 📁 File Structure

```
yolo3d/
├── web_dashboard.py          # Main Flask application
├── start_dashboard.py        # Startup script with dependency checking
├── web_requirements.txt      # Web dashboard dependencies
├── templates/
│   └── dashboard.html        # Web interface template
├── src/                      # Detection modules
├── configs/                  # Configuration files
└── scripts/                  # Command-line scripts
```

## 🔌 API Endpoints

### REST API
- `GET /`: Main dashboard page
- `GET /api/available_streams`: List all available streams (used for multi-stream UI)
- `POST /api/start_detection`: Start detection on a stream
- `POST /api/stop_detection`: Stop detection on a stream
- `GET /api/status`: Get current status

### WebSocket Events
- `connect`: Client connected
- `disconnect`: Client disconnected
- `frame`: New processed frame (includes stream_path)
- `status`: Status updates
- `error`: Error messages
- `fps_update`: FPS updates (includes stream_path)

## 🛠️ Development

### Adding New Features

1. **Backend (Python/Flask)**:
   - Modify `web_dashboard.py`
   - Add new API endpoints
   - Update `StreamProcessor` class

2. **Frontend (HTML/JavaScript)**:
   - Modify `templates/dashboard.html`
   - Add new UI controls
   - Update JavaScript event handlers

### Allowed Models (Local Only)

The dashboard only allows you to select from three local models (relative paths for Docker):

- `data/other_models/default_model/yolov8n.pt` (YOLOv8n Default)
- `data/other_models/cross_model/weights/best.pt` (Cross Model)
- `data/other_models/Infrared/weights/best.onnx` (Infrared Model)

No models are downloaded from the internet. You must place these files in the correct locations before running the dashboard.

### Custom Models

To add custom detection models:

1. Place your `.pt` or `.onnx` file in a new folder under `data/other_models/`
2. Add the relative path to the allowed models list in both `web_dashboard.py` and the dashboard UI dropdown
3. Restart the dashboard

## Security Considerations

- The dashboard runs on `0.0.0.0:7070` by default (accessible from any IP)
- For production use, consider:
  - Using HTTPS
  - Adding authentication
  - Restricting access to specific IPs
  - Using a reverse proxy (nginx)

## Performance Tips

1. **Hardware Acceleration**:
   - Use CUDA-enabled GPU for faster processing
   - Ensure OpenCV is compiled with CUDA support


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

#
---

