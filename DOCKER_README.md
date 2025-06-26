# 🐳 Object Detection Dashboard - Docker Setup

This document explains how to run the Object Detection Dashboard using Docker and Docker Compose.

## 🚀 Quick Start

### Prerequisites
- Docker
- Docker Compose
- At least 4GB RAM
- Model files in the correct locations

### Option 1: Simple Docker Compose (Recommended)
```bash
# Build and run the dashboard
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Option 2: Development Mode
```bash
# Run with source code mounted for development
docker-compose -f docker-compose.dev.yml up --build
```

### Option 3: Production with Nginx
```bash
# Run with nginx reverse proxy
docker-compose --profile production up --build
```

### Option 4: Public Access with Localtunnel (NEW!)
```bash
# Run with localtunnel for public access
docker-compose -f docker-compose.tunnel.yml up --build
```

## 📁 File Structure

```
yolo3d/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Multi-service orchestration
├── .dockerignore             # Files to exclude from build
├── nginx.conf                # Nginx reverse proxy config
├── web_dashboard.py          # Main Flask application
├── templates/
│   └── dashboard.html        # Web interface
├── data/
│   └── other_models/         # Custom models (mounted as volume)
├── yolov8n.pt               # Default YOLOv8 model
└── output/                  # Output directory (mounted as volume)
```

## 🔧 Configuration

### Environment Variables

The following environment variables can be set in `docker-compose.yml`:

```yaml
environment:
  - PYTHONPATH=/app
  - KMP_DUPLICATE_LIB_OK=TRUE
  - FLASK_ENV=development  # For development mode
  - CUDA_VISIBLE_DEVICES=0  # For GPU support
```

### Model Paths

The dashboard expects these model files:

1. **YOLOv8 Nano**: `/app/yolov8n.pt`
2. **Custom Cross Model**: `/app/data/other_models/cross_model/weights/best.pt`
3. **Thermal Infrared Model**: `/app/data/other_models/Infrared/weights/best.pt`

### Volume Mounts

```yaml
volumes:
  - ./data:/app/data                    # Models directory
  - ./yolov8n.pt:/app/yolov8n.pt       # Default model
  - ./output:/app/output               # Output directory
  - ./:/app/src                        # Source code (development mode only)
```

## 🛠️ Docker Commands

### Build and Run

```bash
# Build the image
docker-compose build

# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f object-detection-dashboard

# Stop services
docker-compose down
```

### Development

```bash
# Rebuild after code changes
docker-compose up --build

# Access container shell
docker-compose exec object-detection-dashboard bash

# View real-time logs
docker-compose logs -f
```

### Production

```bash
# Start with nginx
docker-compose --profile production up -d

# Check health
curl http://localhost/health

# View all logs
docker-compose logs -f
```

### Public Access with Localtunnel

```bash
# Start with public access
docker-compose -f docker-compose.tunnel.yml up --build

# Check logs for public URL
docker-compose -f docker-compose.tunnel.yml logs -f
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 7070
   lsof -i :7070
   
   # Change port in docker-compose.yml
   ports:
     - "8080:7070"  # Use port 8080 instead
   ```

2. **Model Files Not Found**
   ```bash
   # Check if models exist
   ls -la data/other_models/
   ls -la yolov8n.pt
   
   # Ensure correct paths in dashboard
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # In Docker Desktop: Settings → Resources → Memory
   ```

4. **GPU Support**
   ```bash
   # For NVIDIA GPU support, use nvidia-docker
   docker-compose -f docker-compose.gpu.yml up
   ```

### Health Checks

```bash
# Check service health
docker-compose ps

# Check health endpoint
curl http://localhost:7070/api/status

# View health check logs
docker inspect object-detection-dashboard | grep Health -A 10
```

## 📊 Monitoring

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs object-detection-dashboard

# Follow logs in real-time
docker-compose logs -f object-detection-dashboard
```

### Metrics

```bash
# Check container stats
docker stats object-detection-dashboard

# Check resource usage
docker-compose exec object-detection-dashboard top
```

## 🔒 Security

### Production Considerations

1. **HTTPS**: Use nginx with SSL certificates
2. **Authentication**: Add authentication to the dashboard
3. **Network**: Use internal networks for service communication
4. **Secrets**: Use Docker secrets for sensitive data

### Example SSL Setup

```bash
# Create SSL directory
mkdir -p ssl

# Add your certificates
cp your-cert.pem ssl/cert.pem
cp your-key.pem ssl/key.pem

# Start with SSL
docker-compose --profile production up -d
```

## 🚀 Deployment

### Local Development

```bash
# Quick start
docker-compose up --build

# With hot reload (mount source code)
docker-compose -f docker-compose.dev.yml up
```

### Production Server

```bash
# Clone repository
git clone <your-repo>
cd yolo3d

# Set up environment
cp .env.example .env
# Edit .env with your settings

# Deploy
docker-compose --profile production up -d

# Monitor
docker-compose logs -f
```

### Cloud Deployment

```bash
# For AWS/GCP/Azure
docker-compose -f docker-compose.prod.yml up -d

# With load balancer
docker-compose -f docker-compose.prod.yml -f docker-compose.lb.yml up -d
```

## 📝 Environment Files

Create a `.env` file for environment variables:

```env
# Dashboard settings
DASHBOARD_PORT=7070
DASHBOARD_HOST=0.0.0.0

# Model paths
YOLOV8_MODEL=/app/yolov8n.pt
CUSTOM_MODEL=/app/data/other_models/cross_model/weights/best.pt
THERMAL_MODEL=/app/data/other_models/Infrared/weights/best.pt

# Detection settings
DEFAULT_CONFIDENCE=0.25
DEFAULT_IOU=0.45

# Performance
WORKER_THREADS=4
MAX_FRAME_RATE=30
```

## 🎯 Usage

1. **Start the dashboard**: `docker-compose up`
2. **Open browser**: http://localhost:7070
3. **Enter RTMP URL**: Your stream URL
4. **Select model**: Choose from available models
5. **Adjust settings**: Confidence and IoU thresholds
6. **Start detection**: Click "Start Detection"

## 📞 Support

For issues and questions:
1. Check the logs: `docker-compose logs`
2. Verify model files exist
3. Check port availability
4. Review Docker resource limits

---

**Happy detecting! 🎯** 