# RTMP Stream Support for 3D Object Detection

This document explains how to use the modified `run_detection.py` script with RTMP streams and other video sources.

## Supported Input Sources

The script now supports multiple input sources:

1. **RTMP Streams**: `rtmp://server.com/app/stream`
2. **RTSP Streams**: `rtsp://server.com/stream`
3. **HTTP/HTTPS Streams**: `http://server.com/stream.m3u8`
4. **Video Files**: `.mp4`, `.avi`, `.mov`, `.mkv`, etc.
5. **Image Folders**: Directories containing image sequences

## Basic Usage

### RTMP Stream
```bash
python scripts/run_detection.py --source rtmp://your-server.com/live/stream1 --output output.mp4
```

### RTSP Stream
```bash
python scripts/run_detection.py --source rtsp://192.168.1.100:554/stream1 --output output.mp4
```

### HTTP Stream
```bash
python scripts/run_detection.py --source http://your-server.com/live/stream.m3u8 --output output.mp4
```

## Advanced Options

### Stream-Specific Parameters

- `--buffer-size`: Buffer size for streams (default: 1024)
- `--timeout`: Connection timeout in seconds (default: 30)
- `--max-frames`: Maximum number of frames to process (default: unlimited)

### Example with Custom Settings
```bash
python scripts/run_detection.py \
  --source rtmp://your-server.com/live/stream1 \
  --output output.mp4 \
  --model yolov8s \
  --conf 0.3 \
  --buffer-size 4096 \
  --timeout 60 \
  --max-frames 1000 \
  --show-fps
```

## Supported Protocols

| Protocol | Example URL | Description |
|----------|-------------|-------------|
| RTMP | `rtmp://server.com/app/stream` | Real-Time Messaging Protocol |
| RTMPS | `rtmps://server.com/app/stream` | RTMP over SSL/TLS |
| RTSP | `rtsp://server.com/stream` | Real-Time Streaming Protocol |
| HTTP | `http://server.com/stream.m3u8` | HTTP Live Streaming |
| HTTPS | `https://server.com/stream.m3u8` | HTTP Live Streaming over SSL |
| UDP | `udp://@:1234` | UDP multicast/broadcast |
| TCP | `tcp://server.com:port` | TCP stream |

## Common Use Cases

### 1. IP Camera Stream
```bash
python scripts/run_detection.py \
  --source rtsp://192.168.1.100:554/stream1 \
  --output camera_detection.mp4 \
  --show-fps
```

### 2. Live Streaming Service
```bash
python scripts/run_detection.py \
  --source rtmp://live-server.com/live/stream_key \
  --output live_detection.mp4 \
  --buffer-size 2048 \
  --timeout 60
```

### 3. Limited Processing (for testing)
```bash
python scripts/run_detection.py \
  --source rtmp://your-server.com/live/stream1 \
  --output test_output.mp4 \
  --max-frames 300 \
  --show-fps
```

## Troubleshooting

### Connection Issues

1. **Timeout Errors**: Increase the timeout value
   ```bash
   --timeout 60
   ```

2. **Buffer Underruns**: Increase buffer size
   ```bash
   --buffer-size 4096
   ```

3. **Network Issues**: Check your network connection and firewall settings

### Performance Issues

1. **Low FPS**: Use a smaller model
   ```bash
   --model yolov8n  # instead of yolov8l or yolov8x
   ```

2. **High CPU Usage**: Disable features you don't need
   ```bash
   --no-bev --no-depth --no-track
   ```

3. **Memory Issues**: Limit the number of frames
   ```bash
   --max-frames 1000
   ```

## Error Handling

The script includes automatic reconnection for streams:

- If a stream connection is lost, it will attempt to reconnect
- Press `q` or `Esc` to exit gracefully
- Use `Ctrl+C` to interrupt processing

## Example Scripts

### Python API Usage
```python
import subprocess

# Run detection on RTMP stream
cmd = [
    "python", "scripts/run_detection.py",
    "--source", "rtmp://your-server.com/live/stream1",
    "--output", "output.mp4",
    "--show-fps",
    "--max-frames", "1000"
]

subprocess.run(cmd)
```

### Batch Processing Multiple Streams
```bash
#!/bin/bash
streams=(
    "rtmp://server1.com/live/stream1"
    "rtmp://server2.com/live/stream2"
    "rtsp://192.168.1.100:554/stream1"
)

for stream in "${streams[@]}"; do
    echo "Processing: $stream"
    python scripts/run_detection.py \
        --source "$stream" \
        --output "output_$(date +%s).mp4" \
        --max-frames 1000
done
```

## Requirements

- OpenCV with FFmpeg support
- Python 3.7+
- CUDA (optional, for GPU acceleration)

## Notes

- The script automatically detects the input source type
- Stream reconnection is handled automatically
- FPS display is available with `--show-fps`
- All original features (tracking, BEV, depth) work with streams
- Output is always saved as a video file

## Testing

To test with a sample RTMP stream, you can use:

1. **Local test stream** (if you have FFmpeg):
   ```bash
   # Start a test stream
   ffmpeg -re -f lavfi -i testsrc -f lavfi -i sine -c:v libx264 -c:a aac -f flv rtmp://localhost/live/test
   
   # Run detection on the test stream
   python scripts/run_detection.py --source rtmp://localhost/live/test --output test_output.mp4
   ```

2. **Public test streams** (if available):
   ```bash
   python scripts/run_detection.py --source http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 --output test_output.mp4
   ``` 