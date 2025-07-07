#!/usr/bin/env python3
"""
Startup script for Object Detection Dashboard with GPU optimization.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch',
        'torchvision', 
        'opencv-python',
        'numpy',
        'Flask',
        'Flask-SocketIO',
        'ultralytics',
        'transformers',
        'Pillow',
        'scipy',
        'filterpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_gpu():
    """Check GPU availability and CUDA support."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 GPU Available: {gpu_name}")
            print(f"   - Count: {gpu_count}")
            print(f"   - Memory: {gpu_memory:.1f} GB")
            print(f"   - CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU/CUDA available - will use CPU")
            return False
    except Exception as e:
        print(f"⚠️  Error checking GPU: {e}")
        return False

def check_models():
    """Check if model files exist."""
    model_paths = [
        'data/other_models/default_model/yolov8n.pt',
        'data/other_models/cross_model/weights/best.pt',
        'data/other_models/Infrared/weights/best.onnx'
    ]
    
    missing_models = []
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"✅ {model_path}")
        else:
            missing_models.append(model_path)
            print(f"❌ {model_path} - MISSING")
    
    if missing_models:
        print(f"\n⚠️  Missing model files: {len(missing_models)}")
        print("Some models may not be available for selection")
    
    return len(missing_models) == 0

def optimize_system():
    """Apply system optimizations for better performance."""
    print("\n🔧 Applying system optimizations...")
    
    # Set environment variables for better performance
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    
    # Try to set CUDA optimizations
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            print("✅ CUDA optimizations applied")
    except:
        pass
    
    print("✅ System optimizations applied")

def main():
    """Main startup function."""
    print("🚀 Object Detection Dashboard - GPU Optimized")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before starting.")
        sys.exit(1)
    
    # Check GPU
    print("\n🖥️  Checking GPU availability...")
    gpu_available = check_gpu()
    
    # Check models
    print("\n🤖 Checking model files...")
    check_models()
    
    # Apply optimizations
    optimize_system()
    
    # Start the dashboard
    print("\n🌐 Starting Object Detection Dashboard...")
    print("   - URL: http://localhost:7070")
    print("   - GPU Mode: " + ("Enabled" if gpu_available else "Disabled"))
    print("   - Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Import and run the dashboard
        from web_dashboard import app, socketio
        
        # Run with optimized settings
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=7070, 
            debug=False, 
            allow_unsafe_werkzeug=True,
            use_reloader=False  # Disable reloader for better performance
        )
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 