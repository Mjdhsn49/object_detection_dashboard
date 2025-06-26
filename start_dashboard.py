#!/usr/bin/env python3
"""
Startup script for the Object Detection Web Dashboard
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    print("🚀 Object Detection Web Dashboard")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("web_dashboard.py").exists():
        print("❌ Error: web_dashboard.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Required packages for web dashboard
    required_packages = [
        "flask",
        "flask_socketio", 
        "opencv-python",
        "numpy"
    ]
    
    print("📦 Checking and installing dependencies...")
    
    missing_packages = []
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package} (already installed)")
        else:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            if not install_package(package):
                print(f"❌ Cannot start dashboard without {package}")
                return
    
    # Check detection modules
    print("\n🔍 Checking detection modules...")
    try:
        from src.models import ObjectDetector, DepthEstimator
        print("✅ Detection modules available")
    except ImportError as e:
        print(f"⚠️  Warning: Detection modules not available: {e}")
        print("The dashboard will run but detection features may not work.")
    
    print("\n🌐 Starting web dashboard...")
    print("The dashboard will be available at: http://localhost:7070")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the dashboard
        from web_dashboard import app, socketio
        socketio.run(app, host='0.0.0.0', port=7070, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main() 