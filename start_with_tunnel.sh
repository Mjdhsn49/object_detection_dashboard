#!/bin/bash

# Start the web dashboard in the background
echo "Starting Object Detection Dashboard..."
python web_dashboard.py &
DASHBOARD_PID=$!

# Wait a moment for the dashboard to start
sleep 5

# Start localtunnel to create a public URL
echo "Starting localtunnel for public access..."
lt --port 7070 --subdomain yolo3d-dashboard &
TUNNEL_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Shutting down..."
    kill $DASHBOARD_PID 2>/dev/null
    kill $TUNNEL_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "Dashboard is running at: http://localhost:7070"
echo "Public URL will be available at: https://yolo3d-dashboard.loca.lt"
echo "Press Ctrl+C to stop"

# Wait for both processes
wait 