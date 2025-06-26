#!/bin/bash

# Dashboard Runner Script
# Usage: ./run_dashboard.sh [local|public|dev|production]

set -e

case "${1:-local}" in
    "local")
        echo "Starting local dashboard..."
        docker-compose down 2>/dev/null || true
        docker-compose up --build -d
        echo "Dashboard available at: http://localhost:7070"
        echo "View logs: docker-compose logs -f"
        ;;
    "public")
        echo "Starting dashboard with public access..."
        docker-compose -f docker-compose.tunnel.yml down 2>/dev/null || true
        docker-compose -f docker-compose.tunnel.yml up --build -d
        echo "Dashboard available at: http://localhost:7070"
        echo "Public URL: https://yolo3d-dashboard.loca.lt"
        echo "View logs: docker-compose -f docker-compose.tunnel.yml logs -f"
        ;;
    "dev")
        echo "Starting development mode..."
        docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
        docker-compose -f docker-compose.dev.yml up --build -d
        echo "Dashboard available at: http://localhost:7070"
        echo "View logs: docker-compose -f docker-compose.dev.yml logs -f"
        ;;
    "production")
        echo "Starting production mode with nginx..."
        docker-compose --profile production down 2>/dev/null || true
        docker-compose --profile production up --build -d
        echo "Dashboard available at: http://localhost"
        echo "View logs: docker-compose logs -f"
        ;;
    "stop")
        echo "Stopping all dashboard containers..."
        docker-compose down 2>/dev/null || true
        docker-compose -f docker-compose.tunnel.yml down 2>/dev/null || true
        docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
        docker-compose --profile production down 2>/dev/null || true
        echo "All containers stopped."
        ;;
    "logs")
        echo "Showing logs for running containers..."
        docker-compose logs -f 2>/dev/null || docker-compose -f docker-compose.tunnel.yml logs -f 2>/dev/null || docker-compose -f docker-compose.dev.yml logs -f 2>/dev/null || echo "No running containers found."
        ;;
    *)
        echo "Usage: $0 [local|public|dev|production|stop|logs]"
        echo ""
        echo "Options:"
        echo "  local      - Run dashboard locally (default)"
        echo "  public     - Run dashboard with public access via localtunnel"
        echo "  dev        - Run in development mode with source mounting"
        echo "  production - Run with nginx reverse proxy"
        echo "  stop       - Stop all dashboard containers"
        echo "  logs       - Show logs for running containers"
        echo ""
        echo "Examples:"
        echo "  $0 local      # http://localhost:7070"
        echo "  $0 public     # https://yolo3d-dashboard.loca.lt"
        echo "  $0 dev        # Development mode"
        echo "  $0 stop       # Stop all containers"
        exit 1
        ;;
esac 