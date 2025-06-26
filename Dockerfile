FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies in stages to better handle errors
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt web_requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r web_requirements.txt

# Copy application code
COPY . .

# Create templates directory
RUN mkdir -p templates

# Expose port
EXPOSE 7070

# Set environment variables
ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7070/api/status || exit 1

# Run the application
CMD ["python", "web_dashboard.py"] 