# PDF Translate Note - Docker Image
# Build: docker build -t pdf-translate-note .
# Run: docker run -p 7000:7000 --env-file .env pdf-translate-note

FROM python:3.11-slim

# System dependencies for OpenCV, PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY garment_dict.json .

# Create output directory
RUN mkdir -p output

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7000/ || exit 1

# Run application
CMD ["python", "app.py"]
