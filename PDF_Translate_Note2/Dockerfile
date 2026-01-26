# Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV FLASK_DEBUG=false

# Install system dependencies (opencv-headless는 GL 불필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY garment_dict.json .
COPY vision-key.json .

# Create output directory
RUN mkdir -p output uploads

# Expose port
EXPOSE 8080

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 app:app
