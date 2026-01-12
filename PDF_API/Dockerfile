
# Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Updated packages: libgl1 (new name for libgl1-mesa-glx in newer Debian)
# libglib2.0-0 is still required for cv2
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fix: Adjust copy paths for Root Context
COPY PDF_API/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from PDF_API to /app
COPY PDF_API/ .

# Environment
ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
