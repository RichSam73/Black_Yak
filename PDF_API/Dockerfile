
# Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fix: Adjust copy paths for Root Context
# Assuming build context is root, we copy from PDF_API/
COPY PDF_API/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from PDF_API to /app
COPY PDF_API/ .

# Dependencies for OpenCV just in case
ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
