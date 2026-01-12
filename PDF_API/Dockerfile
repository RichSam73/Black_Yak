
# Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# cv2 dependencies: libgl1, libglib2.0-0
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Environment
ENV PORT=8080
EXPOSE 8080

# Run
CMD ["python", "main.py"]
