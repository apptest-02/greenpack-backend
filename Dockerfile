# Greenpack Pro — Backend Dockerfile (Mode B Cloud)
FROM python:3.11-slim-bookworm

LABEL maintainer="Aura Tech Labs <hello@auratechlabs.com>"
LABEL description="Greenpack Pro Label Inspection Engine"

# System dependencies for OpenCV, Tesseract, pdf2image
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    zbar-tools \
    libzbar0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data files reports templates models logs temp backups

# Configure Tesseract path for Linux
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV EASYOCR_DOWNLOAD_ENABLED=true
ENV GREENPACK_MODE=server

# Download EasyOCR models at build time
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='models', download_enabled=True)" 2>&1 || echo "Model download will happen at runtime"

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:18080/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18080", "--workers", "2"]
