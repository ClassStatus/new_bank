# Use the official Python slim image to reduce size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for pdf2image, PaddleOCR, and PPStructure
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY simple_pdf_api.py .

# Expose the port Render.com will use
EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "simple_pdf_api:app", "--host", "0.0.0.0", "--port", "1000"]
