# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies for pdf2image and PaddleOCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY simple_pdf_api.py .

# Run FastAPI app with uvicorn
CMD ["uvicorn", "simple_pdf_api:app", "--host", "0.0.0.0", "--port", "10000"]
