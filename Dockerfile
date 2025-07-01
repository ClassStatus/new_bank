FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.24.4 && \
    pip install --no-cache-dir -r requirements.txt

COPY simple_pdf_api.py .

CMD ["uvicorn", "simple_pdf_api:app", "--host", "0.0.0.0", "--port", "10000"]
