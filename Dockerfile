FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.11 \
        python3.11-pip \
        python3.11-dev \
        tesseract-ocr \
        tesseract-ocr-hin \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp_files

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "uvicorn", "new.simple_pdf_api_prod:app", "--host", "0.0.0.0", "--port", "8000"] 
