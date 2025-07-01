from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import tempfile
import os
import cv2
import numpy as np

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Supports English

@app.post("/pdf-ocr/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

    pdf_bytes = await file.read()
    images = convert_from_bytes(pdf_bytes)

    extracted_text = []

    for img in images:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = ocr.ocr(img_cv, cls=True)
        for line in result[0]:
            text = line[1][0]
            extracted_text.append(text)

    return JSONResponse(content={"text": extracted_text})
