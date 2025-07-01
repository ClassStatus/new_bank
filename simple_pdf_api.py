from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import cv2
import numpy as np

app = FastAPI()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>PDF OCR API</h2>
    <p>Go to <a href="/docs">/docs</a> to test the API.</p>
    """

@app.post("/pdf-ocr/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

        pdf_bytes = await file.read()
        images = convert_from_bytes(pdf_bytes)

        extracted_text = []
        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            result = ocr.ocr(img_cv, cls=False)
            for line in result[0]:
                extracted_text.append(line[1][0])  # Get text only

        return {"text": extracted_text}

    except Exception as e:
        print("OCR ERROR:", str(e))
        return JSONResponse(status_code=500, content={"error": "OCR failed", "detail": str(e)})
