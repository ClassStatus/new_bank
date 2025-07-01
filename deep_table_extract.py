import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import re
from pdf2image import convert_from_bytes

# Image Preprocessing Function
def preprocess_for_ocr(img_pil):
    img = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize (Upscale small images)
    if gray.shape[0] < 1000:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(binary)

# Extract Table from Preprocessed Image using OCR
def extract_table_from_image(img_pil):
    config = '--psm 6'
    text = pytesseract.image_to_string(img_pil, config=config)

    rows = [line.strip() for line in text.split('\n') if line.strip()]
    data = []

    for row in rows:
        if '\t' in row:
            cols = row.split('\t')
        elif '  ' in row:
            cols = re.split(r'\s{2,}', row)
        else:
            cols = re.split(r'[|,;]', row)
        cols = [c.strip() for c in cols if c.strip()]
        if cols:
            data.append(cols)

    if not data:
        return pd.DataFrame()

    max_cols = max(len(r) for r in data)
    columns = [f'Column_{i+1}' for i in range(max_cols)]
    df = pd.DataFrame(data, columns=columns)
    return df

# PDF to Table Extraction (Full Pipeline)
def extract_tables_from_pdf(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    all_tables = []

    for page_num, img_pil in enumerate(images, 1):
        print(f"📄 Processing page {page_num}")
        processed_img = preprocess_for_ocr(img_pil)
        df = extract_table_from_image(processed_img)
        if not df.empty:
            print(f"✅ Table found on page {page_num}, rows: {len(df)}")
            all_tables.append({
                'table': len(all_tables)+1,
                'page': page_num,
                'columns': df.columns.tolist(),
                'rows': df.to_dict(orient='records')
            })
        else:
            print(f"⚠️ No table found on page {page_num}")

    return all_tables
