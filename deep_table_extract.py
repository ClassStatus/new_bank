import os
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import re
import traceback
from io import BytesIO
import tempfile
import platform

# Optional: Setup Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✅ Tesseract OCR is available")

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Found Tesseract at: {path}")
                    break
            else:
                print("⚠️ Tesseract found but not in PATH. Add it or specify the path manually.")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("❌ Tesseract OCR is not available")

# Camelot for table extraction
try:
    import camelot
    CAMELOT_AVAILABLE = True
    print("✅ Camelot is available for table extraction")
except ImportError:
    CAMELOT_AVAILABLE = False
    print("❌ Camelot is not available. Only OCR will be used.")

# Preprocess PIL Image
def preprocess_image(img_pil):
    try:
        img = np.array(img_pil.convert('L'))  # grayscale
        img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Deskew
        coords = np.column_stack(np.where(img > 0))
        angle = 0
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return img
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return np.array(img_pil.convert('L'))

# OpenCV Table Detection
def detect_tables_opencv(img):
    try:
        horizontal = img.copy()
        vertical = img.copy()
        scale = 20
        horizontalsize = max(1, img.shape[1] // scale)
        verticalsize = max(1, img.shape[0] // scale)

        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        mask = cv2.add(horizontal, vertical)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [(x, y, x + w, y + h) for cnt in contours if (w := cv2.boundingRect(cnt)[2]) > 40 and (h := cv2.boundingRect(cnt)[3]) > 20]
        return sorted(boxes, key=lambda b: (b[1], b[0]))
    except Exception as e:
        print(f"Error in table detection: {e}")
        return []

# OCR-based table extraction
def extract_table_from_crop(crop_img):
    try:
        if not TESSERACT_AVAILABLE:
            print("❌ Tesseract not available.")
            return pd.DataFrame()

        ocr_text = pytesseract.image_to_string(crop_img, config='--psm 6', lang='eng')
        rows = [line for line in ocr_text.split('\n') if line.strip()]
        if not rows:
            return pd.DataFrame()

        data = []
        for row in rows:
            if '\t' in row:
                columns = row.split('\t')
            elif '  ' in row:
                columns = re.split(r' {2,}', row)
            else:
                columns = re.split(r'[|,;]', row)
            columns = [col.strip() for col in columns if col.strip()]
            if columns:
                data.append(columns)

        if len(data) > 1:
            if len(data[0]) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                max_cols = max(len(row) for row in data)
                df = pd.DataFrame(data, columns=[f'Column_{i+1}' for i in range(max_cols)])
        elif len(data) == 1:
            df = pd.DataFrame([data[0]], columns=[f'Column_{i+1}' for i in range(len(data[0]))])
        else:
            df = pd.DataFrame()

        return df
    except Exception as e:
        print(f"Error in table extraction: {e}")
        return pd.DataFrame()

# Camelot Extraction
def extract_tables_with_camelot(pdf_bytes):
    tables = []
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        print("🔍 [Camelot] Extracting tables...")
        try:
            tables_lattice = camelot.read_pdf(tmp.name, pages='all', flavor='lattice')
            tables_stream = camelot.read_pdf(tmp.name, pages='all', flavor='stream')
            all_tables = tables_lattice + tables_stream
            print(f"✅ [Camelot] Found {len(all_tables)} tables")
            for t in all_tables:
                if not t.df.empty:
                    tables.append(t.df)
            return tables
        except Exception as e:
            print(f"❌ [Camelot] Failed: {e}")
            return []

# Main Function
def deep_table_extract(pdf_bytes):
    if isinstance(pdf_bytes, str):
        raise ValueError("❌ Expected bytes input, not string.")

    if CAMELOT_AVAILABLE:
        tables = extract_tables_with_camelot(pdf_bytes)
        if tables:
            return tables
        else:
            print("⚠️ No tables found with Camelot. Falling back to OCR...")

    if not TESSERACT_AVAILABLE:
        raise Exception("Tesseract OCR is required for image-based PDF processing.")

    try:
        print("🔄 Converting PDF to images...")
        images = convert_from_bytes(pdf_bytes)
        print(f"📄 Converted {len(images)} pages")

        all_tables = []

        for page_num, img_pil in enumerate(images, 1):
            print(f"\n📃 Page {page_num}")
            img = preprocess_image(img_pil)
            boxes = detect_tables_opencv(img)

            if not boxes:
                boxes = [(0, 0, img.shape[1], img.shape[0])]
                print("📋 No tables detected. Using full page.")

            for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):
                crop = img[y1:y2, x1:x2]
                if isinstance(crop, bytes):
                    raise ValueError("Expected image array, got bytes.")
                crop_pil = Image.fromarray(crop)
                df = extract_table_from_crop(crop_pil)
                if not df.empty:
                    all_tables.append(df)
                    print(f"✅ Table {idx} extracted with {len(df)} rows")
                else:
                    print(f"⚠️ Table {idx} is empty")

        print(f"\n🎉 OCR Extraction complete. {len(all_tables)} tables found.")
        return all_tables

    except Exception as e:
        print(f"❌ OCR extraction failed: {e}")
        print(traceback.format_exc())
        return []

# Optional: test entry
if __name__ == "__main__":
    print("🚀 Ready to extract tables. Use deep_table_extract(pdf_bytes)")
