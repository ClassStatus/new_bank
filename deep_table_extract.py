import os
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import re
import traceback
import camelot

# Try to import pytesseract with fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✅ Tesseract OCR is available")
    
    # Try to set Tesseract path if not in PATH (Windows common issue)
    try:
        # Test if tesseract is accessible
        pytesseract.get_tesseract_version()
    except Exception:
        # Try common Windows installation paths
        import platform
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', ''))
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Found Tesseract at: {path}")
                    break
            else:
                print("⚠️ Tesseract found but not in PATH. Please add Tesseract to your system PATH or specify the path manually.")
        else:
            print("⚠️ Tesseract found but not in PATH. Please add Tesseract to your system PATH.")
            
except ImportError:
    TESSERACT_AVAILABLE = False
    print("❌ Tesseract OCR is not available")

# Path to CascadeTabNet weights (download from official repo)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'CascadeTabNet_Simple.pth')

# Try to import Camelot for digital PDF table extraction
try:
    import camelot
    CAMELOT_AVAILABLE = True
    print("✅ Camelot is available for table extraction")
except ImportError:
    CAMELOT_AVAILABLE = False
    print("❌ Camelot is not available. Only OCR will be used for table extraction.")

def preprocess_image(img_pil):
    try:
        # Convert to grayscale
        img = np.array(img_pil.convert('L'))
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
        
        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Threshold
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Deskew (simple)
        coords = np.column_stack(np.where(img > 0))
        angle = 0
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return img
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        # Return original image if preprocessing fails
        return np.array(img_pil.convert('L'))

def detect_tables_opencv(img):
    try:
        # Detect table regions using OpenCV (lines/contours)
        # img: preprocessed grayscale/thresholded image
        # Returns: list of (x1, y1, x2, y2)
        
        # Find horizontal and vertical lines
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
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 40 and h > 20:  # filter small boxes
                boxes.append((x, y, x + w, y + h))
        
        # Sort boxes top to bottom
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes
    except Exception as e:
        print(f"Error in table detection: {e}")
        return []

def extract_table_from_crop(crop_img):
    try:
        if not TESSERACT_AVAILABLE:
            print("❌ Tesseract not available, cannot extract text from image")
            return pd.DataFrame()
        
        # Use pytesseract to extract text (English only)
        ocr_text = pytesseract.image_to_string(crop_img, config='--psm 6', lang='eng')
        
        # Split lines, then split by tab or multiple spaces
        rows = [line for line in ocr_text.split('\n') if line.strip()]
        
        if not rows:
            return pd.DataFrame()
        
        # Try to split columns by multiple spaces or tab
        data = []
        for row in rows:
            # Try different splitting methods
            if '\t' in row:
                # Tab-separated
                columns = row.split('\t')
            elif '  ' in row:
                # Multiple spaces
                columns = re.split(r' {2,}', row)
            else:
                # Single space or other delimiter
                columns = re.split(r'[|,;]', row)
            
            # Clean up columns
            columns = [col.strip() for col in columns if col.strip()]
            if columns:
                data.append(columns)
        
        # Try to create DataFrame
        if len(data) > 1:
            # Use first row as header if it looks like headers
            if len(data[0]) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                # No clear headers, use generic column names
                max_cols = max(len(row) for row in data)
                columns = [f'Column_{i+1}' for i in range(max_cols)]
                df = pd.DataFrame(data, columns=columns)
        elif len(data) == 1:
            # Single row
            df = pd.DataFrame([data[0]], columns=[f'Column_{i+1}' for i in range(len(data[0]))])
        else:
            df = pd.DataFrame()
        
        return df
    except Exception as e:
        print(f"Error in table extraction: {e}")
        return pd.DataFrame()

def extract_tables_with_camelot(pdf_bytes):
    """Extract tables from PDF using Camelot. Returns list of DataFrames."""
    import tempfile
    import io
    tables = []
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        print("🔍 [Camelot] Trying to extract tables from PDF using Camelot...")
        try:
            # Try both flavors for best results
            tables_lattice = camelot.read_pdf(tmp.name, pages='all', flavor='lattice')
            tables_stream = camelot.read_pdf(tmp.name, pages='all', flavor='stream')
            all_tables = tables_lattice + tables_stream
            print(f"✅ [Camelot] Found {len(all_tables)} tables in PDF.")
            for t in all_tables:
                if not t.df.empty:
                    tables.append(t.df)
            return tables
        except Exception as e:
            print(f"❌ [Camelot] Table extraction failed: {e}")
            return []

# Main function: PDF bytes -> list of DataFrames
# Now tries Camelot first, then falls back to OCR

def deep_table_extract(pdf_bytes):
    # Try Camelot first
    if CAMELOT_AVAILABLE:
        tables = extract_tables_with_camelot(pdf_bytes)
        if tables:
            print(f"🎉 [Camelot] Table extraction complete. Found {len(tables)} tables.")
            return tables
        else:
            print("⚠️ [Camelot] No tables found with Camelot. Falling back to OCR.")
    # Fallback to OCR
    try:
        if not TESSERACT_AVAILABLE:
            print("❌ Tesseract OCR is not installed. Please install Tesseract OCR engine to process image-based PDFs.")
            raise Exception("Tesseract OCR is not installed. Please install Tesseract OCR engine to process image-based PDFs.")
        print("🔄 [OCR] Converting PDF to images...")
        import io
        images = convert_from_bytes(io.BytesIO(pdf_bytes))
        print(f"📄 [OCR] Converted {len(images)} pages to images")
        all_tables = []
        for page_num, img_pil in enumerate(images, 1):
            print(f"🔍 [OCR] Processing page {page_num}...")
            try:
                # Preprocess image
                img = preprocess_image(img_pil)
                print(f"🛠️ [OCR] Preprocessed image for page {page_num}")
                # Detect table regions
                table_boxes = detect_tables_opencv(img)
                print(f"📊 [OCR] Found {len(table_boxes)} potential table regions on page {page_num}")
                if not table_boxes:
                    print(f"📋 [OCR] No table regions detected on page {page_num}, treating whole page as table")
                    table_boxes = [(0, 0, img.shape[1], img.shape[0])]
                for box_num, (x1, y1, x2, y2) in enumerate(table_boxes, 1):
                    try:
                        print(f"✂️ [OCR] Cropping table region {box_num} on page {page_num}: ({x1}, {y1}, {x2}, {y2})")
                        crop = img[y1:y2, x1:x2]
                        crop_pil = Image.fromarray(crop)
                        print(f"🔠 [OCR] Running pytesseract OCR on table region {box_num} (English only)...")
                        df = extract_table_from_crop(crop_pil)
                        if not df.empty and len(df) > 0:
                            print(f"✅ [OCR] Extracted table {box_num} from page {page_num} with {len(df)} rows")
                            all_tables.append(df)
                        else:
                            print(f"⚠️ [OCR] Table {box_num} from page {page_num} is empty after OCR")
                    except Exception as crop_error:
                        print(f"❌ [OCR] Error processing table {box_num} on page {page_num}: {crop_error}")
                        continue
            except Exception as page_error:
                print(f"❌ [OCR] Error processing page {page_num}: {page_error}")
                continue
        print(f"🎉 [OCR] OCR extraction complete. Found {len(all_tables)} tables total")
        return all_tables
    except Exception as e:
        print(f"❌ [OCR] OCR extraction failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

# Placeholder for CascadeTabNet integration (advanced deep learning table detection)
def load_cascadetabnet_model():
    raise NotImplementedError("You must integrate CascadeTabNet model class here (see official repo)")

def detect_tables_in_image(model, image_np):
    # Placeholder for CascadeTabNet model inference
    return []

def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # or 'lattice'
    for i, table in enumerate(tables):
        print(f"Table {i+1}")
        print(table.df) 
