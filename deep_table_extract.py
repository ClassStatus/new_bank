import os
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import pandas as pd
import re

# Path to CascadeTabNet weights (download from official repo)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'CascadeTabNet_Simple.pth')

def preprocess_image(img_pil):
    # Convert to grayscale
    img = np.array(img_pil.convert('L'))
    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    # Contrast (CLAHE)
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

def detect_tables_opencv(img):
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

def extract_table_from_crop(crop_img):
    # Use pytesseract to extract text (multi-lang)
    ocr_text = pytesseract.image_to_string(crop_img, config='--psm 6', lang='eng+hin')
    # Split lines, then split by tab or multiple spaces
    rows = [line for line in ocr_text.split('\n') if line.strip()]
    # Try to split columns by multiple spaces or tab
    data = [re.split(r'\t| {2,}', row) for row in rows]
    # Try to create DataFrame
    if len(data) > 1:
        df = pd.DataFrame(data[1:], columns=data[0])
    else:
        df = pd.DataFrame()
    return df

# Main function: PDF bytes -> list of DataFrames
def deep_table_extract(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    all_tables = []
    for img_pil in images:
        img = preprocess_image(img_pil)
        table_boxes = detect_tables_opencv(img)
        if not table_boxes:
            # Fallback: treat whole page as one table
            table_boxes = [(0, 0, img.shape[1], img.shape[0])]
        for (x1, y1, x2, y2) in table_boxes:
            crop = img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)
            df = extract_table_from_crop(crop_pil)
            if not df.empty:
                all_tables.append(df)
    return all_tables

# Placeholder for CascadeTabNet integration (advanced deep learning table detection)
def load_cascadetabnet_model():
    raise NotImplementedError("You must integrate CascadeTabNet model class here (see official repo)")

def detect_tables_in_image(model, image_np):
    # Placeholder for CascadeTabNet model inference
    return [] 
