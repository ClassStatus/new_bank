from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import numpy as np
from PIL import Image
import pandas as pd

def extract_tables_from_pdf(pdf_bytes):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False, det=True, rec=True, structure=True)
    images = convert_from_bytes(pdf_bytes, dpi=300)
    all_tables = []
    for page_num, img_pil in enumerate(images, 1):
        print(f"Processing page {page_num} with PaddleOCR...")
        img_np = np.array(img_pil.convert("RGB"))
        result = ocr.ocr(img_np, cls=True, det=True, rec=True, structure=True)
        if 'structure' in result and result['structure']:
            for table in result['structure']:
                df = pd.DataFrame(table['res'])
                all_tables.append({
                    'table': len(all_tables)+1,
                    'page': page_num,
                    'columns': df.columns.tolist(),
                    'rows': df.to_dict(orient='records')
                })
        else:
            print(f"No table found on page {page_num}")
    return all_tables
