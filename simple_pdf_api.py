import os
import shutil
import uuid
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pdfplumber
import pandas as pd
import io
import re
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR, PPStructure
from PIL import Image
import numpy as np
from transformers import pipeline
import json

# Directory to store temp files
TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# FastAPI app setup
app = FastAPI(
    title="Production PDF Table Extractor API with OCR",
    description="Upload PDF, get unique download links for HTML, Excel, CSV, JSON, TallyXML. Supports text-based and image-based PDFs with OCR and table detection. Files auto-delete after 10 min.",
    version="2.1.0-prod"
)

# Allowed frontend domains
ALLOWED_ORIGINS = {"http://localhost:3000", "http://localhost", "http://127.0.0.1:8000", "https://mywebsite.com"}

def check_origin(request: Request):
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        raise HTTPException(status_code=403, detail="No origin header.")
    if not any(origin.startswith(allowed) for allowed in ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origin not allowed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background cleanup job
CLEANUP_INTERVAL = 600  # seconds (10 min)
FILE_LIFETIME = 600     # seconds (10 min)

def cleanup_temp_files():
    now = time.time()
    for folder in os.listdir(TEMP_DIR):
        folder_path = os.path.join(TEMP_DIR, folder)
        if os.path.isdir(folder_path):
            mtime = os.path.getmtime(folder_path)
            if now - mtime > FILE_LIFETIME:
                try:
                    shutil.rmtree(folder_path)
                except Exception:
                    pass

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_temp_files, 'interval', seconds=CLEANUP_INTERVAL)
scheduler.start()

# Initialize PaddleOCR and PPStructure
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
table_engine = PPStructure(table=True, ocr=True, show_log=False)

# Supported formats
SUPPORTED_FORMATS = ["html", "excel", "csv", "json", "tallyxml"]

# Helper functions
def normalize_date(date_str):
    """Normalize various date formats to YYYY-MM-DD."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        date_str = str(date_str).strip()
        for fmt in [
            '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d',
            '%d %b %Y', '%d %B %Y', '%m/%d/%Y', '%m-%d-%Y'
        ]:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str
    except:
        return date_str

def normalize_amount(amount):
    """Normalize amount by removing currency symbols and handling negative values."""
    if not amount or pd.isna(amount):
        return None
    try:
        amount = str(amount).replace(',', '').replace('$', '').replace('₹', '').replace('€', '').replace('£', '').strip()
        if amount.startswith('(') and amount.endswith(')'):
            amount = '-' + amount[1:-1]
        return float(amount)
    except:
        return amount

def categorize_transaction(description, classifier=None):
    """Categorize transaction using a zero-shot classifier."""
    if not description or not classifier:
        return "Uncategorized"
    labels = ["Groceries", "Utilities", "Rent", "Salary", "Entertainment", "Transport", "Other"]
    result = classifier(description, candidate_labels=labels, multi_label=False)
    return result['labels'][0] if result['scores'][0] > 0.5 else "Uncategorized"

def extract_balances(tables, unique_tables=None):
    if unique_tables:
        merged = None
        for dfs in unique_tables.values():
            merged_df = pd.concat(dfs, ignore_index=True)
            if merged is None or len(merged_df) > len(merged):
                merged = merged_df
        if merged is not None and not merged.empty:
            balance_col = None
            for col in merged.columns:
                if col is not None and 'balance' in str(col).lower():
                    balance老
                    balance_col = col
                    break
            if balance_col:
                opening = normalize_amount(merged[balance_col].iloc[0])
                closing = normalize_amount(merged[balance_col].iloc[-1])
                return opening, closing
    if not tables:
        return None, None
    df = tables[0]['data']
    if df.empty:
        return None, None
    balance_col = None
    for col in df.columns:
        if col is not None and 'balance' in str(col).lower():
            balance_col = col
            break
    if balance_col:
        opening = normalize_amount(df[balance_col].iloc[0])
        closing = normalize_amount(df[balance_col].iloc[-1])
        return opening, closing
    return None, None

def to_tally_xml(tables):
    if not tables:
        return ""
    df = tables[0]['data']
    if df.empty:
        return ""
    date_col = desc_col = debit_col = credit_col = balance_col = None
    for col in df.columns:
        if col is None:
            continue
        lcol = str(col).lower()
        if not date_col and 'date' in lcol:
            date_col = col
        if not desc_col and ('desc' in lcol or 'particular' in lcol or 'narration' in lcol):
            desc_col = col
        if not debit_col and 'debit' in lcol:
            debit_col = col
        if not credit_col and 'credit' in lcol:
            credit_col = col
        if not balance_col and 'balance' in lcol:
            balance_col = col
    if not date_col:
        date_col = df.columns[0] if len(df.columns) > 0 else None
    if not desc_col:
        desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0] if len(df.columns) > 0 else None
    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ENVELOPE>',
        ' <HEADER>',
        '  <TALLYREQUEST>Import Data</TALLYREQUEST>',
        ' </HEADER>',
        ' <BODY>',
        '  <IMPORTDATA>',
        '   <REQUESTDESC>',
        '    <REPORTNAME>Vouchers</REPORTNAME>',
        '   </REQUESTDESC>',
        '   <REQUESTDATA>',
    ]
    for _, row in df.iterrows():
        date_val = normalize_date(row[date_col]) if date_col and date_col in row else ''
        desc_val = str(row[desc_col]) if desc_col and desc_col in row else ''
        debit_val = normalize_amount(row[debit_col]) if debit_col and debit_col in row else ''
        credit_val = normalize_amount(row[credit_col]) if credit_col and credit_col in row else ''
        balance_val = normalize_amount(row[balance_col]) if balance_col and balance_col in row else ''
        xml.append('    <TALLYMESSAGE>')
        xml.append('     <VOUCHER VCHTYPE="Bank Statement" ACTION="Create">')
        xml.append(f'      <DATE>{date_val}</DATE>')
        xml.append(f'      <NARRATION>{desc_val}</NARRATION>')
        if debit_val:
            xml.append(f'      <DEBIT>{debit_val}</DEBIT>')
        if credit_val:
            xml.append(f'      <CREDIT>{credit_val}</CREDIT>')
        if balance_val:
            xml.append(f'      <BALANCE>{balance_val}</BALANCE>')
        xml.append('     </VOUCHER>')
        xml.append('    </TALLYMESSAGE>')
    xml += [
        '   </REQUESTDATA>',
        '  </IMPORTDATA>',
        ' </BODY>',
        '</ENVELOPE>'
    ]
    return '\n'.join(xml)

def extract_and_save(pdf_bytes, out_dir, password=None, file_map=None, categorize=False):
    tables = []
    unique_tables = {}
    non_blank_pages = set()
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") if categorize else None

    # Step 1: Try direct text extraction with pdfplumber
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=password)
        for page_num, page in enumerate(pdf.pages, 1):
            if page is None:
                continue
            found_table = False
            for table in page.find_tables():
                data = table.extract()
                if data and len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                    # Normalize data
                    for col in df.columns:
                        if 'date' in str(col).lower():
                            df[col] = df[col].apply(normalize_date)
                        elif any(k in str(col).lower() for k in ['amount', 'debit', 'credit', 'balance']):
                            df[col] = df[col].apply(normalize_amount)
                    # Optional categorization
                    if categorize and classifier:
                        desc_col = next((col for col in df.columns if 'desc' in str(col).lower() or 'particular' in str(col).lower()), None)
                        if desc_col:
                            df['Category'] = df[desc_col].apply(lambda x: categorize_transaction(x, classifier))
                    tables.append({"page": page_num, "data": df})
                    headers_key = tuple(df.columns)
                    if headers_key not in unique_tables:
                        unique_tables[headers_key] = []
                    unique_tables[headers_key].append(df)
                    found_table = True
            if found_table:
                non_blank_pages.add(page_num)  # Corrected line
        pdf.close()
    except Exception as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            raise Exception("PDF is password protected" if not password else "Incorrect PDF password")
        # If no text/tables found, proceed to OCR

    # Step 2: OCR for image-based PDFs
    if not tables:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300)
            for page_num, img in enumerate(images, 1):
                img_np = np.array(img)
                result = table_engine(img_np)
                table_data = []
                for res in result:
                    if res['type'] == 'table':
                        # Extract table HTML and convert to DataFrame
                        html_table = res['res']['html']
                        dfs = pd.read_html(html_table)
                        if dfs:
                            df = dfs[0]
                            # Normalize data
                            for col in df.columns:
                                if 'date' in str(col).lower():
                                    df[col] = df[col].apply(normalize_date)
                                elif any(k in str(col).lower() for k in ['amount', 'debit', 'credit', 'balance']):
                                    df[col] = df[col].apply(normalize_amount)
                            # Optional categorization
                            if categorize and classifier:
                                desc_col = next((col for col in df.columns if 'desc' in str(col).lower() or 'particular' in str(col).lower()), None)
                                if desc_col:
                                    df['Category'] = df[desc_col].apply(lambda x: categorize_transaction(x, classifier))
                            tables.append({"page": page_num, "data": df})
                            headers_key = tuple(df.columns)
                            if headers_key not in unique_tables:
                                unique_tables[headers_key] = []
                            unique_tables[headers_key].append(df)
                            non_blank_pages.add(page_num)
        except Exception as e:
            raise Exception(f"OCR processing error: {str(e)}")

    if not tables:
        return 0, 0, None, None, {}

    # Step 3: Save outputs
    if file_map is None:
        file_map = {
            "html": "tables.html",
            "excel": "tables.xlsx",
            "csv": "tables.csv",
            "json": "tables.json",
            "tallyxml": "tables_tally.xml"
        }

    # Save HTML
    html = ""
    for i, t in enumerate(tables):
        html += t['data'].to_html(index=False, border=1)
    with open(os.path.join(out_dir, file_map["html"]), "w", encoding="utf-8") as f:
        f.write(html)

    # Save Excel
    with pd.ExcelWriter(os.path.join(out_dir, file_map["excel"]), engine='xlsxwriter') as writer:
        for i, t in enumerate(tables):
            t['data'].to_excel(writer, sheet_name=f"Table_{i+1}_Page_{t['page']}", index=False)

    # Save CSV
    with open(os.path.join(out_dir, file_map["csv"]), "w", encoding="utf-8") as f:
        for headers, dfs in unique_tables.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(f, index=False)
            f.write("\n\n")

    # Save JSON
    json_data = []
    for i, t in enumerate(tables):
        json_data.append({
            "table": i+1,
            "page": t['page'],
            "columns": list(t['data'].columns),
            "rows": t['data'].to_dict(orient='records')
        })
    with open(os.path.join(out_dir, file_map["json"]), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # Save Tally XML
    tally_xml = to_tally_xml(tables)
    with open(os.path.join(out_dir, file_map["tallyxml"]), "w", encoding="utf-8") as f:
        f.write(tally_xml)

    # Extract balances
    opening, closing = extract_balances(tables, unique_tables)
    return len(tables), len(non_blank_pages), opening, closing, unique_tables

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(None),
    categorize: bool = Form(False),
    request: Request = None,
    _: None = Depends(check_origin)
):
    if not file.filename.lower().endswith('.pdf'):
        return {
            "success": False,
            "error_code": "INVALID_FILE_TYPE",
            "message": "Only PDF files are allowed. Please upload a PDF file.",
            "details": "The uploaded file must have a .pdf extension."
        }

    pdf_bytes = await file.read()
    file_id = str(uuid.uuid4())
    out_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(file.filename)[0]
    file_map = {
        "html": f"{base_name}.html",
        "excel": f"{base_name}.xlsx",
        "csv": f"{base_name}.csv",
        "json": f"{base_name}.json",
        "tallyxml": f"{base_name}_tally.xml"
    }

    with open(os.path.join(out_dir, "original.pdf"), "wb") as f:
        f.write(pdf_bytes)

    try:
        tables_found, pages_count, opening_balance, closing_balance, unique_tables = extract_and_save(
            pdf_bytes, out_dir, password=password, file_map=file_map, categorize=categorize)

        merged_tables_json = []
        for headers, dfs in unique_tables.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_tables_json.append({
                "columns": list(merged_df.columns),
                "rows": merged_df.to_dict(orient="records")
            })

    except Exception as e:
        err_msg = str(e).lower()
        shutil.rmtree(out_dir)
        if any(keyword in err_msg for keyword in ["password", "encrypted", "incorrect password", "protected"]):
            return {
                "success": False,
                "error_code": "INCORRECT_PASSWORD" if password else "PASSWORD_REQUIRED",
                "message": "The provided password is incorrect." if password else "This PDF is password protected.",
                "details": "Please check your password and try again." if password else "Please provide the password to extract tables."
            }
        elif any(keyword in err_msg for keyword in ["corrupted", "damaged"]):
            return {
                "success": False,
                "error_code": "CORRUPTED_FILE",
                "message": "The PDF file appears to be corrupted or damaged.",
                "details": "Please try uploading a different PDF file."
            }
        else:
            return {
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "message": "Failed to process the PDF file.",
                "details": f"Error: {str(e)}"
            }

    if tables_found == 0:
        shutil.rmtree(out_dir)
        return {
            "success": False,
            "error_code": "NO_TABLES_FOUND",
            "message": "No tables found in the PDF.",
            "details": f"Processed {pages_count} pages but found no extractable tables.",
            "pages_count": pages_count
        }

    links = {fmt: f"/download/{file_id}/{fmt}" for fmt in SUPPORTED_FORMATS}
    return {
        "success": True,
        "tables_found": tables_found,
        "pages_count": pages_count,
        "file_id": file_id,
        "download_links": links,
        "output_file_names": file_map,
        "opening_balance": opening_balance,
        "closing_balance": closing_balance,
        "merged_tables_json": merged_tables_json
    }

@app.get("/download/{file_id}/{fmt}")
def download_file(file_id: str, fmt: str):
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid format.")

    safe_id = file_id.replace("..", "")
    out_dir = os.path.join(TEMP_DIR, safe_id)

    if not os.path.exists(out_dir):
        raise HTTPException(status_code=404, detail="File not found or expired.")

    files = os.listdir(out_dir)
    file_name = None
    ext_map = {
        "html": ".html",
        "excel": ".xlsx",
        "csv": ".csv",
        "json": ".json",
        "tallyxml": "_tally.xml"
    }

    for f in files:
        if fmt == "tallyxml" and f.endswith(ext_map[fmt]):
            file_name = f
            break
        elif f.endswith(ext_map[fmt]):
            file_name = f
            break

    if not file_name:
        raise HTTPException(status_code=404, detail="Requested format not found.")

    file_path = os.path.join(out_dir, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired.")

    media_types = {
        "html": "text/html",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "json": "application/json",
        "tallyxml": "application/xml"
    }

    return FileResponse(file_path, media_type=media_types[fmt], filename=file_name)

@app.get("/")
def root():
    return {"message": "Production PDF Table Extractor API with OCR. POST /upload with PDF, get download links."}
