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

# Directory to store temp files
TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# FastAPI app setup
app = FastAPI(
    title="Production PDF Table Extractor API",
    description="Upload PDF, get unique download links for HTML, Excel, CSV, JSON. Files auto-delete after 10 min.",
    version="2.0.0-prod"
)

# Allowed frontend domains (add your production domain here later)
ALLOWED_ORIGINS = {"http://localhost:3000", "http://localhost", "http://127.0.0.1:8000", "https://mywebsite.com"}  # Add your real domain

def check_origin(request: Request):
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        raise HTTPException(status_code=403, detail="No origin header.")
    if not any(origin.startswith(allowed) for allowed in ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origin not allowed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),  # Allow localhost and your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background cleanup job: delete files older than 10 min
CLEANUP_INTERVAL = 600  # seconds (10 min)
FILE_LIFETIME = 600     # seconds (10 min)

def cleanup_temp_files():
    now = time.time()
    for folder in os.listdir(TEMP_DIR):
        folder_path = os.path.join(TEMP_DIR, folder)
        if os.path.isdir(folder_path):
            # Check folder creation/modification time
            mtime = os.path.getmtime(folder_path)
            if now - mtime > FILE_LIFETIME:
                try:
                    shutil.rmtree(folder_path)
                except Exception:
                    pass

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_temp_files, 'interval', seconds=CLEANUP_INTERVAL)
scheduler.start()

# Helper: extract tables and save all formats
SUPPORTED_FORMATS = ["html", "excel", "csv", "json", "tallyxml"]

def extract_balances(tables, unique_tables=None):
    # If unique_tables is provided, use the largest merged table for balances
    if unique_tables:
        # Find the largest merged table (most rows)
        merged = None
        for dfs in unique_tables.values():
            merged_df = pd.concat(dfs, ignore_index=True)
            if merged is None or len(merged_df) > len(merged):
                merged = merged_df
        if merged is not None and not merged.empty:
            # Try to find balance column
            balance_col = None
            for col in merged.columns:
                if 'balance' in col.lower():
                    balance_col = col
                    break
            if balance_col:
                opening = merged[balance_col].iloc[0]
                closing = merged[balance_col].iloc[-1]
                return opening, closing
    # Fallback: use first table
    if not tables:
        return None, None
    df = tables[0]['data']
    if df.empty:
        return None, None
    balance_col = None
    for col in df.columns:
        if 'balance' in col.lower():
            balance_col = col
            break
    if balance_col:
        opening = df[balance_col].iloc[0]
        closing = df[balance_col].iloc[-1]
        return opening, closing
    return None, None

def to_tally_xml(tables):
    # Only use the first table for Tally export
    if not tables:
        return ""
    df = tables[0]['data']
    if df.empty:
        return ""
    # Try to find columns
    date_col = None
    desc_col = None
    debit_col = None
    credit_col = None
    balance_col = None
    for col in df.columns:
        lcol = col.lower()
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
    # Fallbacks
    if not date_col:
        date_col = df.columns[0]
    if not desc_col:
        desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    # Build XML
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
        date_val = str(row[date_col]) if date_col in row else ''
        desc_val = str(row[desc_col]) if desc_col in row else ''
        debit_val = str(row[debit_col]) if debit_col and debit_col in row else ''
        credit_val = str(row[credit_col]) if credit_col and credit_col in row else ''
        balance_val = str(row[balance_col]) if balance_col and balance_col in row else ''
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

def extract_and_save(pdf_bytes, out_dir, password=None, file_map=None):
    tables = []
    unique_tables = {}  # key: tuple(headers), value: list of DataFrames
    non_blank_pages = set()
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes), password=password) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                found_table = False
                for table in page.find_tables():
                    data = table.extract()
                    if data and len(data) > 1:
                        df = pd.DataFrame(data[1:], columns=data[0])
                        tables.append({"page": page_num, "data": df})
                        # For CSV merging
                        headers_key = tuple(df.columns)
                        if headers_key not in unique_tables:
                            unique_tables[headers_key] = []
                        unique_tables[headers_key].append(df)
                        found_table = True
                if found_table:
                    non_blank_pages.add(page_num)
    except Exception as e:
        err_msg = str(e) if e is not None else "Unknown error"
        err_msg_lower = err_msg.lower() if err_msg else ""
        if "password" in err_msg_lower or "encrypted" in err_msg_lower or "incorrect password" in err_msg_lower:
            shutil.rmtree(out_dir)
            if password:
                return {"success": False, "message": "Incorrect PDF password."}
            else:
                return {"success": False, "message": "PDF is password protected. Please provide password."}
        else:
            shutil.rmtree(out_dir)
            msg = err_msg or "Unknown error. File may be corrupted or unsupported, or password is incorrect."
            return {"success": False, "message": f"PDF processing error: {msg}"}
    if not tables:
        return 0, 0, None, None
    # Use file_map for output names if provided
    if file_map is None:
        file_map = {
            "html": "tables.html",
            "excel": "tables.xlsx",
            "csv": "tables.csv",
            "json": "tables.json",
            "tallyxml": "tables_tally.xml"
        }
    # Save HTML (only tables, no extra text)
    html = ""
    for i, t in enumerate(tables):
        html += t['data'].to_html(index=False, border=1)
    with open(os.path.join(out_dir, file_map["html"]), "w", encoding="utf-8") as f:
        f.write(html)
    # Save Excel
    with pd.ExcelWriter(os.path.join(out_dir, file_map["excel"]), engine='xlsxwriter') as writer:
        for i, t in enumerate(tables):
            t['data'].to_excel(writer, sheet_name=f"Table_{i+1}_Page_{t['page']}", index=False)
    # Save CSV (merge tables with same headers)
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
    import json
    with open(os.path.join(out_dir, file_map["json"]), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    # Save Tally XML
    tally_xml = to_tally_xml(tables)
    with open(os.path.join(out_dir, file_map["tallyxml"]), "w", encoding="utf-8") as f:
        f.write(tally_xml)
    # Extract balances (use merged tables for closing balance)
    opening, closing = extract_balances(tables, unique_tables)
    return len(tables), len(non_blank_pages), opening, closing

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(None),
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
    
    # Determine output file base name - use the uploaded file name
    base_name = os.path.splitext(file.filename)[0]
    file_map = {
        "html": f"{base_name}.html",
        "excel": f"{base_name}.xlsx", 
        "csv": f"{base_name}.csv",
        "json": f"{base_name}.json",
        "tallyxml": f"{base_name}_tally.xml"
    }
    
    # Save original PDF
    with open(os.path.join(out_dir, "original.pdf"), "wb") as f:
        f.write(pdf_bytes)
    
    # Try to extract tables, handle password-protected PDFs
    try:
        tables_found, pages_count, opening_balance, closing_balance = extract_and_save(
            pdf_bytes, out_dir, password=password, file_map=file_map)
        
        # Re-extract unique_tables for merged tables JSON
        tables = []
        unique_tables = {}
        with pdfplumber.open(io.BytesIO(pdf_bytes), password=password) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for table in page.find_tables():
                    data = table.extract()
                    if data and len(data) > 1:
                        df = pd.DataFrame(data[1:], columns=data[0])
                        tables.append({"page": page_num, "data": df})
                        headers_key = tuple(df.columns)
                        if headers_key not in unique_tables:
                            unique_tables[headers_key] = []
                        unique_tables[headers_key].append(df)
        
        merged_tables_json = []
        for headers, dfs in unique_tables.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_tables_json.append({
                "columns": list(merged_df.columns),
                "rows": merged_df.to_dict(orient="records")
            })
            
    except Exception as e:
        err_msg = str(e) if e is not None else "Unknown error"
        err_msg_lower = err_msg.lower() if err_msg else ""
        
        if "password" in err_msg_lower or "encrypted" in err_msg_lower or "incorrect password" in err_msg_lower:
            shutil.rmtree(out_dir)
            if password:
                return {
                    "success": False,
                    "error_code": "INCORRECT_PASSWORD", 
                    "message": "The provided password is incorrect.",
                    "details": "Please check your password and try again."
                }
            else:
                return {
                    "success": False,
                    "error_code": "PASSWORD_REQUIRED",
                    "message": "This PDF is password protected.",
                    "details": "Please provide the password to extract tables."
                }
        elif "corrupted" in err_msg_lower or "damaged" in err_msg_lower:
            shutil.rmtree(out_dir)
            return {
                "success": False,
                "error_code": "CORRUPTED_FILE",
                "message": "The PDF file appears to be corrupted or damaged.",
                "details": "Please try uploading a different PDF file."
            }
        elif "unsupported" in err_msg_lower or "format" in err_msg_lower:
            shutil.rmtree(out_dir)
            return {
                "success": False,
                "error_code": "UNSUPPORTED_FORMAT",
                "message": "This PDF format is not supported.",
                "details": "Please try with a different PDF file."
            }
        else:
            shutil.rmtree(out_dir)
            return {
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "message": "Failed to process the PDF file.",
                "details": f"Error: {err_msg}"
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
    
    # Return download links
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
    
    safe_id = file_id.replace("..", "")  # Prevent path traversal
    out_dir = os.path.join(TEMP_DIR, safe_id)
    
    if not os.path.exists(out_dir):
        raise HTTPException(status_code=404, detail="File not found or expired.")
    
    # Look for files in the directory
    files = os.listdir(out_dir)
    file_name = None
    
    # Try to find the file with the right extension
    ext_map = {
        "html": ".html",
        "excel": ".xlsx", 
        "csv": ".csv",
        "json": ".json",
        "tallyxml": "_tally.xml"
    }
    
    # Find the file with the correct extension
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
    return {"message": "Production PDF Table Extractor API. POST /upload with PDF, get download links."} 
