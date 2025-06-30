import os
import shutil
import uuid
import time
import re
import tempfile
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pdfplumber
import pandas as pd
import io
from xml.sax.saxutils import escape
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDFExtractor")

# Configuration from environment variables
FILE_LIFETIME = int(os.getenv("FILE_LIFETIME", 600))  # 10 minutes
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", 300))  # 5 minutes
MAX_PAGES = int(os.getenv("MAX_PAGES", 100))  # Max pages to process
TEMP_DIR = os.getenv("TEMP_DIR", "temp_files")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
RATE_LIMIT = os.getenv("RATE_LIMIT", "5/minute")

# Directory to store temp files
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# FastAPI app setup
app = FastAPI(
    title="Production PDF Table Extractor API",
    description="Upload PDF, get unique download links for HTML, Excel, CSV, JSON, Tally XML. Files auto-delete after 10 min.",
    version="3.0.0-prod"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background cleanup job
scheduler = BackgroundScheduler()

def cleanup_temp_files():
    logger.info("Running temp file cleanup")
    now = time.time()
    for folder in os.listdir(TEMP_DIR):
        folder_path = os.path.join(TEMP_DIR, folder)
        if os.path.isdir(folder_path):
            try:
                mtime = os.path.getmtime(folder_path)
                if now - mtime > FILE_LIFETIME:
                    shutil.rmtree(folder_path)
                    logger.info(f"Deleted expired folder: {folder}")
            except Exception as e:
                logger.error(f"Error cleaning {folder_path}: {str(e)}")

scheduler.add_job(cleanup_temp_files, 'interval', seconds=CLEANUP_INTERVAL)
scheduler.start()

# Supported formats
SUPPORTED_FORMATS = ["html", "excel", "csv", "json", "tallyxml"]

# Helper functions
def sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent path traversal and injection"""
    safe_name = re.sub(r'[^\w\-_. ]', '', name)
    return safe_name[:100]  # Limit to 100 characters

def find_balance_column(df: pd.DataFrame) -> str:
    """Identify balance column in a DataFrame"""
    for col in df.columns:
        if col and 'balance' in str(col).lower():
            return col
    return None

def extract_tables(pdf_bytes: bytes, password: str = None) -> tuple:
    """Extract tables from PDF bytes"""
    tables = []
    unique_tables = {}
    non_blank_pages = set()
    processed_pages = 0
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes), password=password) as pdf:
            if not hasattr(pdf, 'pages') or not pdf.pages:
                raise Exception("PDF has no accessible pages - may be password protected")
            
            for page_num, page in enumerate(pdf.pages[:MAX_PAGES], 1):
                processed_pages += 1
                if page is None:
                    continue
                    
                found_table = False
                for table in page.find_tables():
                    data = table.extract()
                    if data and len(data) > 1:
                        df = pd.DataFrame(data[1:], columns=data[0])
                        tables.append({"page": page_num, "data": df})
                        
                        # For merged tables
                        headers_key = tuple(df.columns)
                        if headers_key not in unique_tables:
                            unique_tables[headers_key] = []
                        unique_tables[headers_key].append(df)
                        
                        found_table = True
                
                if found_table:
                    non_blank_pages.add(page_num)
                    
            total_pages = len(pdf.pages)
            
    except Exception as e:
        err_msg = str(e)
        logger.error(f"PDF processing error: {err_msg}")
        
        # Password detection
        password_keywords = [
            "password", "encrypted", "incorrect password", "protected", 
            "authentication", "security", "locked", "restricted",
            "requires password", "password required", "access denied"
        ]
        
        if any(kw in err_msg.lower() for kw in password_keywords):
            if password:
                raise HTTPException(
                    status_code=400, 
                    detail="Incorrect PDF password"
                )
            raise HTTPException(
                status_code=400, 
                detail="PDF is password protected"
            )
        elif "corrupted" in err_msg.lower() or "damaged" in err_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="PDF appears corrupted or damaged"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"PDF processing failed: {err_msg}"
            )
    
    return tables, unique_tables, len(non_blank_pages), processed_pages

def extract_balances(tables: list, unique_tables: dict) -> tuple:
    """Extract opening and closing balances"""
    if not tables:
        return None, None
        
    # Try with merged tables first
    if unique_tables:
        largest_df = None
        for dfs in unique_tables.values():
            merged_df = pd.concat(dfs, ignore_index=True)
            if largest_df is None or len(merged_df) > len(largest_df):
                largest_df = merged_df
        
        if largest_df is not None:
            balance_col = find_balance_column(largest_df)
            if balance_col and not largest_df.empty:
                opening = largest_df[balance_col].iloc[0]
                closing = largest_df[balance_col].iloc[-1]
                return opening, closing
    
    # Fallback to first table
    df = tables[0]['data']
    balance_col = find_balance_column(df)
    if balance_col and not df.empty:
        opening = df[balance_col].iloc[0]
        closing = df[balance_col].iloc[-1]
        return opening, closing
        
    return None, None

def to_tally_xml(tables: list) -> str:
    """Convert tables to Tally XML format"""
    if not tables:
        return ""
    
    df = tables[0]['data']
    if df.empty:
        return ""
    
    # Identify columns
    date_col, desc_col, debit_col, credit_col, balance_col = None, None, None, None, None
    
    for col in df.columns:
        if not col:
            continue
        lcol = str(col).lower()
        if not date_col and ('date' in lcol or 'dt' in lcol):
            date_col = col
        if not desc_col and ('desc' in lcol or 'particular' in lcol or 'narration' in lcol or 'details' in lcol):
            desc_col = col
        if not debit_col and 'debit' in lcol:
            debit_col = col
        if not credit_col and 'credit' in lcol:
            credit_col = col
        if not balance_col and 'balance' in lcol:
            balance_col = col
    
    # Fallback to first columns if needed
    if not date_col and len(df.columns) > 0:
        date_col = df.columns[0]
    if not desc_col:
        desc_col = date_col if len(df.columns) == 1 else df.columns[1] if len(df.columns) > 1 else None
    
    # Build XML
    xml_lines = [
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
        # Get values with proper escaping
        date_val = escape(str(row[date_col])) if date_col and date_col in row else ''
        desc_val = escape(str(row[desc_col])) if desc_col and desc_col in row else ''
        
        # Numeric handling
        try:
            debit_val = float(row[debit_col]) if debit_col and debit_col in row and pd.notna(row[debit_col]) else 0.0
        except (ValueError, TypeError):
            debit_val = 0.0
            
        try:
            credit_val = float(row[credit_col]) if credit_col and credit_col in row and pd.notna(row[credit_col]) else 0.0
        except (ValueError, TypeError):
            credit_val = 0.0
            
        balance_val = str(row[balance_col]) if balance_col and balance_col in row else ''
        
        # Add voucher entry
        xml_lines += [
            '    <TALLYMESSAGE>',
            '     <VOUCHER VCHTYPE="Bank Statement" ACTION="Create">',
            f'      <DATE>{date_val}</DATE>',
            f'      <NARRATION>{desc_val}</NARRATION>',
        ]
        
        if debit_val > 0:
            xml_lines.append(f'      <DEBIT>{debit_val:.2f}</DEBIT>')
        if credit_val > 0:
            xml_lines.append(f'      <CREDIT>{credit_val:.2f}</CREDIT>')
        if balance_val:
            xml_lines.append(f'      <BALANCE>{balance_val}</BALANCE>')
            
        xml_lines += [
            '     </VOUCHER>',
            '    </TALLYMESSAGE>'
        ]
    
    xml_lines += [
        '   </REQUESTDATA>',
        '  </IMPORTDATA>',
        ' </BODY>',
        '</ENVELOPE>'
    ]
    
    return '\n'.join(xml_lines)

def save_outputs(tables: list, unique_tables: dict, out_dir: str, file_map: dict):
    """Save extracted tables in all formats"""
    # HTML
    html_content = ""
    for t in tables:
        html_content += t['data'].to_html(index=False, border=1)
    with open(os.path.join(out_dir, file_map["html"]), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Excel
    with pd.ExcelWriter(os.path.join(out_dir, file_map["excel"]), engine='xlsxwriter') as writer:
        for i, t in enumerate(tables):
            sheet_name = f"Page_{t['page']}_{i+1}"[:31]  # Excel sheet name limit
            t['data'].to_excel(writer, sheet_name=sheet_name, index=False)
    
    # CSV (merged tables)
    with open(os.path.join(out_dir, file_map["csv"]), "w", encoding="utf-8") as f:
        for headers, dfs in unique_tables.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(f, index=False)
            f.write("\n\n")
    
    # JSON
    json_data = []
    for t in tables:
        json_data.append({
            "page": t['page'],
            "columns": list(t['data'].columns),
            "rows": t['data'].to_dict(orient='records')
        })
    with open(os.path.join(out_dir, file_map["json"]), "w", encoding="utf-8") as f:
        import json
        json.dump(json_data, f, indent=2)
    
    # Tally XML
    tally_xml = to_tally_xml(tables)
    with open(os.path.join(out_dir, file_map["tallyxml"]), "w", encoding="utf-8") as f:
        f.write(tally_xml)
    
    # Return file sizes
    file_sizes = {}
    for fmt, fname in file_map.items():
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            file_sizes[fmt] = os.path.getsize(path)
    
    return file_sizes

# API Endpoints
@app.post("/upload")
@limiter.limit(RATE_LIMIT)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    password: str = Form(None),
):
    """Process uploaded PDF and extract tables"""
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_code": "INVALID_FILE_TYPE",
                "message": "Only PDF files are allowed"
            }
        )
    
    # Create processing directory
    file_id = str(uuid.uuid4())
    out_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Sanitize filename
    base_name = sanitize_filename(os.path.splitext(file.filename)[0])
    file_map = {
        "html": f"{base_name}.html",
        "excel": f"{base_name}.xlsx",
        "csv": f"{base_name}.csv",
        "json": f"{base_name}.json",
        "tallyxml": f"{base_name}_tally.xml"
    }
    
    # Save original PDF
    orig_path = os.path.join(out_dir, "original.pdf")
    try:
        # Stream to disk instead of loading full file in memory
        with open(orig_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read back for processing
        with open(orig_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        shutil.rmtree(out_dir)
        logger.error(f"File handling error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "FILE_SAVE_ERROR",
                "message": "Could not save uploaded file"
            }
        )
    
    # Extract tables
    try:
        tables, unique_tables, tables_found, pages_processed = extract_tables(pdf_bytes, password)
    except HTTPException as he:
        shutil.rmtree(out_dir)
        return JSONResponse(
            status_code=he.status_code,
            content={
                "success": False,
                "error_code": "EXTRACTION_ERROR",
                "message": he.detail
            }
        )
    
    # Check if any tables found
    if not tables:
        shutil.rmtree(out_dir)
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error_code": "NO_TABLES_FOUND",
                "message": "No extractable tables found",
                "pages_processed": pages_processed
            }
        )
    
    # Extract balances
    opening, closing = extract_balances(tables, unique_tables)
    
    # Save outputs
    try:
        file_sizes = save_outputs(tables, unique_tables, out_dir, file_map)
    except Exception as e:
        shutil.rmtree(out_dir)
        logger.error(f"Output saving failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "OUTPUT_SAVE_ERROR",
                "message": "Could not generate output files"
            }
        )
    
    # Prepare response
    expiration = datetime.now() + timedelta(seconds=FILE_LIFETIME)
    links = {fmt: f"/download/{file_id}/{fmt}" for fmt in SUPPORTED_FORMATS}
    
    return {
        "success": True,
        "file_id": file_id,
        "tables_found": len(tables),
        "pages_processed": pages_processed,
        "opening_balance": str(opening) if opening else None,
        "closing_balance": str(closing) if closing else None,
        "download_links": links,
        "file_sizes": file_sizes,
        "expires_at": expiration.isoformat(),
        "output_filenames": file_map
    }

@app.get("/download/{file_id}/{fmt}")
def download_file(file_id: str, fmt: str):
    """Serve processed files"""
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid format")
    
    # Sanitize file ID
    safe_id = re.sub(r'[^a-zA-Z0-9\-]', '', file_id)
    if not safe_id or safe_id != file_id:
        raise HTTPException(status_code=400, detail="Invalid file ID")
    
    out_dir = os.path.join(TEMP_DIR, safe_id)
    if not os.path.exists(out_dir):
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    # Find matching file
    for file in os.listdir(out_dir):
        if fmt == "tallyxml" and file.endswith("_tally.xml"):
            file_path = os.path.join(out_dir, file)
            return FileResponse(file_path, media_type="application/xml", filename=file)
        elif file.endswith(f".{fmt}") or (fmt == "excel" and file.endswith(".xlsx")):
            file_path = os.path.join(out_dir, file)
            
            # Set correct media types
            media_types = {
                "html": "text/html",
                "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "csv": "text/csv",
                "json": "application/json",
                "tallyxml": "application/xml"
            }
            
            return FileResponse(
                file_path,
                media_type=media_types.get(fmt, "application/octet-stream"),
                filename=file
            )
    
    raise HTTPException(status_code=404, detail="Requested format not available")

@app.get("/health")
def health_check():
    """Service health check"""
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "temp_files": len(os.listdir(TEMP_DIR))
    }

@app.get("/")
def root():
    """API information"""
    return {
        "service": "PDF Table Extractor API",
        "version": app.version,
        "endpoints": {
            "POST /upload": "Upload PDF for processing",
            "GET /download/{id}/{format}": "Download processed files",
            "GET /health": "Service health check"
        }
    }

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    scheduler.shutdown()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    return response
