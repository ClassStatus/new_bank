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
SUPPORTED_FORMATS = ["html", "excel", "csv", "json"]
def extract_and_save(pdf_bytes, out_dir, password=None):
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
        raise e
    if not tables:
        return 0, 0
    # Save HTML (only tables, no extra text)
    html = ""
    for i, t in enumerate(tables):
        html += t['data'].to_html(index=False, border=1)
    with open(os.path.join(out_dir, "tables.html"), "w", encoding="utf-8") as f:
        f.write(html)
    # Save Excel
    with pd.ExcelWriter(os.path.join(out_dir, "tables.xlsx"), engine='xlsxwriter') as writer:
        for i, t in enumerate(tables):
            t['data'].to_excel(writer, sheet_name=f"Table_{i+1}_Page_{t['page']}", index=False)
    # Save CSV (merge tables with same headers)
    with open(os.path.join(out_dir, "tables.csv"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(out_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    return len(tables), len(non_blank_pages)

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(None),
    request: Request = None,
    _: None = Depends(check_origin)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    pdf_bytes = await file.read()
    file_id = str(uuid.uuid4())
    out_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)
    # Save original PDF
    with open(os.path.join(out_dir, "original.pdf"), "wb") as f:
        f.write(pdf_bytes)
    # Try to extract tables, handle password-protected PDFs
    try:
        tables_found, pages_count = extract_and_save(pdf_bytes, out_dir, password=password)
    except Exception as e:
        err_msg = str(e).lower()
        if "password" in err_msg or "encrypted" in err_msg or "incorrect password" in err_msg:
            shutil.rmtree(out_dir)
            if password:
                return {"success": False, "message": "Incorrect PDF password."}
            else:
                return {"success": False, "message": "PDF is password protected. Please provide password."}
        else:
            shutil.rmtree(out_dir)
            msg = str(e) or "Unknown error. File may be corrupted or unsupported, or password is incorrect."
            return {"success": False, "message": f"PDF processing error: {msg}"}
    if tables_found == 0:
        shutil.rmtree(out_dir)
        return {"success": False, "message": "No tables found in PDF.", "pages_count": pages_count}
    # Return download links
    links = {fmt: f"/download/{file_id}/{fmt}" for fmt in SUPPORTED_FORMATS}
    return {
        "success": True,
        "tables_found": tables_found,
        "pages_count": pages_count,
        "file_id": file_id,
        "download_links": links
    }

@app.get("/download/{file_id}/{fmt}")
def download_file(file_id: str, fmt: str):
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid format.")
    safe_id = file_id.replace("..", "")  # Prevent path traversal
    file_map = {
        "html": "tables.html",
        "excel": "tables.xlsx",
        "csv": "tables.csv",
        "json": "tables.json"
    }
    file_path = os.path.join(TEMP_DIR, safe_id, file_map[fmt])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired.")
    media_types = {
        "html": "text/html",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "json": "application/json"
    }
    return FileResponse(file_path, media_type=media_types[fmt], filename=file_map[fmt])

@app.get("/")
def root():
    return {"message": "Production PDF Table Extractor API. POST /upload with PDF, get download links."} 
