from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pandas as pd
import io
import tempfile
import zipfile

app = FastAPI(
    title="Simple PDF Table Extractor API",
    description="Upload a PDF, get tables as HTML, Excel, CSV, or JSON. No auth, no DB, no limits.",
    version="1.0.0-simple"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_tables(pdf_bytes: bytes):
    tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            for table in page.find_tables():
                data = table.extract()
                if data and len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                    tables.append({
                        "page": page_num,
                        "data": df
                    })
    return tables

def tables_to_html(tables):
    html = "<html><body>"
    for i, t in enumerate(tables):
        html += f"<h3>Table {i+1} (Page {t['page']})</h3>"
        html += t['data'].to_html(index=False, border=1)
    html += "</body></html>"
    return html

def tables_to_excel(tables):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for i, t in enumerate(tables):
            t['data'].to_excel(writer, sheet_name=f"Table_{i+1}_Page_{t['page']}", index=False)
    output.seek(0)
    return output

def tables_to_csv_zip(tables):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as zf:
        for i, t in enumerate(tables):
            csv_bytes = t['data'].to_csv(index=False).encode('utf-8')
            zf.writestr(f"table_{i+1}_page_{t['page']}.csv", csv_bytes)
    output.seek(0)
    return output

def tables_to_json(tables):
    result = []
    for i, t in enumerate(tables):
        result.append({
            "table": i+1,
            "page": t['page'],
            "columns": list(t['data'].columns),
            "rows": t['data'].to_dict(orient='records')
        })
    return result

@app.post("/extract", summary="Upload PDF and extract tables")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    pdf_bytes = await file.read()
    tables = extract_tables(pdf_bytes)
    if not tables:
        return {"success": False, "message": "No tables found in PDF."}
    # Prepare download links (simulate, as FastAPI can't serve files from memory by link directly)
    return {
        "success": True,
        "tables_found": len(tables),
        "message": "Tables extracted. Use /download endpoints to get files.",
        "download_endpoints": [
            {"format": "html", "url": "/download/html"},
            {"format": "excel", "url": "/download/excel"},
            {"format": "csv_zip", "url": "/download/csv"},
            {"format": "json", "url": "/download/json"}
        ]
    }

# In-memory storage for last uploaded tables (for demo, not for production)
from threading import Lock
last_tables = {"tables": None}
tables_lock = Lock()

@app.post("/upload", summary="Upload PDF and store tables for download")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    pdf_bytes = await file.read()
    tables = extract_tables(pdf_bytes)
    if not tables:
        return {"success": False, "message": "No tables found in PDF."}
    with tables_lock:
        last_tables["tables"] = tables
    return {"success": True, "tables_found": len(tables), "message": "Tables stored. Now use /download endpoints."}

@app.get("/download/html", response_class=HTMLResponse)
def download_html():
    with tables_lock:
        tables = last_tables["tables"]
    if not tables:
        return HTMLResponse("<h2>No tables available. Please upload a PDF first.</h2>", status_code=400)
    html = tables_to_html(tables)
    return HTMLResponse(content=html, status_code=200)

@app.get("/download/excel")
def download_excel():
    with tables_lock:
        tables = last_tables["tables"]
    if not tables:
        return Response("No tables available. Please upload a PDF first.", status_code=400)
    excel_bytes = tables_to_excel(tables)
    return StreamingResponse(excel_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=extracted_tables.xlsx"})

@app.get("/download/csv")
def download_csv():
    with tables_lock:
        tables = last_tables["tables"]
    if not tables:
        return Response("No tables available. Please upload a PDF first.", status_code=400)
    csv_zip = tables_to_csv_zip(tables)
    return StreamingResponse(csv_zip, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=extracted_tables_csv.zip"})

@app.get("/download/json")
def download_json():
    with tables_lock:
        tables = last_tables["tables"]
    if not tables:
        return JSONResponse({"error": "No tables available. Please upload a PDF first."}, status_code=400)
    json_data = tables_to_json(tables)
    return JSONResponse(content=json_data)

@app.get("/")
def root():
    return {"message": "Welcome to Simple PDF Table Extractor API. POST /upload with a PDF, then GET /download/{format}"} 
