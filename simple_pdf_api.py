import os
import shutil
import uuid
import time
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pdfplumber
import pandas as pd
import io
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import deep_table_extract, with fallback
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'new'))
    from deep_table_extract import deep_table_extract
    OCR_AVAILABLE = True
    logger.info("OCR module loaded successfully")
except Exception as e:
    logger.warning(f"OCR import failed: {e}")
    # Fallback function if OCR is not available
    def deep_table_extract(pdf_bytes):
        return []
    OCR_AVAILABLE = False

# Try to import required OCR dependencies
try:
    import cv2
    import numpy as np
    from pdf2image import convert_from_bytes
    from PIL import Image
    import pytesseract
    OCR_DEPS_AVAILABLE = True
    logger.info("OCR dependencies loaded successfully")
except ImportError as e:
    logger.warning(f"OCR dependencies missing: {e}")
    OCR_DEPS_AVAILABLE = False
    OCR_AVAILABLE = False

# Directory to store temp files
TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# File size limits (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# FastAPI app setup
app = FastAPI(
    title="Production PDF Table Extractor API",
    description="Upload PDF, get unique download links for HTML, Excel, CSV, JSON. Files auto-delete after 10 min.",
    version="2.0.0-prod"
)

# Allowed frontend domains (add your production domain here later)
ALLOWED_ORIGINS = {"http://localhost:3000", "http://localhost", "http://127.0.0.1:8000", "https://mywebsite.com", "*"}  # Allow all for now

def check_origin(request: Request):
    try:
        origin = request.headers.get("origin") or request.headers.get("referer")
        if not origin:
            logger.warning("No origin header found")
            return  # Allow requests without origin for now
        if "*" in ALLOWED_ORIGINS or any(origin.startswith(allowed) for allowed in ALLOWED_ORIGINS if allowed != "*"):
            return
        logger.warning(f"Origin not allowed: {origin}")
        raise HTTPException(status_code=403, detail="Origin not allowed.")
    except Exception as e:
        logger.error(f"Origin check error: {e}")
        # Don't block requests due to origin check errors
        return

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background cleanup job: delete files older than 10 min
CLEANUP_INTERVAL = 600  # seconds (10 min)
FILE_LIFETIME = 600     # seconds (10 min)

def cleanup_temp_files():
    try:
        now = time.time()
        for folder in os.listdir(TEMP_DIR):
            folder_path = os.path.join(TEMP_DIR, folder)
            if os.path.isdir(folder_path):
                # Check folder creation/modification time
                mtime = os.path.getmtime(folder_path)
                if now - mtime > FILE_LIFETIME:
                    try:
                        shutil.rmtree(folder_path)
                        logger.info(f"Cleaned up expired folder: {folder}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup folder {folder}: {e}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_temp_files, 'interval', seconds=CLEANUP_INTERVAL)
scheduler.start()

# Helper: extract tables and save all formats
SUPPORTED_FORMATS = ["html", "excel", "csv", "json", "tallyxml"]

def extract_balances(tables, unique_tables=None):
    try:
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
                    if col is not None and 'balance' in str(col).lower():
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
            if col is not None and 'balance' in str(col).lower():
                balance_col = col
                break
        if balance_col:
            opening = df[balance_col].iloc[0]
            closing = df[balance_col].iloc[-1]
            return opening, closing
        return None, None
    except Exception as e:
        logger.error(f"Error extracting balances: {e}")
        return None, None

def to_tally_xml(tables):
    try:
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
        # Fallbacks
        if not date_col:
            date_col = df.columns[0] if len(df.columns) > 0 else None
        if not desc_col:
            desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0] if len(df.columns) > 0 else None
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
            date_val = str(row[date_col]) if date_col and date_col in row else ''
            desc_val = str(row[desc_col]) if desc_col and desc_col in row else ''
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
    except Exception as e:
        logger.error(f"Error generating Tally XML: {e}")
        return "<!-- Error generating Tally XML -->"

def extract_and_save(pdf_bytes, out_dir, password=None, file_map=None):
    tables = []
    unique_tables = {}  # key: tuple(headers), value: list of DataFrames
    non_blank_pages = set()
    ocr_used = False
    ocr_message = None
    pdf = None
    
    try:
        # Check file size
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise Exception(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
        
        # Try to open the PDF
        logger.info("Attempting to open PDF")
        logger.info(f"Password provided: {'Yes' if password else 'No'}")
        
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=password)
        except Exception as pdf_open_error:
            logger.error(f"PDF open error: {pdf_open_error}")
            # Check if it's a password issue
            if password:
                # Try without password to see if it's password protected
                try:
                    test_pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=None)
                    test_pdf.close()
                    raise Exception("Incorrect PDF password")
                except Exception as test_error:
                    test_err_msg = str(test_error).lower()
                    if any(keyword in test_err_msg for keyword in ["password", "encrypted", "protected"]):
                        raise Exception("Incorrect PDF password")
                    else:
                        raise Exception(f"PDF open failed: {pdf_open_error}")
            else:
                # No password provided, check if it's password protected
                try:
                    test_pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password="")
                    test_pdf.close()
                    raise Exception("PDF is password protected")
                except Exception as test_error:
                    test_err_msg = str(test_error).lower()
                    if any(keyword in test_err_msg for keyword in ["password", "encrypted", "protected"]):
                        raise Exception("PDF is password protected")
                    else:
                        raise Exception(f"PDF open failed: {pdf_open_error}")
        
        # Check if PDF opened successfully
        if pdf is None:
            raise Exception("Failed to open PDF - may be password protected")
        
        # Check if we can access pages
        if not hasattr(pdf, 'pages') or pdf.pages is None:
            raise Exception("PDF appears to be password protected or corrupted")
        
        logger.info(f"PDF opened successfully. Pages: {len(pdf.pages)}")
        
        for page_num, page in enumerate(pdf.pages, 1):
            if page is None:
                continue
            found_table = False
            try:
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
                logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        logger.info(f"Extracted {len(tables)} tables from {len(non_blank_pages)} pages")
        
    except Exception as e:
        # Clean up if PDF was opened
        if pdf is not None:
            try:
                pdf.close()
            except:
                pass
            
        err_msg = str(e) if e is not None else "Unknown error"
        err_msg_lower = err_msg.lower() if err_msg else ""
        
        logger.error(f"PDF processing error: {err_msg}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Check for specific error types
        if any(keyword in err_msg_lower for keyword in ["password", "encrypted", "incorrect password", "protected"]):
            if password:
                raise Exception("Incorrect PDF password")
            else:
                raise Exception("PDF is password protected")
        elif any(keyword in err_msg_lower for keyword in ["corrupted", "damaged", "invalid"]):
            raise Exception("PDF file is corrupted or invalid")
        elif "too large" in err_msg_lower:
            raise Exception(err_msg)
        else:
            # Generic error with more context
            raise Exception(f"PDF processing failed: {err_msg}")
    
    finally:
        # Always close PDF
        if pdf is not None:
            try:
                pdf.close()
            except:
                pass
    
    if not tables:
        # Try OCR extraction only if no tables found in normal flow
        logger.info("No tables found, attempting OCR")
        ocr_used = True
        if OCR_AVAILABLE and OCR_DEPS_AVAILABLE:
            try:
                ocr_tables = deep_table_extract(pdf_bytes)
                if ocr_tables and len(ocr_tables) > 0:
                    logger.info(f"OCR found {len(ocr_tables)} tables")
                    # Save OCR tables as HTML/Excel/CSV/JSON
                    html = ""
                    for i, df in enumerate(ocr_tables):
                        html += df.to_html(index=False, border=1)
                    with open(os.path.join(out_dir, file_map["html"]), "w", encoding="utf-8") as f:
                        f.write(html)
                    with pd.ExcelWriter(os.path.join(out_dir, file_map["excel"]), engine='xlsxwriter') as writer:
                        for i, df in enumerate(ocr_tables):
                            df.to_excel(writer, sheet_name=f"OCR_Table_{i+1}", index=False)
                    with open(os.path.join(out_dir, file_map["csv"]), "w", encoding="utf-8") as f:
                        for df in ocr_tables:
                            df.to_csv(f, index=False)
                            f.write("\n\n")
                    json_data = []
                    for i, df in enumerate(ocr_tables):
                        json_data.append({
                            "table": i+1,
                            "columns": list(df.columns),
                            "rows": df.to_dict(orient='records')
                        })
                    import json
                    with open(os.path.join(out_dir, file_map["json"]), "w", encoding="utf-8") as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                    # No Tally XML for OCR
                    with open(os.path.join(out_dir, file_map["tallyxml"]), "w", encoding="utf-8") as f:
                        f.write("<!-- OCR se Tally XML generate nahi kiya gaya -->")
                    ocr_message = f"Yeh PDF image-based tha. OCR se {len(ocr_tables)} table mil gayi. Data thoda galat bhi ho sakta hai."
                    return len(ocr_tables), 0, None, None, ocr_used, ocr_message
                else:
                    ocr_message = "Yeh PDF image-based hai aur OCR se bhi koi table nahi mili. Shayad quality low hai ya table nahi hai."
                    return 0, 0, None, None, ocr_used, ocr_message
            except Exception as ocr_e:
                logger.error(f"OCR processing error: {ocr_e}")
                ocr_message = f"OCR processing failed: {str(ocr_e)}"
                return 0, 0, None, None, ocr_used, ocr_message
        else:
            ocr_message = "OCR not available on this server. Cannot process image-based PDFs."
            return 0, 0, None, None, ocr_used, ocr_message
    
    if not tables:
        return 0, 0, None, None, ocr_used, ocr_message
    
    # Use file_map for output names if provided
    if file_map is None:
        file_map = {
            "html": "tables.html",
            "excel": "tables.xlsx",
            "csv": "tables.csv",
            "json": "tables.json",
            "tallyxml": "tables_tally.xml"
        }
    
    try:
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
        return len(tables), len(non_blank_pages), opening, closing, ocr_used, ocr_message
        
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        raise Exception(f"Failed to save extracted data: {str(e)}")

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    password: str = Form(None),
    request: Request = None,
    _: None = Depends(check_origin)
):
    out_dir = None
    try:
        logger.info(f"Upload request received for file: {file.filename}")
        
        if not file.filename.lower().endswith('.pdf'):
            return {
                "success": False, 
                "error_code": "INVALID_FILE_TYPE",
                "message": "Only PDF files are allowed. Please upload a PDF file.",
                "details": "The uploaded file must have a .pdf extension."
            }
        
        pdf_bytes = await file.read()
        logger.info(f"File read successfully. Size: {len(pdf_bytes)} bytes")
        
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
            tables_found, pages_count, opening_balance, closing_balance, ocr_used, ocr_message = extract_and_save(
                pdf_bytes, out_dir, password=password, file_map=file_map)
            
            # Re-extract unique_tables for merged tables JSON (only if tables were found)
            merged_tables_json = []
            if tables_found > 0:
                try:
                    # Open PDF again for merged tables JSON
                    pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=password)
                    if pdf is not None and hasattr(pdf, 'pages') and pdf.pages is not None:
                        tables = []
                        unique_tables = {}
                        
                        for page_num, page in enumerate(pdf.pages, 1):
                            if page is None:
                                continue
                            for table in page.find_tables():
                                data = table.extract()
                                if data and len(data) > 1:
                                    df = pd.DataFrame(data[1:], columns=data[0])
                                    tables.append({"page": page_num, "data": df})
                                    headers_key = tuple(df.columns)
                                    if headers_key not in unique_tables:
                                        unique_tables[headers_key] = []
                                    unique_tables[headers_key].append(df)
                        
                        for headers, dfs in unique_tables.items():
                            merged_df = pd.concat(dfs, ignore_index=True)
                            merged_tables_json.append({
                                "columns": list(merged_df.columns),
                                "rows": merged_df.to_dict(orient="records")
                            })
                        
                        pdf.close()
                except Exception as e:
                    logger.warning(f"Failed to extract merged tables JSON: {e}")
                    # Continue without merged tables JSON
                
        except Exception as e:
            # Clean up if PDF was opened
            try:
                if 'pdf' in locals() and pdf is not None:
                    pdf.close()
            except:
                pass
                
            err_msg = str(e) if e is not None else "Unknown error"
            err_msg_lower = err_msg.lower() if err_msg else ""
            
            logger.error(f"Processing error: {err_msg}")
            
            if "password" in err_msg_lower or "encrypted" in err_msg_lower or "incorrect password" in err_msg_lower or "protected" in err_msg_lower:
                if out_dir and os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                if password:
                    return {
                        "success": False,
                        "error_code": "INCORRECT_PASSWORD", 
                        "message": "❌ Incorrect Password",
                        "details": "The password you provided is wrong. Please check and try again with the correct password."
                    }
                else:
                    return {
                        "success": False,
                        "error_code": "PASSWORD_REQUIRED",
                        "message": "🔒 Password Required",
                        "details": "This PDF is password protected. Please enter the password to extract tables."
                    }
            elif "corrupted" in err_msg_lower or "damaged" in err_msg_lower or "invalid" in err_msg_lower:
                if out_dir and os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                return {
                    "success": False,
                    "error_code": "CORRUPTED_FILE",
                    "message": "⚠️ File Corrupted",
                    "details": "The PDF file appears to be corrupted or damaged. Please try uploading a different file."
                }
            elif "too large" in err_msg_lower:
                if out_dir and os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                return {
                    "success": False,
                    "error_code": "FILE_TOO_LARGE",
                    "message": "📏 File Too Large",
                    "details": f"The PDF file is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
                }
            else:
                if out_dir and os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                return {
                    "success": False,
                    "error_code": "PROCESSING_ERROR",
                    "message": "❌ Processing Failed",
                    "details": f"Failed to process the PDF file. Error: {err_msg}"
                }
        
        if tables_found == 0:
            if out_dir and os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            if ocr_used:
                return {
                    "success": False,
                    "error_code": "NO_TABLES_FOUND",
                    "message": "🔍 No Tables Found",
                    "details": ocr_message if ocr_message else "No tables could be extracted from this PDF, even with OCR.",
                    "pages_count": pages_count
                }
            else:
                return {
                    "success": False,
                    "error_code": "NO_TABLES_FOUND",
                    "message": "📋 No Tables Found",
                    "details": f"Processed {pages_count} pages but found no extractable tables. This PDF might be image-based or contain no tabular data.",
                    "pages_count": pages_count
                }
        
        # Return download links
        links = {fmt: f"/download/{file_id}/{fmt}" for fmt in SUPPORTED_FORMATS}
        if ocr_used and ocr_message:
            return {
                "success": True,
                "tables_found": tables_found,
                "pages_count": pages_count,
                "file_id": file_id,
                "download_links": links,
                "output_file_names": file_map,
                "opening_balance": opening_balance,
                "closing_balance": closing_balance,
                "ocr_message": ocr_message
            }
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
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Clean up on error
        if out_dir and os.path.exists(out_dir):
            try:
                shutil.rmtree(out_dir)
            except:
                pass
        
        return {
            "success": False,
            "error_code": "SERVER_ERROR",
            "message": "Internal server error occurred.",
            "details": f"Error: {str(e)}"
        }

@app.get("/download/{file_id}/{fmt}")
def download_file(file_id: str, fmt: str):
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail="Download failed.")

@app.get("/")
def root():
    return {"message": "Production PDF Table Extractor API. POST /upload with PDF, get download links."}

@app.post("/test-pdf")
async def test_pdf(
    file: UploadFile = File(...),
    password: str = Form(None)
):
    """Test endpoint to debug PDF issues"""
    try:
        logger.info(f"Test request for file: {file.filename}")
        
        if not file.filename.lower().endswith('.pdf'):
            return {"error": "Not a PDF file"}
        
        pdf_bytes = await file.read()
        logger.info(f"File size: {len(pdf_bytes)} bytes")
        
        # Test without password
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=None)
            pdf.close()
            return {"status": "PDF opens without password", "password_protected": False}
        except Exception as e1:
            logger.info(f"Without password error: {e1}")
            
            # Test with empty password
            try:
                pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password="")
                pdf.close()
                return {"status": "PDF opens with empty password", "password_protected": False}
            except Exception as e2:
                logger.info(f"With empty password error: {e2}")
                
                # Test with provided password
                if password:
                    try:
                        pdf = pdfplumber.open(io.BytesIO(pdf_bytes), password=password)
                        pdf.close()
                        return {"status": "PDF opens with provided password", "password_protected": True, "password_correct": True}
                    except Exception as e3:
                        logger.info(f"With provided password error: {e3}")
                        return {
                            "status": "PDF is password protected but provided password is incorrect",
                            "password_protected": True,
                            "password_correct": False,
                            "error": str(e3)
                        }
                else:
                    return {
                        "status": "PDF is password protected but no password provided",
                        "password_protected": True,
                        "password_correct": False,
                        "error": str(e1)
                    }
                    
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return {"error": str(e)} 
