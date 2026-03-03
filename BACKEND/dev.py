import os
import uuid
import zipfile
import shutil
import json
import traceback
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import pandas as pd
import tensorflow as tf
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

# ---- TensorFlow GPU / CPU check ----
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] TensorFlow is using GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"❌ GPU setup issue, falling back to CPU: {e}")
        tf.config.set_visible_devices([], 'GPU')
        print("[INFO] TensorFlow will use CPU.")
else:
    print("[WARNING] No GPU detected. TensorFlow will use CPU.")

# ---- Hugging Face cache directory ----
os.environ["DOCTR_CACHE_DIR"] = "weights1/hf_cache/doctr"
CACHE_DIR = "weights1/hf_cache"

# OCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, from_hub



# Your Hugging Face repo names
DET_REPO  = "Felix92/doctr-torch-db-mobilenet-v3-large"
RECO_REPO = "Felix92/doctr-torch-crnn-mobilenet-v3-large-french"

DET_REV  = None
RECO_REV = None

# Load models strictly from local cache
det_model  = from_hub(DET_REPO,  cache_dir=CACHE_DIR, local_files_only=True, revision=DET_REV)
reco_model = from_hub(RECO_REPO, cache_dir=CACHE_DIR, local_files_only=True, revision=RECO_REV)

print("Models loaded from local cache.")

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


BASE_INPUT = Path(os.getenv("USER_INPUT"))
BASE_OUTPUT = Path(os.getenv("USER_OUTPUT"))


BASE_INPUT.mkdir(exist_ok=True)
BASE_OUTPUT.mkdir(exist_ok=True)

# --- Default fields ---
DEFAULT_FIELDS = [
    "Beneficiary Name",
    "Beneficiary Address", 
    "Proforma Invoice Amount",
    "Proforma Invoice Currency Code",
    "Proforma Invoice Number",
    "Proforma Invoice Date",
    "Description of Goods",
    "Country of Origin",
    "Terms of Payment",
    "Tenor",
    "Incoterms",
    "Port of Loading",
    "Port of Discharge"
]

# --- Load Doctr OCR model globally (once) ---
try:
    # PREDICTOR = ocr_predictor(pretrained=True)  # uses HF_HOME cache
    PREDICTOR = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
    print("[INFO] OCR model loaded successfully")
except Exception as e:
    print(f"❌ Error loading OCR model: {e}")
    PREDICTOR = None

# --- OCR helper ---
def ocr_file_to_text(file_path: Path) -> str:
    """Extract text from PDF or image file using OCR"""
    if PREDICTOR is None:
        raise Exception("OCR model not available")
    
    try:
        doc = (
            DocumentFile.from_pdf(str(file_path))
            if file_path.suffix.lower() == ".pdf"
            else DocumentFile.from_images(str(file_path))
        )
        result = PREDICTOR(doc)
        text = result.render()
        print(f"[INFO] Extracted {len(text)} characters from {file_path.name}")
        return text
    except Exception as e:
        print(f"❌ OCR error for {file_path.name}: {e}")
        raise

# --- OpenAI extraction ---
def run_extraction(text: str, fields: list) -> dict:
    """Extract specific fields from text using OpenAI"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",  # Fixed: was "gpt-4.1" which doesn't exist
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert data extraction assistant. Extract ONLY the following fields from the document text: {', '.join(fields)}. "
                        "Return a valid JSON object with exactly these field names as keys and their extracted values as strings. "
                        "If a field is not found or cannot be determined, use an empty string as the value. "
                        "Output ONLY the JSON object, no additional text or formatting."
                    )
                },
                {"role": "user", "content": f"Extract the requested fields from this document text:\n\n{text}"}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        response = completion.choices[0].message.content.strip()
        
        # Clean response (remove potential markdown formatting)
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        print(f"[INFO] OpenAI response: {response[:200]}...")
        
        # Parse JSON response
        extracted_data = json.loads(response)
        
        # Ensure all requested fields are present
        result = {}
        for field in fields:
            result[field] = extracted_data.get(field, "")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw response: {response}")
        # Return empty values for all fields
        return {field: "" for field in fields}
    except Exception as e:
        print(f"❌ OpenAI extraction error: {e}")
        # Return empty values for all fields
        return {field: "" for field in fields}

# --- API endpoint ---
@app.post("/extract")
async def extract_fields(
    upload_file: UploadFile,
    fields: Optional[str] = Form(None)
):
    """
    Extract fields from uploaded document(s).
    
    Args:
        upload_file: Single PDF/image or ZIP of PDFs/images
        fields: JSON string array of field names to extract (optional)
    
    Returns:
        JSON object with extracted field values
    """
    print(f"[INFO] Processing file upload: {upload_file.filename}")
    print(f"[INFO] File size: {upload_file.size if hasattr(upload_file, 'size') else 'Unknown'}")
    
    try:
        # --- Parse fields from frontend ---
        fields_list = DEFAULT_FIELDS  # Default
        
        if fields and fields.strip():
            try:
                # Frontend sends JSON array, parse it
                parsed_fields = json.loads(fields)
                if isinstance(parsed_fields, list) and parsed_fields:
                    fields_list = [f.strip() for f in parsed_fields if f.strip()]
                    print(f"[INFO] Using custom fields: {fields_list}")
                else:
                    print("[INFO] Invalid fields format, using defaults")
            except json.JSONDecodeError:
                # Fallback: treat as newline-separated
                fields_list = [f.strip() for f in fields.strip().splitlines() if f.strip()]
                print(f"[INFO] Using newline-separated fields: {fields_list}")
        
        print(f"[INFO] Final fields list: {fields_list}")

        # --- Create unique folder for this request ---
        request_id = str(uuid.uuid4())
        input_dir = BASE_INPUT / request_id
        output_dir = BASE_OUTPUT / request_id
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Save uploaded file ---
        file_path = input_dir / upload_file.filename
        with open(file_path, "wb") as f:
            content = await upload_file.read()  # Read file content
            f.write(content)
        
        print(f"[INFO] Saved file to: {file_path}")

        # --- Handle ZIP ---
        files_to_process = []
        unsupported_files = []
        
        if upload_file.filename.lower().endswith(".zip"):
            print("[INFO] Processing ZIP file...")
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(input_dir)
                file_path.unlink()  # remove the original zip
                
                # Collect files to process and remove unsupported ones
                for file in input_dir.rglob("*"):
                    if file.is_file():
                        if file.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]:
                            files_to_process.append(file)
                        else:
                            unsupported_files.append(file.name)
                            try:
                                file.unlink()
                            except PermissionError:
                                print(f"⚠️ Permission denied when deleting {file.name}")
                                
            except zipfile.BadZipFile:
                return JSONResponse({"error": "Invalid ZIP file"}, status_code=400)
                
        # --- Validate single file uploads ---
        elif upload_file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            files_to_process.append(file_path)
        else:
            return JSONResponse(
                {"error": "Unsupported file type. Only PDF, images, or ZIP files are allowed."}, 
                status_code=400
            )

        print(f"[INFO] Files to process: {len(files_to_process)}")
        

        # --- Process files and collect all results ---
        all_extracted_results = []  # List of dicts, one per file
        processed_count = 0
        processing_errors = []

        for file in files_to_process:
            try:
                print(f"[INFO] Processing file: {file.name}")

                # Extract text using OCR
                text = ocr_file_to_text(file)
                if not text.strip():
                    print(f"⚠️ No text extracted from {file.name}")
                    processing_errors.append(f"No text extracted from {file.name}")
                    continue

                # Extract fields using OpenAI
                extracted = run_extraction(text, fields_list)

                # Save individual file results
                json_out = output_dir / f"{file.stem}_output.json"
                with open(json_out, "w", encoding="utf-8") as f:
                    json.dump(extracted, f, ensure_ascii=False, indent=4)

                all_extracted_results.append(extracted)
                processed_count += 1
                print(f"[INFO] Successfully processed {file.name}")

            except Exception as e:
                print(f"❌ Error processing {file.name}: {e}")
                print(traceback.format_exc())
                processing_errors.append(f"Error processing {file.name}: {str(e)}")
                continue

        # --- Save combined results as a list ---
        if all_extracted_results:
            # Save combined JSON (list of dicts)
            combined_json = output_dir / "combined_results.json"
            with open(combined_json, "w", encoding="utf-8") as f:
                json.dump(all_extracted_results, f, ensure_ascii=False, indent=4)

            # Save CSV
            df = pd.DataFrame(all_extracted_results)
            df.to_csv(output_dir / "results.csv", index=False)

        print(f"[INFO] Processing complete. Processed {processed_count} files")
        print(f"[INFO] Final results: {all_extracted_results}")

        # --- Clean up input files (optional) ---
        try:
            shutil.rmtree(input_dir)
        except Exception as e:
            print(f"⚠️ Could not clean up input directory: {e}")

        # --- Return results as a list with metadata as the last element ---
        response_list = all_extracted_results.copy()
        response_list.append({
            "_metadata": {
                "total_files_found": len(files_to_process),
                "successfully_processed": processed_count,
                "processing_errors": processing_errors,
                "unsupported_files": unsupported_files,
                "output_directory": str(output_dir)
            }
        })
        return response_list

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": f"Processing failed: {str(e)}"}, 
            status_code=500
        )

# --- Health check endpoint ---

from fastapi import Body
from pydantic import BaseModel

# --- New API endpoint ---
class ExtractWithUrlRequest(BaseModel):
    file_url: str
    fields: Optional[str] = None


@app.post("/extract_with_url")
async def extract_fields_with_url(body: ExtractWithUrlRequest):
    """
    Extract fields from files in a given directory path.
    Processes all supported files (PDF, PNG, JPG, JPEG) in the directory and subdirectories.
    
    Args:
        file_url: Directory path containing files to process
        fields: JSON string array of field names to extract (optional)
    
    Returns:
        JSON object with combined extracted field values from all processed files
    """
    import pathlib
    import zipfile
    
    file_path_str = body.file_url
    fields = body.fields
    print(f"[INFO] Processing directory path: {file_path_str}")
    
    try:
        # --- Parse fields from request ---
        fields_list = DEFAULT_FIELDS  # Default
        
        if fields and fields.strip():
            try:
                # Parse JSON array
                parsed_fields = json.loads(fields)
                if isinstance(parsed_fields, list) and parsed_fields:
                    fields_list = [f.strip() for f in parsed_fields if f.strip()]
                    print(f"[INFO] Using custom fields: {fields_list}")
                else:
                    print("[INFO] Invalid fields format, using defaults")
            except json.JSONDecodeError:
                # Fallback: treat as newline-separated
                fields_list = [f.strip() for f in fields.strip().splitlines() if f.strip()]
                print(f"[INFO] Using newline-separated fields: {fields_list}")
        
        print(f"[INFO] Final fields list: {fields_list}")


        # --- Create unique input and output folders for this request ---
        request_id = str(uuid.uuid4())
        input_dir = BASE_INPUT / request_id
        output_dir = BASE_OUTPUT / request_id
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Validate and process the path ---
        file_path = pathlib.Path(file_path_str)
        if not file_path.exists():
            return JSONResponse(
                {"error": f"Path does not exist: {file_path_str}"}, 
                status_code=400
            )

        # --- Copy file(s) or directory to USER_INPUTS/<request_id>/ ---
        try:
            if file_path.is_file():
                shutil.copy2(file_path, input_dir / file_path.name)
                print(f"[INFO] Copied file {file_path} to {input_dir / file_path.name}")
            elif file_path.is_dir():
                # Copy all files and subfolders
                for item in file_path.rglob("*"):
                    rel_path = item.relative_to(file_path)
                    dest_path = input_dir / rel_path
                    if item.is_dir():
                        dest_path.mkdir(parents=True, exist_ok=True)
                    else:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_path)
                        print(f"[INFO] Copied file {item} to {dest_path}")
            else:
                return JSONResponse(
                    {"error": f"Invalid path: {file_path_str}"},
                    status_code=400
                )
        except Exception as e:
            print(f"❌ Error copying files to USER_INPUTS: {e}")
            return JSONResponse(
                {"error": f"Failed to copy files to USER_INPUTS: {str(e)}"},
                status_code=500
            )

        # Now, process files from input_dir instead of file_path
        file_path = input_dir

        files_to_process = []
        unsupported_files = []
        temp_extract_dirs = []  # Track temp directories for cleanup
        
        # --- Collect files to process ---
        if file_path.is_dir():
            print(f"[INFO] Processing directory: {file_path}")
            
            # Recursively find all files in the directory
            for file_item in file_path.rglob("*"):
                if not file_item.is_file():
                    continue
                    
                file_ext = file_item.suffix.lower()
                
                if file_ext in [".pdf", ".png", ".jpg", ".jpeg"]:
                    files_to_process.append(file_item)
                    print(f"[INFO] Added file: {file_item.name}")
                    
                elif file_ext == ".zip":
                    # Handle ZIP files within the directory
                    try:
                        print(f"[INFO] Extracting ZIP file: {file_item.name}")
                        extract_dir = file_item.parent / f"temp_unzipped_{file_item.stem}_{uuid.uuid4()}"
                        temp_extract_dirs.append(extract_dir)
                        
                        with zipfile.ZipFile(file_item, "r") as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Add extracted files to processing list
                        for extracted_file in extract_dir.rglob("*"):
                            if extracted_file.is_file() and extracted_file.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]:
                                files_to_process.append(extracted_file)
                                print(f"[INFO] Added extracted file: {extracted_file.name}")
                            elif extracted_file.is_file():
                                unsupported_files.append(f"{file_item.name}/{extracted_file.name}")
                                
                    except zipfile.BadZipFile:
                        print(f"⚠️ Invalid ZIP file: {file_item.name}")
                        unsupported_files.append(file_item.name)
                    except Exception as e:
                        print(f"⚠️ Error processing ZIP {file_item.name}: {e}")
                        unsupported_files.append(file_item.name)
                        
                else:
                    unsupported_files.append(file_item.name)
                    
        elif file_path.is_file():
            # Handle single file case
            file_ext = file_path.suffix.lower()
            
            if file_ext in [".pdf", ".png", ".jpg", ".jpeg"]:
                files_to_process.append(file_path)
                print(f"[INFO] Processing single file: {file_path.name}")
                
            elif file_ext == ".zip":
                # Handle single ZIP file
                try:
                    print(f"[INFO] Extracting single ZIP file: {file_path.name}")
                    extract_dir = file_path.parent / f"temp_unzipped_{file_path.stem}_{uuid.uuid4()}"
                    temp_extract_dirs.append(extract_dir)
                    
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    for extracted_file in extract_dir.rglob("*"):
                        if extracted_file.is_file() and extracted_file.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]:
                            files_to_process.append(extracted_file)
                        elif extracted_file.is_file():
                            unsupported_files.append(extracted_file.name)
                            
                except zipfile.BadZipFile:
                    return JSONResponse(
                        {"error": "Invalid ZIP file"}, 
                        status_code=400
                    )
            else:
                return JSONResponse(
                    {"error": f"Unsupported file type: {file_ext}. Only PDF, images, or ZIP files are allowed."}, 
                    status_code=400
                )
        else:
            return JSONResponse(
                {"error": f"Invalid path: {file_path_str}"}, 
                status_code=400
            )

        if not files_to_process:
            return JSONResponse(
                {"error": "No supported files found in the specified path"}, 
                status_code=400
            )

        print(f"[INFO] Total files to process: {len(files_to_process)}")
        if unsupported_files:
            print(f"[INFO] Unsupported files found: {len(unsupported_files)} files")


        # --- Process each file and collect all results ---
        all_extracted_results = []  # List of dicts, one per file
        processed_count = 0
        processing_errors = []

        for file_item in files_to_process:
            try:
                print(f"[INFO] Processing file: {file_item.name}")

                # Extract text using OCR
                text = ocr_file_to_text(file_item)
                if not text.strip():
                    print(f"⚠️ No text extracted from {file_item.name}")
                    processing_errors.append(f"No text extracted from {file_item.name}")
                    continue

                # Extract fields using OpenAI
                extracted = run_extraction(text, fields_list)

                # Save individual file results
                safe_filename = "".join(c for c in file_item.stem if c.isalnum() or c in (' ', '-', '_')).rstrip()
                json_out = output_dir / f"{safe_filename}_output.json"
                with open(json_out, "w", encoding="utf-8") as f:
                    json.dump(extracted, f, ensure_ascii=False, indent=4)

                all_extracted_results.append(extracted)
                processed_count += 1
                print(f"[INFO] Successfully processed {file_item.name}")

            except Exception as e:
                error_msg = f"Error processing {file_item.name}: {str(e)}"
                print(f"❌ {error_msg}")
                print(traceback.format_exc())
                processing_errors.append(error_msg)
                continue

        # --- Save combined results as a list ---
        if all_extracted_results:
            # Save combined JSON (list of dicts)
            combined_json = output_dir / "combined_results.json"
            with open(combined_json, "w", encoding="utf-8") as f:
                json.dump(all_extracted_results, f, ensure_ascii=False, indent=4)

            # Save CSV
            try:
                df = pd.DataFrame(all_extracted_results)
                df.to_csv(output_dir / "results.csv", index=False, encoding="utf-8")
            except Exception as e:
                print(f"⚠️ Error saving CSV: {e}")

        # --- Cleanup temporary extraction directories ---
        for temp_dir in temp_extract_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    print(f"[INFO] Cleaned up temporary directory: {temp_dir.name}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary directory {temp_dir}: {e}")

        print(f"[INFO] Processing complete. Successfully processed {processed_count}/{len(files_to_process)} files")

        print(f"[INFO] Final results: {all_extracted_results}")

        # --- Prepare response ---
        # --- Return results as a list with metadata as the last element ---
        response_list = all_extracted_results.copy()
        response_list.append({
            "_metadata": {
                "total_files_found": len(files_to_process),
                "successfully_processed": processed_count,
                "processing_errors": processing_errors,
                "unsupported_files": unsupported_files,
                "output_directory": str(output_dir)
            }
        })
        print("\nFinal Results: ", response_list)
        return response_list

    except Exception as e:
        # --- Cleanup temporary directories in case of error ---
        try:
            for temp_dir in temp_extract_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        except:
            pass
            
        print(f"❌ Error during processing: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": f"Processing failed: {str(e)}"}, 
            status_code=500
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_model_loaded": PREDICTOR is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

