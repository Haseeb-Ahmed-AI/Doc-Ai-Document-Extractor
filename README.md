# OCR Document Processing API
## Table of Contents

-   Overview
-   Key Features
-   Architecture
-   Tech Stack
-   Project Structure
-   Installation
-   Running the API
-   API Endpoints
-   Sample Usage
-   Output Files
-   Contributing
-   License

## Overview

This system provides a complete backend pipeline for converting document
files into structured JSON using: 1. **Local OCR inference** (fast,
GPU-enabled, offline-capable)\
2. **OpenAI GPT extraction** for user-defined fields\
3. **FastAPI** for easy integration with front-end apps or automation
workflows

It handles single files, images, PDFs, or entire directories and ZIP
archives.

## Key Features
(Complete project is paid for more query mail at ha2153329@gmail.com)
### OCR Layer

-   Uses **Doctr OCR** with locally stored `.pt` model weights\
-   Supports **PDF**, **PNG**, **JPG**, **JPEG**\
-   GPU acceleration when available\
-   Efficient text extraction for multi-page documents

### AI Extraction Layer

-   Uses OpenAI GPT to extract any custom fields\
-   Returns pure JSON output\
-   Automatically cleans and validates the response

### File Handling

-   Handles:
    -   Single files\
    -   ZIP archives\
    -   Entire directory paths\
-   Auto-sanitizes unsupported files\
-   Creates isolated input/output folders per request

### Output Generation

-   Outputs:
    -   JSON per file\
    -   Combined JSON for all processed files\
    -   CSV (tabular format)\
-   Includes metadata:
    -   Total files processed\
    -   Errors\
    -   Unsupported files\
    -   Output directory path

## Architecture

    PDF / Image / ZIP → OCR (Doctr) → Extracted Text → GPT Field Extraction → JSON / CSV → Response

## Tech Stack

-   FastAPI\
-   Doctr OCR\
-   TensorFlow\
-   OpenAI GPT\
-   Pandas\
-   Python 3.10+

## Project Structure

    weights1/
        hf_cache/

    src/
        check.py
        dev.py

    .env
    USER_INPUT/
    USER_OUTPUT/

## Installation

    pip install -r requirements.txt

Create `.env`:

    OPENAI_API_KEY=your_key
    USER_INPUT=USER_INPUT
    USER_OUTPUT=USER_OUTPUT

## Running the API

    uvicorn dev:app --host 0.0.0.0 --port 8000

## API Endpoints

### `/extract`

Upload PDF/Image/ZIP + optional custom field list.

### `/extract_with_url`

Provide a directory path, process all files.

### `/health`

Health status of OCR + OpenAI.

## Sample Python Request

``` python
import requests, json

files = {'upload_file': open('invoice.pdf','rb')}
fields = json.dumps(["Invoice Number","Beneficiary Name"])

resp = requests.post("http://localhost:8000/extract", files=files, data={"fields": fields})
print(resp.json())
```
##Dashboard
<img width="947" height="418" alt="image" src="https://github.com/user-attachments/assets/69de22df-7862-4640-b09d-37383d788578" />

## Output Files

-   Per-file JSON\
-   Combined JSON\
-   results.csv\
-   Metadata summary

## Contributing

Pull requests are welcome.

## License

MIT License
