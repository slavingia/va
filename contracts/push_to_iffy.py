import os
import requests
import fitz # Use fitz (PyMuPDF)
from dotenv import load_dotenv
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Create a .env file in the 'contracts' directory (alongside this script)
# with the following content:
#
# IFFY_API_URL="http://your-iffy-api-url/api/v1/moderate"
# IFFY_AUTH_TOKEN="iffy_YOUR_TOKEN_HERE"
# IFFY_CLIENT_ID="your_client_id_here"
#
# Determine the script's directory to find the .env file reliably
script_dir = Path(__file__).parent.resolve()
dotenv_path = script_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

IFFY_API_URL = os.getenv("IFFY_API_URL")
IFFY_AUTH_TOKEN = os.getenv("IFFY_AUTH_TOKEN")
IFFY_CLIENT_ID = os.getenv("IFFY_CLIENT_ID")
PDF_FOLDER = script_dir / "pdfs" # Look for pdfs folder within the contracts directory

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extracts text content from a PDF file using fitz."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                page_text = page.get_text("text") # Extract plain text
                if page_text:
                    text += page_text + "\n" # Add newline between pages
            return text.strip() if text else None
    except FileNotFoundError:
        logging.error(f"PDF file not found: {pdf_path}")
        return None

def push_to_iffy(content: str, filename: str) -> bool:
    """Pushes extracted content to the IFFY API."""
    if not all([IFFY_API_URL, IFFY_AUTH_TOKEN, IFFY_CLIENT_ID]):
        logging.error("API URL, Auth Token, or Client ID is missing in environment variables.")
        return False

    headers = {
        "Authorization": f"Bearer {IFFY_AUTH_TOKEN}",
        "Content-Type": "application/json",
    }

    # Construct payload - using filename for name/contract_number and placeholders for others
    # Adjust these fields as needed based on your actual requirements
    payload = {
        "clientId": IFFY_CLIENT_ID,
        "name": filename, # Using filename (without extension) as name
        "entity": "contract", # Defaulting to 'contract' as per example
        "content": content,
        "vendor_name": None, # Placeholder
        "passthrough": False, # Defaulting as per example
        "fy25_value": None, # Placeholder
        "total_value": None, # Placeholder
        "contracting_officer": None, # Placeholder
        "contract_number": filename # Using filename (without extension) as contract_number
    }

    logging.info(f"Pushing data for {filename} to {IFFY_API_URL}...")
    try:
        response = requests.post(IFFY_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        logging.info(f"Successfully pushed {filename}. Status Code: {response.status_code}")
        # logging.debug(f"API Response: {response.text}") # Uncomment for detailed response logging
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to push {filename}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response Status: {e.response.status_code}")
            logging.error(f"Response Body: {e.response.text}")
        return False

# --- Main Execution ---

def main():
    if not PDF_FOLDER.is_dir():
        logging.error(f"PDF folder not found: {PDF_FOLDER}") # Use the Path object directly
        # Attempt to create the directory
        try:
            PDF_FOLDER.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created PDF folder: {PDF_FOLDER}")
            # Add a placeholder file to ensure the directory is kept if empty
            (PDF_FOLDER / ".placeholder").touch()
            logging.info(f"Please add your PDF files to the '{PDF_FOLDER}' directory and run the script again.")
        except OSError as e:
            logging.error(f"Could not create PDF folder {PDF_FOLDER}: {e}")
        return # Exit after creating the folder or if creation failed

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        logging.warning(f"No PDF files found in {PDF_FOLDER}") # Use the Path object directly
        return

    logging.info(f"Found {len(pdf_files)} PDF file(s) in {PDF_FOLDER}") # Use the Path object directly

    success_count = 0
    failure_count = 0

    for pdf_path in pdf_files:
        logging.info(f"Processing {pdf_path.name}...")
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text:
            filename_no_ext = pdf_path.stem # Filename without extension
            if push_to_iffy(extracted_text, filename_no_ext):
                success_count += 1
            else:
                failure_count += 1
        else:
            logging.warning(f"Skipping {pdf_path.name} due to extraction error or empty content.")
            failure_count += 1

    logging.info("--- Processing Complete ---")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Failed/Skipped: {failure_count}")


if __name__ == "__main__":
    main()
