#!/usr/bin/env python3
import os
import csv
import glob
import json
import shutil
import random
import asyncio
import aiohttp
from datetime import datetime
import time
import PyPDF2
import openai
import pandas as pd
from dotenv import load_dotenv
import fitz  # pip install pymupdf
import hashlib
import argparse

# Load environment variables
load_dotenv('.env.local')
openai_api_key = os.getenv('openai_api_key')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Try to install pycryptodome if not available
try:
    import Crypto
except ImportError:
    print("Installing pycryptodome for encrypted PDF support...")
    import subprocess
    subprocess.call(['pip', 'install', 'pycryptodome'])
    print("PyCryptodome installed.")

# Set up OpenAI client
client = openai.AsyncOpenAI(api_key=openai_api_key)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_folders = [
    "ri.foundry.main.dataset.46d7aad4-5b2d-496e-8a0b-dea9faaf37cf",
    "ri.foundry.main.dataset.81bbfe85-891e-4883-8402-e0971d11ca73"
]
output_csv = os.path.join(script_dir, "contract_analysis.csv")
processed_log = os.path.join(script_dir, "processed_files.json")
reviewed_contracts_csv = os.path.join(script_dir, "reviewed_contracts.csv")
reviewed_pdfs_dir = os.path.join(script_dir, "reviewed_pdfs")

# Create directories if they don't exist
os.makedirs(script_dir, exist_ok=True)
os.makedirs(reviewed_pdfs_dir, exist_ok=True)

# Print file paths for debugging
print(f"Script directory: {script_dir}")
print(f"Output CSV path: {output_csv}")
print(f"Processed log path: {processed_log}")
print(f"Reviewed contracts CSV path: {reviewed_contracts_csv}")
print(f"Reviewed PDFs directory: {reviewed_pdfs_dir}")

# Load list of already processed files
processed_files = []
if os.path.exists(processed_log):
    with open(processed_log, 'r') as f:
        processed_files = json.load(f)
        print(f"Loaded {len(processed_files)} already processed files from {processed_log}")
else:
    print(f"No processed files log found at {processed_log}, will create new one")

# Initialize or load reviewed contracts
reviewed_contracts = set()
if os.path.exists(reviewed_contracts_csv):
    df_reviewed = pd.read_csv(reviewed_contracts_csv)
    if 'contract_number' in df_reviewed.columns:
        reviewed_contracts = set(df_reviewed['contract_number'].tolist())
        print(f"Loaded {len(reviewed_contracts)} already reviewed contracts from {reviewed_contracts_csv}")
else:
    print(f"No reviewed contracts CSV found at {reviewed_contracts_csv}, will create new one")

# Add a cache directory for extracted text
text_cache_dir = os.path.join(script_dir, "text_cache")
os.makedirs(text_cache_dir, exist_ok=True)

def get_cached_text_path(pdf_path):
    """Get the path to the cached text file for a PDF"""
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return os.path.join(text_cache_dir, f"{pdf_hash}.txt")

def extract_text_from_pdf_with_cache(pdf_path):
    """Extract text from a PDF file with caching"""
    cache_path = get_cached_text_path(pdf_path)
    
    # Check if cached text exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading cached text for {pdf_path}: {e}")
    
    # Extract text if not cached
    text = extract_text_from_pdf(pdf_path)
    
    # Cache the extracted text
    if text:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            print(f"Error caching text for {pdf_path}: {e}")
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pymupdf (much faster than PyPDF2)"""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                try:
                    text += page.get_text()
                except Exception as page_error:
                    print(f"Error extracting text from page in {pdf_path}: {page_error}")
            return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def get_all_pdf_files():
    """Get all PDF files from the specified dataset folders"""
    all_pdfs = []
    
    for dataset_folder in dataset_folders:
        dataset_path = os.path.join(script_dir, dataset_folder)
        if os.path.exists(dataset_path):
            print(f"Searching for PDFs in {dataset_path}...")
            pdfs = glob.glob(os.path.join(dataset_path, "**/*.pdf"), recursive=True)
            all_pdfs.extend(pdfs)
            print(f"Found {len(pdfs)} PDFs in {dataset_folder}")
        else:
            print(f"Warning: Dataset folder {dataset_folder} not found at {dataset_path}")
    
    return all_pdfs

def update_reviewed_contracts(contract_numbers):
    """Update the reviewed contracts CSV file"""
    new_records = []
    for contract_num in contract_numbers:
        if contract_num != "Not found" and contract_num != "Error" and contract_num not in reviewed_contracts:
            new_records.append({"contract_number": contract_num, "review_date": datetime.now().strftime("%Y-%m-%d")})
    
    if new_records:
        print(f"Adding {len(new_records)} new contract numbers to reviewed_contracts.csv")
        # Create dataframe with new records
        new_df = pd.DataFrame(new_records)
        
        if os.path.exists(reviewed_contracts_csv):
            # Append to existing file
            print(f"Appending to existing file: {reviewed_contracts_csv}")
            existing_df = pd.read_csv(reviewed_contracts_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(reviewed_contracts_csv, index=False)
            print(f"Updated {reviewed_contracts_csv} with {len(combined_df)} total records")
        else:
            # Create new file
            print(f"Creating new file: {reviewed_contracts_csv}")
            new_df.to_csv(reviewed_contracts_csv, index=False)
            print(f"Created {reviewed_contracts_csv} with {len(new_df)} records")
        
        # Update the in-memory set
        reviewed_contracts.update([record["contract_number"] for record in new_records])
    else:
        print("No new contract numbers to add to reviewed_contracts.csv")

def copy_pdf_to_reviewed_folder(pdf_path, contract_number):
    """Copy the PDF to the reviewed folder with the contract number as the filename"""
    if contract_number in ["Not found", "Error", ""]:
        print(f"Skipping copy of {pdf_path} because contract number is invalid: {contract_number}")
        return False
    
    # Clean contract number to make it a valid filename
    # Remove any characters that are not allowed in filenames
    valid_contract_number = ''.join(c for c in contract_number if c.isalnum() or c in '._- ')
    
    # If after cleaning we have an empty string, use original filename
    if not valid_contract_number:
        print(f"Contract number '{contract_number}' contains only invalid characters for a filename")
        return False
    
    # Ensure the reviewed_pdfs directory exists
    try:
        os.makedirs(reviewed_pdfs_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating reviewed_pdfs directory: {e}")
        return False
    
    # Create destination filename
    dest_file = os.path.join(reviewed_pdfs_dir, f"{valid_contract_number}.pdf")
    
    try:
        # Check if source file exists
        if not os.path.exists(pdf_path):
            print(f"Source file does not exist: {pdf_path}")
            return False
            
        # Check if destination file already exists
        if os.path.exists(dest_file):
            print(f"File already exists: {dest_file}")
            return True
        
        # Copy the file
        shutil.copy2(pdf_path, dest_file)
        print(f"Copied {pdf_path} to {dest_file}")
        return True
    except Exception as e:
        print(f"Error copying {pdf_path} to {dest_file}: {e}")
        return False

async def analyze_contract_async(text, session, pass_number=1, max_retries=3):
    """Analyze contract text using LLM asynchronously with retries"""
    if pass_number == 1:
        prompt = f"""
        Analyze the following contract text and extract the basic information below. If you can't find specific information, write "Not found".
        
        CONTRACT TEXT:
        {text[:10000]}  # Using first 10000 chars to stay within token limits
        
        Please extract the following information:
        1. Contract Number/PIID
        2. Parent Contract Number (if this is a child contract)
        3. Contract Description - IMPORTANT: Provide a DETAILED 1-2 sentence description that clearly explains what the contract is for. 
           Include WHO the vendor is, WHAT specific products or services they provide, and WHO the end recipients or beneficiaries are.
           For example, instead of "Custom powered wheelchair", write "Contract with XYZ Medical Equipment Provider to supply custom-powered 
           wheelchairs and related maintenance services to veteran patients at VA medical centers."
        4. Vendor Name
        5. Total Contract Value (in USD)
        6. FY 25 Value (in USD)
        7. Remaining Obligations (in USD)
        8. Contracting Officer Name
        9. Is this an IDIQ contract? (true/false)
        10. Is this a modification? (true/false)
        
        PROVIDE YOUR ANALYSIS IN THE FOLLOWING JSON FORMAT:
        {{
            "contract_number": "",
            "parent_contract_number": "",
            "description": "",
            "vendor_name": "",
            "total_value": "",
            "fy25_value": "",
            "remaining_obligations": "",
            "contracting_officer": "",
            "is_idiq": false,
            "is_modification": false
        }}
        """
    else:
        prompt = f"""
        Based on the following contract information, determine if this contract is "munchable" based on these criteria:
        
        CONTRACT INFORMATION:
        {text[:10000]}  # Using first 10000 chars to stay within token limits
        
        First, determine if this is a contract modification by looking for keywords like:
        - "Modification"
        - "Amendment"
        - "Change Order"
        - "Supplemental Agreement"
        - "Modification No."
        - "Amendment No."
        
        Then, determine if this is an IDIQ (Indefinite Delivery, Indefinite Quantity) contract by looking for:
        - "IDIQ" or "Indefinite Delivery/Indefinite Quantity"
        - "Task Order" or "Delivery Order"
        - "Multiple Award" or "Single Award"
        - "Ordering Period"
        - "Maximum Ordering Value"
        
        Also identify any parent contract relationships by looking for:
        - "Parent Contract" or "Master Contract"
        - "Call Order" or "Order Under"
        - "Task Order Under"
        - "Delivery Order Under"
        
        Then, evaluate if this contract is "munchable" based on these criteria:
        - If this is a contract modification, mark it as "N/A" for munchable status
        - If this is an IDIQ contract:
          * For medical devices/equipment: NOT MUNCHABLE
          * For recruiting/staffing: MUNCHABLE
          * For other services: Consider termination if not core medical/benefits
        - Level 0: Direct patient care (e.g., bedside nurse) - NOT MUNCHABLE
        - Level 1: Necessary consultants that can't be insourced - NOT MUNCHABLE
        - Level 2+: Multiple layers removed from veterans care - MUNCHABLE
        - Contracts related to "diversity, equity, and inclusion" (DEI) initiatives - MUNCHABLE
        - Services that could easily be replaced by in-house W2 employees - MUNCHABLE
        
        IMPORTANT EXCEPTIONS - These are NOT MUNCHABLE:
        - Third-party financial audits and compliance reviews
        - Medical equipment audits and certifications (e.g., MRI, CT scan, nuclear medicine equipment)
        - Nuclear physics and radiation safety audits for medical equipment
        - Medical device safety and compliance audits
        - Healthcare facility accreditation reviews
        - Clinical trial audits and monitoring
        - Medical billing and coding compliance audits
        - Healthcare fraud and abuse investigations
        - Medical records privacy and security audits
        - Healthcare quality assurance reviews
        - Community Living Center (CLC) surveys and inspections
        - State Veterans Home surveys and inspections
        - Long-term care facility quality surveys
        - Nursing home resident safety and care quality reviews
        - Assisted living facility compliance surveys
        - Veteran housing quality and safety inspections
        - Residential care facility accreditation reviews
        
        Key considerations:
        - Direct patient care involves: physical examinations, medical procedures, medication administration
        - Distinguish between medical/clinical and psychosocial support
        - Installation, configuration, or implementation of Electronic Medical Record (EMR) systems or healthcare IT systems directly supporting patient care should be classified as NOT munchable. Contracts related to diversity, equity, and inclusion (DEI) initiatives or services that could be easily handled by in-house W2 employees should be classified as MUNCHABLE. Consider 'soft services' like healthcare technology management, data management, administrative consulting, portfolio management, case management, and product catalog management as MUNCHABLE. For contract modifications, mark the munchable status as 'N/A'. For IDIQ contracts, be more aggressive about termination unless they are for core medical services or benefits processing.
        
        Specific services that should be classified as MUNCHABLE (these are "soft services" or consulting-type services):
        - Healthcare technology management (HTM) services
        - Data Commons Software as a Service (SaaS)
        - Administrative management and consulting services
        - Data management and analytics services
        - Product catalog or listing management
        - Planning and transition support services
        - Portfolio management services
        - Operational management review
        - Technology guides and alerts services
        - Case management administrative services
        - Case abstracts, casefinding, follow-up services
        - Enterprise-level portfolio management
        - Support for specific initiatives (like PACT Act)
        - Administrative updates to product information
        - Research data management platforms or repositories
        - Drug/pharmaceutical lifecycle management and pricing analysis
        - Backup Contracting Officer's Representatives (CORs) or administrative oversight roles
        - Modernization and renovation extensions not directly tied to patient care
        - DEI (Diversity, Equity, Inclusion) initiatives
        - Climate & Sustainability programs
        - Consulting & Research Services
        - Non-Performing/Non-Essential Contracts
        - Recruitment Services
        
        Important clarifications based on past analysis errors:
        2. Lifecycle management of drugs/pharmaceuticals IS MUNCHABLE (different from direct supply)
        3. Backup administrative roles (like alternate CORs) ARE MUNCHABLE as they create duplicative work
        4. Contract extensions for renovations/modernization ARE MUNCHABLE unless directly tied to patient care
        
        Direct patient care that is NOT MUNCHABLE includes:
        - Conducting physical examinations
        - Administering medications and treatments
        - Performing medical procedures and interventions
        - Monitoring and assessing patient responses
        - Supply of actual medical products (pharmaceuticals, medical equipment)
        - Maintenance of critical medical equipment
        - Custom medical devices (wheelchairs, prosthetics)
        - Essential therapeutic services with proven efficacy
        
        For maintenance contracts, consider whether pricing appears reasonable. If maintenance costs seem excessive, flag them as potentially over-priced despite being necessary.
        
        Services that can be easily insourced (MUNCHABLE):
        - Video production and multimedia services
        - Customer support/call centers
        - PowerPoint/presentation creation
        - Recruiting and outreach services
        - Public affairs and communications
        - Administrative support
        - Basic IT support (non-specialized)
        - Content creation and writing
        - Training services (non-specialized)
        - Event planning and coordination
        
        Parent-Child Contract Analysis:
        - If this is a child contract (has a parent contract):
          * If the parent contract is NOT MUNCHABLE, this child contract should also be NOT MUNCHABLE
          * If the parent contract is MUNCHABLE, evaluate this child contract independently
        - If this is a parent contract:
          * Consider the overall scope and purpose of the parent contract
          * If it's a master contract for essential services, it may be NOT MUNCHABLE even if individual task orders could be
          * If it's a master contract for administrative or support services, it may be MUNCHABLE
        
        PROVIDE YOUR ANALYSIS IN THE FOLLOWING JSON FORMAT:
        {{
            "contract_number": "The contract number from the text",
            "munchable": "true" if munchable, "false" if not munchable, "N/A" if modification,
            "munchable_reason": "A clear explanation of why this contract is munchable or not",
            "parent_contract_number": "The parent contract number if found, otherwise empty string",
            "is_child_contract": true/false,
            "parent_munchable_status": "The munchable status of the parent contract if known, otherwise empty string"
        }}
        """
    
    for attempt in range(max_retries):
        try:
            print(f"Calling OpenAI API to analyze contract (Pass {pass_number}, Attempt {attempt + 1}/{max_retries})...")
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes government contracts. Always provide comprehensive few-sentence descriptions that explain WHO the contract is with, WHAT specific services/products are provided, and WHO benefits from these services. Remember that contracts for EMR systems and healthcare IT infrastructure directly supporting patient care should be classified as NOT munchable. Contracts related to diversity, equity, and inclusion (DEI) initiatives or services that could be easily handled by in-house W2 employees should be classified as MUNCHABLE. Consider 'soft services' like healthcare technology management, data management, administrative consulting, portfolio management, case management, and product catalog management as MUNCHABLE. For contract modifications, mark the munchable status as 'N/A'. For IDIQ contracts, be more aggressive about termination unless they are for core medical services or benefits processing."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                timeout=30  # 30 second timeout
            )
            
            result = json.loads(response.choices[0].message.content)
            if pass_number == 1:
                print(f"First pass analysis completed. Contract number: {result.get('contract_number', 'Not found')}")
            else:
                print(f"Second pass analysis completed. Munchable status: {result.get('munchable', 'Error')}")
            return result
            
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"Error analyzing contract on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                if pass_number == 1:
                    return {
                        "contract_number": "Error",
                        "parent_contract_number": "Error",
                        "description": str(e)[:100],
                        "vendor_name": "Error",
                        "total_value": "Error",
                        "fy25_value": "Error",
                        "remaining_obligations": "Error",
                        "contracting_officer": "Error",
                        "is_idiq": False,
                        "is_modification": False
                    }
                else:
                    return {
                        "contract_number": "Error",
                        "munchable": "Error",
                        "munchable_reason": "Error in analysis"
                    }
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def extract_text_from_pdf_async(pdf_path):
    """Extract text from a PDF file asynchronously using pymupdf"""
    try:
        # Run the synchronous PDF extraction in a thread pool to not block
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, lambda: extract_text_from_pdf(pdf_path))
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

async def extract_text_batch_parallel(pdf_batch):
    """Extract text from a batch of PDFs in parallel"""
    tasks = []
    for pdf_path in pdf_batch:
        # Check cache first
        cache_path = get_cached_text_path(pdf_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    tasks.append(asyncio.create_task(asyncio.sleep(0, result=text)))
                    continue
            except Exception as e:
                print(f"Error reading cached text for {pdf_path}: {e}")
        
        # If not in cache or cache read failed, extract text
        task = asyncio.create_task(extract_text_from_pdf_async(pdf_path))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and cache successful extractions
    processed_results = []
    for pdf_path, result in zip(pdf_batch, results):
        if isinstance(result, Exception):
            print(f"Error processing {pdf_path}: {result}")
            processed_results.append("")
            continue
            
        if result:
            # Cache the extracted text
            cache_path = get_cached_text_path(pdf_path)
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(result)
            except Exception as e:
                print(f"Error caching text for {pdf_path}: {e}")
        
        processed_results.append(result)
    
    return processed_results

async def process_contract_batch(contract_batch, pass_number=1):
    """Process a batch of contracts in parallel"""
    batch_start_time = time.time()
    tasks = []
    
    # Extract text from all PDFs in the batch in parallel
    pdf_paths = [result.get('file_path') for result in contract_batch if result.get('file_path')]
    print(f"Extracting text from {len(pdf_paths)} PDFs in parallel...")
    extracted_texts = await extract_text_batch_parallel(pdf_paths)
    
    # For second pass, we need to build a mapping of contract numbers to their text content
    contract_texts = {}
    if pass_number == 2:
        for result, text in zip(contract_batch, extracted_texts):
            if text:
                contract_texts[result.get('contract_number')] = text
    
    for pdf_path, text in zip(pdf_paths, extracted_texts):
        try:
            if not text:
                print(f"No text extracted from {pdf_path}, skipping analysis")
                processed_files.append(pdf_path)
                continue
            
            # For second pass, include parent and child contract text if available
            if pass_number == 2:
                # Find the contract number for this PDF
                current_contract = next((r for r in contract_batch if r.get('file_path') == pdf_path), None)
                if current_contract:
                    contract_number = current_contract.get('contract_number')
                    parent_number = current_contract.get('parent_contract_number')
                    
                    # Add parent contract text if available
                    if parent_number and parent_number in contract_texts:
                        text += f"\n\nPARENT CONTRACT TEXT:\n{contract_texts[parent_number]}"
                    
                    # Find and add child contract texts if any
                    child_contracts = [r for r in contract_batch if r.get('parent_contract_number') == contract_number]
                    if child_contracts:
                        text += "\n\nCHILD CONTRACT TEXTS:"
                        for child in child_contracts:
                            child_number = child.get('contract_number')
                            if child_number in contract_texts:
                                text += f"\n\nChild Contract {child_number}:\n{contract_texts[child_number]}"
            
            # Create task for contract analysis
            task = asyncio.create_task(analyze_contract_async(text, None, pass_number))
            tasks.append((pdf_path, task))
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    # Wait for all tasks to complete
    results = []
    for pdf_path, task in tasks:
        try:
            analysis = await task
            if pass_number == 1:
                contract_number = analysis.get('contract_number', 'Not found')
                
                # Skip if this contract number has already been reviewed
                if contract_number in reviewed_contracts and contract_number not in ["Not found", "Error"]:
                    print(f"Skipping already reviewed contract: {contract_number}")
                    processed_files.append(pdf_path)
                    continue
                
                # Add to results with explicit munchable and reason columns
                results.append({
                    'contract_number': contract_number,
                    'parent_contract_number': analysis.get('parent_contract_number', 'Not found'),
                    'description': analysis.get('description', 'Not found'),
                    'vendor_name': analysis.get('vendor_name', 'Not found'),
                    'contracting_officer': analysis.get('contracting_officer', 'Not found'),
                    'total_value': analysis.get('total_value', 'Not found'),
                    'fy25_value': analysis.get('fy25_value', 'Not found'),
                    'remaining_obligations': analysis.get('remaining_obligations', 'Not found'),
                    'is_idiq': analysis.get('is_idiq', False),
                    'is_modification': analysis.get('is_modification', False),
                    'file_path': pdf_path,
                    'munchable': '',  # Explicitly add blank munchable column
                    'reason': ''  # Explicitly add blank reason column
                })
                # Copy PDF to reviewed folder with contract number as the filename
                if contract_number not in ["Not found", "Error"]:
                    copy_pdf_to_reviewed_folder(pdf_path, contract_number)
            else:
                # Second pass - update munchable status
                contract_number = analysis.get('contract_number', 'Not found')
                munchable_status = analysis.get('munchable', 'Error')
                munchable_reason = analysis.get('munchable_reason', '')
                
                # Find and update the corresponding result from first pass
                for result in contract_batch:
                    if result.get('contract_number') == contract_number:
                        result['munchable'] = str(munchable_status).lower()  # Convert to lowercase string
                        result['reason'] = munchable_reason if munchable_reason and munchable_reason != 'nan' else 'No reason provided'
                        results.append(result)
                        print(f"Updated munchable status for contract {contract_number}: {munchable_status}")
                        print(f"Reason: {result['reason']}")
                        break
            
            # Mark as processed
            processed_files.append(pdf_path)
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    print(f"Batch processing completed in {batch_duration:.2f} seconds")
    return results

async def main_async():
    start_time = time.time()
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Process contract PDFs')
    parser.add_argument('--second-pass-only', action='store_true', help='Skip first pass and only do second pass analysis')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with a small batch of files')
    args = parser.parse_args()
    
    # Get all PDF files from dataset folders
    pdf_files = get_all_pdf_files()
    
    if not pdf_files:
        print("No PDF files found in the specified dataset folders.")
        return
    
    # Filter out already processed files
    files_to_process = [f for f in pdf_files if f not in processed_files]
    
    if not files_to_process and not args.second_pass_only:
        print("No new contracts to process.")
        return
    
    # If in test mode, only process a small batch
    if args.test_mode:
        files_to_process = files_to_process[:3]  # Process only 3 files for testing
        print(f"Test mode: Processing {len(files_to_process)} files")
    
    print(f"Found {len(files_to_process)} new contracts to process.")
    
    all_results = []
    batch_size = 1000
    
    # Load existing results if doing second pass only
    if args.second_pass_only:
        if os.path.exists(output_csv):
            print("Loading existing results for second pass analysis...")
            existing_df = pd.read_csv(output_csv)
            
            # Add munchable and reason columns if they don't exist
            if 'munchable' not in existing_df.columns:
                existing_df['munchable'] = ''
                print("Added munchable column to existing results")
            
            if 'reason' not in existing_df.columns:
                existing_df['reason'] = ''
                print("Added reason column to existing results")
                
            # Save the updated DataFrame with the new columns
            existing_df.to_csv(output_csv, index=False)
            print("Updated CSV with munchable and reason columns")
            
            all_results = existing_df.to_dict('records')
            
            # Validate that we have results to process
            if not all_results:
                print("No existing results found for second pass analysis.")
                return
                
            print(f"Loaded {len(all_results)} existing results for second pass analysis")
            
            # In test mode, only process first batch
            if args.test_mode:
                all_results = all_results[:3]
                print(f"Test mode: Processing {len(all_results)} existing results")
                
                # Ensure the records have munchable and reason fields for test mode
                for result in all_results:
                    if 'munchable' not in result:
                        result['munchable'] = ''
                    if 'reason' not in result:
                        result['reason'] = ''
        else:
            print("No existing results found for second pass analysis.")
            return
    else:
        # First pass: Process all contracts to establish basic information
        print("Starting first pass: Extracting basic contract information...")
        
        # First pass: Process all contracts in batches
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            print(f"Processing first pass batch {i//batch_size + 1}/{(len(files_to_process) + batch_size - 1)//batch_size}")
            
            # Process batch
            batch_results = await process_contract_batch(batch, pass_number=1)
            all_results.extend(batch_results)
            
            # Update processed files log after each batch
            with open(processed_log, 'w') as f:
                json.dump(processed_files, f)
            print(f"Updated processed files log with {len(processed_files)} total files")
            
            # Update reviewed contracts CSV after each batch
            batch_contract_numbers = [r['contract_number'] for r in batch_results]
            update_reviewed_contracts(batch_contract_numbers)
            
            # Save intermediate results after each batch
            if batch_results:
                if os.path.exists(output_csv):
                    # Read existing CSV and append new results
                    print(f"Appending first pass batch results to existing analysis CSV: {output_csv}")
                    existing_df = pd.read_csv(output_csv)
                    
                    # Ensure the required columns exist in existing DataFrame
                    if 'munchable' not in existing_df.columns:
                        existing_df['munchable'] = ''
                    if 'reason' not in existing_df.columns:
                        existing_df['reason'] = ''
                    
                    new_df = pd.DataFrame(batch_results)
                    
                    # Ensure new DataFrame has munchable and reason columns
                    if 'munchable' not in new_df.columns:
                        new_df['munchable'] = ''
                    if 'reason' not in new_df.columns:
                        new_df['reason'] = ''
                    
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(output_csv, index=False)
                    print(f"Updated {output_csv} with {len(combined_df)} total records")
                else:
                    # Create new CSV with results
                    print(f"Creating new analysis CSV with first pass batch results: {output_csv}")
                    df = pd.DataFrame(batch_results)
                    
                    # Ensure DataFrame has munchable and reason columns
                    if 'munchable' not in df.columns:
                        df['munchable'] = ''
                    if 'reason' not in df.columns:
                        df['reason'] = ''
                    
                    df.to_csv(output_csv, index=False)
                    print(f"Created {output_csv} with {len(df)} records")
            
            # In test mode, only process first batch
            if args.test_mode:
                break
        
        print("First pass completed for all contracts.")
        print(f"Total contracts processed in first pass: {len(all_results)}")
    
    # Second pass: Analyze munchable status
    print("Starting second pass: Analyzing munchable status...")
    
    # Process second pass in batches
    for i in range(0, len(all_results), batch_size):
        second_pass_batch = all_results[i:i + batch_size]
        print(f"Processing second pass batch {i//batch_size + 1}/{(len(all_results) + batch_size - 1)//batch_size}")
        
        # Ensure all records in the batch have munchable and reason fields
        for result in second_pass_batch:
            if 'munchable' not in result:
                result['munchable'] = ''
            if 'reason' not in result:
                result['reason'] = ''
        
        # Process batch for munchable analysis
        munchable_results = await process_contract_batch(second_pass_batch, pass_number=2)
        
        # Update results with munchable status
        for result in second_pass_batch:
            contract_num = result.get('contract_number')
            munchable_result = next((r for r in munchable_results if r.get('contract_number') == contract_num), None)
            if munchable_result:
                result['munchable'] = munchable_result.get('munchable', 'Error')
                result['reason'] = munchable_result.get('reason', '')
                print(f"Updated munchable status for contract {contract_num}: {result['munchable']}")
                print(f"Reason: {result['reason']}")
        
        # Save second pass batch results
        if os.path.exists(output_csv):
            # Read existing CSV and update with new munchable status
            print(f"Updating analysis CSV with second pass batch results: {output_csv}")
            existing_df = pd.read_csv(output_csv)
            
            # Ensure munchable and reason columns exist in the DataFrame
            if 'munchable' not in existing_df.columns:
                existing_df['munchable'] = ''
                print("Added munchable column to existing results")
                
            if 'reason' not in existing_df.columns:
                existing_df['reason'] = ''
                print("Added reason column to existing results")
            
            # Ensure reason column is string type
            existing_df['reason'] = existing_df['reason'].fillna('')
            
            # Create a dictionary mapping contract numbers to their updated data
            updates = {r['contract_number']: r for r in second_pass_batch if r['contract_number'] != 'Not found' and r['contract_number'] != 'Error'}
            
            # Update each row in the existing dataframe
            for index, row in existing_df.iterrows():
                contract_num = row['contract_number']
                if contract_num in updates:
                    existing_df.at[index, 'munchable'] = updates[contract_num]['munchable']
                    existing_df.at[index, 'reason'] = str(updates[contract_num]['reason']).replace('nan', '')
            
            # Sort by munchable status if the column exists
            if 'munchable' in existing_df.columns:
                existing_df = existing_df.sort_values(by='munchable', key=lambda x: x.map({'true': 0, 'false': 1, 'n/a': 2, 'error': 3, '': 4}))
            
            existing_df.to_csv(output_csv, index=False)
            print(f"Updated {output_csv} with second pass batch results")
            
            # Validate that munchable status was updated
            final_df = pd.read_csv(output_csv)
            final_df['reason'] = final_df['reason'].fillna('')  # Replace NaN with empty string for display
            munchable_counts = final_df['munchable'].value_counts()
            print("\nMunchable Status Distribution:")
            print(munchable_counts)
            
            # Print detailed test results
            if args.test_mode:
                print("\nTest Mode Results:")
                test_results = final_df[final_df['contract_number'].isin([r['contract_number'] for r in second_pass_batch])]
                print("\nContract Details:")
                for _, row in test_results.iterrows():
                    print(f"\nContract Number: {row['contract_number']}")
                    print(f"Munchable: {row['munchable']}")
                    print(f"Reason: {row['reason'] if row['reason'] and row['reason'] != 'nan' else 'No reason provided'}")
        
        # In test mode, only process first batch
        if args.test_mode:
            break
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Processed {len(all_results)} contracts in {total_duration:.2f} seconds")
    print(f"Average time per contract: {total_duration/len(all_results):.2f} seconds")
    print(f"Results saved to {output_csv}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 
