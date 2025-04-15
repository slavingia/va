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
from openai import AsyncAzureOpenAI
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

# Set up Azure OpenAI client
client = AsyncAzureOpenAI(
    api_key=openai_api_key,
    api_version="2024-02-15-preview",
    azure_endpoint="https://spd-prod-int-east-openai-apim.azure-api.us"
)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, "pdfs")
output_csv = os.path.join(script_dir, "memo_analysis.csv")
processed_log = os.path.join(script_dir, "processed_memos.json")
compliant_memos_csv = os.path.join(script_dir, "compliant_memos.csv")
flagged_pdfs_dir = os.path.join(script_dir, "flagged_memos")

# Create directories if they don't exist
os.makedirs(flagged_pdfs_dir, exist_ok=True)

# Print file paths for debugging
print(f"Script directory: {script_dir}")
print(f"PDF folder: {pdf_folder}")
print(f"Output CSV path: {output_csv}")
print(f"Processed log path: {processed_log}")
print(f"Compliant memos CSV path: {compliant_memos_csv}")
print(f"Flagged PDFs directory: {flagged_pdfs_dir}")

# Load list of already processed files
processed_files = []
if os.path.exists(processed_log):
    with open(processed_log, 'r') as f:
        processed_files = json.load(f)
        print(f"Loaded {len(processed_files)} already processed files from {processed_log}")
else:
    print(f"No processed files log found at {processed_log}, will create new one")

# Initialize or load compliant memos
compliant_memos = set()
if os.path.exists(compliant_memos_csv):
    df_compliant = pd.read_csv(compliant_memos_csv)
    if 'memo_id' in df_compliant.columns:
        compliant_memos = set(df_compliant['memo_id'].tolist())
        print(f"Loaded {len(compliant_memos)} already compliant memos from {compliant_memos_csv}")
else:
    print(f"No compliant memos CSV found at {compliant_memos_csv}, will create new one")

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
    """Get all PDF files from the pdfs folder and its subfolders"""
    all_pdfs = []
    
    pdf_path = os.path.join(script_dir, "pdfs")
    if os.path.exists(pdf_path):
        print(f"Searching for PDFs in {pdf_path}...")
        pdfs = glob.glob(os.path.join(pdf_path, "**/*.pdf"), recursive=True)
        all_pdfs.extend(pdfs)
        print(f"Found {len(pdfs)} PDFs in the pdfs folder and its subfolders")
    else:
        print(f"Warning: PDF folder not found at {pdf_path}")
    
    return all_pdfs

def update_compliant_memos(memo_ids):
    """Update the compliant memos CSV file"""
    new_records = []
    for memo_id in memo_ids:
        if memo_id != "Not found" and memo_id != "Error" and memo_id not in compliant_memos:
            new_records.append({"memo_id": memo_id, "review_date": datetime.now().strftime("%Y-%m-%d")})
    
    if new_records:
        print(f"Adding {len(new_records)} new memo IDs to compliant_memos.csv")
        # Create dataframe with new records
        new_df = pd.DataFrame(new_records)
        
        if os.path.exists(compliant_memos_csv):
            # Append to existing file
            print(f"Appending to existing file: {compliant_memos_csv}")
            existing_df = pd.read_csv(compliant_memos_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(compliant_memos_csv, index=False)
            print(f"Updated {compliant_memos_csv} with {len(combined_df)} total records")
        else:
            # Create new file
            print(f"Creating new file: {compliant_memos_csv}")
            new_df.to_csv(compliant_memos_csv, index=False)
            print(f"Created {compliant_memos_csv} with {len(new_df)} records")
        
        # Update the in-memory set
        compliant_memos.update([record["memo_id"] for record in new_records])
    else:
        print("No new memo IDs to add to compliant_memos.csv")

def copy_pdf_to_flagged_folder(pdf_path, memo_id):
    """Copy the PDF to the flagged folder with the memo ID as the filename"""
    if memo_id in ["Not found", "Error", ""]:
        print(f"Skipping copy of {pdf_path} because memo ID is invalid: {memo_id}")
        return False
    
    # Clean memo ID to make it a valid filename
    # Remove any characters that are not allowed in filenames
    valid_memo_id = ''.join(c for c in memo_id if c.isalnum() or c in '._- ')
    
    # If after cleaning we have an empty string, use original filename
    if not valid_memo_id:
        print(f"Memo ID '{memo_id}' contains only invalid characters for a filename")
        return False
    
    # Ensure the flagged_pdfs directory exists
    try:
        os.makedirs(flagged_pdfs_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating flagged_pdfs directory: {e}")
        return False
    
    # Create destination filename
    dest_file = os.path.join(flagged_pdfs_dir, f"{valid_memo_id}.pdf")
    
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

async def analyze_memo_async(text, session, pass_number=1, max_retries=3):
    """Analyze memo text using LLM asynchronously with retries"""
    if pass_number == 1:
        prompt = f"""
        Analyze this internal memo and extract key information for compliance with new Executive Orders. If information is not found, write "Not found".
        
        MEMO TEXT:
        {text[:10000]}
        
        Extract:
        1. Memo ID or identifier
        2. Memo Title
        3. Memo Description (summary of the content)
        4. Date of Publication/Distribution
        5. Authoring Office/Department
        6. Target Audience
        7. Does this memo contain any DEI, DEIA, or equity-related content? (true/false)
        8. Does this memo reference gender identity, pronouns, or related topics? (true/false)
        9. Does this memo contain COVID policy, vaccine mandates, or extended telework policies? (true/false)
        10. Does this memo reference climate, environmental justice, or sustainability initiatives? (true/false)
        11. Does this memo contain references to WHO partnerships or collaborations? (true/false)
        12. First-pass Non-Compliance Score (1-10, with 10 being most non-compliant)
        13. First-pass Non-Compliance Justification (brief explanation)
        
        Format as JSON:
        {{
            "memo_id": "",
            "title": "",
            "description": "",
            "date": "",
            "authoring_office": "",
            "target_audience": "",
            "contains_dei": false,
            "contains_gender_identity": false,
            "contains_covid_policy": false,
            "contains_climate": false,
            "contains_who_references": false,
            "first_pass_non_compliance_score": 0,
            "first_pass_justification": ""
        }}
        """
    else:
        prompt = f"""
        Based on this memo analysis, provide a detailed assessment of non-compliance with 2025 Executive Orders:
        
        MEMO CONTENT:
        {text[:10000]}
        
        Rules for determining non-compliance:
        
        1. DEI/DEIA Content:
        - References to "diversity", "equity", "inclusion" in policy context
        - Mentions of equity action plans
        - References to Chief Diversity Officers or similar roles
        - Performance criteria based on DEI targets
        
        2. Gender Identity Content:
        - References to gender identity or preferred pronouns
        - Content about gender ideology or training
        - Instructions to use non-binary pronouns
        
        3. COVID Policy/Telework:
        - COVID-19 vaccination requirements
        - Pandemic-era remote work extensions
        - Masking or testing policies that remain in effect
        
        4. Climate/Environmental Content:
        - References to climate resilience, zero-emission requirements
        - Environmental justice language
        - Sustainability criteria in purchasing/contracting
        - Plastic straw bans or similar procurement restrictions
        
        5. WHO Partnerships:
        - References to World Health Organization partnerships
        - Alignment with WHO protocols or frameworks
        
        6. Other Key Areas:
        - Affirmative action or racial/gender preferences in contracting
        - $15 federal contractor minimum wage references
        - Student loan forgiveness expansions
        - Paper check disbursement references
        
        Format as JSON:
        {{
            "memo_id": "The memo ID from the first pass",
            "second_pass_violations": [list specific violations found, e.g. "DEI training mandate", "Pronoun guidance", etc.],
            "second_pass_score": 1-10 score with 10 being most non-compliant,
            "suggested_remediation": "Specific text recommended for removal or revision",
            "priority_level": "high/medium/low - based on severity of non-compliance"
        }}
        """
    
    for attempt in range(max_retries):
        try:
            print(f"Calling OpenAI API to analyze memo (Pass {pass_number}, Attempt {attempt + 1}/{max_retries})...")
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes internal government memos for compliance with 2025 Executive Orders. You identify references to DEI, gender identity, COVID policies, climate initiatives, and WHO partnerships that must be removed or modified according to new directives. Provide detailed analysis and be precise about what needs to be changed."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if pass_number == 1:
                print(f"First pass analysis completed. Memo ID: {result.get('memo_id', 'Not found')}")
            else:
                print(f"Second pass analysis completed. Non-compliance score: {result.get('second_pass_score', 'Error')}")
            return result
            
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"Error analyzing memo on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                if pass_number == 1:
                    return {
                        "memo_id": "Error",
                        "title": "Error",
                        "description": str(e)[:100],
                        "date": "Error",
                        "authoring_office": "Error",
                        "target_audience": "Error",
                        "contains_dei": False,
                        "contains_gender_identity": False,
                        "contains_covid_policy": False,
                        "contains_climate": False,
                        "contains_who_references": False,
                        "first_pass_non_compliance_score": 0,
                        "first_pass_justification": "Error in analysis"
                    }
                else:
                    return {
                        "memo_id": "Error",
                        "second_pass_violations": ["Error in analysis"],
                        "second_pass_score": 0,
                        "suggested_remediation": "Error in analysis",
                        "priority_level": "unknown"
                    }
            await asyncio.sleep(2 ** attempt)

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

async def process_memo_batch(memo_batch, pass_number=1):
    """Process a batch of memos in parallel"""
    batch_start_time = time.time()
    tasks = []
    
    # Extract text from all PDFs in the batch in parallel
    pdf_paths = [result.get('file_path') for result in memo_batch if result.get('file_path')]
    print(f"Extracting text from {len(pdf_paths)} PDFs in parallel...")
    extracted_texts = await extract_text_batch_parallel(pdf_paths)
    
    # For second pass, we need to build a mapping of memo IDs to their text content
    memo_texts = {}
    if pass_number == 2:
        for result, text in zip(memo_batch, extracted_texts):
            if text:
                memo_texts[result.get('memo_id')] = text
    
    results = []
    for pdf_path, text in zip(pdf_paths, extracted_texts):
        try:
            if not text:
                print(f"No text extracted from {pdf_path}, skipping analysis")
                processed_files.append(pdf_path)
                continue
            
            # Create task for memo analysis
            task = asyncio.create_task(analyze_memo_async(text, None, pass_number))
            tasks.append((pdf_path, task))
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    # Wait for all tasks to complete
    for pdf_path, task in tasks:
        try:
            analysis = await task
            if pass_number == 1:
                memo_id = analysis.get('memo_id', 'Not found')
                
                # Skip if this memo ID has already been compliant
                if memo_id in compliant_memos and memo_id not in ["Not found", "Error"]:
                    print(f"Skipping already compliant memo: {memo_id}")
                    processed_files.append(pdf_path)
                    continue
                
                # Add to results with all columns
                results.append({
                    'memo_id': memo_id,
                    'title': analysis.get('title', 'Not found'),
                    'description': analysis.get('description', 'Not found'),
                    'date': analysis.get('date', 'Not found'),
                    'authoring_office': analysis.get('authoring_office', 'Not found'),
                    'target_audience': analysis.get('target_audience', 'Not found'),
                    'contains_dei': analysis.get('contains_dei', False),
                    'contains_gender_identity': analysis.get('contains_gender_identity', False),
                    'contains_covid_policy': analysis.get('contains_covid_policy', False),
                    'contains_climate': analysis.get('contains_climate', False),
                    'contains_who_references': analysis.get('contains_who_references', False),
                    'file_path': pdf_path,
                    'first_pass_non_compliance_score': analysis.get('first_pass_non_compliance_score', 0),
                    'first_pass_justification': analysis.get('first_pass_justification', ''),
                    'second_pass_violations': [],  # Will be filled in second pass
                    'second_pass_score': 0,  # Will be filled in second pass
                    'suggested_remediation': '',  # Will be filled in second pass
                    'priority_level': ''  # Will be filled in second pass
                })
                
                # If non-compliance score is high, copy PDF to flagged folder
                non_compliance_score = analysis.get('first_pass_non_compliance_score', 0)
                if non_compliance_score is not None and non_compliance_score >= 5 and memo_id not in ["Not found", "Error"]:
                    copy_pdf_to_flagged_folder(pdf_path, memo_id)
                    
            else:
                # Second pass - update compliance details
                memo_id = analysis.get('memo_id', 'Not found')
                violations = analysis.get('second_pass_violations', [])
                non_compliance_score = analysis.get('second_pass_score', 0)
                remediation = analysis.get('suggested_remediation', '')
                priority = analysis.get('priority_level', 'low')
                
                # Find and update the corresponding result from first pass
                for result in memo_batch:
                    if result.get('memo_id') == memo_id:
                        result['second_pass_violations'] = violations
                        result['second_pass_score'] = non_compliance_score
                        result['suggested_remediation'] = remediation
                        result['priority_level'] = priority
                        results.append(result)
                        print(f"Updated second pass non-compliance details for memo {memo_id}")
                        print(f"Non-compliance score: {non_compliance_score}")
                        print(f"Priority level: {priority}")
                        
                        # If high priority non-compliance, make sure it's in flagged folder
                        if priority.lower() == 'high' and memo_id not in ["Not found", "Error"]:
                            copy_pdf_to_flagged_folder(result.get('file_path', ''), memo_id)
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
    parser = argparse.ArgumentParser(description='Process internal memos for compliance with Executive Orders')
    parser.add_argument('--second-pass-only', action='store_true', help='Skip first pass and only do second pass analysis')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with a small batch of files')
    args = parser.parse_args()
    
    # Get all PDF files from pdfs folder
    pdf_files = get_all_pdf_files()
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return
    
    # Filter out already processed files
    files_to_process = [f for f in pdf_files if f not in processed_files]
    
    if not files_to_process and not args.second_pass_only:
        print("No new memos to process.")
        return
    
    # If in test mode, only process a small batch
    if args.test_mode:
        files_to_process = files_to_process[:3]  # Process only 3 files for testing
        print(f"Test mode: Processing {len(files_to_process)} files")
    
    print(f"Found {len(files_to_process)} new memos to process.")
    
    all_results = []
    batch_size = 100  # Smaller batch size for more detailed analysis
    
    # Load existing results if doing second pass only
    if args.second_pass_only:
        if os.path.exists(output_csv):
            print("Loading existing results for second pass analysis...")
            existing_df = pd.read_csv(output_csv)
            
            # Add new columns if they don't exist
            if 'first_pass_non_compliance_score' not in existing_df.columns:
                existing_df['first_pass_non_compliance_score'] = 0
                print("Added first_pass_non_compliance_score column to existing results")
            
            if 'first_pass_justification' not in existing_df.columns:
                existing_df['first_pass_justification'] = ''
                print("Added first_pass_justification column to existing results")
            
            if 'second_pass_violations' not in existing_df.columns:
                existing_df['second_pass_violations'] = ''
                print("Added second_pass_violations column to existing results")
            
            if 'second_pass_score' not in existing_df.columns:
                existing_df['second_pass_score'] = 0
                print("Added second_pass_score column to existing results")
                
            if 'suggested_remediation' not in existing_df.columns:
                existing_df['suggested_remediation'] = ''
                print("Added suggested_remediation column to existing results")
                
            if 'priority_level' not in existing_df.columns:
                existing_df['priority_level'] = ''
                print("Added priority_level column to existing results")
                
            # Save the updated DataFrame with the new columns
            existing_df.to_csv(output_csv, index=False)
            print("Updated CSV with new compliance columns")
            
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
        else:
            print("No existing results found for second pass analysis.")
            return
    else:
        # First pass: Process all memos to establish basic information
        print("Starting first pass: Extracting basic memo information and initial compliance check...")
        
        # Create results list with file paths
        memo_batch = [{'file_path': f} for f in files_to_process]
        
        # First pass: Process all memos in batches
        for i in range(0, len(memo_batch), batch_size):
            batch = memo_batch[i:i + batch_size]
            print(f"Processing first pass batch {i//batch_size + 1}/{(len(memo_batch) + batch_size - 1)//batch_size}")
            
            # Process batch
            batch_results = await process_memo_batch(batch, pass_number=1)
            all_results.extend(batch_results)
            
            # Update processed files log after each batch
            with open(processed_log, 'w') as f:
                json.dump(processed_files, f)
            print(f"Updated processed files log with {len(processed_files)} total files")
            
            # Update compliant memos CSV after each batch
            batch_memo_ids = [r['memo_id'] for r in batch_results]
            update_compliant_memos(batch_memo_ids)
            
            # Save intermediate results after each batch
            if batch_results:
                if os.path.exists(output_csv):
                    # Read existing CSV and append new results
                    print(f"Appending first pass batch results to existing analysis CSV: {output_csv}")
                    existing_df = pd.read_csv(output_csv)
                    
                    # Ensure all columns exist in existing DataFrame
                    required_columns = [
                        'memo_id', 'title', 'description', 'date', 'authoring_office', 
                        'target_audience', 'contains_dei', 'contains_gender_identity', 
                        'contains_covid_policy', 'contains_climate', 'contains_who_references',
                        'file_path', 'first_pass_non_compliance_score', 'first_pass_justification',
                        'second_pass_violations', 'second_pass_score', 'suggested_remediation', 'priority_level'
                    ]
                    
                    for col in required_columns:
                        if col not in existing_df.columns:
                            if col in ['contains_dei', 'contains_gender_identity', 'contains_covid_policy', 
                                    'contains_climate', 'contains_who_references']:
                                existing_df[col] = False
                            elif col in ['first_pass_non_compliance_score', 'second_pass_score']:
                                existing_df[col] = 0
                            else:
                                existing_df[col] = ''
                    
                    # Convert list column to string for CSV storage
                    new_df = pd.DataFrame(batch_results)
                    if 'second_pass_violations' in new_df.columns:
                        new_df['second_pass_violations'] = new_df['second_pass_violations'].apply(lambda x: str(x) if isinstance(x, list) else x)
                    
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(output_csv, index=False)
                    print(f"Updated {output_csv} with {len(combined_df)} total records")
                else:
                    # Create new CSV with results
                    print(f"Creating new analysis CSV with first pass batch results: {output_csv}")
                    df = pd.DataFrame(batch_results)
                    
                    # Convert list column to string for CSV storage
                    if 'second_pass_violations' in df.columns:
                        df['second_pass_violations'] = df['second_pass_violations'].apply(lambda x: str(x) if isinstance(x, list) else x)
                    
                    df.to_csv(output_csv, index=False)
                    print(f"Created {output_csv} with {len(df)} records")
            
            # In test mode, only process first batch
            if args.test_mode:
                break
        
        print("First pass completed for all memos.")
        print(f"Total memos processed in first pass: {len(all_results)}")
    
    # Second pass: Detailed compliance analysis
    print("Starting second pass: Performing detailed compliance analysis...")
    
    # Process second pass in batches
    for i in range(0, len(all_results), batch_size):
        second_pass_batch = all_results[i:i + batch_size]
        print(f"Processing second pass batch {i//batch_size + 1}/{(len(all_results) + batch_size - 1)//batch_size}")
        
        # Ensure all records in the batch have the required fields
        for result in second_pass_batch:
            for field in ['memo_id', 'title', 'description', 'date', 'authoring_office', 
                        'target_audience', 'contains_dei', 'contains_gender_identity', 
                        'contains_covid_policy', 'contains_climate', 'contains_who_references',
                        'file_path', 'first_pass_non_compliance_score', 'first_pass_justification',
                        'second_pass_violations', 'second_pass_score', 'suggested_remediation', 'priority_level']:
                if field not in result:
                    if field in ['contains_dei', 'contains_gender_identity', 'contains_covid_policy', 
                                'contains_climate', 'contains_who_references']:
                        result[field] = False
                    elif field in ['first_pass_non_compliance_score', 'second_pass_score']:
                        result[field] = 0
                    elif field == 'second_pass_violations':
                        result[field] = []
                    else:
                        result[field] = ''
        
        # Process batch for compliance analysis
        compliance_results = await process_memo_batch(second_pass_batch, pass_number=2)
        
        # Update results with compliance status
        for result in second_pass_batch:
            memo_id = result.get('memo_id')
            compliance_result = next((r for r in compliance_results if r.get('memo_id') == memo_id), None)
            if compliance_result:
                result['second_pass_violations'] = compliance_result.get('second_pass_violations', [])
                result['second_pass_score'] = compliance_result.get('second_pass_score', 0)
                result['suggested_remediation'] = compliance_result.get('suggested_remediation', '')
                result['priority_level'] = compliance_result.get('priority_level', 'low')
                print(f"Updated second pass compliance status for memo {memo_id}")
                print(f"Non-compliance score: {result['second_pass_score']}")
                print(f"Priority level: {result['priority_level']}")
        
        # Save second pass batch results
        if os.path.exists(output_csv):
            # Read existing CSV and update with new compliance status
            print(f"Updating analysis CSV with second pass batch results: {output_csv}")
            existing_df = pd.read_csv(output_csv)
            
            # Ensure all required columns exist
            for col in ['memo_id', 'second_pass_violations', 'second_pass_score', 'suggested_remediation', 'priority_level']:
                if col not in existing_df.columns:
                    if col == 'second_pass_score':
                        existing_df[col] = 0
                    else:
                        existing_df[col] = ''
            
            # Create a dictionary mapping memo IDs to their updated data
            updates = {r['memo_id']: r for r in second_pass_batch if r['memo_id'] != 'Not found' and r['memo_id'] != 'Error'}
            
            # Update each row in the existing dataframe
            for index, row in existing_df.iterrows():
                memo_id = row['memo_id']
                if memo_id in updates:
                    # Convert list to string for CSV storage
                    violations = updates[memo_id]['second_pass_violations']
                    if isinstance(violations, list):
                        violations_str = str(violations)
                    else:
                        violations_str = violations
                    
                    existing_df.at[index, 'second_pass_violations'] = violations_str
                    existing_df.at[index, 'second_pass_score'] = updates[memo_id]['second_pass_score']
                    existing_df.at[index, 'suggested_remediation'] = updates[memo_id]['suggested_remediation']
                    existing_df.at[index, 'priority_level'] = updates[memo_id]['priority_level']
            
            # Sort by priority level and non-compliance score
            if 'priority_level' in existing_df.columns and 'second_pass_score' in existing_df.columns:
                existing_df = existing_df.sort_values(
                    by=['priority_level', 'second_pass_score'], 
                    key=lambda x: x.map({'high': 0, 'medium': 1, 'low': 2, '': 3}) if x.name == 'priority_level' else -x
                )
            
            existing_df.to_csv(output_csv, index=False)
            print(f"Updated {output_csv} with second pass batch results")
            
            # Print compliance summary
            final_df = pd.read_csv(output_csv)
            
            # Count priority levels
            priority_counts = final_df['priority_level'].value_counts()
            print("\nPriority Level Distribution:")
            print(priority_counts)
            
            # Count compliance violations by type
            print("\nNon-Compliance Type Distribution:")
            dei_count = final_df[final_df['contains_dei'] == True].shape[0]
            gender_count = final_df[final_df['contains_gender_identity'] == True].shape[0]
            covid_count = final_df[final_df['contains_covid_policy'] == True].shape[0]
            climate_count = final_df[final_df['contains_climate'] == True].shape[0]
            who_count = final_df[final_df['contains_who_references'] == True].shape[0]
            
            print(f"DEI/DEIA Content: {dei_count}")
            print(f"Gender Identity Content: {gender_count}")
            print(f"COVID Policy/Telework: {covid_count}")
            print(f"Climate/Environmental Content: {climate_count}")
            print(f"WHO References: {who_count}")
            
            # Print detailed test results
            if args.test_mode:
                print("\nTest Mode Results:")
                test_results = final_df[final_df['memo_id'].isin([r['memo_id'] for r in second_pass_batch])]
                print("\nMemo Details:")
                for _, row in test_results.iterrows():
                    print(f"\nMemo ID: {row['memo_id']}")
                    print(f"Title: {row['title']}")
                    print(f"Non-Compliance Score: {row['second_pass_score']}")
                    print(f"Priority Level: {row['priority_level']}")
                    print(f"Suggested Remediation: {row['suggested_remediation']}")
        
        # In test mode, only process first batch
        if args.test_mode:
            break
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Processed {len(all_results)} memos in {total_duration:.2f} seconds")
    print(f"Average time per memo: {total_duration/len(all_results):.2f} seconds")
    print(f"Results saved to {output_csv}")
    print(f"High-priority non-compliant memos copied to {flagged_pdfs_dir}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 