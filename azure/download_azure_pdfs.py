#!/usr/bin/env python3
import os
import logging
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(".env.local")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure Blob Storage configuration
STORAGE_ACCOUNT_URL = "https://vac20sdpasa201c23301.blob.core.usgovcloudapi.net"
CONTAINER_NAME = "ci-raw"
SAS_TOKEN = os.getenv("SAS_TOKEN")  # Load SAS token from environment variable

# Local directory to store downloaded PDFs
DOWNLOAD_DIR = "azure_pdfs"

def download_blob(blob_client, local_path):
    """Download a single blob to local path"""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"Downloading {blob_client.blob_name} to {local_path}")

        # Check if file already exists and is complete
        if os.path.exists(local_path):
            blob_properties = blob_client.get_blob_properties()
            local_size = os.path.getsize(local_path)
            if local_size == blob_properties.size:
                logger.info(f"File already exists and is complete: {blob_client.blob_name}")
                return True

        # Download the blob
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            f.write(stream.readall())

        logger.info(f"Successfully downloaded {blob_client.blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {blob_client.blob_name}: {str(e)}")
        return False

def list_and_download_blobs(container_client):
    """List all blobs in the container, display their information, and download them immediately"""
    try:
        logger.info("Attempting to list and download blobs in the container...")
        for blob in container_client.list_blobs():
            logger.info(f"Blob Name: {blob.name}, Size: {blob.size} bytes, Last Modified: {blob.last_modified}")
            if blob.name.lower().endswith('.pdf'):
                try:
                    # Create blob client
                    blob_client = container_client.get_blob_client(blob.name)

                    # Create local path
                    local_path = os.path.join(DOWNLOAD_DIR, blob.name)

                    # Download blob
                    if download_blob(blob_client, local_path):
                        logger.info(f"Successfully downloaded: {blob.name}")
                    else:
                        logger.warning(f"Failed to download: {blob.name}")
                except Exception as e:
                    logger.error(f"Error processing blob {blob.name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing and downloading blobs: {str(e)}")

def main():
    start_time = datetime.now()
    logger.info("Starting PDF download process...")

    # Create download directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        # Create BlobServiceClient using SAS token
        logger.info("Creating BlobServiceClient...")
        blob_service_client = BlobServiceClient(
            account_url=STORAGE_ACCOUNT_URL,
            credential=SAS_TOKEN
        )

        # Access container
        logger.info(f"Accessing container: {CONTAINER_NAME}")
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        # List and download all PDF blobs
        logger.info("Listing and downloading PDFs in container...")
        list_and_download_blobs(container_client)

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info(f"Download process completed in {duration}")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()