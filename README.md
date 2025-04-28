Tools I have written to do my work more effectively at the VA. Code exists to make humans more efficient; not to replace them. All code leads to human review. Feedback appreciated!

## Org Charts

Org chart tools are in the [org charts](./org_charts/) directory.

## Contracts

Contract processing and analysis tools are in the [contracts](./contracts) directory.

## Executive Orders

Executive Order compliance analysis tools are in the [eos](./eos) directory.

## Usage: download_azure_pdfs.py

This script connects to an Azure Blob Storage container and downloads all PDF files found within it to a local directory named `azure_pdfs`. It checks for existing files and only downloads if the local file is missing or incomplete.

**Setup:**

1.  Ensure you have a `.env.local` file in the same directory.
2.  Add the necessary Azure Blob Storage SAS token to the `.env.local` file:
    ```
    SAS_TOKEN="your_sas_token_here"
    ```

**Basic Usage:**

```bash
python download_azure_pdfs.py
```

The script will log the download progress and output the downloaded files to the `azure_pdfs` directory in the current working directory.
