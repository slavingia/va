# VA tools

These are tools I have written to do my work more effectively at the VA.

## Usage: process_contracts_v3.py

This script processes PDF contracts found in the `azure_pdfs` directory, extracts text, analyzes them using an LLM, and outputs the results to `contract_analysis.csv`.

**Basic Usage:**

```bash
python process_contracts_v3.py
```

**Environment Variables:**

Make sure you have a `.env.local` file in the same directory with the necessary API keys:

*   `openai_api_key`: Your Azure OpenAI API key (used by default).
*   `openai_api_key_smart`: Your standard OpenAI API key (used with `--smart-mode`).

**Command-line Options:**

*   `--smart-mode`: Use the standard OpenAI client (`o3-mini` model) instead of the default Azure OpenAI client (`gpt-4o` model). Requires `openai_api_key_smart` to be set in `.env.local`.
*   `--second-pass-only`: Skip the initial data extraction (pass 1) and only perform the munchable analysis (pass 2) on existing results in `contract_analysis.csv`.
*   `--test-mode`: Run the script on a small batch of 3 files for testing purposes.
*   `--batch-size <NUMBER>`: Specify the number of contracts to process in each batch (default: 25).

**Example:**

Run in smart mode with a batch size of 10:

```bash
python process_contracts_v3.py --smart-mode --batch-size 10
```

Run only the second pass analysis:

```bash
python process_contracts_v3.py --second-pass-only
```
