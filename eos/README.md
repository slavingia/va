# VA Executive Order Analysis Tools

These tools are used to analyze internal VA memos for compliance with Executive Orders.

## Usage: analyze_eos.py

This script analyzes internal PDF memos found in the `pdfs` directory for compliance with specific Executive Orders (related to DEI, gender identity, COVID policies, climate initiatives, WHO partnerships). It uses an LLM to score memos based on compliance criteria and outputs detailed results to `memo_analysis.csv`. Memos flagged as potentially non-compliant (score >= 5) are copied to the `flagged_memos` directory.

**Basic Usage:**

```bash
python analyze_eos.py
```

**Environment Variables:**

Ensure you have a `.env.local` file with the necessary API key:

*   `openai_api_key`: Your Azure OpenAI API key.

**Command-line Options:**

*   `--second-pass-only`: Skip the initial data extraction and only perform the detailed compliance analysis on existing results in `memo_analysis.csv`.
*   `--test-mode`: Run the script on a small batch of 3 files for testing purposes.

**Example:**

Run only the second pass analysis in test mode:

```bash
python analyze_eos.py --second-pass-only --test-mode
```
