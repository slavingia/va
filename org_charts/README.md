# Org Chart Generator

This script generates filtered CSV files based on cost centers from organizational data stored in a CSV file.

## Requirements

*   Python 3

## Input Data

The script expects an input CSV file named `org_data.csv` in the same directory. If using a different file, specify it with the `--input_file` parameter.

The CSV file should contain positional data, including at least the following columns:
*   `Position Number`: Unique identifier for each position.
*   `Reports To Position Number`: The `Position Number` of the direct supervisor.
*   `Official Position Title`: The title of the position.
*   `Cost Center Desc`: The cost center description associated with the position.
*   `Location Desc`: The location description for the position.
*   `Admin` (or a column ending with `Admin`): Used for filtering (e.g., 'vba').

*Note: The script attempts to handle potential Byte Order Mark (BOM) characters in the header row, particularly for the `Admin` column.*

## Usage

Run the script from the command line with optional filters and parameters.

```bash
python org_chart_generator.py [options]
```

**Options:**

*   `--filter column=value`: (Optional) Filter the positions included in the output based on column values. You can use this option multiple times for multiple filters.
    *   Example: `--filter admin=vba`
    *   Example: `--filter "Location Desc"="New York"` (Use quotes if the value contains spaces)
    *   **Special Filters:**
        *   `admin`: Matches against any column ending with 'Admin' (case-insensitive).
        *   `location`: Matches against the `Location Desc` column (case-insensitive).
*   `--input_file`: (Optional) Specify the input CSV file (default: org_data.csv)
*   `--output_dir`: (Optional) Specify the base output directory (default: cost_center_outputs)

**Examples:**

1.  **Generate filtered CSVs for all cost centers:**
    ```bash
    python org_chart_generator.py
    ```

2.  **Generate filtered CSVs for all cost centers with 'vba' admin:**
    ```bash
    python org_chart_generator.py --filter admin=vba
    ```

3.  **Generate filtered CSVs using a custom input file and output directory:**
    ```bash
    python org_chart_generator.py --input_file custom_data.csv --output_dir my_outputs
    ```

## Output

The script creates a directory structure organized by cost center within the specified output directory. For each cost center, it generates:

*   **`org_data_cc_<cost_center>_<timestamp>.csv`**:
    *   A CSV file containing only the rows from the input CSV that correspond to the positions in that cost center.
    *   Has the same header and format as the input CSV.
    *   Additional filters from the command line are also applied to this data.

The generated files are organized in the following structure:
```
output_dir/
├── cost_center_1/
│   └── org_data_cc_cost_center_1_timestamp.csv
├── cost_center_2/
│   └── org_data_cc_cost_center_2_timestamp.csv
└── ...
```

**Note:** Cost center names are sanitized for use in filenames by replacing non-alphanumeric characters with underscores.
