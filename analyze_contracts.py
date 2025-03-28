import pandas as pd
import numpy as np
import re
from datetime import datetime

def format_currency(x):
    """Format currency in the most appropriate unit with 2 decimal places"""
    if x >= 1_000_000_000:  # Billions
        return f"${x/1_000_000_000:.2f}B"
    elif x >= 1_000_000:  # Millions
        return f"${x/1_000_000:.2f}M"
    elif x >= 1_000:  # Thousands
        return f"${x/1_000:.2f}K"
    else:
        return f"${x:.2f}"

def clean_currency(x):
    if pd.isna(x) or x == 'Not found':
        return 0
    
    # Convert string to lowercase and remove common currency indicators
    x = str(x).lower().strip()
    x = x.replace('usd', '').replace('$', '').strip()
    
    try:
        # Handle "million" values
        if 'million' in x:
            num = float(x.split('million')[0].strip().replace(',', ''))
            return num * 1_000_000
        
        # Handle "m" for million values
        if x.endswith('m'):
            x = x[:-1].strip()  # Remove the 'm' and any trailing spaces
            x = re.sub(r'[^0-9.]', '', x)  # Remove any remaining non-numeric characters
            return float(x) * 1_000_000 if x else 0
        
        # Handle regular currency values - remove any remaining non-numeric characters except decimal points
        x = re.sub(r'[^0-9.]', '', x)
        
        # Handle multiple decimal points by keeping only the first one
        parts = x.split('.')
        if len(parts) > 2:
            x = parts[0] + '.' + ''.join(parts[1:])
        
        return float(x) if x else 0
    except ValueError as e:
        print(f"Warning: Could not convert value: {x}")
        return 0

def analyze_contracts(df, title="All Contracts", output_file=None):
    # Clean both total_value and fy25_value columns
    df['total_value'] = df['total_value'].apply(clean_currency)
    df['fy25_value'] = df['fy25_value'].apply(clean_currency)
    
    # Create a mapping of parent contracts to their values
    parent_contracts = df[df['parent_contract_number'] != 'Not found']
    parent_values = dict(zip(parent_contracts['parent_contract_number'], parent_contracts['total_value']))
    
    # Calculate total value excluding duplicates
    total_value = 0
    total_fy25 = 0
    processed_parents = set()
    
    for _, row in df.iterrows():
        if row['parent_contract_number'] != 'Not found':
            # If this is a child contract, only count it if we haven't processed its parent
            if row['parent_contract_number'] not in processed_parents:
                total_value += row['total_value']
                total_fy25 += row['fy25_value']
                processed_parents.add(row['parent_contract_number'])
        else:
            # If this is not a child contract, count it
            total_value += row['total_value']
            total_fy25 += row['fy25_value']
    
    # Prepare output text
    output = []
    output.append(f"\n{title}")
    output.append("=" * 85)
    output.append(f"Total Contract Value: {format_currency(total_value)}")
    output.append(f"Total FY25 Value: {format_currency(total_fy25)}")
    output.append(f"Number of Unique Contracts: {len(processed_parents) + len(df[df['parent_contract_number'] == 'Not found'])}")
    
    # Calculate vendor totals for both values
    vendor_totals = df.groupby('vendor_name').agg({
        'total_value': 'sum',
        'fy25_value': 'sum'
    }).sort_values('total_value', ascending=False)
    
    output.append("\nTop 100 Vendors by Contract Value:")
    output.append("-" * 85)
    output.append(f"{'Rank':<4} {'Vendor':<50} {'Total Value':>15} {'FY25':>15}")
    output.append("-" * 85)
    
    # Format each value in appropriate units
    top_100 = vendor_totals.head(100)
    for rank, (vendor, row) in enumerate(top_100.iterrows(), 1):
        total_formatted = format_currency(row['total_value'])
        fy25_formatted = format_currency(row['fy25_value'])
        output.append(f"{rank:<4} {vendor[:50]:<50} {total_formatted:>15} {fy25_formatted:>15}")
    
    # Print to console
    print("\n".join(output))
    
    # Write to file if output_file is provided
    if output_file:
        with open(output_file, 'a') as f:
            f.write("\n".join(output) + "\n")

def main():
    # Read the CSV file
    df = pd.read_csv('contract_analysis.csv')
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"contract_analysis_{timestamp}.txt"
    
    # Clear the output file if it exists
    with open(output_file, 'w') as f:
        f.write(f"Contract Analysis Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 85 + "\n\n")
    
    # Analyze all contracts
    analyze_contracts(df, "All Contracts", output_file)
    
    # Analyze munchable contracts only
    munchable_df = df[df['munchable'] == True]
    analyze_contracts(munchable_df, "Munchable Contracts Only", output_file)
    
    print(f"\nAnalysis has been saved to: {output_file}")

if __name__ == "__main__":
    main() 