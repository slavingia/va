#!/usr/bin/env python3

import csv
import os
import sys
import datetime
import argparse
from collections import defaultdict, Counter

# Increase recursion limit to handle deep hierarchies
sys.setrecursionlimit(10000)

def create_d3_csv(d3_csv_file, root_position, positions, children, visited, input_file, filters=None):
    """Create a CSV file with the same format as input but only relevant positions for the given root and filters."""
    # Note: d3_csv_file is now the full path passed from build_org_chart
    print(f"Writing filtered D3 CSV to {d3_csv_file}...")

    # Read the header from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

    # Write the filtered data
    with open(d3_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Read the input file again and write only relevant rows
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                position_number = row.get('Position Number', '').strip()
                if position_number in visited:
                    # Apply additional filters if specified
                    if filters and not match_filters(row, filters):
                        continue
                    # Write the row exactly as it is
                    writer.writerow([row[col] for col in header])

def match_filters(row, filters):
    """Check if a row matches all specified filters"""
    for column, value in filters.items():
        # Handle the special case for Admin column with potential BOM character
        if column.lower() == 'admin':
            # Check all columns that end with 'Admin'
            matched = False
            for col in row:
                if col.endswith('Admin') and row[col].lower() == value.lower():
                    matched = True
                    break
            if not matched:
                return False
        # Handle location filtering
        elif column.lower() == 'location':
            # Check Location Desc column
            if row.get('Location Desc', '').lower() != value.lower():
                return False
        elif column in row:
            if row[column].lower() != value.lower():
                return False
        else:
            # Column doesn't exist in the row
            return False
    return True

def load_data(input_file):
    """Loads data from the input CSV file without applying filters."""
    # Data structures
    positions = {}  # position_number -> position data
    children = defaultdict(list)  # position_number -> [child position numbers]
    cost_centers = defaultdict(list)  # cost_center_desc -> [position numbers]

    # Load data
    print(f"Reading data from {input_file}...")
    admin_col = 'Admin' # Default
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check column names and handle BOM character
            fieldnames = reader.fieldnames
            print(f"Original column names: {fieldnames}")

            # Fix for BOM character in Admin column
            for col in fieldnames:
                if col.endswith('Admin'):
                    admin_col = col
                    print(f"Found Admin column as: '{admin_col}'")
                    break

            row_count = 0
            for row in reader:
                row_count += 1
                position_number = row.get('Position Number', '').strip()
                if not position_number:
                    continue

                # Get reporting relationship
                reports_to = row.get('Reports To Position Number', '').strip()

                # Get position title and cost center
                position_title = row.get('Official Position Title', '')
                cost_center_desc = row.get('Cost Center Desc', '')

                # Store position data
                positions[position_number] = {
                    'position_title': position_title,
                    'reports_to': reports_to,
                    'admin': row.get(admin_col, ''),
                    'cost_center_desc': cost_center_desc,
                    'dept_desc': row.get('Dept Desc', ''),
                    'org_code_desc': row.get('Org Code Desc', ''),
                    'supervisory_level_desc': row.get('Supervisory Level Desc', ''),
                    'location_desc': row.get('Location Desc', '')
                }

                # Build reporting relationships (from parent to children)
                if reports_to:
                    children[reports_to].append(position_number)

                # Group positions by cost center
                if cost_center_desc:
                    cost_centers[cost_center_desc].append(position_number)

                # Debug info
                if row_count % 50000 == 0:
                    print(f"Processed {row_count} rows...")

        print(f"Loaded {len(positions)} positions (after initial filter if applied)")
        print(f"Found {len(cost_centers)} cost centers")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data from {input_file}: {e}")
        return None, None, None, None

    return positions, children, cost_centers, admin_col

def calculate_all_max_depths(positions, children):
    """Calculates the maximum depth underneath every position in the hierarchy using a bottom-up approach."""
    max_depths = {} # Stores max depth *under* each position
    parent_map = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    all_position_keys = set(positions.keys())

    # Build parent map and degrees
    for parent, kids in children.items():
        # Ensure parent exists in positions (handle potential data inconsistencies)
        if parent not in all_position_keys:
            continue
        valid_kids = [kid for kid in kids if kid in all_position_keys]
        if not valid_kids:
            continue
        out_degree[parent] = len(valid_kids)
        for child in valid_kids:
            parent_map[child].append(parent)
            in_degree[child] += 1

    # Initialize queue with leaf nodes (nodes with out_degree 0)
    # These nodes have a max depth of 0 beneath them.
    queue = []
    processed_nodes = set()
    for pos_num in all_position_keys:
        if out_degree[pos_num] == 0:
            max_depths[pos_num] = 0
            queue.append(pos_num)
            processed_nodes.add(pos_num)

    head = 0
    while head < len(queue):
        current_node = queue[head]
        head += 1

        # Process parents of the current node
        for parent in parent_map[current_node]:
            # Calculate potential depth for the parent based on this child
            current_child_depth = max_depths.get(current_node, -1)
            potential_parent_depth = current_child_depth + 1

            # Update parent's depth if this path is deeper
            max_depths[parent] = max(max_depths.get(parent, -1), potential_parent_depth)

            # Decrease in-degree count for the parent
            in_degree[parent] -= 1
            # If all children of the parent have been processed (in-degree becomes 0 for its children links? No, check out-degree)
            # Let's rethink the trigger to add parent to queue.
            # A parent is ready when all its children are processed.
            # We need to track how many children of a parent have been processed.

    # We need a different approach. The bottom-up requires knowing when all children are done.
    # Let's retry the bottom-up using out-degree decrementing. -> No, that's topological sort for dependencies.

    # Retry with explicit tracking of processed children for each parent.
    max_depths = {} # Reset
    processed_children_count = defaultdict(int)
    queue = [] # Start with leaves again
    processed_nodes_depth = set()

    for pos_num in all_position_keys:
        if out_degree[pos_num] == 0:
            max_depths[pos_num] = 0
            queue.append(pos_num)
            processed_nodes_depth.add(pos_num)

    head = 0
    while head < len(queue):
        child_node = queue[head]
        head += 1

        for parent_node in parent_map[child_node]:
            # Increment processed child count for the parent
            processed_children_count[parent_node] += 1
            # Update the parent's max depth based on this child
            max_depths[parent_node] = max(max_depths.get(parent_node, -1), max_depths[child_node] + 1)

            # If all children of this parent have been processed, add parent to queue
            if processed_children_count[parent_node] == out_degree[parent_node]:
                if parent_node not in processed_nodes_depth:
                    queue.append(parent_node)
                    processed_nodes_depth.add(parent_node)

    # Check for unprocessed nodes (cycles)
    if len(processed_nodes_depth) != len(all_position_keys):
        unprocessed_count = len(all_position_keys) - len(processed_nodes_depth)
        print(f"Warning: Could not determine depth for {unprocessed_count} positions. This might indicate cycles in the reporting structure.")
        # Assign default depth for nodes missed
        for pos_num in all_position_keys:
            if pos_num not in max_depths:
                max_depths[pos_num] = -1 # Indicate unknown depth

    print(f"Calculated max depths for {len(max_depths)} positions.")
    return max_depths

def populate_visited_nodes(root_position, positions, children, visited, filters=None):
    """Iteratively traverse the hierarchy to populate the visited set."""
    if root_position not in positions:
        return

    # Stack to track positions to process
    stack = [root_position]

    while stack:
        position_number = stack.pop()

        # Basic check if position exists
        if position_number not in positions:
            continue

        # Check filters ONLY if they are defined (avoids unnecessary checks)
        # Note: Filtering logic might need refinement depending on whether
        # the filter should prune branches or just exclude nodes from the final CSV.
        # Current approach: if a node matches the filter, we visit it and its children.
        # If create_d3_csv handles the filtering based on the visited set,
        # we might not need filter logic here at all. Assuming create_d3_csv does the filtering.
        # pos_data = positions[position_number]
        # if filters and not match_filters(pos_data, filters):
        #     continue # Skip this node if it doesn't match filters

        # Check if already visited *before* adding children
        if position_number in visited:
            continue

        visited.add(position_number)

        # Add children to stack
        if position_number in children:
            for child_position in children[position_number]:
                 # Add child only if it exists in the main positions dictionary
                if child_position in positions:
                    stack.append(child_position)

def build_org_chart(root_position, positions, children, cost_centers, input_file,
                      criteria_type=None, criteria_value=None, filters=None):
    # Create output directory with timestamp, criteria, and position ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_criteria_value = "".join(c if c.isalnum() else '_' for c in str(criteria_value)) if criteria_value else ""
    criteria_prefix = f"{criteria_type}_{safe_criteria_value}_" if criteria_type and safe_criteria_value else ""
    filter_suffix = ""
    if filters:
        filter_suffix = "_" + "_".join([f"{k}_{v}" for k, v in filters.items()])

    output_dir = f'org_chart_{criteria_prefix}{root_position}{filter_suffix}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    d3_csv_file = os.path.join(output_dir, f'org_data_{criteria_prefix}{root_position}{filter_suffix}.csv')

    print(f"Building chart data for root: {root_position}")
    if root_position not in positions:
        print(f"Error: Root position {root_position} not found in the loaded data.")
        return

    # Traverse the hierarchy to populate the 'visited' set
    visited = set()
    try:
        print(f"Traversing hierarchy for root {root_position} to identify relevant positions...")
        populate_visited_nodes(
            root_position, positions, children, visited, filters # Pass filters if needed by populate_visited_nodes
        )
        print(f"Found {len(visited)} positions in the hierarchy for root {root_position}.")
    except Exception as e:
        print(f"Error during hierarchy traversal for root {root_position}: {e}")
        # Decide how to handle traversal errors, e.g., skip this root or exit
        return # Skip generating file for this root if traversal fails

    # Generate the filtered CSV file (this is the only output now)
    print(f"Generating filtered data CSV...")
    if not visited:
         print(f"Warning: No positions visited for root {root_position}. Skipping CSV generation.")
         return

    # Pass the full d3_csv_file path
    create_d3_csv(d3_csv_file, root_position, positions, children, visited, input_file, filters)

    # --- REMOVED --- Generation of hierarchy, cost center, and levels CSV files

    print(f"Filtered data CSV created at: {d3_csv_file}")

def create_filtered_csv(output_csv_path, input_file, filters):
    """Create a CSV file containing only rows from input_file that match the given filters."""
    print(f"Writing filtered CSV for filters {filters} to {output_csv_path}...")

    try:
        # Read the header from the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        # Write the filtered data
        rows_written = 0
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)

            # Read the input file again and write only relevant rows
            with open(input_file, 'r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    # Apply filters
                    if match_filters(row, filters):
                        writer.writerow([row.get(col, '') for col in header])
                        rows_written += 1

        if rows_written > 0:
            print(f"Successfully wrote {rows_written} rows to {output_csv_path}")
        else:
            print(f"Warning: No rows matched the filters {filters} for file {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found during filtered CSV creation.")
    except Exception as e:
        print(f"Error creating filtered CSV {output_csv_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate filtered CSV files based on cost centers.')
    # Removed the mandatory position_id argument
    parser.add_argument('--filter', action='append', help='Filter the output by column=value (e.g., admin=vba)', default=[])
    parser.add_argument('--input_file', default='org_data.csv', help='Specify the input CSV file (default: org_data.csv)')
    parser.add_argument('--output_dir', default='cost_center_outputs', help='Specify the base output directory (default: cost_center_outputs)')
    args = parser.parse_args()

    # Set input file and output directory
    input_file = args.input_file
    base_output_dir = args.output_dir
    print(f"Using input file: {input_file}")
    print(f"Using base output directory: {base_output_dir}")

    # Process command-line filters if provided
    cli_filters = {}
    if args.filter:
        print("Applying base filters:")
        for filter_str in args.filter:
            if '=' in filter_str:
                column, value = filter_str.split('=', 1)
                print(f" - {column}={value}")
                cli_filters[column] = value
            else:
                print(f"Warning: Skipping invalid filter '{filter_str}'. Format should be column=value.")
    else:
        print("No base filters applied.")

    # Load data first
    positions, children, cost_centers, admin_col = load_data(input_file)

    if positions is None:
        print("Exiting due to data loading error.")
        sys.exit(1)

    if not positions:
        print("No positions found in the data. Cannot generate outputs.")
        sys.exit(1)

    if not cost_centers:
        print("No cost centers found in the data. Cannot generate cost center specific outputs.")
        # Decide if you want to exit or perhaps generate one file with just cli_filters
        # sys.exit(1) # Optionally exit if no cost centers are found

    # --- REMOVED Depth calculation, root identification, criteria grouping, final root selection ---

    # --- NEW: Loop through cost centers and generate filtered files ---
    print(f"\nFound {len(cost_centers)} unique cost centers. Generating filtered CSV for each...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for cost_center_desc in sorted(cost_centers.keys()): # Sort for consistent output order
        print(f"\n--- Processing Cost Center: {cost_center_desc} ---")

        # Combine CLI filters with the current cost center filter
        current_filters = cli_filters.copy()
        # Ensure we don't overwrite a cost center filter if provided via CLI
        if 'Cost Center Desc' not in current_filters:
             current_filters['Cost Center Desc'] = cost_center_desc
        elif current_filters['Cost Center Desc'].lower() != cost_center_desc.lower():
             print(f"Warning: CLI filter overrides Cost Center Desc. Generating for '{current_filters['Cost Center Desc']}' instead of '{cost_center_desc}' if Cost Center Desc was specified in --filter.")
             # If CLI filter specified a *different* cost center, we might skip or adjust logic.
             # Current logic uses the CLI filter if present. Let's stick to that.
             pass # Use the CLI filter value

        # Make a safe filename from the cost center description
        safe_cc_desc = "".join(c if c.isalnum() else '_' for c in cost_center_desc)
        if not safe_cc_desc:
            safe_cc_desc = "unknown_cost_center" # Handle empty cost center descriptions

        # Define output path
        output_filename = f'org_data_cc_{safe_cc_desc}_{timestamp}.csv'
        output_csv_path = os.path.join(base_output_dir, safe_cc_desc, output_filename) # Subdirectory per cost center

        # Create the filtered CSV
        create_filtered_csv(output_csv_path, input_file, current_filters)


    print("\nAll Done!")
