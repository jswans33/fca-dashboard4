#!/usr/bin/env python3
import csv
import re
import os
import sys

def extract_table_data(markdown_file):
    """Extract data from markdown table, handling hierarchical structure."""
    with open(markdown_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find all table sections (they start with a header row)
    table_sections = []
    in_table = False
    start_line = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        # Detect table headers
        if line.startswith('| NUMBER | TITLE |'):
            if in_table:
                # End previous section
                table_sections.append((start_line, i-1))
            # Start new section
            in_table = True
            start_line = i+2  # +2 to skip header and separator rows
        # End of file or section
        elif in_table and (not line or not line.startswith('|')):
            if i > start_line:
                table_sections.append((start_line, i-1))
                in_table = False
    
    # Add the last section if file ends while in a table
    if in_table and start_line < len(lines):
        table_sections.append((start_line, len(lines)-1))
    
    # Process each table section
    data = []
    
    for start, end in table_sections:
        current_number = None
        current_title = None
        current_mf_number = None
        
        for i in range(start, end+1):
            line = lines[i].strip()
            
            # Skip empty lines or non-table lines
            if not line or not line.startswith('|'):
                continue
                
            # Split the line by pipe character and strip whitespace
            cells = [cell.strip() for cell in line.split('|')]
            
            # Remove empty strings from the beginning and end (from the outer pipes)
            cells = [cell for cell in cells if cell]
            
            # Skip if we don't have enough cells
            if len(cells) < 4:
                continue
                
            # Extract cell values
            number, title, mf_number, explanation = cells[0:4]
            
            # Skip separator rows or headers that consist of dashes
            if all(cell.startswith('---') or cell == '---------' for cell in cells):
                continue
            
            # Update current values if they are not empty
            if number and number != "--------":
                current_number = number
            
            if title and title != "-------":
                current_title = title
                
            if mf_number and mf_number != "-----------":
                current_mf_number = mf_number
            
            # Add row to data with all copied down values
            data.append({
                'NUMBER': current_number,
                'TITLE': current_title,
                'MF NUMBER': mf_number if mf_number and mf_number != "-----------" else current_mf_number,
                'EXPLANATION': explanation if explanation != "-------------" else ""
            })
    
    return data

def write_csv(data, output_file):
    """Write data to CSV file."""
    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        fieldnames = ['NUMBER', 'TITLE', 'MF NUMBER', 'EXPLANATION']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for markdown files in the current directory
    md_files = [f for f in os.listdir(script_dir) if f.endswith('.md') and os.path.isfile(os.path.join(script_dir, f))]
    
    if not md_files:
        print("Error: No markdown files found in the directory.")
        sys.exit(1)
    
    # Print available files and let user choose
    if len(md_files) > 1:
        print("Available markdown files:")
        for i, file in enumerate(md_files):
            print(f"{i+1}. {file}")
        
        try:
            choice = int(input("Enter the number of the file to process: "))
            if choice < 1 or choice > len(md_files):
                print("Invalid choice. Exiting.")
                sys.exit(1)
            input_file = md_files[choice-1]
        except ValueError:
            print("Invalid input. Exiting.")
            sys.exit(1)
    else:
        input_file = md_files[0]
    
    input_path = os.path.join(script_dir, input_file)
    output_file = os.path.join(script_dir, os.path.splitext(input_file)[0] + '.csv')
    
    # Process the file
    data = extract_table_data(input_path)
    
    # Write to CSV
    write_csv(data, output_file)
    
    print(f"Successfully converted {input_file} to {os.path.basename(output_file)}")
    print(f"Extracted {len(data)} rows of data")

if __name__ == "__main__":
    main() 