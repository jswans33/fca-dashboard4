#!/usr/bin/env python3
"""
Simple script to extract data from a markdown table and convert it to CSV.
This script specifically targets the "Services Pages from UniFormat_redesign.pdf.md" file
and handles the hierarchical structure where NUMBER and TITLE values don't repeat for child rows.
"""
import csv
import os

# Input and output file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Services Pages from UniFormat_redesign.pdf.md")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "services_uniformat.csv")

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
    # Process the file
    data = extract_table_data(INPUT_FILE)
    
    # Write to CSV
    write_csv(data, OUTPUT_FILE)
    
    print(f"Successfully converted markdown table to {os.path.basename(OUTPUT_FILE)}")
    print(f"Extracted {len(data)} rows of data")

if __name__ == "__main__":
    main() 