# SMACNA Manufacturer Data

This directory contains SMACNA (Sheet Metal and Air Conditioning Contractors'
National Association) manufacturer data files used by the NexusML classification
system.

## File Format

SMACNA manufacturer data should be provided as JSON files (`.json`) with the
following structure:

```json
[
  {
    "Manufacturer": "Company Name",
    "Representative": "Representative Name or Company",
    "Product_Description": "Product 1, Product 2, Product 3"
  },
  {
    "Manufacturer": "Another Company",
    "Representative": "Another Representative",
    "Product_Description": "Product A, Product B"
  }
]
```

Each entry in the array represents a manufacturer with:

- Manufacturer name
- Representative name or company
- Comma-separated list of products

## Example Files

- smacna_manufacturers_2023.json
- smacna_directory.json

## Usage

The system will automatically load and combine all JSON files in this directory.
The data is used to:

1. Find manufacturers that produce specific products
2. Find products made by specific manufacturers
3. Enrich equipment data with manufacturer information

This helps the classification system understand relationships between equipment
types and manufacturers.

## Data Extraction

This data is typically extracted from SMACNA directories or membership guides.
You can use PDF extraction tools to convert the directories to text, then parse
the text into the required JSON format.

Example script for parsing SMACNA directory text:

```python
import json
import re

def parse_smacna_directory(text_file, output_file):
    with open(text_file, 'r') as f:
        content = f.read()

    # Extract manufacturer sections
    sections = re.findall(r'MANUFACTURER: (.*?)\nREPRESENTATIVE: (.*?)\nPRODUCTS: (.*?)(?=\n\nMANUFACTURER:|$)', content, re.DOTALL)

    manufacturers = []
    for manuf, rep, products in sections:
        manufacturers.append({
            "Manufacturer": manuf.strip(),
            "Representative": rep.strip(),
            "Product_Description": products.strip()
        })

    with open(output_file, 'w') as f:
        json.dump(manufacturers, f, indent=2)
```
