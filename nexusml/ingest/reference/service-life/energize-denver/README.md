# Energize Denver Service Life Data

This directory contains Energize Denver service life data for equipment, used by
the NexusML classification system.

## File Format

Energize Denver service life data is typically extracted from the Energize
Denver Technical Guidance document and converted to a structured format. The
data is provided as CSV files with the following columns:

```
Equipment Type,Median Years,Min Years,Max Years,Source
Chiller,23,15,30,Energize Denver
Cooling Tower,20,15,25,Energize Denver
Boiler,25,20,30,Energize Denver
```

## Example Files

- energize_denver_service_life.csv

## Usage

The system will automatically load all CSV files in this directory. Make sure
all files follow the same column structure.

## Data Source

This data is typically extracted from the Energize Denver Technical Guidance
document, which provides service life estimates for various building systems and
equipment. The document is published by the City and County of Denver as part of
their building performance standards.

Energize Denver service life data is used as a secondary source when ASHRAE data
is not available for a specific equipment type.

## Extraction Process

The service life data is extracted from the PDF file using the
`extract_service_life.py` script in this directory. The script uses PyPDF2 to
extract text from the PDF and regular expressions to identify service life
tables. The extracted data is then saved to a CSV file.

If the automatic extraction fails, the script will create a CSV file with sample
data based on the examples in this README.
