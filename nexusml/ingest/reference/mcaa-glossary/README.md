# MCAA Glossary and Abbreviations

This directory contains MCAA (Mechanical Contractors Association of America)
glossary and abbreviations files used by the NexusML classification system.

## File Format

MCAA glossary and abbreviations are provided as CSV files with the following
format:

```
Term,Definition
Absorber,A device containing liquid for absorbing refrigerant vapor
ABW / Orbital Welding,Automatic buttweld. Often used with fittings to indicated if they are sized to fit the weld head of an orbital welding machine
```

Each file has a header row with "Term" and "Definition" columns, followed by
rows containing the terms and their definitions.

## Example Files

- Glossary.csv (converted from PDF)
- Abbreviations.csv (converted from PDF)

## Usage

The system will automatically load and parse all CSV files in this directory.
Terms are case-insensitive for lookup purposes.

### Glossary

The glossary contains industry terms and their definitions. For example:

```
Term,Definition
Absorber,A device containing liquid for absorbing refrigerant vapor
Air conditioner,An assembly of equipment for a simultaneous control of air temperature and relative humidity
```

### Abbreviations

The abbreviations file contains industry acronyms and their meanings. For
example:

```
Term,Definition
ABS,Acrylonitrile Butadiene Styrene
AC,Alternating Current
```

## Conversion from PDF

The original files (Glossary.pdf.md and Abbreviations.pdf.md) were extracted
from PDF files and were in a table format. These have been converted to CSV
format for optimal use with the reference manager.

## Why CSV is Preferred

CSV files are preferred for several reasons:

1. **Structured data**: CSV provides a clear structure for tabular data
2. **Easy to parse**: CSV is a standard format that can be easily parsed by the
   reference manager
3. **Version control friendly**: Changes to CSV files can be tracked in version
   control systems
4. **Widely supported**: CSV files can be opened and edited with many tools,
   including Excel
5. **Programmatic access**: The reference manager can directly access and query
   the content

The system is designed to work with CSV files, providing the best performance
and reliability.

## Implementation Notes

The `MCAAGlossaryDataSource` and `MCAAAbbrDataSource` classes in the reference
manager are configured to load and parse these CSV files. They extract the terms
and definitions and make them available through the reference manager API.

These glossaries help the classification system understand industry terminology
and improve classification accuracy.
