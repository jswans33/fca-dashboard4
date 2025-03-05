# OmniClass Description Generator Scripts

This directory contains scripts for generating descriptions for OmniClass codes
using the Claude API.

## Overview

The OmniClass Construction Classification System is a means of organizing and
retrieving information specifically designed for the construction industry.
These scripts help generate plain-English descriptions for OmniClass codes,
which can be used for training classification models.

## Available Scripts

### 1. `run_omniclass_descriptions.py`

This script processes all OmniClass tables and divisions, saving each to a
separate output file.

```bash
python fca_dashboard/examples/omniclass_scripts/run_omniclass_descriptions.py
```

Features:

- Processes all tables and divisions in sequence
- Saves each to a separate output file
- Creates a detailed log file
- Provides a summary of successful and failed processes

### 2. `process_omniclass_section.py`

This script allows processing a specific OmniClass table or division.

```bash
python fca_dashboard/examples/omniclass_scripts/process_omniclass_section.py [section_name]
```

Available sections:

- `table21_elements` - Table 21 (Elements)
- `table22_work_results` - Table 22 (Work Results)
- `table23_products` - Table 23 (Products)
- `table22_div22_plumbing` - Table 22, Division 22 (Plumbing)
- `table22_div23_hvac` - Table 22, Division 23 (HVAC)
- `table23_div23_hvac` - Table 23, Division 23 (HVAC Products)
- `all` - Process all sections

Example:

```bash
# Process just the HVAC section
python fca_dashboard/examples/omniclass_scripts/process_omniclass_section.py table22_div23_hvac
```

## OmniClass Tables and Divisions

### Table 21 - Elements (Rows 2050-2689)

Elements are major components, assemblies, or "systems" of a facility. This
table includes systems like foundations, exterior walls, HVAC systems, etc.

### Table 22 - Work Results (Rows 2690-11635)

Work Results are construction results achieved in the various phases of the
construction process. This is similar to MasterFormat and describes construction
results by trade.

#### Division 22 - Plumbing (Rows 6510-6797)

This division covers plumbing systems and components.

#### Division 23 - HVAC (Rows 6798-7231)

This division covers heating, ventilation, and air conditioning systems and
components.

### Table 23 - Products (Rows 11636-18532)

Products are the individual products and materials used in construction.

#### Division 23 - HVAC Products (Rows 14473-14662)

This division covers products specifically related to HVAC systems.

## Output Files

All output files are saved to the `fca_dashboard/generator/ingest/output/`
directory with filenames that indicate the table or division they contain:

- `omniclass_table21_elements.csv`
- `omniclass_table22_work_results.csv`
- `omniclass_table23_products.csv`
- `omniclass_table22_div22_plumbing.csv`
- `omniclass_table22_div23_hvac.csv`
- `omniclass_table23_div23_hvac.csv`

## Troubleshooting

If you encounter issues:

1. Make sure you're running the scripts from the project root directory
   (`c:/Repos/fca-dashboard4`)
2. Ensure your virtual environment is activated and located at `.venv` in the
   project root
3. Check that the required packages are installed in the virtual environment:
   ```bash
   .venv/Scripts/pip install pandas anthropic python-dotenv tqdm loguru
   ```
4. If you're still having issues with the virtual environment, you can modify
   the scripts to use your system Python instead:
   - Open
     `fca_dashboard/examples/omniclass_scripts/process_omniclass_section.py`
   - Change `.venv/Scripts/python` to `python` in the `cmd` list
   - Do the same for
     `fca_dashboard/examples/omniclass_scripts/run_omniclass_descriptions.py`
5. Check the log file for detailed error messages

### Create output directory first

```bash
mkdir -p fca_dashboard/generator/ingest/output
```

### Process Table 21 (Elements)

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py --start 2050 --end 2690 --output-file fca_dashboard/generator/ingest/output/omniclass_table21_elements.csv
```

### Process Table 22 (Work Results)

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py --start 2690 --end 11636 --output-file fca_dashboard/generator/ingest/output/omniclass_table22_work_results.csv
```

### Process Table 23 (Products)

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py --start 11636 --end 18533 --output-file fca_dashboard/generator/ingest/output/omniclass_table23_products.csv
```

### Process Table 22, Division 22 (Plumbing)

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py --start 6510 --end 6798 --output-file fca_dashboard/generator/ingest/output/omniclass_table22_div22_plumbing.csv
```

### Process Table 22, Division 23 (HVAC)

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py --start 6798 --end 7232 --output-file fca_dashboard/generator/ingest/output/omniclass_table22_div23_hvac.csv
```

### Process Table 23, Division 23 (HVAC Products)

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py --start 14473 --end 14663 --output-file fca_dashboard/generator/ingest/output/omniclass_table23_div23_hvac.csv
```
