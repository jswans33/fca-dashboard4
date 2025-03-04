# SOP-002: Data Extraction Pipeline Creation and Usage Guide

## Purpose

This procedure documents the steps required to create, configure, and run data extraction pipelines in the FCA Dashboard system. These pipelines extract data from various sources (CSV, Excel, etc.), analyze it, validate it, and export it to a database. This standardized approach ensures consistent data processing across different data sources.

## Scope

This SOP covers:

- Creating a new extraction pipeline
- Configuration setup for pipelines
- Running pipelines
- Understanding the output files
- Interpreting analysis and validation reports
- Troubleshooting common issues

## Prerequisites

- Python 3.9+ installed
- FCA Dashboard environment set up (see SOP-001)
- Access to the source data files
- Appropriate permissions to create and modify files in the output directory

## Procedure

### 1. Creating a New Pipeline

1. Create a new Python file in the `fca_dashboard/pipelines` directory:

   ```bash
   touch fca_dashboard/pipelines/pipeline_[source_name].py
   ```

   Replace `[source_name]` with a descriptive name for your data source (e.g., `medtronics`, `wichita`).

2. Use the following template structure for your pipeline:

   ```python
   """
   [Source Name] Data Pipeline.
   
   This pipeline extracts data from [source description],
   analyzes and validates it, and then exports it to a database.
   """
   
   import os
   import sys
   from pathlib import Path
   
   # Add the project root to the Python path
   sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
   
   import pandas as pd
   
   from fca_dashboard.config.settings import settings
   from fca_dashboard.utils.database import (
       get_table_schema,
       save_dataframe_to_database,
   )
   from fca_dashboard.utils.excel import (
       analyze_column_statistics,
       analyze_text_columns,
       analyze_unique_values,
       validate_dataframe,
   )
   from fca_dashboard.utils.logging_config import get_logger
   from fca_dashboard.utils.path_util import get_root_dir, resolve_path
   
   
   class [SourceName]Pipeline:
       """Pipeline for processing [Source Name] Data."""
       
       def __init__(self):
           """Initialize the pipeline."""
           self.logger = get_logger("[source_name]_pipeline")
           
           # Get settings from configuration
           self.input_file = settings.get("[source_name].input_file", "default_path.csv")
           self.output_dir = settings.get("[source_name].output_dir", "outputs/pipeline/[source_name]")
           self.db_name = settings.get("[source_name].db_name", "[source_name]_data.db")
           
           # Initialize data storage
           self.extracted_data = None
           self.analysis_results = None
           self.validation_results = None
       
       def extract(self):
           """Extract data from the source file."""
           # Implementation specific to your data source
           pass
       
       def analyze(self, df):
           """Analyze the extracted data."""
           # Implementation specific to your data source
           pass
       
       def validate(self, df):
           """Validate the extracted data."""
           # Implementation specific to your data source
           pass
       
       def export(self, df):
           """Export the data to a database."""
           # Implementation specific to your data source
           pass
       
       def save_reports(self, df):
           """Save analysis and validation reports."""
           # Implementation specific to your data source
           pass
       
       def run(self):
           """Run the pipeline."""
           # Standard pipeline execution flow
           pass
   
   
   def main():
       """Main function."""
       pipeline = [SourceName]Pipeline()
       results = pipeline.run()
       # Print results
       return 0 if results["status"] == "success" else 1
   
   
   if __name__ == "__main__":
       sys.exit(main())
   ```

3. Implement the specific methods for your data source.

### 2. Configuration Setup

1. Add configuration for your pipeline in `fca_dashboard/config/settings.yml`:

   ```yaml
   # [Source Name] pipeline settings
   [source_name]:
     input_file: "path/to/your/input/file.csv"  # or .xlsx
     output_dir: "outputs/pipeline/[source_name]"
     db_name: "[source_name]_data.db"
     columns_to_extract: []  # Empty list means use all columns
     drop_na_columns: ["Important Column 1", "Important Column 2"]  # Drop rows where these columns have NaN values
   ```

   Example (Wichita Animal Shelter):
   ```yaml
   # Wichita Animal Shelter pipeline settings
   wichita:
     input_file: "uploads/Asset_List Wichita Animal Shelter (1).csv"
     output_dir: "outputs/pipeline/wichita"
     db_name: "wichita_assets.db"
     columns_to_extract: []  # Empty list means use all columns
     drop_na_columns: ["Asset Name", "Asset Category Name"]
   ```

2. Verify that the input file exists at the specified path:

   ```bash
   ls -la [path/to/your/input/file]
   ```

3. Create the output directory if it doesn't exist:

   ```bash
   mkdir -p [output_directory]
   ```

### 3. Running a Pipeline

1. Activate the virtual environment:

   ```bash
   source ./.venv/scripts/activate  # Windows
   # OR
   source ./.venv/bin/activate      # macOS/Linux
   ```

2. Run the pipeline:

   ```bash
   python -m fca_dashboard.pipelines.pipeline_[source_name]
   ```

3. Alternatively, you can use the Python interpreter directly:

   ```bash
   python fca_dashboard/pipelines/pipeline_[source_name].py
   ```

4. Monitor the console output for progress and any errors.

### 4. Understanding the Output Files

The pipeline generates several output files in the configured output directory:

1. **SQLite Database**: `[source_name]_data.db`
   - Contains the processed data in a structured format
   - Can be queried using SQL tools or libraries

2. **Database Schema**: `[table_name]_schema.sql`
   - Contains the SQL schema definition for the database table
   - Useful for understanding the structure of the data

3. **Analysis Report**: `[source_name]_analysis_report.txt`
   - Contains statistical analysis of the data
   - Includes unique value counts, column statistics, and text pattern analysis

4. **Validation Report**: `[source_name]_validation_report.txt`
   - Contains validation results for the data
   - Includes missing value checks, duplicate row detection, and data type validation

### 5. Interpreting the Reports

#### Analysis Report

The analysis report provides insights into the data structure and content:

1. **Unique Values Analysis**:
   - Shows the distribution of values in categorical columns
   - Identifies the number of unique values and their frequencies
   - Helps identify potential data quality issues (e.g., misspellings, inconsistent naming)

2. **Column Statistics**:
   - Provides statistical measures for numeric columns (min, max, mean, median, etc.)
   - Identifies potential outliers
   - Helps understand the range and distribution of numeric data
3. **Text Analysis**:
   - Analyzes text patterns in string columns
   - Identifies minimum, maximum, and average text lengths
   - Helps identify inconsistencies in text formatting

Example from Wichita Animal Shelter pipeline:

```text
Unique Values Analysis:
--------------------------------------------------
  Column: Asset Category Name
    Unique value count: 4
    Null count: 0 (0.00%)
    Unique values: Terminal & Package Units, HVAC Distribution Systems, Plumbing, Cooling Systems
    Value counts (top 5):
      HVAC Distribution Systems: 25
      Terminal & Package Units: 17
      Plumbing: 3
      Cooling Systems: 1
```
   - Helps identify inconsistencies in text formatting

#### Validation Report

The validation report highlights potential data quality issues:

1. **Missing Values Report**:
   - Shows the percentage of missing values in key columns
   - Helps identify data completeness issues

2. **Duplicate Rows Report**:
   - Identifies duplicate entries in the data
   - Helps ensure data integrity

3. **Data Types Report**:
   - Verifies that columns have the expected data types
   - Helps identify data type inconsistencies

Example from Wichita Animal Shelter pipeline:
```
Missing Values Report:
--------------------------------------------------
  Asset Name: 0.00% missing
  Asset Category Name: 0.00% missing
  Type: 0.00% missing
  Manufacturer: 10.87% missing
```

### 6. Using the Database

To query the SQLite database:

1. Using the SQLite command-line tool:

   ```bash
   sqlite3 [output_directory]/[database_name]
   ```

   Example:
   ```bash
   sqlite3 outputs/pipeline/wichita/wichita_assets.db
   ```

2. In the SQLite shell:

   ```sql
   -- View all tables
   .tables
   
   -- View schema
   .schema [table_name]
   
   -- Query data
   SELECT [column_name], COUNT(*) FROM [table_name] GROUP BY [column_name];
   
   -- Exit
   .exit
   ```

   Example:
   ```sql
   .schema wichita_assets
   SELECT "Asset Category Name", COUNT(*) FROM wichita_assets GROUP BY "Asset Category Name";
   ```

3. Using Python:

   ```python
   import sqlite3
   
   # Connect to the database
   conn = sqlite3.connect('[output_directory]/[database_name]')
   cursor = conn.cursor()
   
   # Query data
   cursor.execute('SELECT [column_name], COUNT(*) FROM [table_name] GROUP BY [column_name]')
   results = cursor.fetchall()
   
   # Print results
   for row in results:
       print(f"{row[0]}: {row[1]}")
   
   # Close connection
   conn.close()
   ```

## Verification

1. Verify the pipeline ran successfully:
   - Check for the "Pipeline completed successfully" message in the console output
   - Verify that all expected output files exist in the output directory

2. Verify the database was created correctly:
   - Check that the database file exists
   - Verify that the table contains the expected number of rows
   - Verify that the table contains the expected columns (should match the source file)

3. Verify the reports contain meaningful information:
   - Check that the analysis report includes statistics for key columns
   - Verify that the validation report identifies any data quality issues

## Troubleshooting

1. **Input file not found**:
   - Verify the file path in the settings.yml file
   - Check that the file exists in the specified location
   - Ensure the file name and extension are correct (case-sensitive)

2. **Permission errors**:
   - Verify that you have write permissions for the output directory
   - Try running the script with elevated privileges if necessary

3. **Database errors**:
   - If the database is locked, ensure no other processes are using it
   - If the database is corrupted, delete it and run the pipeline again
   - Check for disk space issues if the database fails to write

4. **Data quality issues**:
   - Review the validation report for missing values or data type issues
   - Check the source file for formatting problems
   - Consider preprocessing the data to fix issues before running the pipeline

5. **Memory errors**:
   - For large files, consider increasing available memory
   - Process the data in smaller batches if necessary

6. **Pipeline-specific errors**:
   - Check the pipeline implementation for any source-specific handling
   - Verify that the data format matches what the pipeline expects
   - Consult the pipeline's documentation for specific requirements

## Examples

### Example 1: Wichita Animal Shelter Pipeline

```bash
# Run the Wichita pipeline
python -m fca_dashboard.pipelines.pipeline_wichita
```

Expected output:
- SQLite database: `outputs/pipeline/wichita/wichita_assets.db`
- Schema file: `outputs/pipeline/wichita/wichita_assets_schema.sql`
- Analysis report: `outputs/pipeline/wichita/wichita_assets_analysis_report.txt`
- Validation report: `outputs/pipeline/wichita/wichita_assets_validation_report.txt`

### Example 2: Medtronics Pipeline

```bash
# Run the Medtronics pipeline
python -m fca_dashboard.pipelines.pipeline_medtronics
```

Expected output:
- SQLite database: `outputs/pipeline/medtronic/medtronics_assets.db`
- Schema file: `outputs/pipeline/medtronic/asset_data_schema.sql`
- Analysis report: `outputs/pipeline/medtronic/asset_data_analysis_report.txt`
- Validation report: `outputs/pipeline/medtronic/asset_data_validation_report.txt`

## References

- [ETL Pipeline v4 Implementation Guide](../../docs/guide/guide.md)
- [Database Utilities Documentation](../../fca_dashboard/utils/database/README.md)
- [Excel Utilities Documentation](../../fca_dashboard/utils/excel/README.md)
- [SOP-001: ETL Pipeline Environment Setup](./001-env-setup.md)

## Revision History

| Version | Date       | Author   | Changes                                              |
| ------- | ---------- | -------- | ---------------------------------------------------- |
| 1.0     | 2025-03-04 | ETL Team | Initial version                                      |
| 1.1     | 2025-03-04 | ETL Team | Updated to be a general pipeline guide with examples |