# SOP-002: Wichita Animal Shelter Asset Data Pipeline

## Purpose

This procedure documents the steps required to run the Wichita Animal Shelter Asset Data Pipeline, which extracts data from a CSV file, analyzes it, validates it, and exports it to a SQLite database. This pipeline helps facility managers track and manage HVAC and other building equipment assets.

## Scope

This SOP covers:

- Configuration setup for the Wichita pipeline
- Running the pipeline
- Understanding the output files
- Interpreting analysis and validation reports
- Troubleshooting common issues

## Prerequisites

- Python 3.9+ installed
- FCA Dashboard environment set up (see SOP-001)
- Access to the Wichita Animal Shelter asset data CSV file
- Appropriate permissions to create and modify files in the output directory

## Procedure

### 1. Configuration Setup

1. Ensure the Wichita pipeline configuration is present in `fca_dashboard/config/settings.yml`:

   ```yaml
   # Wichita Animal Shelter pipeline settings
   wichita:
     input_file: "uploads/Asset_List Wichita Animal Shelter (1).csv"
     output_dir: "outputs/pipeline/wichita"
     db_name: "wichita_assets.db"
     columns_to_extract: []  # Empty list means use all columns
     drop_na_columns: ["Asset Name", "Asset Category Name"]  # Drop rows where these columns have NaN values
   ```

2. If the configuration is not present, add it to the settings file.

3. Verify that the input file exists at the specified path:

   ```bash
   ls -la uploads/Asset_List\ Wichita\ Animal\ Shelter\ \(1\).csv
   ```

4. Create the output directory if it doesn't exist:

   ```bash
   mkdir -p outputs/pipeline/wichita
   ```

### 2. Running the Pipeline

1. Activate the virtual environment:

   ```bash
   source ./.venv/scripts/activate  # Windows
   # OR
   source ./.venv/bin/activate      # macOS/Linux
   ```

2. Run the pipeline:

   ```bash
   python -m fca_dashboard.pipelines.pipeline_wichita
   ```

3. Alternatively, you can use the Python interpreter directly:

   ```bash
   python fca_dashboard/pipelines/pipeline_wichita.py
   ```

4. Monitor the console output for progress and any errors.

### 3. Understanding the Output Files

The pipeline generates several output files in the `outputs/pipeline/wichita` directory:

1. **SQLite Database**: `wichita_assets.db`
   - Contains the processed asset data in a structured format
   - Can be queried using SQL tools or libraries

2. **Database Schema**: `wichita_assets_schema.sql`
   - Contains the SQL schema definition for the database table
   - Useful for understanding the structure of the data

3. **Analysis Report**: `wichita_assets_analysis_report.txt`
   - Contains statistical analysis of the data
   - Includes unique value counts, column statistics, and text pattern analysis

4. **Validation Report**: `wichita_assets_validation_report.txt`
   - Contains validation results for the data
   - Includes missing value checks, duplicate row detection, and data type validation

### 4. Interpreting the Reports

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

### 5. Using the Database

To query the SQLite database:

1. Using the SQLite command-line tool:

   ```bash
   sqlite3 outputs/pipeline/wichita/wichita_assets.db
   ```

2. In the SQLite shell:

   ```sql
   -- View all tables
   .tables
   
   -- View schema
   .schema wichita_assets
   
   -- Query data
   SELECT "Asset Category Name", COUNT(*) FROM wichita_assets GROUP BY "Asset Category Name";
   
   -- Exit
   .exit
   ```

3. Using Python:

   ```python
   import sqlite3
   
   # Connect to the database
   conn = sqlite3.connect('outputs/pipeline/wichita/wichita_assets.db')
   cursor = conn.cursor()
   
   # Query data
   cursor.execute('SELECT "Asset Category Name", COUNT(*) FROM wichita_assets GROUP BY "Asset Category Name"')
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
   - Verify that the table contains the expected number of rows (should be 46 rows)
   - Verify that the table contains the expected columns (should match the CSV file)

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
   - Check the CSV file for formatting problems
   - Consider preprocessing the data to fix issues before running the pipeline

5. **Memory errors**:
   - For large files, consider increasing available memory
   - Process the data in smaller batches if necessary

## References

- [ETL Pipeline v4 Implementation Guide](../../docs/guide/guide.md)
- [Database Utilities Documentation](../../fca_dashboard/utils/database/README.md)
- [Excel Utilities Documentation](../../fca_dashboard/utils/excel/README.md)
- [SOP-001: ETL Pipeline Environment Setup](./001-env-setup.md)

## Revision History

| Version | Date       | Author   | Changes                                 |
|---------|------------|----------|----------------------------------------|
| 1.0     | 2025-03-04 | ETL Team | Initial version                         |