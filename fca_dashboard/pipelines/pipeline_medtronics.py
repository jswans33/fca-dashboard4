"""
Medtronics Asset Data Pipeline.

This pipeline extracts data from the Medtronics Asset Log Uploader Excel file,
analyzes and validates it, and then exports it to a SQLite database.
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
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    extract_excel_with_config,
    validate_dataframe,
)
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


class MedtronicsPipeline:
    """
    Pipeline for processing Medtronics Asset Data.
    
    This pipeline extracts data from the Medtronics Asset Log Uploader Excel file,
    analyzes and validates it, and then exports it to a SQLite database.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.logger = get_logger("medtronics_pipeline")
        
        # Get file paths from settings
        self.input_file = settings.get("medtronics.input_file", "uploads/Medtronics - Asset Log Uploader.xlsx")
        self.output_dir = settings.get("medtronics.output_dir", "outputs/pipeline/medtronic")
        self.db_name = settings.get("medtronics.db_name", "medtronics_assets.db")
        
        # Get sheet name from settings
        self.sheet_name = settings.get("medtronics.sheet_name", "Asset Data")
        
        # Get extraction configuration from settings
        self.extraction_config = settings.get("excel_utils.extraction", {})
        
        # Get validation configuration from settings
        self.validation_config = settings.get("excel_utils.validation", {})
        
        # Get analysis configuration from settings
        self.analysis_config = settings.get("excel_utils.analysis", {})
        
        # Get columns to extract from settings
        self.columns_to_extract = settings.get("medtronics.columns_to_extract", [])
        
        # Get columns to drop NaN values from settings
        self.drop_na_columns = settings.get("medtronics.drop_na_columns", [])
        
        # Initialize data storage
        self.extracted_data = None
        self.analysis_results = None
        self.validation_results = None
    
    def extract(self):
        """
        Extract data from the Medtronics Excel file.
        
        Returns:
            The extracted DataFrame for the Asset Data sheet.
        """
        self.logger.info(f"Extracting data from {self.input_file}")
        
        # Resolve the file path
        file_path = resolve_path(self.input_file)
        
        # Analyze the Excel file structure
        analysis = analyze_excel_structure(file_path)
        self.logger.info(f"File type: {analysis['file_type']}")
        self.logger.info(f"Sheet names: {analysis['sheet_names']}")
        
        # Create extraction configuration
        config = {}
        
        # Add default settings
        if "default" in self.extraction_config:
            config["default"] = self.extraction_config["default"]
        
        # Add sheet-specific settings
        for sheet_name in analysis['sheet_names']:
            # Convert sheet name to the format used in settings (lowercase with underscores)
            settings_key = sheet_name.lower().replace(" ", "_")
            
            # If there are settings for this sheet, add them to the config
            if settings_key in self.extraction_config:
                config[sheet_name] = self.extraction_config[settings_key]
        
        self.logger.info(f"Using extraction configuration: {config}")
        
        # Extract data from the Excel file
        extracted_data = extract_excel_with_config(file_path, config)
        
        # Store the extracted data
        self.extracted_data = extracted_data
        
        # Return the Asset Data sheet
        # The extraction process converts sheet names to lowercase with underscores
        normalized_sheet_name = self.sheet_name.lower().replace(" ", "_")
        
        df = None
        if normalized_sheet_name in extracted_data:
            df = extracted_data[normalized_sheet_name]
        elif self.sheet_name in extracted_data:
            df = extracted_data[self.sheet_name]
        else:
            self.logger.error(f"Sheet '{self.sheet_name}' not found in extracted data")
            self.logger.error(f"Available sheets: {list(extracted_data.keys())}")
            return None
        
        # Filter columns if columns_to_extract is specified
        if self.columns_to_extract:
            self.logger.info(f"Filtering columns to: {self.columns_to_extract}")
            
            # Convert Excel column letters to column indices
            if all(isinstance(col, str) and len(col) <= 3 and col.isalpha() for col in self.columns_to_extract):
                # Convert Excel column letters to 0-based indices
                def excel_col_to_index(col_str):
                    """Convert Excel column letter to 0-based index."""
                    col_str = col_str.upper()
                    result = 0
                    for c in col_str:
                        result = result * 26 + (ord(c) - ord('A') + 1)
                    return result - 1
                
                # Get column indices
                col_indices = [excel_col_to_index(col) for col in self.columns_to_extract]
                
                # Get column names from indices
                if len(df.columns) > max(col_indices):
                    col_names = [df.columns[i] for i in col_indices if i < len(df.columns)]
                    df = df[col_names]
                    self.logger.info(f"Filtered to {len(col_names)} columns: {col_names}")
                else:
                    self.logger.warning(f"Some column indices are out of range. Max index: {len(df.columns) - 1}")
            else:
                # Assume column names are provided
                existing_cols = [col for col in self.columns_to_extract if col in df.columns]
                if existing_cols:
                    df = df[existing_cols]
                    self.logger.info(f"Filtered to {len(existing_cols)} columns: {existing_cols}")
                else:
                    self.logger.warning(f"None of the specified columns exist in the DataFrame")
        
        # Normalize column names to lowercase
        if df is not None:
            self.logger.info("Normalizing column names to lowercase")
            df.columns = [col.lower() for col in df.columns]
            self.logger.info(f"Normalized column names: {list(df.columns)}")
        
        # Drop rows with NaN values in specified columns
        if self.drop_na_columns and df is not None:
            original_row_count = len(df)
            
            # Convert drop_na_columns to lowercase for consistency
            drop_na_columns_lower = [col.lower() for col in self.drop_na_columns]
            
            # Check if the specified columns exist in the DataFrame
            existing_cols = [col for col in drop_na_columns_lower if col in df.columns]
            
            if existing_cols:
                self.logger.info(f"Dropping rows with NaN values in columns: {existing_cols}")
                df = df.dropna(subset=existing_cols)
                self.logger.info(f"Dropped {original_row_count - len(df)} rows with NaN values")
            else:
                self.logger.warning(f"None of the specified columns for dropping NaN values exist in the DataFrame")
                self.logger.warning(f"Available columns: {list(df.columns)}")
        
        return df
    
    def analyze(self, df):
        """
        Analyze the extracted data.
        
        Args:
            df: The DataFrame to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info(f"Analyzing data from sheet '{self.sheet_name}'")
        
        # Get default analysis settings
        default_analysis = self.analysis_config.get("default", {})
        
        # Initialize results dictionary
        results = {}
        
        # 1. Analyze unique values
        self.logger.info("Analyzing unique values...")
        unique_values_settings = default_analysis.get("unique_values", {})
        max_unique_values = unique_values_settings.get("max_unique_values", 20)
        
        # Select columns for unique value analysis
        # For demonstration purposes, we'll analyze the first 5 columns
        unique_columns = list(df.columns[:5])
        unique_values_results = analyze_unique_values(
            df, 
            columns=unique_columns,
            max_unique_values=max_unique_values
        )
        results['unique_values'] = unique_values_results
        
        # 2. Analyze column statistics for numeric columns
        self.logger.info("Analyzing column statistics...")
        column_stats_settings = default_analysis.get("column_statistics", {})
        include_outliers = column_stats_settings.get("include_outliers", True)
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns[:5])
            results['column_statistics'] = column_stats_results
        
        # 3. Analyze text columns
        self.logger.info("Analyzing text columns...")
        text_analysis_settings = default_analysis.get("text_analysis", {})
        include_pattern_analysis = text_analysis_settings.get("include_pattern_analysis", True)
        
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            text_analysis_results = analyze_text_columns(df, columns=text_columns[:5])
            results['text_analysis'] = text_analysis_results
        
        # Store the analysis results
        self.analysis_results = results
        
        return results
    
    def validate(self, df):
        """
        Validate the extracted data.
        
        Args:
            df: The DataFrame to validate.
            
        Returns:
            A dictionary containing the validation results.
        """
        self.logger.info(f"Validating data from sheet '{self.sheet_name}'")
        
        # Create validation configuration
        validation_config = {}
        
        # Add default settings
        if "default" in self.validation_config:
            default_settings = self.validation_config["default"]
            
            # Add missing values check
            if "missing_values" in default_settings:
                missing_values_settings = default_settings["missing_values"]
                validation_config["missing_values"] = {
                    "columns": missing_values_settings.get("columns") or list(df.columns),
                    "threshold": missing_values_settings.get("threshold", 0.5)
                }
            
            # Add duplicate rows check
            if "duplicate_rows" in default_settings:
                duplicate_rows_settings = default_settings["duplicate_rows"]
                validation_config["duplicate_rows"] = {
                    "subset": duplicate_rows_settings.get("subset")
                }
            
            # Add data types check
            if "data_types" in default_settings:
                data_types_settings = default_settings["data_types"]
                
                # Create type specifications based on settings
                type_specs = {}
                
                # Add date columns
                for col in data_types_settings.get("date_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "date"
                
                # Add numeric columns
                for col in data_types_settings.get("numeric_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "float"
                
                # Add string columns
                for col in data_types_settings.get("string_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "str"
                
                # Add boolean columns
                for col in data_types_settings.get("boolean_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "bool"
                
                if type_specs:
                    validation_config["data_types"] = type_specs
        
        # Add sheet-specific validation
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = self.sheet_name.lower().replace(" ", "_")
        
        if settings_key in self.validation_config:
            sheet_settings = self.validation_config[settings_key]
            
            # Add value ranges check
            if "value_ranges" in sheet_settings:
                # Convert keys to lowercase
                value_ranges = {}
                for col, range_values in sheet_settings["value_ranges"].items():
                    value_ranges[col.lower()] = range_values
                
                validation_config["value_ranges"] = value_ranges
            
            # Add required columns check
            if "required_columns" in sheet_settings:
                validation_config["required_columns"] = sheet_settings["required_columns"]
        
        self.logger.info(f"Using validation configuration: {validation_config}")
        
        # Validate the DataFrame
        results = validate_dataframe(df, validation_config)
        
        # Store the validation results
        self.validation_results = results
        
        return results
    
    def export(self, df):
        """
        Export the data to a SQLite database.
        
        Args:
            df: The DataFrame to export.
            
        Returns:
            The path to the SQLite database file.
        """
        self.logger.info(f"Exporting data to SQLite database")
        
        # Create the output directory if it doesn't exist
        output_dir = resolve_path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the database path
        db_path = os.path.join(output_dir, self.db_name)
        
        # Create the connection string
        connection_string = f"sqlite:///{db_path}"
        
        # Export the data to the database
        table_name = self.sheet_name.lower().replace(" ", "_")
        save_dataframe_to_database(
            df=df,
            table_name=table_name,
            connection_string=connection_string,
            if_exists="replace"
        )
        
        self.logger.info(f"Data exported to {db_path}, table: {table_name}")
        
        # Get the schema of the table
        try:
            schema = get_table_schema(connection_string, table_name)
            self.logger.info(f"Database schema:\n{schema}")
            
            # Save the schema to a file
            schema_path = os.path.join(output_dir, f"{table_name}_schema.sql")
            with open(schema_path, "w") as f:
                f.write(schema)
            
            self.logger.info(f"Schema saved to {schema_path}")
        except Exception as e:
            self.logger.error(f"Error getting schema: {str(e)}")
        
        return db_path
    
    def save_reports(self, df):
        """
        Save analysis and validation reports.
        
        Args:
            df: The DataFrame that was analyzed and validated.
            
        Returns:
            A dictionary containing the paths to the report files.
        """
        self.logger.info(f"Saving analysis and validation reports")
        
        # Create the output directory if it doesn't exist
        output_dir = resolve_path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize a dictionary to store the report paths
        report_paths = {}
        
        # Save analysis report
        if self.analysis_results:
            analysis_report_path = os.path.join(output_dir, f"{self.sheet_name.lower().replace(' ', '_')}_analysis_report.txt")
            
            with open(analysis_report_path, "w") as f:
                f.write(f"Analysis Report for Sheet: {self.sheet_name}\n")
                f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                
                # Write unique values report
                if 'unique_values' in self.analysis_results:
                    f.write("Unique Values Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    unique_values = self.analysis_results['unique_values']
                    for col, res in unique_values.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Unique value count: {res['count']}\n")
                        f.write(f"    Null count: {res['null_count']} ({res['null_percentage'] * 100:.2f}%)\n")
                        
                        if 'values' in res:
                            f.write(f"    Unique values: {', '.join(res['values'][:10])}")
                            if len(res['values']) > 10:
                                f.write(f" ... and {len(res['values']) - 10} more")
                            f.write("\n")
                        
                        if 'value_counts' in res:
                            f.write(f"    Value counts (top 5):\n")
                            sorted_counts = sorted(res['value_counts'].items(), key=lambda x: x[1], reverse=True)
                            for val, count in sorted_counts[:5]:
                                f.write(f"      {val}: {count}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write column statistics report
                if 'column_statistics' in self.analysis_results:
                    f.write("Column Statistics Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    column_stats = self.analysis_results['column_statistics']
                    for col, stats in column_stats.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Min: {stats['min']}\n")
                        f.write(f"    Max: {stats['max']}\n")
                        f.write(f"    Mean: {stats['mean']}\n")
                        f.write(f"    Median: {stats['median']}\n")
                        f.write(f"    Standard deviation: {stats['std']}\n")
                        f.write(f"    Q1 (25th percentile): {stats['q1']}\n")
                        f.write(f"    Q3 (75th percentile): {stats['q3']}\n")
                        f.write(f"    IQR: {stats['iqr']}\n")
                        f.write(f"    Outliers count: {stats['outliers_count']}\n")
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write text analysis report
                if 'text_analysis' in self.analysis_results:
                    f.write("Text Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    text_analysis = self.analysis_results['text_analysis']
                    for col, analysis in text_analysis.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Min length: {analysis['min_length']}\n")
                        f.write(f"    Max length: {analysis['max_length']}\n")
                        f.write(f"    Average length: {analysis['avg_length']:.2f}\n")
                        f.write(f"    Empty strings: {analysis['empty_count']}\n")
                        
                        if 'pattern_analysis' in analysis:
                            f.write(f"    Pattern analysis:\n")
                            for pattern, count in analysis['pattern_analysis'].items():
                                if count > 0:
                                    f.write(f"      {pattern}: {count}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Analysis report saved to {analysis_report_path}")
            report_paths['analysis_report'] = analysis_report_path
        
        # Save validation report
        if self.validation_results:
            validation_report_path = os.path.join(output_dir, f"{self.sheet_name.lower().replace(' ', '_')}_validation_report.txt")
            
            with open(validation_report_path, "w") as f:
                f.write(f"Validation Report for Sheet: {self.sheet_name}\n")
                f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                
                # Write missing values report
                if "missing_values" in self.validation_results:
                    f.write("Missing Values Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    missing_values = self.validation_results["missing_values"]
                    for col, pct in missing_values.items():
                        f.write(f"  {col}: {pct * 100:.2f}% missing\n")
                    
                    f.write("\n")
                
                # Write duplicate rows report
                if "duplicate_rows" in self.validation_results:
                    f.write("Duplicate Rows Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    duplicate_rows = self.validation_results["duplicate_rows"]
                    f.write(f"  Duplicate rows: {duplicate_rows['duplicate_count']}\n")
                    
                    if duplicate_rows["duplicate_count"] > 0:
                        f.write(f"  Duplicate indices: {duplicate_rows['duplicate_indices'][:10]}")
                        if len(duplicate_rows["duplicate_indices"]) > 10:
                            f.write(f" ... and {len(duplicate_rows['duplicate_indices']) - 10} more")
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write value ranges report
                if "value_ranges" in self.validation_results:
                    f.write("Value Ranges Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    value_ranges = self.validation_results["value_ranges"]
                    for col, res in value_ranges.items():
                        f.write(f"  {col}:\n")
                        f.write(f"    Below minimum: {res['below_min']}\n")
                        f.write(f"    Above maximum: {res['above_max']}\n")
                        f.write(f"    Total outside range: {res['total_outside_range']}\n")
                    
                    f.write("\n")
                
                # Write data types report
                if "data_types" in self.validation_results:
                    f.write("Data Types Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    data_types = self.validation_results["data_types"]
                    for col, res in data_types.items():
                        f.write(f"  {col}:\n")
                        f.write(f"    Expected type: {res['expected_type']}\n")
                        f.write(f"    Current type: {res['current_type']}\n")
                        f.write(f"    Error count: {res['error_count']}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Validation report saved to {validation_report_path}")
            report_paths['validation_report'] = validation_report_path
        
        return report_paths
    
    def run(self):
        """
        Run the pipeline.
        
        Returns:
            A dictionary containing the results of the pipeline.
        """
        try:
            self.logger.info("Starting Medtronics Asset Data Pipeline")
            
            # Extract data
            try:
                df = self.extract()
                
                if df is None or len(df) == 0:
                    self.logger.error(f"No data extracted from sheet '{self.sheet_name}'")
                    return {
                        "status": "error",
                        "message": f"No data extracted from sheet '{self.sheet_name}'"
                    }
            except Exception as e:
                self.logger.error(f"Error extracting data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error extracting data: {str(e)}"
                }
            
            # Analyze data
            try:
                analysis_results = self.analyze(df)
            except Exception as e:
                self.logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error analyzing data: {str(e)}"
                }
            
            # Validate data
            try:
                validation_results = self.validate(df)
            except Exception as e:
                self.logger.error(f"Error validating data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error validating data: {str(e)}"
                }
            
            # Export data
            try:
                db_path = self.export(df)
            except Exception as e:
                self.logger.error(f"Error exporting data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error exporting data: {str(e)}"
                }
            
            # Save reports
            try:
                report_paths = self.save_reports(df)
            except Exception as e:
                self.logger.error(f"Error saving reports: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error saving reports: {str(e)}"
                }
            
            self.logger.info("Medtronics Asset Data Pipeline completed successfully")
            
            # Return results
            return {
                "status": "success",
                "message": "Pipeline completed successfully",
                "data": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "db_path": db_path,
                    "sheet_name": self.sheet_name.lower().replace(" ", "_"),
                    "report_paths": report_paths
                }
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error in pipeline: {str(e)}"
            }


def main():
    """Main function."""
    try:
        # Create and run the pipeline
        pipeline = MedtronicsPipeline()
        results = pipeline.run()
        
        # Print the results
        print("\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Message: {results['message']}")
        
        if results['status'] == 'success':
            print(f"\nData:")
            print(f"  Rows: {results['data']['rows']}")
            print(f"  Columns: {results['data']['columns']}")
            print(f"  Database: {results['data']['db_path']}")
            
            # Check if schema file exists
            schema_path = os.path.join(os.path.dirname(results['data']['db_path']), f"{results['data']['sheet_name']}_schema.sql")
            if os.path.exists(schema_path):
                print(f"  Schema: {schema_path}")
                
                # Print the schema
                print(f"\nDatabase Schema:")
                with open(schema_path, "r") as f:
                    schema = f.read()
                print(schema)
            
            print(f"\nReports:")
            for report_type, report_path in results['data']['report_paths'].items():
                print(f"  {report_type}: {report_path}")
            
            return 0
        else:
            # Return non-zero exit code for errors
            return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())