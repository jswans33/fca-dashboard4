"""
Medtronics Asset Data Pipeline.

This pipeline extracts data from the Medtronics Asset Log Uploader Excel file,
analyzes and validates it, and then exports it to a SQLite database.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.pipelines import BasePipeline
from fca_dashboard.utils.database import (
    get_table_schema,
    save_dataframe_to_database,
)
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    extract_excel_with_config,
    validate_dataframe,
)
from fca_dashboard.utils.path_util import resolve_path


class MedtronicsPipeline(BasePipeline):
    """
    Pipeline for processing Medtronics Asset Data.
    
    This pipeline extracts data from the Medtronics Asset Log Uploader Excel file,
    analyzes and validates it, and then exports it to a SQLite database.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        # Call the parent class constructor with the pipeline name
        super().__init__("medtronics")
        
        # Get sheet name from settings
        self.sheet_name = settings.get("medtronics.sheet_name", "Asset Data")
        
        # Get staging configuration from settings
        self.staging_config = settings.get("medtronics.staging", {})
        
        # Set default values if not provided in settings
        if "enabled" not in self.staging_config:
            self.staging_config["enabled"] = True
        if "db_path" not in self.staging_config:
            self.staging_config["db_path"] = os.path.join(self.output_dir, "staging.db")
        if "source_system" not in self.staging_config:
            self.staging_config["source_system"] = "Medtronics"
        if "batch_id_prefix" not in self.staging_config:
            self.staging_config["batch_id_prefix"] = "MEDTRONICS-BATCH-"
        
        # Set up verification configuration
        self.verification_config = {
            "no_placeholder_values": {
                "columns": ["usassetid", "asset tag"],
                "placeholder_values": ["no id", "none", "n/a", "unknown", ""]
            },
            "no_null_values": {
                "columns": ["usassetid", "asset tag", "asset name"]
            }
        }
        
        # Create a SQLiteStagingManager instance
        self.staging_manager = SQLiteStagingManager()
        
        # Initialize staging results
        self.staging_results = None
    
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
        
        # Drop rows with NaN values or placeholder values in specified columns
        if self.drop_na_columns and df is not None:
            original_row_count = len(df)
            
            # Convert drop_na_columns to lowercase for consistency
            drop_na_columns_lower = [col.lower() for col in self.drop_na_columns]
            
            # Check if the specified columns exist in the DataFrame
            existing_cols = [col for col in drop_na_columns_lower if col in df.columns]
            
            if existing_cols:
                # First drop rows with NaN values
                self.logger.info(f"Dropping rows with NaN values in columns: {existing_cols}")
                df = df.dropna(subset=existing_cols)
                dropped_count = original_row_count - len(df)
                self.logger.info(f"Dropped {dropped_count} rows with NaN values")
                
                # Now filter out rows with placeholder values like "NO ID"
                current_count = len(df)
                placeholder_values = ['no id', 'none', 'n/a', 'unknown', '']
                
                # Create a mask for each column to identify rows with placeholder values
                for col in existing_cols:
                    if col in df.columns:
                        # Convert values to lowercase for case-insensitive comparison
                        mask = df[col].astype(str).str.lower().isin(placeholder_values)
                        if mask.any():
                            self.logger.info(f"Dropping {mask.sum()} rows with placeholder values in column '{col}'")
                            df = df[~mask]
                
                total_dropped = original_row_count - len(df)
                self.logger.info(f"Total rows dropped: {total_dropped} ({dropped_count} NaN, {total_dropped - dropped_count} placeholder values)")
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
    
    def stage_data(self, df):
        """
        Stage the data in the SQLite staging database.
        
        Args:
            df: The DataFrame to stage.
            
        Returns:
            A dictionary containing the staging results.
        """
        self.logger.info(f"Staging data in SQLite staging database")
        
        if not self.staging_config.get("enabled", True):
            self.logger.info("Staging is disabled in configuration, skipping")
            return {
                "status": "skipped",
                "message": "Staging is disabled in configuration"
            }
        
        try:
            # Create the output directory if it doesn't exist
            output_dir = resolve_path(self.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the staging database path
            db_path = self.staging_config.get("db_path", os.path.join(output_dir, "staging.db"))
            
            # Initialize the staging database if it doesn't exist
            if not os.path.exists(db_path):
                self.logger.info(f"Initializing staging database at {db_path}")
                self.staging_manager.initialize_db(db_path)
            
            # Create the connection string
            connection_string = f"sqlite:///{db_path}"
            
            # Generate a batch ID
            batch_id_prefix = self.staging_config.get("batch_id_prefix", "MEDTRONICS-BATCH-")
            batch_id = f"{batch_id_prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Get the source system name
            source_system = self.staging_config.get("source_system", "Medtronics")
            
            # Prepare the data for staging
            staging_df = df.copy()
            
            # Add any additional metadata columns needed for staging
            # For example, you might want to add equipment-specific columns
            if 'equipment_tag' not in staging_df.columns and 'asset_id' in staging_df.columns:
                staging_df['equipment_tag'] = staging_df['asset_id']
            
            # Convert any JSON-like columns to actual JSON
            for col in staging_df.columns:
                if col.endswith('_data') or col in ['attributes', 'metadata']:
                    if staging_df[col].notna().any():
                        # If the column contains dictionaries or lists, convert to JSON
                        if isinstance(staging_df[col].iloc[0], (dict, list)):
                            self.logger.info(f"Converting column {col} to JSON format")
                            staging_df[col] = staging_df[col].apply(lambda x: x if pd.isna(x) else x)
            
            # Save the data to the staging table
            self.logger.info(f"Saving {len(staging_df)} rows to staging table")
            self.staging_manager.save_dataframe_to_staging(
                df=staging_df,
                connection_string=connection_string,
                source_system=source_system,
                import_batch_id=batch_id
            )
            
            # Get the count of pending items
            pending_items = self.staging_manager.get_pending_items(connection_string)
            pending_count = len(pending_items)
            
            self.logger.info(f"Successfully staged {len(staging_df)} rows, {pending_count} pending items")
            
            # Store the staging results
            self.staging_results = {
                "db_path": db_path,
                "connection_string": connection_string,
                "batch_id": batch_id,
                "source_system": source_system,
                "rows_staged": len(staging_df),
                "pending_items": pending_count
            }
            
            return {
                "status": "success",
                "message": f"Successfully staged {len(staging_df)} rows",
                "db_path": db_path,
                "connection_string": connection_string,
                "batch_id": batch_id,
                "pending_items": pending_count
            }
        except Exception as e:
            error_msg = f"Error staging data: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "message": error_msg
            }
    
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
    
    def run(self, clear_output=False, preserve_db=True):
        """
        Run the pipeline.
        
        Args:
            clear_output: Whether to clear the output directory before running
            preserve_db: Whether to preserve database files when clearing output
            
        Returns:
            A dictionary containing the results of the pipeline.
        """
        # Call the base class run method
        base_result = super().run(clear_output, preserve_db)
        
        # If the pipeline was successful, add staging information to the result
        if base_result["status"] == "success" and self.staging_results:
            base_result["data"]["staging"] = {
                "db_path": self.staging_results["db_path"],
                "batch_id": self.staging_results["batch_id"],
                "source_system": self.staging_results["source_system"],
                "rows_staged": self.staging_results["rows_staged"],
                "pending_items": self.staging_results["pending_items"]
            }
        
        return base_result


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
            
            # Print staging information if available
            if 'staging' in results['data']:
                print(f"\nStaging Information:")
                staging = results['data']['staging']
                print(f"  Staging Database: {staging['db_path']}")
                print(f"  Batch ID: {staging['batch_id']}")
                print(f"  Source System: {staging['source_system']}")
                print(f"  Rows Staged: {staging['rows_staged']}")
                print(f"  Pending Items: {staging['pending_items']}")
            
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