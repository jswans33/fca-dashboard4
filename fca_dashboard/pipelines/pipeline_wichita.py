"""
Wichita Animal Shelter Asset Data Pipeline.

This pipeline extracts data from the Wichita Animal Shelter Asset List CSV file,
analyzes and validates it, and then exports it to a SQLite database.
"""

import os
import sys
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
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_text_columns,
    analyze_unique_values,
    validate_dataframe,
)
from fca_dashboard.utils.path_util import resolve_path


class WichitaPipeline(BasePipeline):
    """
    Pipeline for processing Wichita Animal Shelter Asset Data.
    
    This pipeline extracts data from the Wichita Animal Shelter Asset List CSV file,
    analyzes and validates it, and then exports it to a SQLite database.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        # Call the parent class constructor with the pipeline name
        super().__init__("wichita")
        
        # Set the table name for the database
        self.table_name = "wichita_assets"
    
    def extract(self):
        """
        Extract data from the Wichita Animal Shelter CSV file.
        
        Returns:
            The extracted DataFrame.
        """
        self.logger.info(f"Extracting data from {self.input_file}")
        
        # Resolve the file path
        file_path = resolve_path(self.input_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns")
            
            # Filter columns if columns_to_extract is specified
            if self.columns_to_extract:
                self.logger.info(f"Filtering columns to: {self.columns_to_extract}")
                existing_cols = [col for col in self.columns_to_extract if col in df.columns]
                if existing_cols:
                    df = df[existing_cols]
                    self.logger.info(f"Filtered to {len(existing_cols)} columns: {existing_cols}")
                else:
                    self.logger.warning(f"None of the specified columns exist in the DataFrame")
            
            # Drop rows with NaN values in specified columns
            if self.drop_na_columns:
                original_row_count = len(df)
                
                # Check if the specified columns exist in the DataFrame
                existing_cols = [col for col in self.drop_na_columns if col in df.columns]
                
                if existing_cols:
                    self.logger.info(f"Dropping rows with NaN values in columns: {existing_cols}")
                    df = df.dropna(subset=existing_cols)
                    self.logger.info(f"Dropped {original_row_count - len(df)} rows with NaN values")
                else:
                    self.logger.warning(f"None of the specified columns for dropping NaN values exist in the DataFrame")
                    self.logger.warning(f"Available columns: {list(df.columns)}")
            
            # Store the extracted data
            self.extracted_data = df
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error extracting data from CSV file: {str(e)}", exc_info=True)
            raise
    
    def analyze(self, df):
        """
        Analyze the extracted data.
        
        Args:
            df: The DataFrame to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info(f"Analyzing data from Wichita Animal Shelter asset list")
        
        # Get default analysis settings
        default_analysis = self.analysis_config.get("default", {})
        
        # Initialize results dictionary
        results = {}
        
        # 1. Analyze unique values
        self.logger.info("Analyzing unique values...")
        unique_values_settings = default_analysis.get("unique_values", {})
        max_unique_values = unique_values_settings.get("max_unique_values", 20)
        
        # Select important columns for unique value analysis
        unique_columns = [
            "Building Name", "Asset Category Name", "Type", 
            "Floor", "Room Number", "Manufacturer"
        ]
        # Filter to only columns that exist in the DataFrame
        unique_columns = [col for col in unique_columns if col in df.columns]
        
        unique_values_results = analyze_unique_values(
            df, 
            columns=unique_columns,
            max_unique_values=max_unique_values
        )
        results['unique_values'] = unique_values_results
        
        # 2. Analyze column statistics for numeric columns
        self.logger.info("Analyzing column statistics...")
        
        # Find numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns)
            results['column_statistics'] = column_stats_results
        
        # 3. Analyze text columns
        self.logger.info("Analyzing text columns...")
        
        # Find text columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        # Select a subset of important text columns
        important_text_columns = [
            "Asset Name", "Asset Category Name", "Type", 
            "Manufacturer", "Model", "Description"
        ]
        # Filter to only columns that exist in the DataFrame
        text_columns = [col for col in important_text_columns if col in text_columns]
        
        if text_columns:
            text_analysis_results = analyze_text_columns(df, columns=text_columns)
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
        self.logger.info(f"Validating data from Wichita Animal Shelter asset list")
        
        # Create validation configuration
        validation_config = {}
        
        # Add default settings
        if "default" in self.validation_config:
            default_settings = self.validation_config["default"]
            
            # Add missing values check
            if "missing_values" in default_settings:
                missing_values_settings = default_settings["missing_values"]
                validation_config["missing_values"] = {
                    "columns": missing_values_settings.get("columns") or [
                        "Asset Name", "Asset Category Name", "Type", "Manufacturer"
                    ],
                    "threshold": missing_values_settings.get("threshold", 0.5)
                }
            
            # Add duplicate rows check
            if "duplicate_rows" in default_settings:
                duplicate_rows_settings = default_settings["duplicate_rows"]
                validation_config["duplicate_rows"] = {
                    "subset": duplicate_rows_settings.get("subset") or ["Asset Name", "ID"]
                }
            
            # Add data types check
            if "data_types" in default_settings:
                data_types_settings = default_settings["data_types"]
                
                # Create type specifications based on settings
                type_specs = {}
                
                # Add date columns
                date_columns = ["Installation Date", "Warranty Expiration Date", "Estimated Replacement Date"]
                for col in date_columns:
                    if col in df.columns:
                        type_specs[col] = "date"
                
                # Add numeric columns
                numeric_columns = ["Cost", "Service Life", "Quantity", "Square Feet"]
                for col in numeric_columns:
                    if col in df.columns:
                        type_specs[col] = "float"
                
                # Add string columns
                string_columns = ["Asset Name", "Asset Category Name", "Type", "Manufacturer", "Model"]
                for col in string_columns:
                    if col in df.columns:
                        type_specs[col] = "str"
                
                if type_specs:
                    validation_config["data_types"] = type_specs
        
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
        
        # Prepare the output directory
        output_dir = self.prepare_output_directory()
        
        # Create the database path
        db_path = os.path.join(output_dir, self.db_name)
        
        # Create the connection string
        connection_string = f"sqlite:///{db_path}"
        
        # Export the data to the database
        save_dataframe_to_database(
            df=df,
            table_name=self.table_name,
            connection_string=connection_string,
            if_exists="replace"
        )
        
        self.logger.info(f"Data exported to {db_path}, table: {self.table_name}")
        
        # Get the schema of the table
        try:
            schema = get_table_schema(connection_string, self.table_name)
            self.logger.info(f"Database schema:\n{schema}")
            
            # Save the schema to a file
            schema_path = os.path.join(output_dir, f"{self.table_name}_schema.sql")
            with open(schema_path, "w") as f:
                f.write(schema)
            
            self.logger.info(f"Schema saved to {schema_path}")
        except Exception as e:
            self.logger.error(f"Error getting schema: {str(e)}")
        
        return db_path


def main():
    """Main function."""
    try:
        # Create and run the pipeline
        pipeline = WichitaPipeline()
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
            schema_path = os.path.join(
                os.path.dirname(results['data']['db_path']), 
                f"{pipeline.table_name}_schema.sql"
            )
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