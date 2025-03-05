"""
Example demonstrating how to use the base pipeline class with pipeline utilities.

This example shows how to:
1. Create a new pipeline by inheriting from BasePipeline
2. Use pipeline utilities to manage output directories
3. Run the pipeline with different options
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.pipelines import BasePipeline
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_text_columns,
    analyze_unique_values,
    validate_dataframe,
)
from fca_dashboard.utils.database import (
    get_table_schema,
    save_dataframe_to_database,
)
from fca_dashboard.utils.logging_config import get_logger


class SamplePipeline(BasePipeline):
    """
    Sample pipeline implementation using BasePipeline.
    
    This pipeline demonstrates how to create a new pipeline by inheriting from BasePipeline.
    It processes a sample CSV file and exports it to a SQLite database.
    """
    
    def __init__(self):
        """Initialize the sample pipeline."""
        # Call the parent class constructor with the pipeline name
        super().__init__("sample")
        
        # Set up pipeline-specific configuration
        self.table_name = "sample_data"
    
    def extract(self):
        """
        Extract data from a sample CSV file.
        
        Returns:
            The extracted DataFrame.
        """
        self.logger.info("Extracting sample data")
        
        # Create a sample DataFrame
        df = pd.DataFrame({
            "id": range(1, 11),
            "name": [f"Sample {i}" for i in range(1, 11)],
            "value": [i * 10.5 for i in range(1, 11)],
            "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"]
        })
        
        self.logger.info(f"Extracted {len(df)} rows with {len(df.columns)} columns")
        return df
    
    def analyze(self, df):
        """
        Analyze the extracted data.
        
        Args:
            df: The DataFrame to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info("Analyzing sample data")
        
        # Initialize results dictionary
        results = {}
        
        # Analyze unique values
        unique_columns = ["name", "category"]
        unique_values_results = analyze_unique_values(
            df, 
            columns=unique_columns,
            max_unique_values=20
        )
        results['unique_values'] = unique_values_results
        
        # Analyze column statistics for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns)
            results['column_statistics'] = column_stats_results
        
        # Analyze text columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
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
        self.logger.info("Validating sample data")
        
        # Create validation configuration
        validation_config = {
            "missing_values": {
                "columns": list(df.columns),
                "threshold": 0.5
            },
            "duplicate_rows": {
                "subset": ["id"]
            },
            "data_types": {
                "id": "int",
                "name": "str",
                "value": "float",
                "category": "str"
            }
        }
        
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
        self.logger.info("Exporting sample data to SQLite database")
        
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
    """Run the sample pipeline example."""
    logger = get_logger("pipeline_refactor_example")
    
    try:
        # Create the pipeline
        pipeline = SamplePipeline()
        
        # Run the pipeline with different options
        logger.info("Running pipeline with default options")
        results = pipeline.run()
        
        if results["status"] == "success":
            logger.info("Pipeline completed successfully")
            logger.info(f"Processed {results['data']['rows']} rows")
            logger.info(f"Database: {results['data']['db_path']}")
            
            # Run again with clear_output=True
            logger.info("\nRunning pipeline with clear_output=True")
            results = pipeline.run(clear_output=True)
            
            if results["status"] == "success":
                logger.info("Pipeline with clear_output completed successfully")
                
                # Run again with clear_output=True but preserve_db=False
                logger.info("\nRunning pipeline with clear_output=True, preserve_db=False")
                results = pipeline.run(clear_output=True, preserve_db=False)
                
                if results["status"] == "success":
                    logger.info("Pipeline with clear_output and no DB preservation completed successfully")
                else:
                    logger.error(f"Pipeline failed: {results['message']}")
            else:
                logger.error(f"Pipeline failed: {results['message']}")
        else:
            logger.error(f"Pipeline failed: {results['message']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running sample pipeline: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())