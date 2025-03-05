"""
Base Pipeline Module.

This module provides a base class for all data pipelines in the FCA Dashboard application.
It implements common functionality such as output directory management, logging, and
standard pipeline steps.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path
from fca_dashboard.utils.pipeline_util import clear_output_directory, get_pipeline_output_dir
from fca_dashboard.utils.verification_util import verify_pipeline_output


class BasePipeline(ABC):
    """
    Base class for all data pipelines.
    
    This abstract class provides common functionality for data pipelines, including:
    - Configuration loading from settings
    - Output directory management
    - Standard pipeline steps (extract, analyze, validate, export)
    - Logging
    
    Subclasses must implement the abstract methods to provide pipeline-specific functionality.
    """
    
    def __init__(self, pipeline_name: str):
        """
        Initialize the base pipeline.
        
        Args:
            pipeline_name: Name of the pipeline (e.g., 'medtronics', 'wichita')
        """
        self.pipeline_name = pipeline_name
        self.logger = get_logger(f"{pipeline_name}_pipeline")
        
        # Get file paths from settings
        self.input_file = settings.get(f"{pipeline_name}.input_file", "")
        self.output_dir = settings.get(f"{pipeline_name}.output_dir", f"outputs/pipeline/{pipeline_name}")
        self.db_name = settings.get(f"{pipeline_name}.db_name", f"{pipeline_name}_assets.db")
        
        # Get configuration from settings
        self.extraction_config = settings.get("excel_utils.extraction", {})
        self.validation_config = settings.get("excel_utils.validation", {})
        self.analysis_config = settings.get("excel_utils.analysis", {})
        
        # Get columns to extract and drop NaN values from settings
        self.columns_to_extract = settings.get(f"{pipeline_name}.columns_to_extract", [])
        self.drop_na_columns = settings.get(f"{pipeline_name}.drop_na_columns", [])
        
        # Get verification configuration from settings
        self.verification_config = settings.get(f"{pipeline_name}.verification", {})
        
        # Initialize data storage
        self.extracted_data = None
        self.analysis_results = None
        self.validation_results = None
        self.verification_results = None
    
    def clear_output_directory(self, preserve_db: bool = True, preserve_files: Optional[List[str]] = None) -> List[str]:
        """
        Clear the pipeline's output directory.
        
        Args:
            preserve_db: Whether to preserve database files (.db extension)
            preserve_files: List of specific filenames to preserve
            
        Returns:
            List of files that were deleted
        """
        self.logger.info(f"Clearing output directory for {self.pipeline_name} pipeline")
        
        # Determine which extensions to preserve
        preserve_extensions = [".db"] if preserve_db else []
        
        # Clear the output directory
        return clear_output_directory(
            self.pipeline_name,
            preserve_files=preserve_files,
            preserve_extensions=preserve_extensions
        )
    
    def prepare_output_directory(self) -> Path:
        """
        Prepare the output directory for the pipeline.
        
        This method ensures the output directory exists and returns its path.
        
        Returns:
            Path object representing the output directory
        """
        # Get the output directory
        output_dir = get_pipeline_output_dir(self.pipeline_name)
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Extract data from the source.
        
        This method must be implemented by subclasses to extract data from the source.
        
        Returns:
            The extracted DataFrame.
        """
        pass
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the extracted data.
        
        This method must be implemented by subclasses to analyze the extracted data.
        
        Args:
            df: The DataFrame to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the extracted data.
        
        This method must be implemented by subclasses to validate the extracted data.
        
        Args:
            df: The DataFrame to validate.
            
        Returns:
            A dictionary containing the validation results.
        """
        pass
    
    @abstractmethod
    def export(self, df: pd.DataFrame) -> str:
        """
        Export the data to a destination.
        
        This method must be implemented by subclasses to export the data.
        
        Args:
            df: The DataFrame to export.
            
        Returns:
            The path to the exported data.
        """
        pass
    
    def verify(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify the pipeline output.
        
        This method verifies the pipeline output against the verification configuration.
        
        Args:
            table_name: The name of the table to verify. If None, uses the pipeline name.
            
        Returns:
            A dictionary containing the verification results.
        """
        self.logger.info(f"Verifying pipeline output")
        
        # If no verification configuration is provided, use default configuration
        if not self.verification_config:
            # Default verification configuration
            self.verification_config = {
                "no_placeholder_values": {
                    "columns": self.drop_na_columns,
                    "placeholder_values": ['no id', 'none', 'n/a', 'unknown', '']
                },
                "no_null_values": {
                    "columns": self.drop_na_columns
                }
            }
            self.logger.info(f"Using default verification configuration: {self.verification_config}")
        
        # If table_name is not provided, use a default table name
        if table_name is None:
            # For Medtronics pipeline, the table name is "asset_data"
            if self.pipeline_name == "medtronics":
                table_name = "asset_data"
            # For Wichita pipeline, the table name is "wichita_assets"
            elif self.pipeline_name == "wichita":
                table_name = "wichita_assets"
            else:
                # Default to pipeline_name_assets
                table_name = f"{self.pipeline_name}_assets"
        
        # Verify the pipeline output
        self.verification_results = verify_pipeline_output(
            self.pipeline_name,
            self.db_name,
            table_name,
            self.verification_config,
            self.output_dir,
            self.analysis_results,
            self.validation_results
        )
        
        return self.verification_results
    
    def save_reports(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Save analysis and validation reports.
        
        This method saves the analysis and validation results to report files.
        Subclasses can override this method to provide custom report formats.
        
        Args:
            df: The DataFrame that was analyzed and validated.
            
        Returns:
            A dictionary containing the paths to the report files.
        """
        self.logger.info(f"Saving analysis and validation reports")
        
        # Prepare the output directory
        output_dir = self.prepare_output_directory()
        
        # Initialize a dictionary to store the report paths
        report_paths = {}
        
        # Save analysis report if available
        if self.analysis_results:
            analysis_report_path = os.path.join(output_dir, f"{self.pipeline_name}_analysis_report.txt")
            
            with open(analysis_report_path, "w") as f:
                f.write(f"Analysis Report for {self.pipeline_name.capitalize()} Pipeline\n")
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
                            f.write(f"    Unique values: {', '.join(str(v) for v in res['values'][:10])}")
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
        
        # Save validation report if available
        if self.validation_results:
            validation_report_path = os.path.join(output_dir, f"{self.pipeline_name}_validation_report.txt")
            
            with open(validation_report_path, "w") as f:
                f.write(f"Validation Report for {self.pipeline_name.capitalize()} Pipeline\n")
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
    
    def run(self, clear_output: bool = False, preserve_db: bool = True) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        This method executes the pipeline steps in sequence and handles errors.
        
        Args:
            clear_output: Whether to clear the output directory before running
            preserve_db: Whether to preserve database files when clearing output
            
        Returns:
            A dictionary containing the results of the pipeline.
        """
        try:
            self.logger.info(f"Starting {self.pipeline_name.capitalize()} Pipeline")
            
            # Clear output directory if requested
            if clear_output:
                self.clear_output_directory(preserve_db=preserve_db)
            
            # Extract data
            try:
                df = self.extract()
                
                if df is None or len(df) == 0:
                    self.logger.error(f"No data extracted")
                    return {
                        "status": "error",
                        "message": f"No data extracted"
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
            
            # Verify the pipeline output
            try:
                # For Medtronics pipeline, the table name is "asset_data"
                if self.pipeline_name == "medtronics":
                    table_name = "asset_data"
                # For Wichita pipeline, the table name is "wichita_assets"
                elif self.pipeline_name == "wichita":
                    table_name = "wichita_assets"
                else:
                    # Extract table name from the database path
                    table_name = os.path.splitext(os.path.basename(db_path))[0]
                    if table_name == self.db_name.replace('.db', ''):
                        # If the table name is the same as the database name, use a default table name
                        table_name = f"{self.pipeline_name}_assets"
                
                verification_results = self.verify(table_name)
                self.logger.info(f"Verification completed: {len(verification_results.get('verifications', {}))} checks performed")
            except Exception as e:
                self.logger.error(f"Error verifying pipeline output: {str(e)}", exc_info=True)
                # Continue with the pipeline even if verification fails
                verification_results = {
                    "status": "error",
                    "message": f"Error verifying pipeline output: {str(e)}"
                }
            
            self.logger.info(f"{self.pipeline_name.capitalize()} Pipeline completed successfully")
            
            # Prepare result data
            result_data = {
                "rows": len(df),
                "columns": len(df.columns),
                "db_path": db_path,
                "report_paths": report_paths
            }
            
            # Add verification results if available
            if verification_results:
                result_data["verification"] = verification_results
            
            return {
                "status": "success",
                "message": "Pipeline completed successfully",
                "data": result_data
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error in pipeline: {str(e)}"
            }