"""
Excel extractor module for the FCA Dashboard application.

This module provides functionality for extracting data from Excel files
and loading it into pandas DataFrames, with features like error handling,
logging, and optional file upload.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.base_extractor import DataExtractor, ExtractionError, extractor_factory

# Alias for backward compatibility
ExcelExtractionError = ExtractionError
from fca_dashboard.utils.excel import get_excel_file_type, is_excel_file
from fca_dashboard.utils.path_util import get_root_dir, resolve_path
from fca_dashboard.utils.upload_util import upload_file


class ExcelExtractor(DataExtractor):
    """
    Extractor for Excel files (XLSX, XLS, XLSM, XLSB) and CSV files.
    
    This extractor can handle various Excel formats and CSV files,
    with options for sheet selection, column filtering, and file uploading.
    """
    
    def __init__(self, upload_service=None, logger=None):
        """
        Initialize the Excel extractor.
        
        Args:
            upload_service: Optional service for uploading files. If None, uses the default upload_file function.
            logger: Optional logger instance. If None, a default logger will be created.
        """
        super().__init__(logger)
        self.upload_service = upload_service or upload_file
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this extractor can handle the file, False otherwise.
        """
        return is_excel_file(file_path)
    
    def extract(
        self,
        file_path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        columns: Optional[List[str]] = None,
        upload: bool = False,
        target_filename: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from an Excel file into a pandas DataFrame.
        
        Args:
            file_path: Path to the Excel file. Can be absolute or relative.
            sheet_name: Name or index of the sheet to extract (default: 0, first sheet).
            columns: Optional list of column names to extract. If None, extracts all columns.
            upload: Whether to upload the file to the uploads directory.
            target_filename: Optional filename to use when uploading. If None, uses the source filename.
            **kwargs: Additional extraction options passed to pandas.read_excel or pandas.read_csv.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            FileNotFoundError: If the source file does not exist.
            ExtractionError: If an error occurs during the extraction process.
        """
        # Resolve the file path
        source_path = resolve_path(file_path)
        
        # Validate source file
        if not source_path.is_file():
            self.logger.error(f"Source file not found: {source_path}")
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Log the extraction operation
        self.logger.info(f"Extracting data from Excel file: {source_path}")
        
        try:
            # Determine the file type
            file_type = get_excel_file_type(source_path)
            
            # Read the file into a DataFrame
            if file_type == "csv":
                df = pd.read_csv(source_path, **kwargs)
            else:
                df = pd.read_excel(source_path, sheet_name=sheet_name, **kwargs)
            
            # Filter columns if specified
            if columns:
                # Validate that all requested columns exist
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    error_msg = f"Columns not found in Excel file: {missing_columns}"
                    self.logger.error(error_msg)
                    raise ExtractionError(error_msg)
                
                # Select only the specified columns
                df = df[columns]
            
            # Upload the file if requested
            if upload:
                self._upload_file(source_path, target_filename)
            
            # Log success and return the DataFrame
            self.logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
            return df
        
        except (EmptyDataError, ParserError) as e:
            # Handle pandas-specific errors
            error_msg = f"Error parsing Excel file {source_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e
        
        except ExtractionError:
            # Re-raise ExtractionError
            raise
        
        except Exception as e:
            # Handle any other errors
            error_msg = f"Error extracting data from Excel file {source_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e
    
    def _upload_file(self, source_path: Path, target_filename: Optional[str] = None) -> None:
        """
        Upload a file to the uploads directory.
        
        Args:
            source_path: Path to the file to upload.
            target_filename: Optional filename to use when uploading.
                If None, uses the source filename.
        
        Raises:
            ExtractionError: If an error occurs during the upload process.
        """
        try:
            # Get the uploads directory from settings
            uploads_dir = get_root_dir() / settings.get("file_paths.uploads_dir", "uploads")
            
            # Ensure the uploads directory exists
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Upload the file
            self.logger.info(f"Uploading Excel file to: {uploads_dir}")
            self.upload_service(source_path, uploads_dir, target_filename)
        except Exception as e:
            error_msg = f"Error uploading file {source_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e


# For backward compatibility
def extract_excel_to_dataframe(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    columns: Optional[List[str]] = None,
    upload: bool = False,
    target_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract data from an Excel file into a pandas DataFrame.
    
    This function is maintained for backward compatibility.
    New code should use the ExcelExtractor class directly.
    
    Args:
        file_path: Path to the Excel file. Can be absolute or relative.
        sheet_name: Name or index of the sheet to extract (default: 0, first sheet).
        columns: Optional list of column names to extract. If None, extracts all columns.
        upload: Whether to upload the file to the uploads directory.
        target_filename: Optional filename to use when uploading. If None, uses the source filename.
        
    Returns:
        pandas DataFrame containing the extracted data.
        
    Raises:
        FileNotFoundError: If the source file does not exist.
        ExtractionError: If an error occurs during the extraction process.
    """
    extractor = ExcelExtractor()
    return extractor.extract(
        file_path=file_path,
        sheet_name=sheet_name,
        columns=columns,
        upload=upload,
        target_filename=target_filename
    )


# Register the Excel extractor with the factory
excel_extractor = ExcelExtractor()
extractor_factory.register_extractor(excel_extractor)