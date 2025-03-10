"""
Example script demonstrating the use of the Excel extractor.

This script shows how to use the Excel extractor to load data from an Excel file
into a pandas DataFrame and perform basic operations on it.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.excel_extractor import extract_excel_to_dataframe
from fca_dashboard.utils.path_util import get_root_dir


def main():
    """
    Demonstrate the Excel extractor functionality.
    
    This function creates a sample Excel file, extracts data from it,
    and shows how to work with the resulting DataFrame.
    """
    try:
        # Import pandas here to avoid import errors in the module
        import pandas as pd
        
        # Get settings for examples
        examples_dir_name = settings.get("file_paths.examples_dir", "examples")
        examples_dir = get_root_dir() / examples_dir_name
        os.makedirs(examples_dir, exist_ok=True)
        
        # Get sample filename from settings
        sample_filename = settings.get("examples.excel.sample_filename", "sample_data.xlsx")
        
        # Create a sample Excel file
        sample_data = {
            "ID": [1, 2, 3, 4, 5],
            "Product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
            "Price": [10.99, 20.50, 15.75, 8.25, 30.00],
            "InStock": [True, False, True, True, False],
            "Category": ["Electronics", "Tools", "Electronics", "Office", "Tools"]
        }
        
        # Create a DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save the DataFrame to an Excel file
        sample_file = examples_dir / sample_filename
        df.to_excel(sample_file, index=False)
        print(f"Created sample Excel file: {sample_file}")
        
        # Extract data from the Excel file
        print("\nExtracting data from Excel file...")
        # Get uploads directory from settings for informational purposes
        uploads_dir_name = settings.get("file_paths.uploads_dir", "uploads")
        uploads_dir = get_root_dir() / uploads_dir_name
        print(f"File will be uploaded to: {uploads_dir}")
        extracted_df = extract_excel_to_dataframe(sample_file, upload=True)
        
        # Display the extracted data
        print("\nExtracted DataFrame:")
        print(extracted_df)
        
        # Demonstrate filtering columns
        print("\nExtracting only specific columns...")
        # Get columns to extract from settings
        columns_to_extract = settings.get("examples.excel.columns_to_extract", ["ID", "Product", "Price"])
        filtered_df = extract_excel_to_dataframe(sample_file, columns=columns_to_extract)
        print("\nFiltered DataFrame:")
        print(filtered_df)
        
        # Demonstrate basic DataFrame operations
        print("\nPerforming basic DataFrame operations:")
        
        # Filter rows based on a condition
        # Get price threshold from settings
        price_threshold = settings.get("examples.excel.price_threshold", 15)
        print(f"\n1. Filter products with price > {price_threshold}:")
        expensive_products = extracted_df[extracted_df["Price"] > price_threshold]
        print(expensive_products)
        
        # Group by a column and calculate statistics
        print("\n2. Group by Category and calculate average price:")
        category_stats = extracted_df.groupby("Category")["Price"].mean()
        print(category_stats)
        
        # Sort the DataFrame
        print("\n3. Sort by Price (descending):")
        sorted_df = extracted_df.sort_values("Price", ascending=False)
        print(sorted_df)
        
        print("\nExample completed successfully!")
        
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Please ensure pandas and openpyxl are installed.")
        return 1
    except Exception as e:
        print(f"Error in example: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())