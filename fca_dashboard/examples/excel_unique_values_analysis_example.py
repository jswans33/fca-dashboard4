"""
Example script for analyzing Excel data.

This script demonstrates how to use the Excel analysis utilities to analyze data
from Excel files, including analyzing unique values, column statistics, and text patterns.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    extract_excel_with_config,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def extract_and_analyze_data(file_path, output_dir):
    """
    Extract data from an Excel file and analyze it.
    
    Args:
        file_path: Path to the Excel file to analyze.
        output_dir: Directory to save the analysis reports.
        
    Returns:
        A dictionary containing the analysis results.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Extracting and analyzing data from Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get extraction configuration from settings
    extraction_config = settings.get("excel_utils.extraction", {})
    
    # Convert sheet names to match the format in the settings
    # The settings use lowercase with underscores, but the Excel file uses spaces and title case
    config = {}
    
    # Add default settings
    if "default" in extraction_config:
        config["default"] = extraction_config["default"]
    
    # Add sheet-specific settings
    for sheet_name in analysis['sheet_names']:
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = sheet_name.lower().replace(" ", "_")
        
        # If there are settings for this sheet, add them to the config
        if settings_key in extraction_config:
            config[sheet_name] = extraction_config[settings_key]
    
    print(f"Using extraction configuration from settings: {config}")
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a dictionary to store the analysis results
    analysis_results = {}
    
    # Analyze each sheet
    for sheet_name, df in extracted_data.items():
        print(f"\nAnalyzing sheet: {sheet_name}")
        
        # Skip empty sheets
        if len(df) == 0:
            print(f"  Skipping empty sheet: {sheet_name}")
            continue
        
        # Initialize sheet results
        sheet_results = {}
        
        # Get analysis configuration from settings
        analysis_config = settings.get("excel_utils.analysis", {})
        
        # Get default analysis settings
        default_analysis = analysis_config.get("default", {})
        
        # 1. Analyze unique values
        print(f"  Analyzing unique values...")
        # Get unique values settings
        unique_values_settings = default_analysis.get("unique_values", {})
        max_unique_values = unique_values_settings.get("max_unique_values", 20)
        
        # Select a subset of columns for unique value analysis
        # For demonstration purposes, we'll analyze the first 5 columns
        unique_columns = list(df.columns[:5])
        unique_values_results = analyze_unique_values(
            df,
            columns=unique_columns,
            max_unique_values=max_unique_values
        )
        sheet_results['unique_values'] = unique_values_results
        
        # 2. Analyze column statistics for numeric columns
        print(f"  Analyzing column statistics...")
        # Get column statistics settings
        column_stats_settings = default_analysis.get("column_statistics", {})
        include_outliers = column_stats_settings.get("include_outliers", True)
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns[:5])
            sheet_results['column_statistics'] = column_stats_results
        
        # 3. Analyze text columns
        print(f"  Analyzing text columns...")
        # Get text analysis settings
        text_analysis_settings = default_analysis.get("text_analysis", {})
        include_pattern_analysis = text_analysis_settings.get("include_pattern_analysis", True)
        
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            text_analysis_results = analyze_text_columns(df, columns=text_columns[:5])
            sheet_results['text_analysis'] = text_analysis_results
        
        # Store the results
        analysis_results[sheet_name] = sheet_results
        
        # Save the analysis report
        save_analysis_report(sheet_name, df, sheet_results, output_dir)
    
    return analysis_results


def save_analysis_report(sheet_name, df, results, output_dir):
    """
    Save an analysis report for a sheet.
    
    Args:
        sheet_name: Name of the sheet.
        df: DataFrame that was analyzed.
        results: Analysis results.
        output_dir: Directory to save the report.
    """
    # Create a report file
    report_path = os.path.join(output_dir, f"{sheet_name}_analysis_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Analysis Report for Sheet: {sheet_name}\n")
        f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Write unique values report
        if 'unique_values' in results:
            f.write("Unique Values Analysis:\n")
            f.write("-" * 50 + "\n")
            
            unique_values = results['unique_values']
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
        if 'column_statistics' in results:
            f.write("Column Statistics Analysis:\n")
            f.write("-" * 50 + "\n")
            
            column_stats = results['column_statistics']
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
        if 'text_analysis' in results:
            f.write("Text Analysis:\n")
            f.write("-" * 50 + "\n")
            
            text_analysis = results['text_analysis']
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
    
    print(f"  Saved analysis report to: {report_path}")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Get the output directory
    output_dir = os.path.join(get_root_dir(), "examples", "output")
    
    # Extract and analyze data
    analysis_results = extract_and_analyze_data(file_path, output_dir)
    
    print("\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())