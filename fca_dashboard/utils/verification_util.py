"""
Verification utilities for pipeline outputs.

This module provides functions to verify the results of pipeline operations,
including checking for placeholder values, validating data quality, and
generating verification reports.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from fca_dashboard.utils.logging_config import get_logger

logger = get_logger(__name__)


def verify_no_placeholder_values(
    df: pd.DataFrame,
    columns: List[str],
    placeholder_values: Optional[List[str]] = None,
    case_sensitive: bool = False,
) -> Dict[str, Dict[str, Union[int, List[str], Dict[str, int]]]]:
    """
    Verify that the specified columns do not contain placeholder values.
    
    Args:
        df: The DataFrame to verify
        columns: The columns to check for placeholder values
        placeholder_values: A list of placeholder values to check for.
            If None, defaults to ['no id', 'none', 'n/a', 'unknown', '']
        case_sensitive: Whether to perform case-sensitive comparison
        
    Returns:
        A dictionary with verification results for each column
    """
    if placeholder_values is None:
        placeholder_values = ['no id', 'none', 'n/a', 'unknown', '']
    
    results = {}
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Convert to string to handle non-string columns
        col_values = df[col].astype(str)
        
        if not case_sensitive:
            col_values = col_values.str.lower()
            placeholder_set = set(v.lower() for v in placeholder_values)
        else:
            placeholder_set = set(placeholder_values)
        
        # Find rows with placeholder values
        mask = col_values.isin(placeholder_set)
        placeholder_rows = df[mask]
        
        # Count occurrences of each placeholder value
        value_counts = {}
        for placeholder in placeholder_set:
            if not case_sensitive:
                count = sum(col_values.str.lower() == placeholder)
            else:
                count = sum(col_values == placeholder)
            
            if count > 0:
                value_counts[placeholder] = count
        
        results[col] = {
            'total_rows': len(df),
            'placeholder_count': len(placeholder_rows),
            'placeholder_percentage': len(placeholder_rows) / len(df) if len(df) > 0 else 0,
            'value_counts': value_counts,
            'placeholder_indices': placeholder_rows.index.tolist()
        }
    
    return results


def verify_no_null_values(
    df: pd.DataFrame,
    columns: List[str]
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Verify that the specified columns do not contain null values.
    
    Args:
        df: The DataFrame to verify
        columns: The columns to check for null values
        
    Returns:
        A dictionary with verification results for each column
    """
    results = {}
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Find rows with null values
        null_rows = df[df[col].isna()]
        
        results[col] = {
            'total_rows': len(df),
            'null_count': len(null_rows),
            'null_percentage': len(null_rows) / len(df) if len(df) > 0 else 0,
            'null_indices': null_rows.index.tolist()
        }
    
    return results


def verify_database_table(
    db_path: str,
    table_name: str,
    verification_config: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Verify a database table against a verification configuration.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table to verify
        verification_config: A dictionary with verification configurations
            Example:
            {
                'no_placeholder_values': {
                    'columns': ['asset_id', 'equipment_tag'],
                    'placeholder_values': ['no id', 'none', 'n/a', 'unknown', '']
                },
                'no_null_values': {
                    'columns': ['asset_name', 'asset_id']
                }
            }
            
    Returns:
        A dictionary with verification results
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return {'error': f"Database file not found: {db_path}"}
    
    try:
        # Connect to the database and read the table
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        results = {
            'database': db_path,
            'table': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'verifications': {}
        }
        
        # Run verifications based on the configuration
        if 'no_placeholder_values' in verification_config:
            config = verification_config['no_placeholder_values']
            columns = config.get('columns', [])
            placeholder_values = config.get('placeholder_values')
            case_sensitive = config.get('case_sensitive', False)
            
            placeholder_results = verify_no_placeholder_values(
                df, columns, placeholder_values, case_sensitive
            )
            results['verifications']['no_placeholder_values'] = placeholder_results
        
        if 'no_null_values' in verification_config:
            config = verification_config['no_null_values']
            columns = config.get('columns', [])
            
            null_results = verify_no_null_values(df, columns)
            results['verifications']['no_null_values'] = null_results
        
        return results
    
    except Exception as e:
        logger.error(f"Error verifying database table: {str(e)}", exc_info=True)
        return {'error': f"Error verifying database table: {str(e)}"}


def generate_verification_report(
    verification_results: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a human-readable verification report.
    
    Args:
        verification_results: The results from verify_database_table
        output_path: Path to save the report. If None, the report is only returned
        
    Returns:
        The report as a string
    """
    if 'error' in verification_results:
        report = f"Verification Error: {verification_results['error']}\n"
        return report
    
    # Generate the report
    report = "Database Verification Report\n"
    report += "=" * 50 + "\n\n"
    
    report += f"Database: {verification_results['database']}\n"
    report += f"Table: {verification_results['table']}\n"
    report += f"Row count: {verification_results['row_count']}\n"
    report += f"Column count: {verification_results['column_count']}\n\n"
    
    # Report on placeholder values
    if 'no_placeholder_values' in verification_results['verifications']:
        report += "Placeholder Values Check\n"
        report += "-" * 50 + "\n"
        
        placeholder_results = verification_results['verifications']['no_placeholder_values']
        for col, result in placeholder_results.items():
            report += f"  Column: {col}\n"
            report += f"    Total rows: {result['total_rows']}\n"
            report += f"    Placeholder count: {result['placeholder_count']}\n"
            report += f"    Placeholder percentage: {result['placeholder_percentage'] * 100:.2f}%\n"
            
            if result['value_counts']:
                report += f"    Placeholder value counts:\n"
                for value, count in result['value_counts'].items():
                    report += f"      '{value}': {count}\n"
            
            report += "\n"
    
    # Report on null values
    if 'no_null_values' in verification_results['verifications']:
        report += "Null Values Check\n"
        report += "-" * 50 + "\n"
        
        null_results = verification_results['verifications']['no_null_values']
        for col, result in null_results.items():
            report += f"  Column: {col}\n"
            report += f"    Total rows: {result['total_rows']}\n"
            report += f"    Null count: {result['null_count']}\n"
            report += f"    Null percentage: {result['null_percentage'] * 100:.2f}%\n"
            report += "\n"
    
    # Report on analysis results
    if 'analysis' in verification_results:
        report += "Analysis Results\n"
        report += "-" * 50 + "\n"
        
        analysis_results = verification_results['analysis']
        for key, result in analysis_results.items():
            report += f"  {key}:\n"
            for metric, value in result.items():
                if isinstance(value, float):
                    report += f"    {metric}: {value * 100:.2f}%\n"
                else:
                    report += f"    {metric}: {value}\n"
            report += "\n"
    
    # Report on validation results
    if 'validation' in verification_results:
        report += "Validation Results\n"
        report += "-" * 50 + "\n"
        
        validation_results = verification_results['validation']
        for key, result in validation_results.items():
            if key.startswith('missing_values_'):
                col = key.replace('missing_values_', '')
                report += f"  Missing values in {col}: {result * 100:.2f}%\n"
            elif key == 'duplicate_rows':
                report += f"  Duplicate rows: {result['count']}\n"
            else:
                if isinstance(result, dict):
                    report += f"  {key}:\n"
                    for metric, value in result.items():
                        report += f"    {metric}: {value}\n"
                else:
                    report += f"  {key}: {result}\n"
            report += "\n"
    
    # Save the report if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Verification report saved to {output_path}")
    
    return report


def verify_pipeline_output(
    pipeline_name: str,
    db_name: str,
    table_name: str,
    verification_config: Dict[str, Dict],
    output_dir: Optional[str] = None,
    analysis_results: Optional[Dict] = None,
    validation_results: Optional[Dict] = None
) -> Dict:
    """
    Verify the output of a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        db_name: Name of the database file
        table_name: Name of the table to verify
        verification_config: Configuration for verification
        output_dir: Directory where the pipeline output is stored.
            If None, defaults to 'outputs/pipeline/{pipeline_name}'
        analysis_results: Results from the pipeline's analyze method
        validation_results: Results from the pipeline's validate method
            
    Returns:
        The verification results
    """
    if output_dir is None:
        output_dir = os.path.join('outputs', 'pipeline', pipeline_name)
    
    db_path = os.path.join(output_dir, db_name)
    
    # Verify the database table
    results = verify_database_table(db_path, table_name, verification_config)
    
    # Add analysis results if available
    if analysis_results:
        results['analysis'] = {}
        
        # Check for unique values in key columns
        if 'unique_values' in analysis_results:
            unique_values = analysis_results['unique_values']
            for col, res in unique_values.items():
                if col.lower() in [c.lower() for c in verification_config.get('no_placeholder_values', {}).get('columns', [])]:
                    results['analysis'][f'unique_values_{col}'] = {
                        'count': res['count'],
                        'null_count': res['null_count'],
                        'null_percentage': res['null_percentage']
                    }
    
    # Add validation results if available
    if validation_results:
        results['validation'] = {}
        
        # Check for missing values in key columns
        if 'missing_values' in validation_results:
            missing_values = validation_results['missing_values']
            for col, pct in missing_values.items():
                if col.lower() in [c.lower() for c in verification_config.get('no_null_values', {}).get('columns', [])]:
                    results['validation'][f'missing_values_{col}'] = pct
        
        # Check for duplicate rows
        if 'duplicate_rows' in validation_results:
            duplicate_rows = validation_results['duplicate_rows']
            results['validation']['duplicate_rows'] = {
                'count': duplicate_rows['duplicate_count']
            }
    
    # Generate and save the report
    report_path = os.path.join(output_dir, f"{table_name}_verification_report.txt")
    generate_verification_report(results, report_path)
    
    return results