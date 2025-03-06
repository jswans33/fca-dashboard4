"""
CSV Utilities for NexusML

This module provides utilities for working with CSV files, including cleaning,
verification, and safe reading of potentially malformed CSV files.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame

from nexusml.utils.logging import get_logger

logger = get_logger(__name__)


def verify_csv_file(
    filepath: Union[str, Path],
    expected_columns: Optional[List[str]] = None,
    expected_field_count: Optional[int] = None,
    fix_issues: bool = False,
    output_filepath: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verify a CSV file for common issues and optionally fix them.

    Args:
        filepath: Path to the CSV file
        expected_columns: List of column names that should be present
        expected_field_count: Expected number of fields per row
        fix_issues: Whether to attempt to fix issues
        output_filepath: Path to save the fixed file (if fix_issues is True)
                         If None, will use the original filename with "_fixed" appended

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if the file is valid or was fixed successfully
        - error_message: Description of the issue if not valid, or None if valid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return False, f"File not found: {filepath}"

    if output_filepath is None and fix_issues:
        output_filepath = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"

    try:
        # First, try to read the file with pandas to check for basic validity
        try:
            df = pd.read_csv(filepath)

            # Check expected columns if provided
            if expected_columns:
                missing_columns = [
                    col for col in expected_columns if col not in df.columns
                ]
                if missing_columns:
                    if not fix_issues:
                        return (
                            False,
                            f"Missing expected columns: {', '.join(missing_columns)}",
                        )
                    logger.warning(
                        f"Missing expected columns: {', '.join(missing_columns)}"
                    )

            # If we got here without errors and don't need to check field count, the file is valid
            if expected_field_count is None:
                return True, None

        except Exception as e:
            # If pandas can't read it, we'll try a more manual approach
            logger.warning(f"Pandas couldn't read the CSV file: {e}")

            if not fix_issues:
                return False, str(e)

        # Manually check each row for the correct number of fields
        issues = []
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Get the header row
            try:
                header = next(reader)
                field_count = len(header)

                # If expected_field_count wasn't provided, use the header length
                if expected_field_count is None:
                    expected_field_count = field_count

                # Check if header has the expected number of fields
                if field_count != expected_field_count:
                    issues.append(
                        (
                            0,
                            header,
                            f"Header has {field_count} fields, expected {expected_field_count}",
                        )
                    )

                # Check each data row
                for i, row in enumerate(reader, start=1):
                    if len(row) != expected_field_count:
                        issues.append(
                            (
                                i,
                                row,
                                f"Line {i+1} has {len(row)} fields, expected {expected_field_count}",
                            )
                        )

            except Exception as e:
                return False, f"Error reading CSV file: {e}"

        # If there are no issues, the file is valid
        if not issues:
            return True, None

        # Log the issues found
        logger.warning(f"Found {len(issues)} issues in {filepath}")
        for line_num, row, msg in issues[:5]:  # Log first 5 issues
            logger.warning(f"  {msg}")

        if len(issues) > 5:
            logger.warning(f"  ... and {len(issues) - 5} more issues")

        # If we're not fixing issues, return False
        if not fix_issues:
            return False, f"Found {len(issues)} rows with incorrect field count"

        # Fix the issues by reading and writing the file manually
        logger.info(f"Fixing issues and writing to {output_filepath}")
        with open(filepath, "r", newline="", encoding="utf-8") as f_in, open(
            output_filepath, "w", newline="", encoding="utf-8"
        ) as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            # Write the header
            header = next(reader)
            if len(header) != expected_field_count:
                # Fix header if needed
                if len(header) < expected_field_count:
                    # Add empty columns if there are too few
                    header.extend([""] * (expected_field_count - len(header)))
                else:
                    # Combine extra columns if there are too many
                    header = header[: expected_field_count - 1] + [
                        " ".join(header[expected_field_count - 1 :])
                    ]

            writer.writerow(header)

            # Write each data row, fixing as needed
            for i, row in enumerate(reader, start=1):
                if len(row) != expected_field_count:
                    if len(row) < expected_field_count:
                        # Add empty columns if there are too few
                        row.extend([""] * (expected_field_count - len(row)))
                    else:
                        # Combine extra columns if there are too many
                        row = row[: expected_field_count - 1] + [
                            " ".join(row[expected_field_count - 1 :])
                        ]

                writer.writerow(row)

        logger.info(f"Fixed CSV file saved to {output_filepath}")
        return True, None

    except Exception as e:
        return False, f"Error verifying CSV file: {e}"


def read_csv_safe(
    filepath: Union[str, Path],
    expected_columns: Optional[List[str]] = None,
    expected_field_count: Optional[int] = None,
    fix_issues: bool = True,
    **kwargs: Any,
) -> DataFrame:
    """
    Safely read a CSV file, handling common issues.

    Args:
        filepath: Path to the CSV file
        expected_columns: List of column names that should be present
        expected_field_count: Expected number of fields per row
        fix_issues: Whether to attempt to fix issues
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame containing the CSV data

    Raises:
        ValueError: If the file is invalid and couldn't be fixed
    """
    filepath = Path(filepath)

    # First, verify the file
    is_valid, error_message = verify_csv_file(
        filepath,
        expected_columns=expected_columns,
        expected_field_count=expected_field_count,
        fix_issues=fix_issues,
    )

    if is_valid:
        # If the file is valid, read it directly
        return pd.read_csv(filepath, **kwargs)
    elif fix_issues:
        # If we tried to fix issues but still got an error, try reading the fixed file
        fixed_filepath = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"
        if fixed_filepath.exists():
            logger.info(f"Reading fixed CSV file: {fixed_filepath}")
            return pd.read_csv(fixed_filepath, **kwargs)

    # If we got here, the file is invalid and couldn't be fixed
    raise ValueError(f"Invalid CSV file: {error_message}")


def clean_omniclass_csv(
    input_filepath: Union[str, Path],
    output_filepath: Optional[Union[str, Path]] = None,
    expected_columns: Optional[List[str]] = None,
) -> str:
    """
    Clean the OmniClass CSV file, handling specific issues with this format.

    Args:
        input_filepath: Path to the input OmniClass CSV file
        output_filepath: Path to save the cleaned file (if None, will use input_filepath with "_cleaned" appended)
        expected_columns: List of expected column names

    Returns:
        Path to the cleaned CSV file

    Raises:
        ValueError: If the file couldn't be cleaned
    """
    input_filepath = Path(input_filepath)

    if output_filepath is None:
        output_filepath = (
            input_filepath.parent
            / f"{input_filepath.stem}_cleaned{input_filepath.suffix}"
        )
    else:
        output_filepath = Path(output_filepath)

    logger.info(f"Cleaning OmniClass CSV file: {input_filepath}")

    # Determine the expected field count
    if expected_columns:
        expected_field_count = len(expected_columns)
    else:
        # Try to determine from the header row
        try:
            with open(input_filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                expected_field_count = len(header)
        except Exception:
            # Default to 3 fields for OmniClass CSV (code, title, description)
            expected_field_count = 3

    # Verify and fix the CSV file
    is_valid, error_message = verify_csv_file(
        input_filepath,
        expected_columns=expected_columns,
        expected_field_count=expected_field_count,
        fix_issues=True,
        output_filepath=output_filepath,
    )

    if not is_valid:
        raise ValueError(f"Failed to clean OmniClass CSV file: {error_message}")

    logger.info(f"OmniClass CSV file cleaned and saved to: {output_filepath}")
    return str(output_filepath)


if __name__ == "__main__":
    # If run as a script, clean the OmniClass CSV file
    import argparse

    parser = argparse.ArgumentParser(description="Clean and verify CSV files")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", help="Path to the output CSV file")
    parser.add_argument(
        "--fields", "-f", type=int, help="Expected number of fields per row"
    )
    parser.add_argument("--columns", "-c", nargs="+", help="Expected column names")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        output_file = clean_omniclass_csv(
            args.input_file, args.output, expected_columns=args.columns
        )
        print(f"CSV file cleaned and saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
