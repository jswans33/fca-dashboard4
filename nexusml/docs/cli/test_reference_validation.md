# Command-Line Tool: test_reference_validation.py

## Overview

The `test_reference_validation.py` script demonstrates how to use the reference data validation functionality to ensure data quality across all reference data sources in the NexusML system. It provides a simple way to validate all reference data sources, view validation results, and save them to a file for further analysis.

Key features include:

1. **Reference Data Loading**: Loads all reference data sources using the ReferenceManager
2. **Comprehensive Validation**: Validates all reference data sources to ensure data quality
3. **Detailed Reporting**: Provides detailed validation results for each reference data source
4. **Result Persistence**: Saves validation results to a JSON file for further analysis

## Usage

```bash
python test_reference_validation.py
```

The script doesn't take any command-line arguments. It simply runs the validation process and outputs the results to the console and a JSON file.

### Example

```bash
python test_reference_validation.py
```

This will load all reference data, validate it, print the validation results to the console, and save them to a JSON file.

## Functions

### `test_reference_validation()`

Test the reference data validation functionality.

**Returns:**
- Dict: Validation results for all reference data sources

**Example:**
```python
from nexusml.test_reference_validation import test_reference_validation

# Run the reference data validation
validation_results = test_reference_validation()

# Check if a specific data source has issues
if validation_results["omniclass"]["issues"]:
    print("OmniClass data has issues:")
    for issue in validation_results["omniclass"]["issues"]:
        print(f"  - {issue}")
else:
    print("OmniClass data is valid")
```

**Notes:**
- This function creates a ReferenceManager instance
- It loads all reference data sources
- It validates all reference data sources
- It prints the validation results to the console
- It saves the validation results to a JSON file
- It returns the validation results as a dictionary

### `main()`

Main function to run the reference data validation.

**Example:**
```python
from nexusml.test_reference_validation import main

# Run the reference data validation
main()
```

**Notes:**
- This function is called when the script is run directly
- It simply calls the test_reference_validation function

## Output Format

The script outputs validation results in two formats:

### Console Output

The console output includes:

1. **Loading Status**: Indicates that reference data is being loaded
2. **Validation Status**: Indicates that reference data is being validated
3. **Validation Results**: For each reference data source:
   - Source name
   - Loading status
   - Issues (if any)
   - Statistics about the data

Example console output:
```
Testing reference data validation...

Loading reference data...

Validating reference data...

Validation Results:
==================

OMNICLASS:
  Loaded: True
  Issues: None
  Stats:
    - row_count: 1532
    - column_count: 5
    - columns: ['code', 'title', 'level', 'parent_code', 'description']
    - null_counts: {'code': 0, 'title': 0, 'level': 0, 'parent_code': 132, 'description': 0}
    - unique_counts: {'code': 1532, 'title': 1532, 'level': 6, 'parent_code': 131, 'description': 1532}

UNIFORMAT:
  Loaded: True
  Issues: None
  Stats:
    - row_count: 297
    - column_count: 4
    - columns: ['uniformat_code', 'title', 'level', 'parent_code']
    - null_counts: {'uniformat_code': 0, 'title': 0, 'level': 0, 'parent_code': 23}
    - unique_counts: {'uniformat_code': 297, 'title': 297, 'level': 4, 'parent_code': 23}

...

Validation results saved to /path/to/nexusml/test_output/reference_validation_results.json
```

### JSON Output

The validation results are also saved to a JSON file at `test_output/reference_validation_results.json`. The JSON file contains the same information as the console output, but in a structured format that can be easily parsed and analyzed.

Example JSON output:
```json
{
  "omniclass": {
    "loaded": true,
    "issues": [],
    "stats": {
      "row_count": 1532,
      "column_count": 5,
      "columns": ["code", "title", "level", "parent_code", "description"],
      "null_counts": {
        "code": 0,
        "title": 0,
        "level": 0,
        "parent_code": 132,
        "description": 0
      },
      "unique_counts": {
        "code": 1532,
        "title": 1532,
        "level": 6,
        "parent_code": 131,
        "description": 1532
      }
    }
  },
  "uniformat": {
    "loaded": true,
    "issues": [],
    "stats": {
      "row_count": 297,
      "column_count": 4,
      "columns": ["uniformat_code", "title", "level", "parent_code"],
      "null_counts": {
        "uniformat_code": 0,
        "title": 0,
        "level": 0,
        "parent_code": 23
      },
      "unique_counts": {
        "uniformat_code": 297,
        "title": 297,
        "level": 4,
        "parent_code": 23
      }
    }
  },
  ...
}
```

## Reference Data Sources

The script validates the following reference data sources:

1. **OmniClass**: OmniClass construction classification system
2. **Uniformat**: Uniformat construction classification system
3. **MasterFormat**: MasterFormat construction classification system
4. **MCAA Glossary**: Mechanical Contractors Association of America glossary
5. **MCAA Abbreviations**: Mechanical Contractors Association of America abbreviations
6. **SMACNA**: Sheet Metal and Air Conditioning Contractors' National Association data
7. **ASHRAE**: American Society of Heating, Refrigerating and Air-Conditioning Engineers data
8. **Energize Denver**: Energize Denver building performance data
9. **Equipment Taxonomy**: Equipment taxonomy data

## Validation Checks

For each reference data source, the validation process checks:

1. **Loading Status**: Whether the data was loaded successfully
2. **Required Columns**: Whether all required columns are present
3. **Data Types**: Whether the data types are correct
4. **Null Values**: Whether there are any null values in required columns
5. **Duplicate Values**: Whether there are any duplicate values in unique columns
6. **Referential Integrity**: Whether references to other data sources are valid
7. **Data Quality**: Whether the data meets specific quality criteria for each source

## Process Flow

1. **Create ReferenceManager**: Create a ReferenceManager instance
2. **Load Reference Data**: Load all reference data sources
3. **Validate Reference Data**: Validate all reference data sources
4. **Print Validation Results**: Print the validation results to the console
5. **Save Validation Results**: Save the validation results to a JSON file

## Dependencies

- **json**: Used for JSON serialization
- **sys**: Used for system operations
- **pathlib**: Used for path manipulation
- **nexusml.core.reference.manager**: Used for ReferenceManager

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path.
- The script creates a `test_output` directory in the project root if it doesn't exist.
- The validation results are saved to `test_output/reference_validation_results.json`.
- The script doesn't take any command-line arguments.
- The validation process can take some time, especially if there are many reference data sources or if they are large.
- The validation results include statistics about each reference data source, which can be useful for understanding the data.
- If a reference data source has issues, they will be listed in the validation results.
- The script doesn't fail if there are validation issues; it simply reports them.

## Use Cases

1. **Data Quality Assurance**: Ensure that all reference data sources meet quality standards
2. **Data Exploration**: Understand the structure and content of reference data sources
3. **Troubleshooting**: Identify issues with reference data sources
4. **Documentation**: Generate documentation about reference data sources
5. **Testing**: Test that reference data sources are loaded and validated correctly

## Example Integration

```python
import json
from pathlib import Path
from nexusml.core.reference.manager import ReferenceManager

def validate_reference_data():
    """Validate reference data and return results."""
    # Create reference manager
    manager = ReferenceManager()
    
    # Load all reference data
    manager.load_all()
    
    # Validate all reference data
    validation_results = manager.validate_data()
    
    # Check if there are any issues
    has_issues = any(result["issues"] for result in validation_results.values())
    
    # Save validation results to file
    output_file = Path("validation_results.json")
    with open(output_file, "w") as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    return validation_results, has_issues

# Validate reference data
results, has_issues = validate_reference_data()

# Print summary
print(f"Reference data validation {'failed' if has_issues else 'passed'}")
if has_issues:
    for source_name, result in results.items():
        if result["issues"]:
            print(f"\n{source_name.upper()} issues:")
            for issue in result["issues"]:
                print(f"  - {issue}")