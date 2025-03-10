# Domain-Specific Examples

This document provides documentation for the domain-specific examples in NexusML, which demonstrate specialized functionality for equipment classification and related domains.

## OmniClass Generator Example

### Overview

The `omniclass_generator_example.py` script demonstrates how to use the OmniClass generator in NexusML to extract OmniClass data from Excel files and generate descriptions using the Claude API.

### Key Features

- OmniClass data extraction from Excel files
- Description generation using the Claude API
- Batch processing of OmniClass codes
- CSV output for processed data

### Usage

```python
# Run the example
python -m nexusml.examples.omniclass_generator_example
```

### Code Walkthrough

#### OmniClass Data Extraction

The example first extracts OmniClass data from Excel files:

```python
# Extract OmniClass data from Excel files
print(f"Extracting OmniClass data from {input_dir}...")
df = extract_omniclass_data(input_dir=input_dir, output_file=output_csv, file_pattern="*.xlsx")
print(f"Extracted {len(df)} OmniClass codes to {output_csv}")
```

#### API Key Verification

It checks if the ANTHROPIC_API_KEY environment variable is set:

```python
# Check if ANTHROPIC_API_KEY is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Warning: ANTHROPIC_API_KEY environment variable not set.")
    print("Description generation will not work without an API key.")
    print("Please set the ANTHROPIC_API_KEY environment variable and try again.")
    return
```

#### Description Generation

It generates descriptions for a small subset of the data:

```python
# Generate descriptions for a small subset of the data
print("Generating descriptions for a sample of OmniClass codes...")
result_df = generate_descriptions(
    input_file=output_csv,
    output_file=output_with_descriptions,
    start_index=0,
    end_index=5,  # Only process 5 rows for this example
    batch_size=5,
    description_column="Description",
)
```

#### Results Display

It displays the results:

```python
# Display sample results
print("\nSample results:")
for _, row in result_df.head().iterrows():
    print(f"Code: {row['OmniClass_Code']}")
    print(f"Title: {row['OmniClass_Title']}")
    print(f"Description: {row['Description']}")
    print("-" * 50)
```

### Dependencies

- nexusml: For OmniClass data extraction and description generation
- os, pathlib: For file system operations
- anthropic: For Claude API access (via nexusml)

### Notes and Warnings

- Requires an ANTHROPIC_API_KEY environment variable to be set for description generation
- The example processes only a small subset of the data (5 rows) for demonstration purposes
- In a production environment, you would process the entire dataset
- The Claude API has rate limits and costs associated with it

## OmniClass Hierarchy Example

### Overview

The `omniclass_hierarchy_example.py` script demonstrates how to use the OmniClass hierarchy visualization tools to display OmniClass data in a hierarchical tree structure.

### Key Features

- OmniClass data loading and cleaning
- Hierarchical tree construction
- Terminal and Markdown visualization formats
- Filtering by OmniClass code and keywords
- Output to files for further use

### Usage

```python
# Run the example
python -m nexusml.examples.omniclass_hierarchy_example
```

### Code Walkthrough

#### Data Loading and Cleaning

The example first loads and cleans the OmniClass data:

```python
# Try to read the CSV file safely
try:
    logger.info("Attempting to read CSV file")
    df = read_csv_safe(omniclass_file)
    logger.info(f"Successfully loaded {len(df)} rows")
except Exception as e:
    logger.warning(f"Error reading CSV file: {e}")
    logger.info("Cleaning the CSV file...")
    print("CSV file has issues. Cleaning it...")

    # Clean the CSV file
    cleaned_file = clean_omniclass_csv(omniclass_file)
    logger.info(f"Using cleaned file: {cleaned_file}")
    print(f"Using cleaned file: {cleaned_file}")

    # Read the cleaned file
    df = read_csv_safe(cleaned_file)
    logger.info(f"Successfully loaded {len(df)} rows from cleaned file")
```

#### Data Filtering

It filters the data to focus on specific OmniClass codes and keywords:

```python
# Filter data (optional)
# For example, filter to only show Table 23 (Products) entries
filter_value = "23-"
logger.info(f"Filtering by: {filter_value}")
print(f"\nFiltering by: {filter_value}")
filtered_df = df[df[code_col].str.contains(filter_value, na=False)]
logger.info(f"Filtered to {len(filtered_df)} rows")

# Further filter to limit the number of entries for the example
# For example, only show entries related to HVAC
hvac_filter = "HVAC|mechanical|boiler|pump|chiller"
logger.info(f"Further filtering by: {hvac_filter}")
print(f"Further filtering by: {hvac_filter}")
hvac_df = filtered_df[
    (filtered_df[title_col].str.contains(hvac_filter, case=False, na=False))
    | (filtered_df[desc_col].str.contains(hvac_filter, case=False, na=False))
]
logger.info(f"Final dataset has {len(hvac_df)} rows")
```

#### Tree Building and Visualization

It builds and visualizes the hierarchical tree:

```python
# Build the tree
logger.info("Building hierarchy tree...")
print("\nBuilding hierarchy tree...")
tree = build_tree(hvac_df, code_col, title_col, desc_col)

# Display the tree in terminal format
logger.info("Generating terminal output...")
print("\nOmniClass Hierarchy Tree (Terminal Format):")
print_tree_terminal(tree)

# Display the tree in markdown format
logger.info("Generating markdown output...")
print("\nOmniClass Hierarchy Tree (Markdown Format):")
markdown_lines = print_tree_markdown(tree)
print("\n".join(markdown_lines))
```

#### Output to Files

It saves the visualizations to files:

```python
# Save to file
output_file = os.path.join(output_dir, "omniclass_hvac_hierarchy.md")
with open(output_file, "w") as f:
    f.write("\n".join(markdown_lines))
logger.info(f"Saved markdown output to {output_file}")
print(f"\nSaved to {output_file}")

# Save terminal output to file as well
terminal_output_file = os.path.join(output_dir, "omniclass_hvac_hierarchy.txt")
with open(terminal_output_file, "w", encoding="utf-8") as f:
    # Redirect stdout to file temporarily
    import contextlib

    with contextlib.redirect_stdout(f):
        print("OmniClass Hierarchy Tree (Terminal Format):")
        print_tree_terminal(tree)
logger.info(f"Saved terminal output to {terminal_output_file}")
print(f"Saved terminal output to {terminal_output_file}")
```

### Dependencies

- nexusml.ingest.generator.omniclass_hierarchy: For tree building and visualization
- nexusml.utils: For CSV cleaning and logging
- pandas: For data manipulation
- os, sys, pathlib: For file system operations

### Notes and Warnings

- The example includes error handling for CSV file issues
- It uses filtering to focus on specific OmniClass codes and keywords
- The output is saved in both Markdown and terminal formats
- The example uses logging for tracking the process

## Uniformat Keywords Example

### Overview

The `uniformat_keywords_example.py` script demonstrates how to use the Uniformat keywords functionality to find Uniformat codes by keyword and enrich equipment data.

### Key Features

- Uniformat code lookup by keyword
- Equipment data enrichment with Uniformat and MasterFormat information
- Reference data management
- Automatic code assignment based on equipment names

### Usage

```python
# Run the example
python -m nexusml.examples.uniformat_keywords_example
```

### Code Walkthrough

#### Reference Manager Initialization

The example first initializes the reference manager and loads all reference data:

```python
# Initialize the reference manager
ref_manager = ReferenceManager()

# Load all reference data
ref_manager.load_all()
```

#### Finding Uniformat Codes by Keyword

It demonstrates finding Uniformat codes by keyword:

```python
# Example 1: Find Uniformat codes by keyword
print("\nExample 1: Find Uniformat codes by keyword")
print("-------------------------------------------")

keywords = ["Air Barriers", "Boilers", "Elevators", "Pumps"]

for keyword in keywords:
    print(f"\nSearching for keyword: {keyword}")
    results = ref_manager.find_uniformat_codes_by_keyword(keyword)

    if results:
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  - Keyword: {result['keyword']}")
            print(f"    Uniformat Code: {result['uniformat_code']}")
            print(f"    MasterFormat Code: {result['masterformat_code']}")

            # Get the description for the Uniformat code
            if result["uniformat_code"]:
                description = ref_manager.get_uniformat_description(
                    result["uniformat_code"]
                )
                if description:
                    print(f"    Description: {description}")
    else:
        print("No results found.")
```

#### Enriching Equipment Data

It demonstrates enriching equipment data with Uniformat and MasterFormat information:

```python
# Example 2: Enrich equipment data with Uniformat and MasterFormat information
print("\nExample 2: Enrich equipment data")
print("--------------------------------")

# Create a sample DataFrame with equipment data
equipment_data = [
    {
        "equipment_id": "EQ001",
        "equipment_name": "Air Handling Unit",
        "uniformat_code": None,
        "masterformat_code": None,
    },
    # ... other equipment data
]

df = pd.DataFrame(equipment_data)
print("\nOriginal DataFrame:")
print(df)

# Enrich the DataFrame with reference information
enriched_df = ref_manager.enrich_equipment_data(df)

print("\nEnriched DataFrame:")
print(
    enriched_df[
        [
            "equipment_id",
            "equipment_name",
            "uniformat_code",
            "uniformat_description",
            "masterformat_code",
            "masterformat_description",
        ]
    ]
)
```

#### Showing Keyword Matching Results

It shows which codes were found by keyword matching:

```python
# Show which codes were found by keyword matching
print("\nCodes found by keyword matching:")
for _, row in enriched_df.iterrows():
    if pd.notna(row["uniformat_code"]) and row["equipment_id"] in [
        "EQ001",
        "EQ002",
        "EQ004",
        "EQ005",
    ]:
        print(
            f"{row['equipment_name']}: {row['uniformat_code']} - {row['uniformat_description']}"
        )
```

### Dependencies

- nexusml.core.reference.manager: For reference data management
- pandas: For data manipulation

### Notes and Warnings

- The example assumes that reference data is available and properly formatted
- The keyword matching is case-insensitive but requires exact keyword matches
- The enrichment process adds new columns to the DataFrame
- Some equipment may not have matching Uniformat or MasterFormat codes

## Validation Example

### Overview

The `validation_example.py` script demonstrates how to use the validation components in the NexusML suite to validate data quality and integrity.

### Key Features

- Multiple validation rules for different data quality checks
- Validators for combining multiple rules
- Configuration-driven validation
- Schema validation
- Custom validation rules
- Validation reporting

### Usage

```python
# Run the example
python -m nexusml.examples.validation_example
```

### Code Walkthrough

#### Individual Validation Rules

The example first demonstrates using individual validation rules:

```python
# Example 1: Using individual validation rules
print("\nExample 1: Using individual validation rules")
print("-------------------------------------------")

# Check if 'name' column exists
rule = ColumnExistenceRule('name')
result = rule.validate(df)
print(f"Column 'name' exists: {result.valid}")

# Check if 'email' column exists
rule = ColumnExistenceRule('email')
result = rule.validate(df)
print(f"Column 'email' exists: {result.valid}")

# Check if 'age' column has no null values
rule = NonNullRule('age')
result = rule.validate(df)
print(f"Column 'age' has no nulls: {result.valid}")

# Check if 'age' column has values between 20 and 50
rule = ValueRangeRule('age', min_value=20, max_value=50)
result = rule.validate(df)
print(f"Column 'age' values between 20 and 50: {result.valid}")
```

#### Validator with Multiple Rules

It demonstrates using a validator with multiple rules:

```python
# Example 2: Using a validator with multiple rules
print("\nExample 2: Using a validator with multiple rules")
print("----------------------------------------------")

validator = BaseValidator("SampleValidator")
validator.add_rule(ColumnExistenceRule('id'))
validator.add_rule(ColumnExistenceRule('name'))
validator.add_rule(ColumnExistenceRule('age'))
validator.add_rule(ColumnTypeRule('id', 'int'))
validator.add_rule(ColumnTypeRule('name', 'str'))
validator.add_rule(ColumnTypeRule('age', 'float'))
validator.add_rule(UniqueValuesRule('id'))

report = validator.validate(df)
print(f"Validation passed: {report.is_valid()}")
print(f"Number of errors: {len(report.get_errors())}")
print(f"Number of warnings: {len(report.get_warnings())}")
print(f"Number of info messages: {len(report.get_info())}")

# Print all validation results
print("\nValidation results:")
for result in report.results:
    print(f"  {result}")
```

#### Configuration-Driven Validator

It demonstrates using a configuration-driven validator:

```python
# Example 3: Using a configuration-driven validator
print("\nExample 3: Using a configuration-driven validator")
print("-----------------------------------------------")

config = {
    'required_columns': [
        {'name': 'id', 'data_type': 'int', 'required': True},
        {'name': 'name', 'data_type': 'str', 'required': True},
        {'name': 'age', 'data_type': 'float', 'min_value': 0, 'max_value': 100},
        {'name': 'grade', 'allowed_values': ['A', 'B', 'C', 'D', 'F']},
        {'name': 'email', 'required': False},  # Optional column
    ],
    'row_count': {'min': 1},
    'column_count': {'min': 4},
}

validator = ConfigDrivenValidator(config)
report = validator.validate(df)

print(f"Validation passed: {report.is_valid()}")
print(f"Number of errors: {len(report.get_errors())}")
print(f"Number of warnings: {len(report.get_warnings())}")
print(f"Number of info messages: {len(report.get_info())}")

# Print error messages
if not report.is_valid():
    print("\nValidation errors:")
    for error in report.get_errors():
        print(f"  {error.message}")
```

#### Schema Validator

It demonstrates using a schema validator:

```python
# Example 4: Using a schema validator
print("\nExample 4: Using a schema validator")
print("----------------------------------")

schema = {
    'id': 'int',
    'name': 'str',
    'age': 'float',
    'score': 'int',
    'grade': 'str',
}

validator = SchemaValidator(schema)
report = validator.validate_dataframe(df)

print(f"Schema validation passed: {report.is_valid()}")
```

#### Convenience Functions

It demonstrates using convenience functions:

```python
# Example 5: Using convenience functions
print("\nExample 5: Using convenience functions")
print("------------------------------------")

# Validate a specific column
report = validate_column(df, 'age', config={'required': True, 'type': 'float', 'min_value': 0})
print(f"Column 'age' validation passed: {report.is_valid()}")

# Validate the entire DataFrame
report = validate_dataframe(df, config=config)
print(f"DataFrame validation passed: {report.is_valid()}")
```

#### Custom Validation Rule

It demonstrates creating and using a custom validation rule:

```python
# Example 6: Creating a custom validation rule
print("\nExample 6: Creating a custom validation rule")
print("------------------------------------------")

# Define a custom rule that checks if the average score is above a threshold
class AverageScoreRule(ValidationRule):
    def __init__(self, column, threshold, level=ValidationLevel.ERROR):
        self.column = column
        self.threshold = threshold
        self.level = level
    
    def validate(self, data):
        # Implementation...
    
    def get_name(self):
        return f"AverageScore({self.column}, {self.threshold})"
    
    def get_description(self):
        return f"Checks if the average value in column '{self.column}' is above {self.threshold}"

# Use the custom rule
rule = AverageScoreRule('score', 85)
result = rule.validate(df)
print(f"Average score above 85: {result.valid}")
print(f"Message: {result.message}")

# Use a higher threshold
rule = AverageScoreRule('score', 90)
result = rule.validate(df)
print(f"Average score above 90: {result.valid}")
print(f"Message: {result.message}")
```

### Dependencies

- nexusml.core.validation: For validation components
- pandas: For data manipulation
- numpy: For numerical operations

### Notes and Warnings

- The validation system supports different levels of validation (ERROR, WARNING, INFO)
- Rules can be combined into validators for more complex validation
- Configuration-driven validation allows for flexible validation without code changes
- Custom rules can be created for specific validation needs
- Validation reports provide detailed information about validation results

## Common Patterns

Across all four domain-specific examples, you can observe these common patterns:

1. **Domain-Specific Data Handling**: All examples deal with domain-specific data formats and structures, such as OmniClass codes, Uniformat codes, and equipment data.

2. **Reference Data Management**: Many examples use reference data management to look up codes, descriptions, and relationships.

3. **Data Enrichment**: Several examples demonstrate enriching data with additional information from reference sources.

4. **Visualization and Reporting**: The examples include visualization and reporting capabilities to make the results more accessible.

5. **Configuration-Driven Behavior**: Many components are configured through configuration objects or files.

6. **Error Handling**: All examples include error handling to gracefully handle failures.

## Best Practices

### When to Use Each Example

1. **OmniClass Generator**: Use when you need to extract OmniClass data from Excel files and generate descriptions.

2. **OmniClass Hierarchy**: Use when you need to visualize OmniClass data in a hierarchical structure.

3. **Uniformat Keywords**: Use when you need to find Uniformat codes by keyword and enrich equipment data.

4. **Validation**: Use when you need to validate data quality and integrity.

### Integration with Other Components

The domain-specific components can be integrated with other components:

1. **Data Loading**: Use the data loading components to load domain-specific data.

2. **Feature Engineering**: Use the feature engineering components to transform domain-specific data into features.

3. **Model Building**: Use the model building components to create and configure models for domain-specific tasks.

4. **Pipeline**: Use the pipeline components to create end-to-end workflows for domain-specific tasks.

## Next Steps

After understanding domain-specific examples, you might want to explore:

1. **Usage Guide**: Learn how to use NexusML in your own projects.

2. **API Reference**: Explore the complete API reference for NexusML.

3. **Custom Components**: Learn how to create custom components for your specific domain.