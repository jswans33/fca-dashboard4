# Data Loading Examples

This document provides documentation for the data loading examples in NexusML, which demonstrate different approaches to loading and processing data for machine learning tasks.

## Basic Data Loader Example

### Overview

The `data_loader_example.py` script demonstrates how to find and load data files from different locations in a project. It provides a flexible way to discover data files with specific extensions and load them into pandas DataFrames.

### Key Features

- Automatic discovery of data files in specified locations
- Support for multiple file formats (CSV, Excel)
- Utility functions for finding the project root directory
- Simple data loading interface

### Usage

```python
# Run the example
python -m nexusml.examples.data_loader_example
```

### Code Walkthrough

#### Project Root Discovery

The example first defines a utility function to find the project root directory:

```python
def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    # Assuming this script is in nexusml/examples
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the project root
    return os.path.dirname(script_dir)
```

#### Finding Data Files

It then provides a function to discover data files with specific extensions in given locations:

```python
def find_data_files(
    locations: List[str], extensions: List[str] = [".xlsx", ".csv"]
) -> Dict[str, str]:
    """Find all data files with specified extensions in the given locations."""
    data_files = {}
    for location in locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(location, file)
                    data_files[file] = file_path
    return data_files
```

#### Loading Data

The example includes a function to load data based on file extension:

```python
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a file based on its extension."""
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
```

#### Main Workflow

The main workflow:
1. Gets the project root directory
2. Defines locations to search for data files
3. Finds all data files in those locations
4. Displays available files
5. Selects a file (first one by default)
6. Loads the selected file
7. Displays information about the loaded data

### Dependencies

- pandas: For data manipulation and analysis
- os, sys: For file system operations

### Notes and Warnings

- The example assumes a specific project structure with the script in the `nexusml/examples` directory
- In a real application, you might want to add more error handling and user interaction
- The example currently only supports CSV and Excel files

## Enhanced Data Loader Example

### Overview

The `enhanced_data_loader_example.py` script demonstrates how to use the `StandardDataLoader` class from NexusML's core pipeline components to discover and load data files. This provides a more integrated approach compared to the basic example.

### Key Features

- Uses NexusML's built-in `StandardDataLoader` class
- Automatic discovery of data files
- Support for multiple file formats
- Simplified API for data loading

### Usage

```python
# Run the example
python -m nexusml.examples.enhanced_data_loader_example
```

### Code Walkthrough

#### Setup

The example first ensures the project root is in the Python path:

```python
# Add the project root to the Python path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
```

#### Creating a Data Loader

It then creates an instance of the `StandardDataLoader`:

```python
# Create a data loader instance
data_loader = StandardDataLoader()
print(f"Created {data_loader.get_name()}: {data_loader.get_description()}\n")
```

#### Discovering Data Files

The example demonstrates how to discover available data files:

```python
# Discover available data files
print("Discovering available data files...")
available_files = data_loader.discover_data_files()
```

#### Loading Data

It shows two ways to load data:
1. Loading a specific file:
   ```python
   data = data_loader.load_data(first_file_path)
   ```

2. Automatic discovery and loading:
   ```python
   auto_data = data_loader.load_data(discover_files=True)
   ```

### Dependencies

- nexusml.core.pipeline.components.data_loader: For the StandardDataLoader class
- pandas: For data manipulation (used internally by StandardDataLoader)
- os, sys, pathlib: For file system operations

### Notes and Warnings

- The `StandardDataLoader` will automatically search in default locations defined in the NexusML configuration
- The automatic discovery feature will select the first available file if multiple files are found
- For production use, you should specify the exact file to load rather than relying on automatic discovery

## Staging Data Example

### Overview

The `staging_data_example.py` script demonstrates how to use NexusML with staging data that has different column names than what the model expects. It shows the complete workflow from staging data to master database field mapping.

### Key Features

- Creates sample staging data with realistic equipment information
- Maps staging data fields to model input fields
- Uses the EquipmentClassifier for predictions
- Demonstrates master database field mapping

### Usage

```python
# Run the example
python -m nexusml.examples.staging_data_example
```

### Code Walkthrough

#### Creating Test Data

The example first creates sample staging data:

```python
def create_test_staging_data():
    """Create a sample staging data CSV file for testing."""
    data = [
        {
            "Asset Name": "Centrifugal Chiller",
            "Asset Tag": "CH-01",
            # ... other fields
        },
        # ... other records
    ]
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / "test_staging_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path
```

#### Processing Equipment Records

It then processes each equipment record:

```python
for i, (idx, row) in enumerate(staging_df.iterrows()):
    print(f"\nProcessing equipment {i+1}: {row['Asset Name']}")
    
    # Create description from relevant fields
    description_parts = []
    for field in [
        "Asset Name",
        "System Category",
        "Sub System Type",
        "Manufacturer",
        "Model Number",
    ]:
        if field in row and pd.notna(row[field]) and row[field] != "":
            description_parts.append(str(row[field]))
    
    description = " ".join(description_parts)
    service_life = (
        float(row.get("Service Life", 20.0))
        if pd.notna(row.get("Service Life", 0))
        else 20.0
    )
    asset_tag = str(row.get("Asset Tag", ""))
    
    # Get prediction with master DB mapping
    prediction = classifier.predict(description, service_life, asset_tag)
```

#### Saving Results

Finally, it saves the results to a JSON file:

```python
# Save results to JSON
results_file = (
    Path(staging_data_path).parent / "staging_classification_results.json"
)
with open(results_file, "w") as f:
    json.dump(results, f, indent=2, default=str)
```

### Dependencies

- nexusml.core.data_mapper: For mapping staging data to model input
- nexusml.core.model: For the EquipmentClassifier
- pandas: For data manipulation
- json, os, sys, pathlib: For file system and data operations

### Notes and Warnings

- The example creates a new CSV file in the output directory each time it runs
- It assumes the EquipmentClassifier has been properly trained
- In a production environment, you would need to define proper field mappings between your staging data and the model's expected input format
- The service life defaults to 20.0 if not provided or invalid

## Common Patterns

Across all three examples, you can observe these common patterns:

1. **File Discovery**: All examples demonstrate ways to find data files, from basic directory scanning to using NexusML's built-in discovery mechanisms.

2. **Format Handling**: The examples show how to handle different file formats, particularly CSV and Excel files.

3. **Data Loading**: Each example demonstrates loading data into pandas DataFrames, which is the standard data structure used throughout NexusML.

4. **Project Structure Awareness**: The examples use various techniques to locate files relative to the project structure, making them more portable.

5. **Progressive Complexity**: The examples progress from basic file operations to using NexusML's built-in components to a complete workflow with classification.

## Next Steps

After understanding these data loading examples, you might want to explore:

1. **Feature Engineering Examples**: Learn how to transform raw data into features suitable for machine learning.

2. **Model Building Examples**: See how to build and train models using the loaded data.

3. **Pipeline Examples**: Understand how to combine data loading, feature engineering, and model building into complete pipelines.