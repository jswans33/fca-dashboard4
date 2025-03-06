# NexusML Generator Module

This module provides utilities for generating data for the NexusML module,
including OmniClass data extraction, description generation, and hierarchy
visualization.

## Components

### OmniClass Data Extraction

The `omniclass.py` module provides functionality to extract OmniClass data from
Excel files and generate a unified CSV file for classifier training.

Key functions:

- `extract_omniclass_data`: Extract OmniClass data from Excel files and save to
  a CSV file.

### OmniClass Description Generator

The `omniclass_description_generator.py` module provides functionality to
generate plain-English descriptions for OmniClass codes using the Claude API.

Key components:

- `OmniClassDescriptionGenerator`: Generator for OmniClass descriptions using
  Claude API.
- `BatchProcessor`: Processor for batch processing data.
- `AnthropicClient`: Client for the Anthropic Claude API.
- `generate_descriptions`: Generate descriptions for OmniClass codes.

### OmniClass Hierarchy Visualization

The module includes several tools for visualizing OmniClass data in a
hierarchical tree structure:

- `omniclass_hierarchy.py`: Interactive tool for visualizing OmniClass data in a
  hierarchical tree structure.
- `omniclass_tree.py`: Command-line tool for quickly generating OmniClass
  hierarchy trees.
- `omniclass_example.py`: Example tool with hardcoded medical equipment data to
  demonstrate hierarchy visualization.

Key functions:

- `parse_omniclass_code`: Parse OmniClass codes in the format xx-yy yy yy-zz.
- `build_tree`: Build a hierarchical tree from OmniClass data.
- `print_tree_terminal`: Display the hierarchy tree in terminal format.
- `print_tree_markdown`: Generate a markdown representation of the hierarchy
  tree.

## Usage

### OmniClass Data Extraction

```python
from nexusml import extract_omniclass_data

# Extract OmniClass data from Excel files
df = extract_omniclass_data(
    input_dir="files/omniclass_tables",
    output_file="nexusml/ingest/generator/data/omniclass.csv",
    file_pattern="*.xlsx"
)
```

### OmniClass Description Generation

```python
from nexusml import generate_descriptions

# Generate descriptions for OmniClass codes
result_df = generate_descriptions(
    input_file="nexusml/ingest/generator/data/omniclass.csv",
    output_file="nexusml/ingest/generator/data/omniclass_with_descriptions.csv",
    start_index=0,
    end_index=None,  # Process all rows
    batch_size=50,
    description_column="Description"
)
```

### OmniClass Hierarchy Visualization

```python
from nexusml.ingest.generator.omniclass_hierarchy import build_tree, print_tree_terminal

# Load OmniClass data
import pandas as pd
df = pd.read_csv("nexusml/ingest/data/omniclass.csv")

# Filter data (optional)
filtered_df = df[df["OmniClass_Code"].str.contains("23-", na=False)]

# Build the hierarchy tree
tree = build_tree(filtered_df, "OmniClass_Code", "OmniClass_Title", "Description")

# Display the tree in terminal format
print_tree_terminal(tree)
```

## Requirements

- Python 3.8+
- pandas
- anthropic
- dotenv
- tqdm
- re

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for the Anthropic Claude API (only needed for
  description generation).

## Examples

- See `nexusml/examples/omniclass_generator_example.py` for a complete example
  of how to use the generator module.
- See `nexusml/examples/omniclass_hierarchy_example.py` for an example of how to
  use the hierarchy visualization tools.
