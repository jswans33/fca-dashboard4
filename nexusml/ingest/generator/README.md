# NexusML Generator Module

This module provides utilities for generating data for the NexusML module,
including OmniClass data extraction and description generation.

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

## Requirements

- Python 3.8+
- pandas
- anthropic
- dotenv
- tqdm

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for the Anthropic Claude API.

## Example

See `nexusml/examples/omniclass_generator_example.py` for a complete example of
how to use the generator module.
