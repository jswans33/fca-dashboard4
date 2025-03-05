# SOP 006: Generating OmniClass Descriptions with Claude API

## Purpose

This Standard Operating Procedure (SOP) outlines the process for generating
descriptions for OmniClass codes using the Claude API. The process involves
batch processing of OmniClass data, calling the Claude API with optimized
prompts, and updating the CSV file with generated descriptions.

## Scope

This SOP applies to the generation of descriptions for OmniClass codes in the
unified training data format.

## Prerequisites

- Python 3.8+
- Anthropic API key
- OmniClass data in CSV format (generated from
  `fca_dashboard/examples/omniclass_generator_example.py`)

## Required Packages

```bash
pip install pandas anthropic python-dotenv tqdm
```

## Environment Setup

1. Create a `.env` file in the project root directory:

```
ANTHROPIC_API_KEY=your_api_key_here
```

2. Ensure the `.env` file is added to `.gitignore` to prevent committing API
   keys to version control.

## Procedure

### 1. Create the Description Generator Script

Create a new file `fca_dashboard/utils/omniclass_description_generator.py` with
the following content:

```python
"""
Utility for generating descriptions for OmniClass codes using the Claude API.

This module provides functions to generate plain-English descriptions for OmniClass codes
using the Claude API. It processes the data in batches to manage API rate limits and costs.
"""
import os
import time
import json
import re
import pandas as pd
import anthropic
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Constants
BATCH_SIZE = 50  # Process 50 items at a time
MODEL = "claude-haiku-v1"  # Cheapest Claude model
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# System prompt for Claude
SYSTEM_PROMPT = """
You are an expert in construction and building systems with deep knowledge of OmniClass classification.
Your task is to write clear, concise descriptions for OmniClass codes.
Each description should explain what the item is in plain English, suitable for non-experts.
Keep descriptions factual, informative, and under 100 characters when possible.
"""

def create_client():
    """Create and return an Anthropic client."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_prompt(batch_data):
    """
    Generate a prompt for the Claude API based on the batch data.

    Args:
        batch_data: DataFrame containing OmniClass codes and titles

    Returns:
        str: Formatted prompt for the Claude API
    """
    prompt_items = []
    for _, row in batch_data.iterrows():
        prompt_items.append(f"Code: {row['OmniClass_Code']}, Title: {row['OmniClass_Title']}")

    prompt = f"""
    Write brief, clear descriptions for these OmniClass codes.
    Each description should be 1-2 sentences explaining what the item is in plain English.
    Format your response as a JSON array of strings, with each string being a description.

    Here are the items:
    {chr(10).join(prompt_items)}
    """
    return prompt

def call_claude_api(client, prompt):
    """
    Call the Claude API with retry logic.

    Args:
        client: Anthropic client
        prompt: Prompt for the Claude API

    Returns:
        str: Response from the Claude API
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                temperature=0.2,  # Low temperature for consistent outputs
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Error: {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed after {MAX_RETRIES} attempts: {e}")
                return None

def parse_response(response_text):
    """
    Parse the response from the Claude API.

    Args:
        response_text: Response text from the Claude API

    Returns:
        list: List of descriptions
    """
    try:
        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            print("Could not extract JSON from response")
            return []
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []

def generate_descriptions(input_file, output_file=None, start_index=0, end_index=None):
    """
    Generate descriptions for OmniClass codes.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: input_file with '_with_descriptions' suffix)
        start_index: Index to start processing from (default: 0)
        end_index: Index to end processing at (default: None, process all rows)

    Returns:
        DataFrame: DataFrame with generated descriptions
    """
    # Set default output file if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_descriptions{input_path.suffix}")

    # Load the CSV
    df = pd.read_csv(input_file)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows")

    # Set end_index to total_rows if not provided
    if end_index is None:
        end_index = total_rows

    # Create Anthropic client
    client = create_client()

    # Process in batches
    for i in tqdm(range(start_index, end_index, BATCH_SIZE)):
        batch = df.iloc[i:min(i+BATCH_SIZE, end_index)].copy()

        # Skip rows that already have descriptions
        batch = batch[batch['Description'].isna() | (batch['Description'] == '')]

        if len(batch) == 0:
            continue

        # Generate prompt
        prompt = generate_prompt(batch)

        # Call Claude API
        response_text = call_claude_api(client, prompt)
        if response_text is None:
            continue

        # Parse response
        descriptions = parse_response(response_text)

        # Update the dataframe
        if descriptions:
            for idx, desc in zip(batch.index, descriptions):
                df.at[idx, 'Description'] = desc

        # Save progress periodically
        if i % (BATCH_SIZE * 10) == 0:
            df.to_csv(output_file, index=False)
            print(f"Progress saved: {i+len(batch)}/{total_rows} rows processed")

        # Rate limiting - be nice to the API
        time.sleep(1)

    # Save the final result
    df.to_csv(output_file, index=False)
    print(f"Processing complete! Output saved to {output_file}")

    return df

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate descriptions for OmniClass codes')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, help='Path to the output CSV file')
    parser.add_argument('--start', type=int, default=0, help='Index to start processing from')
    parser.add_argument('--end', type=int, help='Index to end processing at')

    args = parser.parse_args()

    generate_descriptions(args.input, args.output, args.start, args.end)

if __name__ == "__main__":
    main()
```

### 2. Create a Runner Script

Create a new file
`fca_dashboard/examples/omniclass_description_generator_example.py` with the
following content:

```python
"""
Example script for generating descriptions for OmniClass codes.

This script demonstrates how to use the OmniClass description generator to generate
descriptions for OmniClass codes using the Claude API.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fca_dashboard.utils.omniclass_description_generator import generate_descriptions
from fca_dashboard.utils.path_util import resolve_path
from fca_dashboard.utils.logging_config import get_logger

def main():
    """Main function."""
    logger = get_logger("omniclass_description_generator_example")

    # Define input and output files
    input_file = resolve_path("fca_dashboard/generator/ingest/omniclass.csv")
    output_file = resolve_path("fca_dashboard/generator/ingest/omniclass_with_descriptions.csv")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Generate descriptions
    logger.info("Generating descriptions for OmniClass codes...")
    df = generate_descriptions(input_file, output_file)

    logger.info(f"Generated descriptions for {len(df)} rows")
    logger.info(f"Output file saved to: {output_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 3. Running the Description Generator

1. Ensure you have the required packages installed:

```bash
pip install pandas anthropic python-dotenv tqdm
```

2. Create a `.env` file in the project root directory with your Anthropic API
   key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

3. Run the description generator:

```bash
python fca_dashboard/examples/omniclass_description_generator_example.py
```

4. For more control, you can use command-line arguments:

```bash
python -m fca_dashboard.utils.omniclass_description_generator --input fca_dashboard/generator/ingest/omniclass.csv --output fca_dashboard/generator/ingest/omniclass_with_descriptions.csv --start 0 --end 100
```

### 4. Batch Processing Strategies

For large datasets, consider these batch processing strategies:

1. **Process in chunks**: Process a subset of the data at a time to manage
   memory usage and allow for easier recovery if the process is interrupted.

```bash
# Process first 1000 rows
python -m fca_dashboard.utils.omniclass_description_generator --input omniclass.csv --output omniclass_with_descriptions.csv --start 0 --end 1000

# Process next 1000 rows
python -m fca_dashboard.utils.omniclass_description_generator --input omniclass_with_descriptions.csv --output omniclass_with_descriptions.csv --start 1000 --end 2000
```

2. **Distributed processing**: Split the data into multiple files and process
   them in parallel on different machines.

3. **Scheduled processing**: Use a task scheduler (e.g., cron) to run the script
   at regular intervals to avoid hitting API rate limits.

## System Prompt Guidelines

The system prompt is crucial for getting high-quality descriptions. The current
system prompt is:

```
You are an expert in construction and building systems with deep knowledge of OmniClass classification.
Your task is to write clear, concise descriptions for OmniClass codes.
Each description should explain what the item is in plain English, suitable for non-experts.
Keep descriptions factual, informative, and under 100 characters when possible.
```

Guidelines for modifying the system prompt:

1. **Be specific about domain expertise**: Clearly state the domain expertise
   required (construction, building systems, OmniClass).
2. **Define the output format**: Specify that descriptions should be clear,
   concise, and in plain English.
3. **Set length constraints**: Specify a character limit to keep descriptions
   concise.
4. **Maintain consistency**: Ensure the system prompt encourages consistent
   formatting across all descriptions.

## Cost Optimization

To optimize costs when using the Claude API:

1. **Use Claude Haiku**: This is the cheapest Claude model and is sufficient for
   generating short descriptions.
2. **Batch processing**: Process multiple items in a single API call to reduce
   the number of calls.
3. **Low temperature**: Set temperature=0.2 to keep responses concise and
   predictable.
4. **Skip existing descriptions**: Only generate descriptions for rows that
   don't already have them.
5. **Efficient prompting**: Keep prompts concise while providing enough context.

## Troubleshooting

1. **API rate limits**: If you encounter rate limit errors, increase the sleep
   time between API calls.
2. **JSON parsing errors**: If the response cannot be parsed as JSON, check the
   raw response and adjust the prompt to encourage proper JSON formatting.
3. **Missing descriptions**: If some rows don't get descriptions, check if the
   batch size is too large or if there are issues with the prompt.

## References

- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Claude Models and Pricing](https://www.anthropic.com/api)
- [OmniClass Classification System](https://www.csiresources.org/standards/omniclass)
