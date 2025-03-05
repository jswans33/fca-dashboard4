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

## SOLID Principles and Best Practices

The current implementation works well but can be improved by applying SOLID
principles and best practices. This section outlines a refactored approach that
enhances maintainability, testability, and adherence to software engineering
principles while leveraging existing utility modules.

### SOLID Principles Assessment

#### Single Responsibility Principle (SRP)

The current `generate_descriptions` function has multiple responsibilities
(loading data, batching, API calling, and saving results). Breaking this down
into smaller, single-responsibility methods would improve maintainability.

#### Open/Closed Principle (OCP)

The current implementation is tightly coupled to Claude API specifics. Adding
abstraction layers would allow changing APIs or prompt methods without altering
core logic.

#### Liskov Substitution Principle (LSP)

Not directly violated but closely tied to the Open/Closed principle. With proper
abstraction, different generators could be easily substituted.

#### Interface Segregation Principle (ISP)

The current implementation doesn't have large interfaces that violate this
principle.

#### Dependency Inversion Principle (DIP)

The code directly depends on concrete implementations (Anthropic client).
Implementing dependency injection would improve testability and flexibility.

### Refactored Implementation

The following refactored implementation leverages existing utility modules in
the `fca_dashboard/utils/` directory while applying SOLID principles.

#### 1. API Client (`fca_dashboard/utils/api_client.py`)

```python
"""
API client utilities for external API interactions.

This module provides base classes and implementations for interacting with
external APIs, with features like retry logic, error handling, and logging.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.env_utils import get_env_var

class ApiClientError(FCADashboardError):
    """Exception raised for API client errors."""
    pass

class ApiClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, logger_name: str = "api_client"):
        """
        Initialize the API client.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = get_logger(logger_name)

    @abstractmethod
    def call(self, *args, **kwargs) -> Any:
        """
        Make an API call.

        Args:
            *args: Positional arguments for the API call
            **kwargs: Keyword arguments for the API call

        Returns:
            API response

        Raises:
            ApiClientError: If the API call fails
        """
        pass

class AnthropicClient(ApiClient):
    """Client for interacting with the Anthropic API."""

    def __init__(self, api_key: Optional[str] = None, logger_name: str = "anthropic_client"):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key (if None, will be loaded from environment)
            logger_name: Name for the logger instance
        """
        super().__init__(logger_name)

        # Import here to avoid requiring anthropic for all utils
        import anthropic

        # Get API key from environment if not provided
        self.api_key = api_key or get_env_var("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ApiClientError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.logger.debug("Anthropic client initialized")

    def call(self,
             prompt: str,
             system_prompt: str,
             model: str = "claude-haiku-v1",
             max_tokens: int = 1024,
             temperature: float = 0.2,
             retries: int = 3,
             delay: int = 5) -> Optional[str]:
        """
        Call the Anthropic API with retry logic.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            retries: Number of retries
            delay: Delay between retries in seconds

        Returns:
            Response text or None if all retries fail

        Raises:
            ApiClientError: If the API call fails after all retries
        """
        for attempt in range(retries):
            try:
                self.logger.debug(f"Making API call to Anthropic (attempt {attempt+1}/{retries})")

                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )

                self.logger.debug("API call successful")
                return response.content[0].text

            except Exception as e:
                if attempt < retries - 1:
                    self.logger.warning(f"API call failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"API call failed after {retries} attempts: {str(e)}")
                    return None
```

#### 2. Description Generator (`fca_dashboard/utils/description_generator.py`)

```python
"""
Description generator utilities for generating descriptions using LLMs.

This module provides base classes and implementations for generating descriptions
for various types of data using large language models.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.utils.api_client import AnthropicClient, ApiClientError
from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.json_utils import json_deserialize
from fca_dashboard.utils.logging_config import get_logger

class DescriptionGeneratorError(FCADashboardError):
    """Exception raised for description generator errors."""
    pass

class DescriptionGenerator(ABC):
    """Abstract base class for description generators."""

    def __init__(self, logger_name: str = "description_generator"):
        """
        Initialize the description generator.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = get_logger(logger_name)

    @abstractmethod
    def generate(self, data: pd.DataFrame, **kwargs) -> List[Optional[str]]:
        """
        Generate descriptions for the given data.

        Args:
            data: DataFrame containing data to generate descriptions for
            **kwargs: Additional keyword arguments

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If description generation fails
        """
        pass

class OmniClassDescriptionGenerator(DescriptionGenerator):
    """Generator for OmniClass descriptions using Claude API."""

    # Default system prompt
    DEFAULT_SYSTEM_PROMPT = """
    You are an expert in construction and building systems with deep knowledge of OmniClass classification.
    Your task is to write clear, concise descriptions for OmniClass codes.
    Each description should explain what the item is in plain English, suitable for non-experts.
    Keep descriptions factual, informative, and under 100 characters when possible.
    """

    def __init__(self,
                 api_client: Optional[AnthropicClient] = None,
                 system_prompt: Optional[str] = None,
                 model: str = "claude-haiku-v1",
                 logger_name: str = "omniclass_description_generator"):
        """
        Initialize the OmniClass description generator.

        Args:
            api_client: Anthropic API client (if None, a new one will be created)
            system_prompt: System prompt for Claude (if None, uses default)
            model: Model to use
            logger_name: Name for the logger instance
        """
        super().__init__(logger_name)

        self.api_client = api_client or AnthropicClient()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.model = model

        self.logger.debug("OmniClass description generator initialized")

    def generate_prompt(self, data: pd.DataFrame) -> str:
        """
        Generate a prompt for the Claude API based on the data.

        Args:
            data: DataFrame containing OmniClass codes and titles

        Returns:
            Formatted prompt for the Claude API
        """
        prompt_items = []
        for _, row in data.iterrows():
            prompt_items.append(f"Code: {row['OmniClass_Code']}, Title: {row['OmniClass_Title']}")

        prompt = f"""
        Write brief, clear descriptions for these OmniClass codes.
        Each description should be 1-2 sentences explaining what the item is in plain English.
        Format your response as a JSON array of strings, with each string being a description.

        Here are the items:
        {chr(10).join(prompt_items)}
        """
        return prompt

    def parse_response(self, response_text: str) -> List[Optional[str]]:
        """
        Parse the response from the Claude API.

        Args:
            response_text: Response text from the Claude API

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If parsing fails
        """
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json_deserialize(json_match.group(0), [])
            else:
                self.logger.warning("Could not extract JSON from response")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            return []

    def generate(self, data: pd.DataFrame, **kwargs) -> List[Optional[str]]:
        """
        Generate descriptions for OmniClass codes.

        Args:
            data: DataFrame containing OmniClass codes and titles
            **kwargs: Additional keyword arguments

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If description generation fails
        """
        if data.empty:
            self.logger.warning("Empty DataFrame provided, returning empty list")
            return []

        # Check required columns
        required_columns = ['OmniClass_Code', 'OmniClass_Title']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DescriptionGeneratorError(f"Missing required columns: {missing_columns}")

        # Generate prompt
        prompt = self.generate_prompt(data)

        # Call API
        try:
            response_text = self.api_client.call(
                prompt=prompt,
                system_prompt=self.system_prompt,
                model=self.model
            )

            if response_text is None:
                self.logger.warning("API call returned None")
                return [None] * len(data)

            # Parse response
            descriptions = self.parse_response(response_text)

            # If we got fewer descriptions than expected, pad with None
            if len(descriptions) < len(data):
                self.logger.warning(f"Got {len(descriptions)} descriptions for {len(data)} items")
                descriptions.extend([None] * (len(data) - len(descriptions)))

            return descriptions

        except ApiClientError as e:
            self.logger.error(f"API client error: {str(e)}")
            raise DescriptionGeneratorError(f"Failed to generate descriptions: {str(e)}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise DescriptionGeneratorError(f"Failed to generate descriptions: {str(e)}") from e
```

#### 3. Batch Processor (`fca_dashboard/utils/batch_processor.py`)

```python
"""
Batch processing utilities for handling large datasets.

This module provides utilities for processing large datasets in batches,
with features like progress tracking, error handling, and logging.
"""

import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import pandas as pd
from tqdm import tqdm

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger

T = TypeVar('T')

class BatchProcessorError(FCADashboardError):
    """Exception raised for batch processor errors."""
    pass

class BatchProcessor:
    """Processor for batch processing data."""

    def __init__(self,
                 processor_func: Callable[[pd.DataFrame], List[T]],
                 batch_size: int = 50,
                 logger_name: str = "batch_processor"):
        """
        Initialize the batch processor.

        Args:
            processor_func: Function to process each batch
            batch_size: Size of batches to process
            logger_name: Name for the logger instance
        """
        self.processor_func = processor_func
        self.batch_size = batch_size
        self.logger = get_logger(logger_name)

        self.logger.debug(f"Batch processor initialized with batch size {batch_size}")

    def process(self,
               df: pd.DataFrame,
               target_column: str,
               filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
               start: int = 0,
               end: Optional[int] = None,
               save_callback: Optional[Callable[[pd.DataFrame], None]] = None,
               save_interval: int = 10,
               delay: float = 1.0) -> pd.DataFrame:
        """
        Process data in batches.

        Args:
            df: DataFrame to process
            target_column: Column to store results in
            filter_func: Function to filter rows in each batch (if None, processes all rows)
            start: Index to start processing from
            end: Index to end processing at
            save_callback: Callback function to save progress
            save_interval: Number of batches between saves
            delay: Delay between batches in seconds

        Returns:
            Processed DataFrame

        Raises:
            BatchProcessorError: If batch processing fails
        """
        end = end or len(df)
        result_df = df.copy()

        self.logger.info(f"Processing {end - start} rows in batches of {self.batch_size}")

        try:
            for i in tqdm(range(start, end, self.batch_size)):
                batch = result_df.iloc[i:min(i+self.batch_size, end)].copy()

                # Apply filter if provided
                if filter_func:
                    batch_to_process = filter_func(batch)
                else:
                    batch_to_process = batch

                if batch_to_process.empty:
                    self.logger.debug(f"Batch {i//self.batch_size + 1} is empty after filtering, skipping")
                    continue

                self.logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_to_process)} rows")

                # Process batch
                try:
                    results = self.processor_func(batch_to_process)

                    # Update the dataframe
                    for idx, result in zip(batch_to_process.index, results):
                        if result is not None:
                            result_df.at[idx, target_column] = result

                    self.logger.debug(f"Batch {i//self.batch_size + 1} processed successfully")

                except Exception as e:
                    self.logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                    # Continue with next batch

                # Save progress if callback provided
                if save_callback and i % (self.batch_size * save_interval) == 0:
                    self.logger.info(f"Saving progress after {i + len(batch)} rows")
                    save_callback(result_df)

                # Delay between batches
                if delay > 0:
                    time.sleep(delay)

            self.logger.info(f"Processing complete, processed {end - start} rows")
            return result_df

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise BatchProcessorError(f"Batch processing failed: {str(e)}") from e
```

#### 4. Refactored OmniClass Description Generator (`fca_dashboard/utils/omniclass_description_generator.py`)

```python
"""
Utility for generating descriptions for OmniClass codes using the Claude API.

This module provides functions to generate plain-English descriptions for OmniClass codes
using the Claude API. It processes the data in batches to manage API rate limits and costs.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv

from fca_dashboard.utils.api_client import AnthropicClient
from fca_dashboard.utils.batch_processor import BatchProcessor
from fca_dashboard.utils.description_generator import OmniClassDescriptionGenerator
from fca_dashboard.utils.error_handler import ErrorHandler, FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path

# Load environment variables from .env file
load_dotenv()

# Constants
DEFAULT_BATCH_SIZE = 50
DEFAULT_MODEL = "claude-haiku-v1"

# Initialize logger
logger = get_logger("omniclass_description_generator")

# Initialize error handler
error_handler = ErrorHandler("omniclass_description_generator")

class OmniClassDescriptionGeneratorError(FCADashboardError):
    """Exception raised for OmniClass description generator errors."""
    pass

@error_handler.with_error_handling
def generate_descriptions(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    description_column: str = 'Description'
) -> pd.DataFrame:
    """
    Generate descriptions for OmniClass codes.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: input_file with '_with_descriptions' suffix)
        start_index: Index to start processing from (default: 0)
        end_index: Index to end processing at (default: None, process all rows)
        batch_size: Size of batches to process (default: 50)
        model: Claude model to use (default: "claude-haiku-v1")
        system_prompt: System prompt for Claude (default: None, uses default prompt)
        description_column: Column to store descriptions in (default: 'Description')

    Returns:
        DataFrame: DataFrame with generated descriptions

    Raises:
        OmniClassDescriptionGeneratorError: If description generation fails
    """
    try:
        # Resolve paths
        input_path = resolve_path(input_file)

        # Set default output file if not provided
        if output_file is None:
            output_path = Path(input_path).parent / f"{Path(input_path).stem}_with_descriptions{Path(input_path).suffix}"
        else:
            output_path = resolve_path(output_file)

        logger.info(f"Input file: {input_path}")
        logger.info(f"Output file: {output_path}")

        # Load the CSV
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        total_rows = len(df)
        logger.info(f"Loaded {total_rows} rows")

        # Set end_index to total_rows if not provided
        if end_index is None:
            end_index = total_rows

        # Create API client
        api_client = AnthropicClient()

        # Create description generator
        description_generator = OmniClassDescriptionGenerator(
            api_client=api_client,
            system_prompt=system_prompt,
            model=model
        )

        # Define filter function to skip rows that already have descriptions
        def filter_empty_descriptions(batch_df: pd.DataFrame) -> pd.DataFrame:
            return batch_df[batch_df[description_column].isna() | (batch_df[description_column] == '')]

        # Define save callback
        def save_progress(current_df: pd.DataFrame) -> None:
            current_df.to_csv(output_path, index=False)
            logger.info(f"Progress saved to {output_path}")

        # Create batch processor
        batch_processor = BatchProcessor(
            processor_func=description_generator.generate,
            batch_size=batch_size
        )

        # Process data
        logger.info(f"Processing data from index {start_index} to {end_index}")
        result_df = batch_processor.process(
            df=df,
            target_column=description_column,
            filter_func=filter_empty_descriptions,
            start=start_index,
            end=end_index,
            save_callback=save_progress
        )

        # Save final result
        result_df.to_csv(output_path, index=False)
        logger.info(f"Processing complete! Output saved to {output_path}")

        return result_df

    except Exception as e:
        logger.error(f"Failed to generate descriptions: {str(e)}")
        raise OmniClassDescriptionGeneratorError(f"Failed to generate descriptions: {str(e)}") from e

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate descriptions for OmniClass codes')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, help='Path to the output CSV file')
    parser.add_argument('--start', type=int, default=0, help='Index to start processing from')
    parser.add_argument('--end', type=int, help='Index to end processing at')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Claude model to use')

    args = parser.parse_args()

    generate_descriptions(
        input_file=args.input,
        output_file=args.output,
        start_index=args.start,
        end_index=args.end,
        batch_size=args.batch_size,
        model=args.model
    )

if __name__ == "__main__":
    main()
```

#### 5. Updated Example Script (`fca_dashboard/examples/omniclass_description_generator_example.py`)

```python
"""
Example script for generating descriptions for OmniClass codes.

This script demonstrates how to use the refactored OmniClass description generator
to generate descriptions for OmniClass codes using the Claude API.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fca_dashboard.utils.omniclass_description_generator import generate_descriptions
from fca_dashboard.utils.path_util import resolve_path
from fca_dashboard.utils.logging_config import get_logger, configure_logging
from fca_dashboard.utils.error_handler import ErrorHandler

# Configure logging
configure_logging(level="INFO")

# Initialize logger
logger = get_logger("omniclass_description_generator_example")

# Initialize error handler
error_handler = ErrorHandler("omniclass_description_generator_example")

@error_handler.with_error_handling
def main():
    """Main function."""
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

### Benefits of the Refactored Implementation

The refactored implementation offers several advantages:

1. **Better Separation of Concerns**:

   - `AnthropicClient` handles API communication
   - `OmniClassDescriptionGenerator` handles description generation
   - `BatchProcessor` handles batch processing
   - `omniclass_description_generator.py` provides a high-level interface

2. **Improved Error Handling**:

   - Uses the existing `error_handler.py` module
   - Custom exception classes for each component
   - Detailed error messages and logging

3. **Enhanced Logging**:

   - Uses the existing `logging_config.py` module
   - Consistent logging across all components
   - Different log levels for different types of messages

4. **Dependency Injection**:

   - Components accept dependencies through constructor parameters
   - Makes testing easier by allowing mock objects
   - Follows the Dependency Inversion Principle

5. **Abstraction Layers**:

   - Abstract base classes for API clients and description generators
   - Allows for easy substitution of different implementations
   - Follows the Open/Closed Principle

6. **Configuration Management**:

   - Uses environment variables for API keys
   - Command-line arguments for runtime configuration
   - Default values for optional parameters

7. **Integration with Existing Utils**:
   - Uses `path_util.py` for path resolution
   - Uses `json_utils.py` for JSON operations
   - Uses `env_utils.py` for environment variables

## Additional Recommendations

### Testing

Implement unit tests for individual components to ensure robustness:

```python
# Example test for AnthropicClient
import unittest
from unittest.mock import patch, MagicMock
from fca_dashboard.utils.api_client import AnthropicClient, ApiClientError

class TestAnthropicClient(unittest.TestCase):
    @patch('anthropic.Anthropic')
    def test_call_success(self, mock_anthropic):
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_client.messages.create.return_value = mock_response

        # Create client and call API
        client = AnthropicClient("test_key")
        result = client.call("test prompt", "test system prompt")

        # Assert
        self.assertEqual(result, "Test response")
        mock_client.messages.create.assert_called_once()

    @patch('anthropic.Anthropic')
    @patch('time.sleep')
    def test_call_retry(self, mock_sleep, mock_anthropic):
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [
            Exception("Test error"),
            MagicMock(content=[MagicMock(text="Test response")])
        ]

        # Create client and call API
        client = AnthropicClient("test_key")
        result = client.call("test prompt", "test system prompt")

        # Assert
        self.assertEqual(result, "Test response")
        self.assertEqual(mock_client.messages.create.call_count, 2)
        mock_sleep.assert_called_once()
```

### Configuration Management

Use a configuration file (e.g., YAML or JSON) for more flexible configuration:

```python
# Add to fca_dashboard/config/settings.yml
omniclass_description_generator:
  api:
    model: "claude-haiku-v1"
    max_tokens: 1024
    temperature: 0.2
    retries: 3
    delay: 5
  processing:
    batch_size: 50
    save_interval: 10
    description_column: "Description"
  system_prompt: |
    You are an expert in construction and building systems with deep knowledge of OmniClass classification.
    Your task is to write clear, concise descriptions for OmniClass codes.
    Each description should explain what the item is in plain English, suitable for non-experts.
    Keep descriptions factual, informative, and under 100 characters when possible.
```

Then load the configuration in your code:

```python
from fca_dashboard.config.settings import settings

# Get configuration
config = settings.get("omniclass_description_generator", {})
api_config = config.get("api", {})
processing_config = config.get("processing", {})

# Use configuration
model = api_config.get("model", DEFAULT_MODEL)
batch_size = processing_config.get("batch_size", DEFAULT_BATCH_SIZE)
system_prompt = config.get("system_prompt")
```

### Performance Optimization

Consider these performance optimizations:

1. **Parallel Processing**: Use Python's `concurrent.futures` to process
   multiple batches in parallel:

```python
import concurrent.futures
from typing import List, Tuple

def process_batch_parallel(batches: List[pd.DataFrame], max_workers: int = 4) -> List[Tuple[pd.DataFrame, List[str]]]:
    """Process batches in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches to the executor
        future_to_batch = {
            executor.submit(description_generator.generate, batch): batch
            for batch in batches
        }

        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                descriptions = future.result()
                results.append((batch, descriptions))
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                results.append((batch, [None] * len(batch)))

        return results
```

2. **Caching**: Cache API responses to avoid redundant calls:

```python
import functools
import hashlib

@functools.lru_cache(maxsize=1000)
def cached_api_call(prompt_hash: str, system_prompt: str, model: str) -> Optional[str]:
    """Cached API call."""
    # Use the hash to look up the original prompt
    prompt = prompt_cache.get(prompt_hash)
    if not prompt:
        return None

    return api_client.call(prompt, system_prompt, model)

# When making API calls
prompt = generate_prompt(batch_data)
prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
prompt_cache[prompt_hash] = prompt  # Store the original prompt
response_text = cached_api_call(prompt_hash, system_prompt, model)
```

## References

- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Claude Models and Pricing](https://www.anthropic.com/api)
- [OmniClass Classification System](https://www.csiresources.org/standards/omniclass)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Python Testing Documentation](https://docs.python.org/3/library/unittest.html)
