"""
Utility for generating descriptions for OmniClass codes using the Claude API.

This module provides functions to generate plain-English descriptions for OmniClass codes
using the Claude API. It processes the data in batches to manage API rate limits and costs.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anthropic
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


# Define custom error classes
class NexusMLError(Exception):
    """Base exception for NexusML errors."""

    pass


class ApiClientError(NexusMLError):
    """Exception raised for API client errors."""

    pass


class DescriptionGeneratorError(NexusMLError):
    """Exception raised for description generator errors."""

    pass


# Load settings from config file if available
def load_settings():
    """
    Load settings from the config file.

    Returns:
        dict: Settings dictionary
    """
    try:
        # Try to load from fca_dashboard settings if available
        try:
            from fca_dashboard.config.settings import settings

            return settings
        except ImportError:
            # Not running in fca_dashboard context, load from local config
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
            if config_path.exists():
                with open(config_path, "r") as file:
                    return yaml.safe_load(file)
            else:
                return {}
    except Exception:
        return {}


# Initialize settings
settings = load_settings()

# Import utilities if available, otherwise define minimal versions
try:
    from fca_dashboard.utils.logging_config import get_logger
    from fca_dashboard.utils.path_util import resolve_path
except ImportError:
    # Define minimal versions of required functions
    def get_logger(name):
        """Simple logger function."""
        import logging

        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

    def resolve_path(path):
        """Resolve a path to an absolute path."""
        if isinstance(path, str):
            path = Path(path)
        return path.resolve()


# Load settings from config file
config = settings.get("generator", {}).get("omniclass_description_generator", {})
api_config = config.get("api", {})

# Constants with defaults from config
BATCH_SIZE = config.get("batch_size", 50)
MODEL = api_config.get("model", "claude-3-haiku-20240307")
MAX_RETRIES = api_config.get("retries", 3)
RETRY_DELAY = api_config.get("delay", 5)
DEFAULT_INPUT_FILE = config.get("input_file", "nexusml/ingest/generator/data/omniclass.csv")
DEFAULT_OUTPUT_FILE = config.get("output_file", "nexusml/ingest/generator/data/omniclass_with_descriptions.csv")
DEFAULT_DESCRIPTION_COLUMN = config.get("description_column", "Description")

# System prompt for Claude
SYSTEM_PROMPT = config.get(
    "system_prompt",
    """
You are an expert in construction and building systems with deep knowledge of OmniClass classification.
Your task is to write clear, concise descriptions for OmniClass codes that will be used in a classification model.
Each description should:
1. Explain what the item is in plain English, suitable for non-experts
2. Include distinctive features that would help a model differentiate between similar categories
3. Be factual, informative, and under 100 characters when possible
4. Use consistent terminology across related items to help the model recognize patterns
5. Highlight the hierarchical relationship to parent categories when relevant

These descriptions will serve as training data for a machine learning model to classify construction elements.
""",
)

# Initialize logger
logger = get_logger("omniclass_description_generator")


class ApiClient(ABC):
    """Abstract base class for API clients."""

    @abstractmethod
    def call(self, prompt: str, system_prompt: str, **kwargs) -> Optional[str]:
        """
        Make an API call.

        Args:
            prompt: The prompt to send to the API
            system_prompt: The system prompt to use
            **kwargs: Additional keyword arguments for the API call

        Returns:
            The API response text or None if the call fails

        Raises:
            ApiClientError: If the API call fails
        """
        pass


class AnthropicClient(ApiClient):
    """Client for the Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.

        Args:
            api_key: The API key to use. If None, uses the ANTHROPIC_API_KEY environment variable.

        Raises:
            ApiClientError: If the API key is not provided and not found in environment variables
        """
        # Get API key from environment variables if not provided
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ApiClientError("ANTHROPIC_API_KEY environment variable not set")

        # Create the client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.debug("Anthropic client initialized")

    def call(
        self, prompt: str, system_prompt: str, model: str = MODEL, max_tokens: int = 1024, temperature: float = 0.2
    ) -> Optional[str]:
        """
        Call the Anthropic API with retry logic.

        Args:
            prompt: The prompt to send to the API
            system_prompt: The system prompt to use
            model: The model to use
            max_tokens: The maximum number of tokens to generate
            temperature: The temperature to use for generation

        Returns:
            The API response text or None if all retries fail

        Raises:
            ApiClientError: If the API call fails after all retries
        """
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Making API call to Anthropic (attempt {attempt + 1}/{MAX_RETRIES})")

                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )

                logger.debug("API call successful")
                return response.content[0].text

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"API call failed: {str(e)}. Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"API call failed after {MAX_RETRIES} attempts: {str(e)}")
                    return None


class DescriptionGenerator(ABC):
    """Abstract base class for description generators."""

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> List[Optional[str]]:
        """
        Generate descriptions for the given data.

        Args:
            data: DataFrame containing data to generate descriptions for

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If description generation fails
        """
        pass


class OmniClassDescriptionGenerator(DescriptionGenerator):
    """Generator for OmniClass descriptions using Claude API."""

    def __init__(self, api_client: Optional[ApiClient] = None, system_prompt: Optional[str] = None):
        """
        Initialize the OmniClass description generator.

        Args:
            api_client: The API client to use. If None, creates a new AnthropicClient.
            system_prompt: The system prompt to use. If None, uses the default SYSTEM_PROMPT.
        """
        self.api_client = api_client or AnthropicClient()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        logger.debug("OmniClass description generator initialized")

    def generate_prompt(self, data: pd.DataFrame) -> str:
        """
        Generate a prompt for the API based on the data.

        Args:
            data: DataFrame containing OmniClass codes and titles

        Returns:
            Formatted prompt for the API
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
        Parse the response from the API.

        Args:
            response_text: Response text from the API

        Returns:
            List of descriptions
        """
        try:
            # Extract JSON array from response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.warning("Could not extract JSON from response")
                return []
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return []

    def generate(self, data: pd.DataFrame) -> List[Optional[str]]:
        """
        Generate descriptions for OmniClass codes.

        Args:
            data: DataFrame containing OmniClass codes and titles

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If description generation fails
        """
        if data.empty:
            logger.warning("Empty DataFrame provided, returning empty list")
            return []

        # Check required columns
        required_columns = ["OmniClass_Code", "OmniClass_Title"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DescriptionGeneratorError(f"Missing required columns: {missing_columns}")

        # Generate prompt
        prompt = self.generate_prompt(data)

        # Call API
        try:
            response_text = self.api_client.call(prompt=prompt, system_prompt=self.system_prompt)

            if response_text is None:
                logger.warning("API call returned None")
                return [None] * len(data)

            # Parse response
            descriptions = self.parse_response(response_text)

            # If we got fewer descriptions than expected, pad with None
            if len(descriptions) < len(data):
                logger.warning(f"Got {len(descriptions)} descriptions for {len(data)} items")
                descriptions.extend([None] * (len(data) - len(descriptions)))

            return descriptions

        except ApiClientError as e:
            logger.error(f"API client error: {str(e)}")
            raise DescriptionGeneratorError(f"Failed to generate descriptions: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise DescriptionGeneratorError(f"Failed to generate descriptions: {str(e)}") from e


class BatchProcessor:
    """Processor for batch processing data."""

    def __init__(self, generator: DescriptionGenerator, batch_size: int = BATCH_SIZE):
        """
        Initialize the batch processor.

        Args:
            generator: The description generator to use
            batch_size: The size of batches to process
        """
        self.generator = generator
        self.batch_size = batch_size
        logger.debug(f"Batch processor initialized with batch size {batch_size}")

    def process(
        self,
        df: pd.DataFrame,
        description_column: str = "Description",
        start_index: int = 0,
        end_index: Optional[int] = None,
        save_callback: Optional[callable] = None,
        save_interval: int = 10,
    ) -> pd.DataFrame:
        """
        Process data in batches.

        Args:
            df: DataFrame to process
            description_column: Column to store descriptions in
            start_index: Index to start processing from
            end_index: Index to end processing at
            save_callback: Callback function to save progress
            save_interval: Number of batches between saves

        Returns:
            Processed DataFrame
        """
        end_index = end_index or len(df)
        result_df = df.copy()

        logger.info(f"Processing {end_index - start_index} rows in batches of {self.batch_size}")

        try:
            for i in tqdm(range(start_index, end_index, self.batch_size)):
                batch = result_df.iloc[i : min(i + self.batch_size, end_index)].copy()

                # Process all rows regardless of existing descriptions
                batch_to_process = batch

                if batch_to_process.empty:
                    logger.debug(f"Batch {i // self.batch_size + 1} is empty after filtering, skipping")
                    continue

                logger.debug(f"Processing batch {i // self.batch_size + 1} with {len(batch_to_process)} rows")

                # Process batch
                try:
                    descriptions = self.generator.generate(batch_to_process)

                    # Update the dataframe
                    for idx, desc in zip(batch_to_process.index, descriptions):
                        if desc is not None:
                            # Convert column to string type if needed to avoid dtype warning
                            if pd.api.types.is_numeric_dtype(result_df[description_column].dtype):
                                result_df[description_column] = result_df[description_column].astype(str)
                            result_df.at[idx, description_column] = desc

                    logger.debug(f"Batch {i // self.batch_size + 1} processed successfully")

                except Exception as e:
                    logger.error(f"Error processing batch {i // self.batch_size + 1}: {str(e)}")
                    # Continue with next batch

                # Save progress if callback provided
                if save_callback and i % (self.batch_size * save_interval) == 0:
                    logger.info(f"Saving progress after {i + len(batch)} rows")
                    save_callback(result_df)

                # No rate limiting for Tier 4 API access
                # time.sleep(1)

            logger.info(f"Processing complete, processed {end_index - start_index} rows")
            return result_df

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise DescriptionGeneratorError(f"Batch processing failed: {str(e)}") from e


# Convenience functions for backward compatibility and ease of use


def create_client() -> anthropic.Anthropic:
    """Create and return an Anthropic client."""
    return AnthropicClient().client


def generate_prompt(batch_data: pd.DataFrame) -> str:
    """
    Generate a prompt for the Claude API based on the batch data.

    Args:
        batch_data: DataFrame containing OmniClass codes and titles

    Returns:
        str: Formatted prompt for the Claude API
    """
    return OmniClassDescriptionGenerator().generate_prompt(batch_data)


def call_claude_api(client: anthropic.Anthropic, prompt: str) -> Optional[str]:
    """
    Call the Claude API with retry logic.

    Args:
        client: Anthropic client
        prompt: Prompt for the Claude API

    Returns:
        str: Response from the Claude API
    """
    api_client = AnthropicClient(api_key=client.api_key)
    return api_client.call(prompt=prompt, system_prompt=SYSTEM_PROMPT)


def parse_response(response_text: str) -> List[Optional[str]]:
    """
    Parse the response from the Claude API.

    Args:
        response_text: Response text from the Claude API

    Returns:
        list: List of descriptions
    """
    return OmniClassDescriptionGenerator().parse_response(response_text)


def generate_descriptions(
    input_file: Union[str, Path] = DEFAULT_INPUT_FILE,
    output_file: Optional[Union[str, Path]] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    description_column: str = DEFAULT_DESCRIPTION_COLUMN,
) -> pd.DataFrame:
    """
    Generate descriptions for OmniClass codes.

    Args:
        input_file: Path to the input CSV file (default from config)
        output_file: Path to the output CSV file (default from config or input_file with '_with_descriptions' suffix)
        start_index: Index to start processing from (default: 0)
        end_index: Index to end processing at (default: None, process all rows)
        batch_size: Size of batches to process (default from config)
        description_column: Column to store descriptions in (default from config)

    Returns:
        DataFrame: DataFrame with generated descriptions

    Raises:
        DescriptionGeneratorError: If description generation fails
    """
    try:
        # Resolve paths
        input_path = resolve_path(input_file)

        # Set default output file if not provided
        if output_file is None:
            output_path = (
                Path(input_path).parent / f"{Path(input_path).stem}_with_descriptions{Path(input_path).suffix}"
            )
        else:
            output_path = resolve_path(output_file)

        logger.info(f"Input file: {input_path}")
        logger.info(f"Output file: {output_path}")

        # Create the output directory if it doesn't exist
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load the CSV
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        total_rows = len(df)
        logger.info(f"Loaded {total_rows} rows")

        # Create generator and processor
        generator = OmniClassDescriptionGenerator()
        processor = BatchProcessor(generator, batch_size=batch_size)

        # Define save callback
        def save_progress(current_df: pd.DataFrame) -> None:
            current_df.to_csv(output_path, index=False)
            logger.info(f"Progress saved to {output_path}")

        # Process data
        logger.info(f"Processing data from index {start_index} to {end_index or total_rows}")
        result_df = processor.process(
            df=df,
            description_column=description_column,
            start_index=start_index,
            end_index=end_index,
            save_callback=save_progress,
        )

        # Save final result
        result_df.to_csv(output_path, index=False)
        logger.info(f"Processing complete! Output saved to {output_path}")

        return result_df

    except Exception as e:
        logger.error(f"Failed to generate descriptions: {str(e)}")
        raise DescriptionGeneratorError(f"Failed to generate descriptions: {str(e)}") from e


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate descriptions for OmniClass codes")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input CSV file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to the output CSV file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument("--start", type=int, default=0, help="Index to start processing from")
    parser.add_argument("--end", type=int, help="Index to end processing at")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument(
        "--description-column",
        type=str,
        default=DEFAULT_DESCRIPTION_COLUMN,
        help=f"Column to store descriptions in (default: {DEFAULT_DESCRIPTION_COLUMN})",
    )
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to process")

    args = parser.parse_args()

    # Use max-rows as end_index if provided
    end_index = args.max_rows if args.max_rows is not None else args.end

    generate_descriptions(
        input_file=args.input,
        output_file=args.output,
        start_index=args.start,
        end_index=end_index,
        batch_size=args.batch_size,
        description_column=args.description_column,
    )


if __name__ == "__main__":
    main()
