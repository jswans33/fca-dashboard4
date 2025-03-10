"""
Example script for generating descriptions for OmniClass codes.

This script demonstrates how to use the OmniClass description generator to generate
descriptions for OmniClass codes using the Claude API.

Before running this script, make sure you have:
1. Set up your ANTHROPIC_API_KEY environment variable or created a .env file
2. Installed required packages: pandas, anthropic, python-dotenv, tqdm
3. Prepared your OmniClass data in CSV format

Usage:
    python omniclass_description_generator_example.py [options]

Options:
    --dry-run       Run in dry run mode (doesn't call the API)
    --max-rows N    Process only N rows from the input file
    --sample-size N Number of rows to process in dry run mode (default: 100)
    --input-file F  Path to input CSV file
    --output-file F Path to output CSV file
"""

import os
import sys
import json
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import from the compatibility layer
from fca_dashboard.generator import (
    generate_descriptions, ApiClientError, DescriptionGeneratorError,
    OmniClassDescriptionGenerator, AnthropicClient
)

# Update default paths to use nexusml
DEFAULT_INPUT_FILE = "nexusml/ingest/generator/data/omniclass.csv"
DEFAULT_OUTPUT_FILE = "nexusml/ingest/generator/data/omniclass_with_descriptions.csv"
from fca_dashboard.utils.path_util import resolve_path
from fca_dashboard.utils.logging_config import get_logger, configure_logging
from fca_dashboard.config.settings import settings

# Load environment variables from .env file
load_dotenv()

# Load settings from config file
config = settings.get('generator', {}).get('omniclass_description_generator', {})

# Configure logging
configure_logging(level="INFO")

# Initialize logger
logger = get_logger("omniclass_description_generator_example")


def check_api_key():
    """Check if the ANTHROPIC_API_KEY environment variable is set."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is not set!")
        logger.error("Please set the ANTHROPIC_API_KEY environment variable or create a .env file with your API key.")
        logger.error("Example .env file content:")
        logger.error("ANTHROPIC_API_KEY=your_api_key_here")
        return False
    return True


def setup_mock_api_client():
    """Set up a mock API client for dry run mode."""
    # Create a mock API client that returns fake descriptions
    mock_client = MagicMock()
    
    # Make the mock return a dynamic number of descriptions based on the batch size
    def mock_call(*args, **kwargs):
        # Extract the prompt to determine how many items are in the batch
        prompt = args[0] if args else kwargs.get('prompt', '')
        # Count the number of "Code:" occurrences to estimate batch size
        count = prompt.count('Code:')
        # Generate that many descriptions
        descriptions = [f"Sample description for item {i+1}" for i in range(count)]
        return json.dumps(descriptions)
    
    mock_client.call.side_effect = mock_call
    
    # Patch the AnthropicClient to return our mock
    return patch.object(AnthropicClient, '__new__', return_value=mock_client)


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate descriptions for OmniClass codes')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry run mode (no API calls)')
    parser.add_argument('--start', type=int, default=0, help='Index to start processing from')
    parser.add_argument('--end', type=int, default=None, help='Index to end processing at (exclusive)')
    parser.add_argument('--max-rows', type=int, default=None, help='Maximum number of rows to process')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of rows to process in dry run mode')
    parser.add_argument('--input-file', type=str, default=DEFAULT_INPUT_FILE, help=f'Path to input CSV file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output-file', type=str, default=DEFAULT_OUTPUT_FILE, help=f'Path to output CSV file (default: {DEFAULT_OUTPUT_FILE})')
    args = parser.parse_args()
    
    # Check if API key is set (only if not in dry run mode)
    if not args.dry_run and not check_api_key():
        return 1

    # Define input and output files
    input_file = resolve_path(args.input_file)
    output_file = resolve_path(args.output_file)

    # Ensure the input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please make sure the input file exists or update the path.")
        return 1

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Determine end_index based on arguments
    end_index = args.end
    
    # If end is not specified, calculate it from other parameters
    if end_index is None:
        if args.dry_run and not args.max_rows:
            end_index = args.sample_size
            logger.info(f"Dry run mode: Processing only {end_index} rows")
        elif args.max_rows:
            if args.start > 0:
                end_index = args.start + args.max_rows
                logger.info(f"Processing rows {args.start} to {end_index} ({args.max_rows} rows)")
            else:
                end_index = args.max_rows
                logger.info(f"Processing limited to {end_index} rows")
    else:
        # End index was explicitly provided
        rows_to_process = end_index - args.start
        logger.info(f"Processing rows {args.start} to {end_index} ({rows_to_process} rows)")

    # Generate descriptions
    try:
        if args.dry_run:
            logger.info("Running in DRY RUN mode (no API calls will be made)")
            with setup_mock_api_client():
                logger.info("Generating sample descriptions for OmniClass codes...")
                df = generate_descriptions(input_file, output_file, start_index=args.start, end_index=end_index)
                logger.info(f"Generated sample descriptions for {len(df[:end_index]) if end_index else len(df)} rows")
                logger.info(f"Output file saved to: {output_file}")
        else:
            logger.info("Generating descriptions for OmniClass codes using Claude API...")
            df = generate_descriptions(input_file, output_file, start_index=args.start, end_index=end_index)
            logger.info(f"Generated descriptions for {len(df[:end_index]) if end_index else len(df)} rows")
            logger.info(f"Output file saved to: {output_file}")
        
        return 0
        
    except ApiClientError as e:
        logger.error(f"API client error: {str(e)}")
        logger.error("Please check your API key and internet connection.")
        return 1
        
    except DescriptionGeneratorError as e:
        logger.error(f"Description generator error: {str(e)}")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())