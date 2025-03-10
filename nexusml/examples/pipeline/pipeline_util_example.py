"""
Example script demonstrating the use of pipeline utilities.

This script shows how to use the pipeline utility functions to clear output
directories and manage pipeline state.
"""

import argparse
import sys
from pathlib import Path

from fca_dashboard.utils import (
    clear_output_directory,
    get_pipeline_output_dir,
    PipelineUtilError,
    get_logger
)


def main():
    """Run the pipeline utility example."""
    # Set up logging
    logger = get_logger("pipeline_util_example")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pipeline Utility Example")
    parser.add_argument(
        "pipeline",
        choices=["medtronics", "wichita"],
        help="Pipeline name to operate on"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run (don't actually delete files)"
    )
    parser.add_argument(
        "--preserve-db",
        action="store_true",
        help="Preserve database files (.db extension)"
    )
    parser.add_argument(
        "--preserve-files",
        nargs="+",
        help="List of specific filenames to preserve"
    )
    
    args = parser.parse_args()
    
    try:
        # Get the output directory for the pipeline
        output_dir = get_pipeline_output_dir(args.pipeline)
        logger.info(f"Pipeline output directory: {output_dir}")
        
        # Prepare preserve extensions list
        preserve_extensions = []
        if args.preserve_db:
            preserve_extensions.append(".db")
        
        # Clear the output directory
        deleted_files = clear_output_directory(
            args.pipeline,
            preserve_files=args.preserve_files,
            preserve_extensions=preserve_extensions,
            dry_run=args.dry_run
        )
        
        # Print summary
        action = "Would delete" if args.dry_run else "Deleted"
        logger.info(f"{action} {len(deleted_files)} files from {output_dir}")
        
        if deleted_files:
            logger.info("Files affected:")
            for file in deleted_files[:10]:  # Show first 10 files
                logger.info(f"  - {file}")
            
            if len(deleted_files) > 10:
                logger.info(f"  ... and {len(deleted_files) - 10} more files")
        
        return 0
    
    except PipelineUtilError as e:
        logger.error(f"Pipeline utility error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())