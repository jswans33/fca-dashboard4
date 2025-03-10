"""
Example script for running multiple pipelines with the pipeline utility.

This script demonstrates how to:
1. Run multiple pipelines in sequence
2. Use the pipeline utility to clear output directories
3. Preserve database files when clearing output directories
4. Run pipelines with different options
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fca_dashboard.pipelines import MedtronicsPipeline, WichitaPipeline
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.pipeline_util import clear_output_directory, get_pipeline_output_dir


def run_pipeline(pipeline, clear_output=False, preserve_db=True):
    """
    Run a pipeline with the specified options.
    
    Args:
        pipeline: The pipeline instance to run
        clear_output: Whether to clear the output directory before running
        preserve_db: Whether to preserve database files when clearing output
        
    Returns:
        The results of the pipeline run
    """
    logger = get_logger("pipeline_runner")
    
    pipeline_name = pipeline.__class__.__name__
    logger.info(f"Running {pipeline_name}")
    
    # Clear the output directory if requested
    if clear_output:
        pipeline_id = pipeline.pipeline_name
        output_dir = get_pipeline_output_dir(pipeline_id)
        
        logger.info(f"Clearing output directory: {output_dir}")
        deleted_files = clear_output_directory(
            pipeline_id,
            preserve_extensions=[".db"] if preserve_db else []
        )
        
        if deleted_files:
            logger.info(f"Deleted {len(deleted_files)} files")
            for file in deleted_files[:5]:
                logger.info(f"  - {file}")
            if len(deleted_files) > 5:
                logger.info(f"  - ... and {len(deleted_files) - 5} more")
    
    # Run the pipeline
    results = pipeline.run()
    
    # Log the results
    if results["status"] == "success":
        logger.info(f"{pipeline_name} completed successfully")
        logger.info(f"Processed {results['data']['rows']} rows")
        logger.info(f"Database: {results['data']['db_path']}")
        
        # Log staging information if available
        if "staging" in results["data"]:
            staging = results["data"]["staging"]
            logger.info(f"Staging database: {staging['db_path']}")
            logger.info(f"Batch ID: {staging['batch_id']}")
            logger.info(f"Rows staged: {staging['rows_staged']}")
    else:
        logger.error(f"{pipeline_name} failed: {results['message']}")
    
    return results


def main():
    """Run the example."""
    logger = get_logger("pipeline_runner_example")
    logger.info("Starting pipeline runner example")
    
    try:
        # Run the Medtronics pipeline
        logger.info("\n" + "=" * 50)
        logger.info("Running Medtronics pipeline")
        logger.info("=" * 50)
        
        medtronics_pipeline = MedtronicsPipeline()
        medtronics_results = run_pipeline(medtronics_pipeline, clear_output=True, preserve_db=True)
        
        # Run the Wichita pipeline
        logger.info("\n" + "=" * 50)
        logger.info("Running Wichita pipeline")
        logger.info("=" * 50)
        
        wichita_pipeline = WichitaPipeline()
        wichita_results = run_pipeline(wichita_pipeline, clear_output=True, preserve_db=True)
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("Pipeline Runner Summary")
        logger.info("=" * 50)
        
        logger.info(f"Medtronics pipeline: {medtronics_results['status']}")
        if medtronics_results["status"] == "success":
            logger.info(f"  Rows: {medtronics_results['data']['rows']}")
            logger.info(f"  Database: {medtronics_results['data']['db_path']}")
        
        logger.info(f"Wichita pipeline: {wichita_results['status']}")
        if wichita_results["status"] == "success":
            logger.info(f"  Rows: {wichita_results['data']['rows']}")
            logger.info(f"  Database: {wichita_results['data']['db_path']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running pipelines: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())