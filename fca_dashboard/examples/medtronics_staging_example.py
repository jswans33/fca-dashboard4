"""
Example script demonstrating how to use the SQLite staging system with the Medtronics pipeline.

This script shows how to:
1. Run the Medtronics pipeline with staging enabled
2. Query the staging database for pending items
3. Process the pending items
4. Update the status of processed items
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.pipelines.pipeline_medtronics import MedtronicsPipeline
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path

# Create a logger
logger = get_logger("medtronics_staging_example")


def run_pipeline_with_staging():
    """Run the Medtronics pipeline with staging enabled."""
    logger.info("Running Medtronics pipeline with staging enabled")
    
    # Create and run the pipeline
    pipeline = MedtronicsPipeline()
    
    # Make sure staging is enabled
    pipeline.staging_config["enabled"] = True
    
    # Run the pipeline
    results = pipeline.run()
    
    # Check if the pipeline was successful
    if results["status"] != "success":
        logger.error(f"Pipeline failed: {results['message']}")
        return None
    
    # Return the staging results
    if "staging" in results["data"]:
        logger.info("Pipeline completed successfully with staging")
        return results["data"]["staging"]
    else:
        logger.warning("Pipeline completed successfully but staging information is not available")
        return None


def query_pending_items(staging_results):
    """Query the staging database for pending items."""
    if not staging_results:
        logger.error("No staging results available")
        return None
    
    logger.info(f"Querying pending items from staging database: {staging_results['db_path']}")
    
    # Create a SQLiteStagingManager
    staging_manager = SQLiteStagingManager()
    
    # Get the connection string
    connection_string = staging_results["connection_string"]
    
    # Get pending items
    pending_items = staging_manager.get_pending_items(connection_string)
    
    logger.info(f"Found {len(pending_items)} pending items")
    
    return pending_items


def process_pending_items(staging_results, pending_items):
    """Process the pending items."""
    if not staging_results or pending_items is None or len(pending_items) == 0:
        logger.error("No pending items to process")
        return
    
    logger.info(f"Processing {len(pending_items)} pending items")
    
    # Create a SQLiteStagingManager
    staging_manager = SQLiteStagingManager()
    
    # Get the connection string
    connection_string = staging_results["connection_string"]
    
    # Process each pending item
    for index, item in pending_items.iterrows():
        staging_id = item["staging_id"]
        equipment_tag = item.get("equipment_tag", "Unknown")
        
        logger.info(f"Processing item {staging_id} (Equipment Tag: {equipment_tag})")
        
        try:
            # Update the status to PROCESSING
            staging_manager.update_item_status(
                connection_string=connection_string,
                staging_id=staging_id,
                status="PROCESSING"
            )
            
            # Simulate processing
            logger.info(f"Simulating processing for item {staging_id}")
            
            # In a real application, you would perform actual processing here
            # For example, transforming the data, validating it, etc.
            
            # For demonstration purposes, we'll just mark the item as COMPLETED
            staging_manager.update_item_status(
                connection_string=connection_string,
                staging_id=staging_id,
                status="COMPLETED",
                is_processed=True
            )
            
            logger.info(f"Item {staging_id} processed successfully")
        except Exception as e:
            logger.error(f"Error processing item {staging_id}: {str(e)}")
            
            # Mark the item as ERROR
            staging_manager.update_item_status(
                connection_string=connection_string,
                staging_id=staging_id,
                status="ERROR",
                error_message=str(e)
            )


def clear_old_processed_items(staging_results):
    """Clear old processed items from the staging database."""
    if not staging_results:
        logger.error("No staging results available")
        return
    
    logger.info(f"Clearing old processed items from staging database: {staging_results['db_path']}")
    
    # Create a SQLiteStagingManager
    staging_manager = SQLiteStagingManager()
    
    # Get the connection string
    connection_string = staging_results["connection_string"]
    
    # Get the retention days from settings
    retention_days = settings.get("databases.staging.processed_retention_days", 30)
    
    # Clear processed items
    cleared_count = staging_manager.clear_processed_items(
        connection_string=connection_string,
        days_to_keep=retention_days
    )
    
    logger.info(f"Cleared {cleared_count} processed items older than {retention_days} days")


def main():
    """Main function."""
    try:
        # Run the pipeline with staging
        staging_results = run_pipeline_with_staging()
        
        if not staging_results:
            logger.error("No staging results available")
            return 1
        
        # Print staging information
        print("\nStaging Information:")
        print(f"  Staging Database: {staging_results['db_path']}")
        print(f"  Batch ID: {staging_results['batch_id']}")
        print(f"  Source System: {staging_results['source_system']}")
        print(f"  Rows Staged: {staging_results['rows_staged']}")
        print(f"  Pending Items: {staging_results['pending_items']}")
        
        # Query pending items
        pending_items = query_pending_items(staging_results)
        
        if pending_items is not None and len(pending_items) > 0:
            # Process pending items
            process_pending_items(staging_results, pending_items)
        
        # Clear old processed items
        clear_old_processed_items(staging_results)
        
        print("\nStaging process completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())