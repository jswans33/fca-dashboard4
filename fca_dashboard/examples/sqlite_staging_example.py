"""
Example script demonstrating how to use the SQLite staging manager.

This script shows how to:
1. Initialize a SQLite staging database
2. Load data into the staging table
3. Query pending items
4. Update item status
5. Reset error items
6. Clear processed items

This example demonstrates both the function-based API (for backward compatibility)
and the new class-based API with dependency injection.
"""

import os
import pandas as pd
from datetime import datetime

# Import the SQLiteStagingManager class
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.logging_config import get_logger

logger = get_logger("sqlite_staging_example")


def example_with_functions():
    """Example using the function-based API (for backward compatibility)."""
    # Import the function-based API
    from fca_dashboard.utils.database.sqlite_staging_manager import (
        initialize_sqlite_staging_db,
        save_dataframe_to_staging,
        get_pending_items,
        update_item_status,
        reset_error_items,
        clear_processed_items
    )
    
    # Define the path to the SQLite database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'outputs', 'staging_functions.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # SQLite connection string
    connection_string = f"sqlite:///{db_path}"
    
    # Initialize the SQLite staging database
    logger.info("Initializing SQLite staging database...")
    initialize_sqlite_staging_db(db_path)
    
    # Create a sample DataFrame with equipment data
    logger.info("Creating sample data...")
    data = {
        'equipment_tag': ['EQ-001', 'EQ-002', 'EQ-003'],
        'manufacturer': ['Manufacturer A', 'Manufacturer B', 'Manufacturer C'],
        'model': ['Model X', 'Model Y', 'Model Z'],
        'serial_number': ['SN123456', 'SN789012', 'SN345678'],
        'capacity': [100.0, 200.0, 300.0],
        'install_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'status': ['Active', 'Active', 'Inactive'],
        'category_name': ['HVAC', 'Electrical', 'Plumbing'],
        'building_name': ['Building A', 'Building B', 'Building C'],
        'floor': ['1', '2', '3'],
        'room': ['101', '202', '303'],
        'attributes': [
            {'color': 'red', 'weight': 50},
            {'color': 'blue', 'weight': 75},
            {'color': 'green', 'weight': 100}
        ]
    }
    df = pd.DataFrame(data)
    
    # Save the DataFrame to the staging table
    logger.info("Saving data to staging table...")
    save_dataframe_to_staging(
        df=df,
        connection_string=connection_string,
        source_system='Example System',
        import_batch_id=f'BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    )
    
    # Get pending items
    logger.info("Getting pending items...")
    pending_items = get_pending_items(connection_string)
    logger.info(f"Found {len(pending_items)} pending items")
    
    # Update item status
    if not pending_items.empty:
        # Update the first item to 'PROCESSING'
        first_item_id = pending_items.iloc[0]['staging_id']
        logger.info(f"Updating item {first_item_id} to 'PROCESSING'...")
        update_item_status(
            connection_string=connection_string,
            staging_id=first_item_id,
            status='PROCESSING'
        )
        
        # Update the second item to 'COMPLETED' and mark as processed
        if len(pending_items) > 1:
            second_item_id = pending_items.iloc[1]['staging_id']
            logger.info(f"Updating item {second_item_id} to 'COMPLETED' and marking as processed...")
            update_item_status(
                connection_string=connection_string,
                staging_id=second_item_id,
                status='COMPLETED',
                is_processed=True
            )
        
        # Update the third item to 'ERROR'
        if len(pending_items) > 2:
            third_item_id = pending_items.iloc[2]['staging_id']
            logger.info(f"Updating item {third_item_id} to 'ERROR'...")
            update_item_status(
                connection_string=connection_string,
                staging_id=third_item_id,
                status='ERROR',
                error_message='Example error message'
            )
    
    # Reset error items
    logger.info("Resetting error items...")
    reset_count = reset_error_items(connection_string)
    logger.info(f"Reset {reset_count} error items")
    
    # Clear processed items (for demonstration, using 0 days to keep all processed items)
    logger.info("Clearing processed items...")
    cleared_count = clear_processed_items(connection_string, days_to_keep=0)
    logger.info(f"Cleared {cleared_count} processed items")
    
    logger.info("Function-based example completed successfully!")
    logger.info(f"SQLite staging database created at: {db_path}")


def example_with_class():
    """Example using the class-based API with dependency injection."""
    # Define the path to the SQLite database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'outputs', 'staging_class.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # SQLite connection string
    connection_string = f"sqlite:///{db_path}"
    
    # Create a custom logger
    custom_logger = get_logger("sqlite_staging_class_example")
    
    # Create a SQLiteStagingManager instance with the custom logger
    manager = SQLiteStagingManager(logger=custom_logger)
    
    # Initialize the SQLite staging database
    custom_logger.info("Initializing SQLite staging database...")
    manager.initialize_db(db_path)
    
    # Create a sample DataFrame with equipment data
    custom_logger.info("Creating sample data...")
    data = {
        'equipment_tag': ['EQ-001', 'EQ-002', 'EQ-003'],
        'manufacturer': ['Manufacturer A', 'Manufacturer B', 'Manufacturer C'],
        'model': ['Model X', 'Model Y', 'Model Z'],
        'serial_number': ['SN123456', 'SN789012', 'SN345678'],
        'capacity': [100.0, 200.0, 300.0],
        'install_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'status': ['Active', 'Active', 'Inactive'],
        'category_name': ['HVAC', 'Electrical', 'Plumbing'],
        'building_name': ['Building A', 'Building B', 'Building C'],
        'floor': ['1', '2', '3'],
        'room': ['101', '202', '303'],
        'attributes': [
            {'color': 'red', 'weight': 50},
            {'color': 'blue', 'weight': 75},
            {'color': 'green', 'weight': 100}
        ]
    }
    df = pd.DataFrame(data)
    
    # Save the DataFrame to the staging table
    custom_logger.info("Saving data to staging table...")
    manager.save_dataframe_to_staging(
        df=df,
        connection_string=connection_string,
        source_system='Example System',
        import_batch_id=f'BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    )
    
    # Get pending items
    custom_logger.info("Getting pending items...")
    pending_items = manager.get_pending_items(connection_string)
    custom_logger.info(f"Found {len(pending_items)} pending items")
    
    # Update item status
    if not pending_items.empty:
        # Update the first item to 'PROCESSING'
        first_item_id = pending_items.iloc[0]['staging_id']
        custom_logger.info(f"Updating item {first_item_id} to 'PROCESSING'...")
        manager.update_item_status(
            connection_string=connection_string,
            staging_id=first_item_id,
            status='PROCESSING'
        )
        
        # Update the second item to 'COMPLETED' and mark as processed
        if len(pending_items) > 1:
            second_item_id = pending_items.iloc[1]['staging_id']
            custom_logger.info(f"Updating item {second_item_id} to 'COMPLETED' and marking as processed...")
            manager.update_item_status(
                connection_string=connection_string,
                staging_id=second_item_id,
                status='COMPLETED',
                is_processed=True
            )
        
        # Update the third item to 'ERROR'
        if len(pending_items) > 2:
            third_item_id = pending_items.iloc[2]['staging_id']
            custom_logger.info(f"Updating item {third_item_id} to 'ERROR'...")
            manager.update_item_status(
                connection_string=connection_string,
                staging_id=third_item_id,
                status='ERROR',
                error_message='Example error message'
            )
    
    # Reset error items
    custom_logger.info("Resetting error items...")
    reset_count = manager.reset_error_items(connection_string)
    custom_logger.info(f"Reset {reset_count} error items")
    
    # Clear processed items (for demonstration, using 0 days to keep all processed items)
    custom_logger.info("Clearing processed items...")
    cleared_count = manager.clear_processed_items(connection_string, days_to_keep=0)
    custom_logger.info(f"Cleared {cleared_count} processed items")
    
    custom_logger.info("Class-based example completed successfully!")
    custom_logger.info(f"SQLite staging database created at: {db_path}")


def main():
    """Run both examples."""
    # Run the function-based example
    logger.info("Running function-based example...")
    example_with_functions()
    
    # Run the class-based example
    logger.info("Running class-based example...")
    example_with_class()
    
    logger.info("All examples completed successfully!")


if __name__ == "__main__":
    main()