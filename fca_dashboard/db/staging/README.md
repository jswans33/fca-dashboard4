# SQLite Staging System for FCA Dashboard

This directory contains the schema and utilities for the SQLite staging system used in the FCA Dashboard application.

## Overview

The staging system provides a temporary storage area for incoming data before it is transformed and loaded into the master schema tables. It consists of:

1. A single comprehensive staging table (`equipment_staging`) to hold all incoming data
2. Views for pending and error items
3. Utility functions for managing the staging data

## Schema Files

- `staging_schema.sql`: PostgreSQL version of the staging schema
- `staging_schema_sqlite.sql`: SQLite version of the staging schema

## Key Differences Between PostgreSQL and SQLite Implementations

| Feature | PostgreSQL | SQLite |
|---------|------------|--------|
| Data Types | SERIAL, TIMESTAMPTZ, JSONB | INTEGER PRIMARY KEY AUTOINCREMENT, TIMESTAMP, TEXT |
| Indexes | Supports GIN indexes for JSON | No GIN indexes, uses regular indexes |
| Functions | Stored procedures in plpgsql | Implemented in Python code |
| Schema | Uses schema namespace | No schema namespace |
| Boolean | Native boolean type | Uses INTEGER (0=false, 1=true) |

## Using the SQLite Staging System

The SQLite staging system can be used in two ways:

1. **Function-based API** (for backward compatibility)
2. **Class-based API with Dependency Injection** (recommended for new code)

### Class-based API with Dependency Injection

The recommended way to use the SQLite staging system is through the `SQLiteStagingManager` class, which supports dependency injection:

```python
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.logging_config import get_logger

# Create a custom logger
logger = get_logger("my_custom_logger")

# Create a SQLiteStagingManager instance with the custom logger
manager = SQLiteStagingManager(logger=logger)

# Initialize the database
db_path = "path/to/staging.db"
manager.initialize_db(db_path)

# Create a DataFrame with your data
import pandas as pd
df = pd.DataFrame({
    'equipment_tag': ['EQ-001', 'EQ-002'],
    'manufacturer': ['Manufacturer A', 'Manufacturer B'],
    # ... other columns ...
})

# SQLite connection string
connection_string = f"sqlite:///{db_path}"

# Save to staging table
manager.save_dataframe_to_staging(
    df=df,
    connection_string=connection_string,
    source_system='Example System',
    import_batch_id='BATCH-20250304'
)

# Get pending items
pending_items = manager.get_pending_items(connection_string)

# Update item status
manager.update_item_status(
    connection_string=connection_string,
    staging_id=1,
    status='COMPLETED',
    is_processed=True
)

# Reset error items
reset_count = manager.reset_error_items(connection_string)

# Clear processed items older than 7 days
cleared_count = manager.clear_processed_items(connection_string, days_to_keep=7)
```

### Function-based API (for backward compatibility)

For backward compatibility, the SQLite staging system can also be used through a set of functions:

```python
from fca_dashboard.utils.database.sqlite_staging_manager import (
    initialize_sqlite_staging_db,
    save_dataframe_to_staging,
    get_pending_items,
    update_item_status,
    reset_error_items,
    clear_processed_items
)

# Initialize the database
db_path = "path/to/staging.db"
initialize_sqlite_staging_db(db_path)

# Create a DataFrame with your data
import pandas as pd
df = pd.DataFrame({
    'equipment_tag': ['EQ-001', 'EQ-002'],
    'manufacturer': ['Manufacturer A', 'Manufacturer B'],
    # ... other columns ...
})

# SQLite connection string
connection_string = f"sqlite:///{db_path}"

# Save to staging table
save_dataframe_to_staging(
    df=df,
    connection_string=connection_string,
    source_system='Example System',
    import_batch_id='BATCH-20250304'
)

# Get pending items
pending_items = get_pending_items(connection_string)

# Update item status
update_item_status(
    connection_string=connection_string,
    staging_id=1,
    status='COMPLETED',
    is_processed=True
)

# Reset error items
reset_count = reset_error_items(connection_string)

# Clear processed items older than 7 days
cleared_count = clear_processed_items(connection_string, days_to_keep=7)
```

## Example Script

See `fca_dashboard/examples/sqlite_staging_example.py` for a complete example of how to use the SQLite staging system with both the function-based API and the class-based API.

## Testing

The SQLite staging system has comprehensive unit and integration tests:

- Unit tests: `fca_dashboard/tests/unit/test_sqlite_staging_manager.py`
- Integration tests: `fca_dashboard/tests/integration/test_sqlite_staging_integration.py`

You can run these tests using the following command:

```bash
make test-sqlite-staging
```

## Integration with ETL Pipeline

The staging system is designed to be integrated with the ETL (Extract, Transform, Load) pipeline:

1. **Extract**: Data is extracted from source systems and loaded into the staging table
2. **Transform**: Data in the staging table is transformed and validated
3. **Load**: Transformed data is loaded into the master schema tables

The staging table provides a buffer between the extraction and transformation/loading phases, allowing for better error handling and recovery.

## Transformation Phase

For the transformation phase, you can use the `staging_to_master_sqlite.py` module, which provides Python functions that replicate the functionality of the PostgreSQL stored procedures for the ETL process.

```python
from fca_dashboard.db.staging.procedures.staging_to_master_sqlite import process_staging_data

# SQLite connection string
connection_string = f"sqlite:///{db_path}"

# Process staging data
result = process_staging_data(
    connection_string=connection_string,
    batch_id='BATCH-20250304',
    limit_rows=1000
)

print(f"Processed {result['processed_count']} records, {result['error_count']} errors")
```