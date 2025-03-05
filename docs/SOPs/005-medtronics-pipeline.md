# SOP 005: Medtronics Pipeline and Mapper

## 1. Overview

This Standard Operating Procedure (SOP) document provides detailed information
about the Medtronics ETL (Extract, Transform, Load) pipeline and the dynamic
mapping system. The pipeline extracts data from Medtronics Excel files, maps the
data to a standardized schema, and loads it into a staging database for further
processing.

## 2. Pipeline Architecture

The Medtronics pipeline consists of the following components:

1. **Extraction**: Reads data from the Medtronics Excel file
2. **Mapping**: Transforms the data to match the staging schema
3. **Loading**: Saves the transformed data to the staging database
4. **Analysis & Validation**: Performs data quality checks and generates reports

### 2.1 File Locations

| Component               | File Path                                                   |
| ----------------------- | ----------------------------------------------------------- |
| Pipeline Implementation | `fca_dashboard/pipelines/pipeline_medtronics.py`            |
| Mapper Implementation   | `fca_dashboard/mappers/medtronics_mapper.py`                |
| Staging Manager         | `fca_dashboard/utils/database/sqlite_staging_manager.py`    |
| Configuration           | `fca_dashboard/config/settings.yml`                         |
| Staging Schema          | `fca_dashboard/db/staging/schema/staging_schema_sqlite.sql` |

### 2.2 Data Flow

```
Medtronics Excel File → Extraction → Mapping → Staging Database → Analysis & Validation
```

## 3. Running the Pipeline

### 3.1 Prerequisites

- Python 3.8+
- Required packages installed (see `requirements.txt`)
- Medtronics Excel file in the correct location

### 3.2 Execution

To run the Medtronics pipeline:

```bash
python -m fca_dashboard.pipelines.pipeline_medtronics
```

### 3.3 Configuration

The pipeline is configured through the `settings.yml` file. Key configuration
sections:

```yaml
# Medtronics pipeline settings
medtronics:
  input_file: 'uploads/Medtronics - Asset Log Uploader.xlsx'
  output_dir: 'outputs/pipeline/medtronic'
  db_name: 'medtronics_assets.db'
  sheet_name: 'Asset Data'
  columns_to_extract: ['A', 'B', 'C', ...] # Excel column letters
  drop_na_columns: ['asset name'] # Drop rows where these columns have NaN values
  # Staging configuration
  staging:
    enabled: true
    db_path: 'outputs/pipeline/medtronic/staging.db'
    source_system: 'Medtronics'
    batch_id_prefix: 'MEDTRONICS-BATCH-'

# Mapper settings
mappers:
  medtronics:
    column_mapping:
      asset_name: 'equipment_type'
      asset_tag: 'equipment_tag'
      # ... other explicit mappings
```

## 4. Dynamic Mapping System

### 4.1 How the Mapper Works

The Medtronics mapper uses a combination of explicit and dynamic mapping:

1. **Explicit Mapping**: Columns defined in the
   `mappers.medtronics.column_mapping` section of `settings.yml` are mapped
   directly to specific fields in the staging database.

2. **Dynamic Mapping**: Any columns in the source data that don't have explicit
   mappings are automatically stored in the `attributes` JSON field in the
   staging database.

### 4.2 Attributes JSON Structure

The `attributes` field uses the following structure:

```json
{
  "medtronics_attributes": {
    "column_name_1": { ... values by row index ... },
    "column_name_2": { ... values by row index ... },
    ...
  }
}
```

### 4.3 Adding New Explicit Mappings

To add a new explicit mapping:

1. Open `fca_dashboard/config/settings.yml`
2. Navigate to the `mappers.medtronics.column_mapping` section
3. Add a new mapping in the format: `source_column: 'destination_column'`

Example:

```yaml
mappers:
  medtronics:
    column_mapping:
      asset_name: 'equipment_type'
      new_column: 'new_destination' # Add this line
```

## 5. Troubleshooting

### 5.1 Common Issues and Solutions

#### Issue: Pipeline fails during extraction

**Symptoms:**

- Error message about file not found
- Error message about sheet not found

**Solutions:**

1. Check if the Excel file exists at the path specified in `settings.yml`
2. Verify the sheet name in `settings.yml` matches the actual sheet name in the
   Excel file
3. Check the log for specific error messages

#### Issue: Columns are not being mapped correctly

**Symptoms:**

- Missing data in the staging database
- Data appears in the wrong columns

**Solutions:**

1. Check the column mapping in `settings.yml`
2. Verify the column names in the Excel file match the source column names in
   the mapping
3. Check the log for warnings about unmapped columns
4. Examine the `attributes` field to see if data is being stored there instead

#### Issue: Data validation errors

**Symptoms:**

- Pipeline completes but reports validation errors
- Missing or incorrect data in the staging database

**Solutions:**

1. Check the validation report at
   `outputs/pipeline/medtronic/asset_data_validation_report.txt`
2. Verify the data types in the Excel file match the expected types
3. Check for missing required values in the source data

### 5.2 Logging and Debugging

The pipeline generates detailed logs that can help with troubleshooting:

1. **Console Output**: When running the pipeline, check the console output for
   errors and warnings
2. **Log Files**: Check the log files in the `logs` directory
3. **Analysis Report**: Review
   `outputs/pipeline/medtronic/asset_data_analysis_report.txt`
4. **Validation Report**: Review
   `outputs/pipeline/medtronic/asset_data_validation_report.txt`

### 5.3 Examining the Staging Database

To examine the staging database directly:

```bash
# List all tables
sqlite3 outputs/pipeline/medtronic/staging.db ".tables"

# View schema of the equipment_staging table
sqlite3 outputs/pipeline/medtronic/staging.db ".schema equipment_staging"

# Query the most recent batch
sqlite3 outputs/pipeline/medtronic/staging.db "SELECT * FROM equipment_staging ORDER BY import_timestamp DESC LIMIT 10;"

# Check for pending items
sqlite3 outputs/pipeline/medtronic/staging.db "SELECT COUNT(*) FROM equipment_staging WHERE processing_status = 'PENDING';"

# Examine the attributes field
sqlite3 outputs/pipeline/medtronic/staging.db "SELECT attributes FROM equipment_staging WHERE attributes IS NOT NULL LIMIT 1;"
```

## 6. Advanced Topics

### 6.1 Customizing the Mapper

The `MedtronicsMapper` class can be customized by modifying
`fca_dashboard/mappers/medtronics_mapper.py`. Key methods:

- `_load_column_mapping()`: Loads the mapping configuration
- `map_dataframe()`: Performs the actual mapping

### 6.2 Adding New Validation Rules

To add new validation rules:

1. Open `fca_dashboard/config/settings.yml`
2. Navigate to the `excel_utils.validation` section
3. Add new validation rules under the appropriate subsection

Example:

```yaml
excel_utils:
  validation:
    asset_data:
      value_ranges:
        new_column:
          min: 0
          max: 100
```

### 6.3 Extending the Pipeline

To extend the pipeline with new functionality:

1. Modify `fca_dashboard/pipelines/pipeline_medtronics.py`
2. Add new methods to the `MedtronicsPipeline` class
3. Update the `run()` method to include the new steps

## 7. Maintenance

### 7.1 Clearing Processed Items

To clear processed items from the staging database:

```python
from fca_dashboard.utils.database.sqlite_staging_manager import clear_processed_items

# Clear items processed more than 30 days ago
clear_processed_items("sqlite:///outputs/pipeline/medtronic/staging.db", days_to_keep=30)
```

### 7.2 Resetting Error Items

To reset items that encountered errors:

```python
from fca_dashboard.utils.database.sqlite_staging_manager import reset_error_items

# Reset all error items to 'PENDING'
reset_error_items("sqlite:///outputs/pipeline/medtronic/staging.db")
```

## 8. Migrating from SQLite to PostgreSQL

The current implementation uses SQLite for the staging database. Migrating to
PostgreSQL would provide better performance, concurrency, and enterprise-level
features. This section outlines the steps required to migrate from SQLite to
PostgreSQL.

### 8.1 Required Changes

The codebase has been designed with database abstraction in mind, making the
migration relatively straightforward. Here are the key changes needed:

1. **Schema Definition**:

   - Create a PostgreSQL version of the staging schema at
     `fca_dashboard/db/staging/schema/staging_schema_postgres.sql`
   - Update JSON column definitions to use PostgreSQL's native JSONB type
     instead of TEXT
   - Adjust any SQLite-specific SQL syntax to PostgreSQL syntax

2. **Configuration Updates**:

   - Update the `settings.yml` file to include PostgreSQL connection details:

   ```yaml
   databases:
     postgresql:
       url: 'postgresql://username:password@localhost:5432/fca_dashboard'
     staging:
       type: 'postgresql' # Change from 'sqlite'
       schema_path: 'fca_dashboard/db/staging/schema/staging_schema_postgres.sql'

   medtronics:
     staging:
       db_path: 'postgresql://username:password@localhost:5432/fca_dashboard'
       # Other settings remain the same
   ```

3. **Code Modifications**:
   - Update `SQLiteStagingManager` to handle PostgreSQL connections
   - Modify JSON handling to use PostgreSQL's JSONB functions
   - Update any SQLite-specific queries to use PostgreSQL syntax

### 8.2 Implementation Steps

1. **Install PostgreSQL Dependencies**:

   ```bash
   pip install psycopg2-binary
   ```

2. **Create PostgreSQL Schema**:

   - Convert the SQLite schema to PostgreSQL syntax
   - Replace TEXT JSON fields with JSONB type
   - Add appropriate indexes for PostgreSQL

3. **Update Connection String Handling**:

   - Modify the `SQLiteStagingManager` to detect and handle PostgreSQL
     connection strings
   - Update the connection factory to create the appropriate engine type

4. **Modify JSON Handling**:

   - Update code that processes JSON data to use PostgreSQL's native JSONB
     functions
   - Ensure proper escaping and formatting for PostgreSQL

5. **Update Queries**:
   - Modify any database-specific queries to use PostgreSQL syntax
   - Update any pagination or transaction handling code

### 8.3 Affected Files

Based on a thorough analysis of the codebase, the following files would need to
be modified for the migration:

#### Core Database Files

1. **`fca_dashboard/utils/database/sqlite_staging_manager.py`**

   - This is the primary file that would need to be modified or replaced with a
     PostgreSQL version
   - Contains SQLite-specific connection handling, schema initialization, and
     query execution

2. **`fca_dashboard/utils/database/base.py`**

   - Contains database abstraction logic that already has conditional handling
     for SQLite vs PostgreSQL
   - Minimal changes needed as it already supports PostgreSQL connections

3. **`fca_dashboard/utils/database/sqlite_utils.py`**

   - Contains SQLite-specific utility functions
   - A PostgreSQL equivalent would need to be created or expanded

4. **`fca_dashboard/utils/database/postgres_utils.py`**
   - Already exists but may need to be expanded to support staging operations

#### Schema Files

5. **`fca_dashboard/db/staging/schema/staging_schema_sqlite.sql`**
   - The SQLite schema definition
   - A PostgreSQL version would need to be created at
     `fca_dashboard/db/staging/schema/staging_schema_postgres.sql`

#### Pipeline Files

6. **`fca_dashboard/pipelines/pipeline_medtronics.py`**

   - Contains references to SQLite connection strings and the
     SQLiteStagingManager
   - Would need updates to use PostgreSQL connection strings

7. **`fca_dashboard/pipelines/pipeline_wichita.py`**
   - Similar to the Medtronics pipeline, contains SQLite-specific code

#### Configuration Files

8. **`fca_dashboard/config/settings.yml`**
   - Contains database configuration settings
   - Would need updates to include PostgreSQL connection details

#### Mapper Files

9. **`fca_dashboard/mappers/medtronics_mapper.py`**
   - Contains JSON handling that might need to be updated for PostgreSQL's JSONB
     type
   - Minimal changes needed as most JSON handling is in the staging manager

#### Procedure Files

10. **`fca_dashboard/db/staging/procedures/staging_to_master_sqlite.py`**
    - Contains SQLite-specific procedures for transforming data
    - A PostgreSQL version would need to be created

### 8.4 Detailed Migration Steps

The migration from SQLite to PostgreSQL can be broken down into the following
concrete steps:

#### 1. Schema Migration

1.1. **Copy Existing Schema**

- Copy `fca_dashboard/db/staging/schema/staging_schema.sql` to
  `fca_dashboard/db/staging/schema/staging_schema_postgres.sql`
- The schema is already PostgreSQL-compatible with `SERIAL`, `TIMESTAMPTZ`,
  `JSONB`, and GIN indexes

  1.2. **Verify PostgreSQL Compatibility**

- Ensure all SQL syntax is PostgreSQL-compatible
- Verify that all data types are appropriate for PostgreSQL

  1.3. **Add PostgreSQL-Specific Optimizations**

- Add additional indexes if needed for PostgreSQL performance
- Consider adding foreign key constraints if appropriate

#### 2. Database Abstraction Layer

2.1. **Create Abstract Base Class**

- Create `fca_dashboard/utils/database/base_staging_manager.py`
- Define interface with abstract methods for all staging operations:

  ```python
  class BaseStagingManager(ABC):
      @abstractmethod
      def initialize_db(self, db_path: str) -> None:
          pass

      @abstractmethod
      def save_dataframe_to_staging(self, df: pd.DataFrame, connection_string: str, source_system: str, import_batch_id: str) -> None:
          pass

      @abstractmethod
      def get_pending_items(self, connection_string: str) -> pd.DataFrame:
          pass

      # Other abstract methods...
  ```

  2.2. **Update SQLiteStagingManager**

- Modify `fca_dashboard/utils/database/sqlite_staging_manager.py` to inherit
  from the base class
- Ensure all SQLite-specific code is properly isolated:

  ```python
  class SQLiteStagingManager(BaseStagingManager):
      def initialize_db(self, db_path: str) -> None:
          # SQLite-specific implementation

      def save_dataframe_to_staging(self, df: pd.DataFrame, connection_string: str, source_system: str, import_batch_id: str) -> None:
          # SQLite-specific implementation

      # Other method implementations...
  ```

  2.3. **Create PostgreSQLStagingManager**

- Create `fca_dashboard/utils/database/postgres_staging_manager.py`
- Implement all abstract methods with PostgreSQL-specific code:

  ```python
  class PostgreSQLStagingManager(BaseStagingManager):
      def initialize_db(self, db_path: str) -> None:
          # PostgreSQL-specific implementation

      def save_dataframe_to_staging(self, df: pd.DataFrame, connection_string: str, source_system: str, import_batch_id: str) -> None:
          # PostgreSQL-specific implementation

      # Other method implementations...
  ```

- Use native JSONB functions for JSON handling

#### 3. Factory Implementation

3.1. **Create StagingManagerFactory**

- Create `fca_dashboard/utils/database/staging_manager_factory.py`
- Implement factory method to create appropriate manager based on database type:

  ```python
  class StagingManagerFactory:
      @staticmethod
      def create_manager(db_type: str, **kwargs) -> BaseStagingManager:
          if db_type == 'postgresql':
              return PostgreSQLStagingManager(**kwargs)
          else:
              return SQLiteStagingManager(**kwargs)
  ```

  3.2. **Update Database Utilities**

- Modify `fca_dashboard/utils/database/__init__.py` to expose the factory:

  ```python
  from fca_dashboard.utils.database.staging_manager_factory import StagingManagerFactory

  __all__ = [
      # Existing exports
      "StagingManagerFactory",
  ]
  ```

- Update imports in dependent modules

#### 4. Pipeline Updates

4.1. **Update Medtronics Pipeline**

- Modify `fca_dashboard/pipelines/pipeline_medtronics.py` to use the factory:

  ```python
  # Get database type from settings
  db_type = settings.get("databases.staging.type", "sqlite")

  # Create staging manager using factory
  self.staging_manager = StagingManagerFactory.create_manager(db_type)
  ```

- Update connection string handling to support PostgreSQL format:

  ```python
  if db_type == 'postgresql':
      connection_string = self.staging_config.get("db_path", "postgresql://localhost:5432/fca_dashboard")
  else:
      connection_string = f"sqlite:///{db_path}"
  ```

  4.2. **Update Wichita Pipeline**

- Make similar changes to `fca_dashboard/pipelines/pipeline_wichita.py`
- Ensure consistent approach across all pipelines

#### 5. Configuration Updates

5.1. **Update Settings Schema**

- Modify `fca_dashboard/config/settings.yml` to include database type
  configuration:

  ```yaml
  databases:
    staging:
      type: 'postgresql' # or 'sqlite'
      schema_path: 'fca_dashboard/db/staging/schema/staging_schema_postgres.sql'
      postgresql:
        host: 'localhost'
        port: 5432
        database: 'fca_dashboard'
        username: 'postgres'
        password: 'password'
  ```

  5.2. **Update Settings Loading**

- Ensure settings are properly loaded and validated
- Add defaults for backward compatibility:
  ```python
  # In settings.py or similar
  def get_database_connection_string():
      db_type = settings.get("databases.staging.type", "sqlite")
      if db_type == 'postgresql':
          host = settings.get("databases.staging.postgresql.host", "localhost")
          port = settings.get("databases.staging.postgresql.port", 5432)
          database = settings.get("databases.staging.postgresql.database", "fca_dashboard")
          username = settings.get("databases.staging.postgresql.username", "postgres")
          password = settings.get("databases.staging.postgresql.password", "password")
          return f"postgresql://{username}:{password}@{host}:{port}/{database}"
      else:
          db_path = settings.get("databases.staging.sqlite.path", "outputs/staging.db")
          return f"sqlite:///{db_path}"
  ```

#### 6. Procedure Migration

6.1. **Create PostgreSQL Procedures**

- Create `fca_dashboard/db/staging/procedures/staging_to_master_postgres.py`
- Port SQLite procedures to PostgreSQL:

  ```python
  def process_staging_data(connection_string: str, batch_id: Optional[str] = None):
      """
      Process data from the staging table to the master tables.

      Args:
          connection_string: The PostgreSQL connection string.
          batch_id: Optional batch ID to filter records.
      """
      # PostgreSQL-specific implementation
  ```

  6.2. **Update Procedure Factory**

- Create or update factory for selecting appropriate procedures:
  ```python
  def get_staging_procedure(db_type: str):
      if db_type == 'postgresql':
          from fca_dashboard.db.staging.procedures.staging_to_master_postgres import process_staging_data
      else:
          from fca_dashboard.db.staging.procedures.staging_to_master_sqlite import process_staging_data
      return process_staging_data
  ```

#### 7. Testing Framework

7.1. **Update Unit Tests**

- Modify database-related tests to support both SQLite and PostgreSQL:

  ```python
  @pytest.mark.parametrize("db_type", ["sqlite", "postgresql"])
  def test_staging_manager(db_type):
      # Create appropriate staging manager
      manager = StagingManagerFactory.create_manager(db_type)

      # Test with appropriate connection string
      if db_type == 'postgresql':
          connection_string = "postgresql://postgres:password@localhost:5432/test_db"
      else:
          connection_string = "sqlite:///test.db"

      # Run tests
      # ...
  ```

  7.2. **Create Integration Tests**

- Develop tests that verify end-to-end functionality with PostgreSQL
- Test data migration between SQLite and PostgreSQL

  7.3. **Performance Testing**

- Create benchmarks for comparing SQLite and PostgreSQL performance
- Test with various dataset sizes

### 8.5 Required PostgreSQL Utility Functions

Based on the analysis of the existing codebase in
`fca_dashboard/utils/database/`, the following PostgreSQL utility functions need
to be created:

#### 1. Create `postgres_staging_manager.py`

This file will be the PostgreSQL equivalent of `sqlite_staging_manager.py` and
should implement the following methods:

```python
class PostgreSQLStagingManager(BaseStagingManager):
    def __init__(self, logger=None, connection_factory=None, schema_path=None):
        # Similar to SQLiteStagingManager.__init__
        # But use PostgreSQL-specific schema path

    def _default_connection_factory(self, connection_string: str):
        # Create PostgreSQL connection

    def initialize_db(self, db_path: str) -> None:
        # Initialize PostgreSQL database with schema
        # Use psycopg2 instead of sqlite3

    def reset_error_items(self, connection_string: str) -> int:
        # Use PostgreSQL syntax for UPDATE

    def clear_processed_items(self, connection_string: str, days_to_keep: int = None) -> int:
        # Use PostgreSQL syntax for DELETE
        # Use PostgreSQL date functions

    def save_dataframe_to_staging(self, df: pd.DataFrame, connection_string: str,
                                 source_system: str, import_batch_id: str, **kwargs) -> None:
        # Handle JSONB columns properly
        # Use PostgreSQL-specific table info query

    def get_pending_items(self, connection_string: str) -> pd.DataFrame:
        # Use PostgreSQL view
        # Handle JSONB parsing

    def update_item_status(self, connection_string: str, staging_id: int,
                          status: str, error_message: Optional[str] = None,
                          is_processed: Optional[bool] = None) -> None:
        # Use PostgreSQL syntax for UPDATE
        # Handle PostgreSQL timestamp functions
```

#### 2. Enhance `postgres_utils.py`

The existing `postgres_utils.py` file needs to be enhanced with:

```python
def initialize_postgres_staging_db(connection_string: str, schema_path: str) -> None:
    """
    Initialize the PostgreSQL staging database by executing the schema SQL file.

    Args:
        connection_string: PostgreSQL connection string
        schema_path: Path to the schema SQL file
    """
    logger = get_logger("postgres_utils")

    try:
        # Read the schema file
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)

        # Execute the schema
        with engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()

        logger.info(f"Successfully initialized PostgreSQL staging database")
    except Exception as e:
        error_msg = f"Error initializing PostgreSQL staging database: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e

def postgres_json_to_dict(json_data):
    """
    Convert PostgreSQL JSONB data to Python dictionary.

    Args:
        json_data: JSONB data from PostgreSQL

    Returns:
        Python dictionary
    """
    if json_data is None:
        return None

    if isinstance(json_data, dict):
        return json_data

    try:
        return json.loads(json_data)
    except (TypeError, json.JSONDecodeError):
        return json_data

def dict_to_postgres_json(data):
    """
    Convert Python dictionary to PostgreSQL JSONB format.

    Args:
        data: Python dictionary

    Returns:
        JSONB-compatible string
    """
    if data is None:
        return None

    if isinstance(data, str):
        return data

    return json.dumps(data)

def get_postgres_table_columns(connection_string: str, table_name: str, schema: str = "public") -> List[str]:
    """
    Get the column names of a PostgreSQL table.

    Args:
        connection_string: PostgreSQL connection string
        table_name: Table name
        schema: Schema name

    Returns:
        List of column names
    """
    logger = get_logger("postgres_utils")

    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)

        # Get the column names
        with engine.connect() as conn:
            query = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                AND table_schema = '{schema}'
                ORDER BY ordinal_position
            """
            result = conn.execute(text(query))
            columns = [row[0] for row in result]

        logger.info(f"Successfully retrieved column names for PostgreSQL table {table_name}")
        return columns
    except Exception as e:
        error_msg = f"Error getting column names for table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e
```

#### 3. Create `base_staging_manager.py`

Create an abstract base class for staging managers:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import pandas as pd

class BaseStagingManager(ABC):
    """
    Abstract base class for staging database managers.

    This class defines the interface that all staging managers must implement.
    """

    @abstractmethod
    def initialize_db(self, db_path: str) -> None:
        """
        Initialize the staging database.

        Args:
            db_path: Path to the database file or connection string.
        """
        pass

    @abstractmethod
    def reset_error_items(self, connection_string: str) -> int:
        """
        Reset the status of error items to 'PENDING'.

        Args:
            connection_string: The database connection string.

        Returns:
            The number of rows updated.
        """
        pass

    @abstractmethod
    def clear_processed_items(self, connection_string: str, days_to_keep: int = None) -> int:
        """
        Clear processed items from the staging table that are older than the specified number of days.

        Args:
            connection_string: The database connection string.
            days_to_keep: The number of days to keep processed items.

        Returns:
            The number of rows deleted.
        """
        pass

    @abstractmethod
    def save_dataframe_to_staging(self, df: pd.DataFrame, connection_string: str,
                                 source_system: str, import_batch_id: str, **kwargs) -> None:
        """
        Save a DataFrame to the staging table.

        Args:
            df: The DataFrame to save.
            connection_string: The database connection string.
            source_system: The source system identifier.
            import_batch_id: The import batch identifier.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def get_pending_items(self, connection_string: str) -> pd.DataFrame:
        """
        Get pending items from the staging table.

        Args:
            connection_string: The database connection string.

        Returns:
            A DataFrame containing the pending items.
        """
        pass

    @abstractmethod
    def update_item_status(self, connection_string: str, staging_id: int,
                          status: str, error_message: Optional[str] = None,
                          is_processed: Optional[bool] = None) -> None:
        """
        Update the status of an item in the staging table.

        Args:
            connection_string: The database connection string.
            staging_id: The ID of the item to update.
            status: The new status.
            error_message: The error message (if status is 'ERROR').
            is_processed: Whether the item has been processed.
        """
        pass
```

### 8.6 Benefits of PostgreSQL

1. **Performance**: Better performance for large datasets and complex queries
2. **Concurrency**: Multiple users/processes can access the database
   simultaneously
3. **JSON Support**: Native JSONB type with efficient indexing and querying
4. **Enterprise Features**: Replication, backup, and recovery options
5. **Scalability**: Better suited for growing datasets and increased load

## 9. Conclusion

The Medtronics pipeline provides a robust system for extracting, transforming,
and loading data from Medtronics Excel files into a standardized staging
database. The dynamic mapping system ensures that all columns are properly
mapped, either explicitly to specific fields or dynamically to the attributes
JSON field.

By following this SOP, you should be able to run, configure, and troubleshoot
the Medtronics pipeline effectively.
