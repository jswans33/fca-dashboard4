# ETL Pipeline v4 Implementation Guide

## 1. Environment Setup

### 1.1. Repository Setup

1. Create a new repository on GitHub/GitLab
2. Clone the repository locally:

   ```bash
   git clone <repository-url>
   cd fca-dashboard4
   ```

3. Create a `.gitignore` file with Python-specific entries:

   ```text
   # Python
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   env/
   venv/
   ENV/
   build/
   develop-eggs/
   dist/
   downloads/
   eggs/
   .eggs/
   lib/
   lib64/
   parts/
   sdist/
   var/
   *.egg-info/
   .installed.cfg
   *.egg
   
   # SQLite
   *.db
   *.sqlite3
   
   # Logs
   *.log
   
   # IDE
   .idea/
   .vscode/
   *.swp
   *.swo
   ```

### 1.2. Virtual Environment

1. Create a virtual environment:

   ```bash
   python -m .venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

### 1.3. Dependencies Setup

1. Create a `requirements.txt` file:

   ```text
   # Core dependencies
   sqlalchemy>=2.0.0
   alembic>=1.9.0
   pandas>=1.5.0
   openpyxl>=3.1.0  # For Excel support in pandas
   pyyaml>=6.0
   psycopg2-binary>=2.9.5  # PostgreSQL driver
   
   # Development dependencies
   pytest>=7.0.0
   black>=23.0.0
   isort>=5.12.0
   flake8>=6.0.0
   mypy>=1.0.0
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 1.4. Linter Configuration

1. Create `pyproject.toml` with linting configurations:

   ```toml
   [tool.black]
   line-length = 120
   target-version = ['py39']
   include = '\.pyi?$'
   
   [tool.isort]
   profile = "black"
   line_length = 120
   multi_line_output = 3
   
   [tool.flake8]
   max-line-length = 120
   extend-ignore = "E203"
   exclude = [".git", "__pycache__", "build", "dist", "alembic"]
   
   [tool.mypy]
   python_version = "3.9"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = true
   disallow_incomplete_defs = true
   
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = "test_*.py"
   ```

2. Test linters:

   ```bash
   black --check .
   isort --check .
   flake8 .
   mypy .
   ```

## 2. Project Structure Setup

### 2.1. Create Directory Structure

```bash
mkdir -p fca_dashboard/{alembic,config,core/{interfaces,models},extractors/{strategies},mappers/{strategies},loaders,pipelines,utils,tests/{unit,integration}}
touch fca_dashboard/__init__.py
touch fca_dashboard/main.py
touch fca_dashboard/core/__init__.py
touch fca_dashboard/extractors/__init__.py
touch fca_dashboard/mappers/__init__.py
touch fca_dashboard/loaders/__init__.py
touch fca_dashboard/pipelines/__init__.py
touch fca_dashboard/utils/__init__.py
touch fca_dashboard/tests/__init__.py
touch fca_dashboard/tests/unit/__init__.py
touch fca_dashboard/tests/integration/__init__.py
```

### 2.2. Create Configuration Files

1. Create `fca_dashboard/config/settings.yaml`:

   ```yaml
   # Database settings
   databases:
     sqlite:
       url: "sqlite:///etl.db"
     postgresql:
       url: "postgresql://user:password@localhost/etl"
       
   # Pipeline settings
   pipeline_settings:
     batch_size: 5000
     log_level: "INFO"
     
   # Table mappings
   tables:
     equipment:
       mapping_type: "direct"
       column_mappings:
         tag: "Tag"
         name: "Name"
         description: "Description"
   ```

2. Create `fca_dashboard/config/settings.py`:

   ```python
   import yaml
   from pathlib import Path
   from typing import Dict, Any
   
   def load_settings(config_path: str = None) -> Dict[str, Any]:
       """Load settings from YAML file."""
       if config_path is None:
           config_path = str(Path(__file__).parent / "settings.yaml")
       
       with open(config_path, 'r') as f:
           return yaml.safe_load(f)
   ```

## 3. Core Components Implementation

### 3.1. Create Interfaces

1. Create `fca_dashboard/core/interfaces.py`:

   ```python
   from abc import ABC, abstractmethod
   from typing import Generic, TypeVar, Optional, List, Any, Dict
   
   T = TypeVar('T')
   
   class Repository(Generic[T], ABC):
       """Base repository interface."""
       
       @abstractmethod
       def add(self, entity: T) -> T:
           """Add an entity to the repository."""
           pass
       
       @abstractmethod
       def get_by_id(self, id: Any) -> Optional[T]:
           """Get an entity by its ID."""
           pass
       
       @abstractmethod
       def get_all(self) -> List[T]:
           """Get all entities."""
           pass
       
       @abstractmethod
       def update(self, entity: T) -> T:
           """Update an entity."""
           pass
       
       @abstractmethod
       def delete(self, entity: T) -> None:
           """Delete an entity."""
           pass
   
   class UnitOfWork(ABC):
       """Unit of Work interface."""
       
       @abstractmethod
       def __enter__(self):
           """Enter the context."""
           pass
       
       @abstractmethod
       def __exit__(self, exc_type, exc_val, exc_tb):
           """Exit the context."""
           pass
       
       @abstractmethod
       def commit(self):
           """Commit the transaction."""
           pass
       
       @abstractmethod
       def rollback(self):
           """Rollback the transaction."""
           pass
   ```

### 3.2. Create SQLAlchemy Models

1. Create `fca_dashboard/core/models.py`:

   ```python
   from sqlalchemy import Column, Integer, String, Text, ForeignKey
   from sqlalchemy.ext.declarative import declarative_base
   from sqlalchemy.orm import relationship
   
   Base = declarative_base()
   
   class Equipment(Base):
       """Equipment model."""
       __tablename__ = 'equipment'
       
       id = Column(Integer, primary_key=True)
       tag = Column(String(50), nullable=False, unique=True)
       name = Column(String(100), nullable=False)
       description = Column(Text, nullable=True)
   
       def __repr__(self):
           return f"<Equipment(tag='{self.tag}', name='{self.name}')>"
   ```

### 3.3. Implement Repository

1. Create `fca_dashboard/core/repository.py`:

   ```python
   from typing import Generic, TypeVar, Optional, List, Any, Type
   from sqlalchemy.orm import Session
   
   from fca_dashboard.core.interfaces import Repository
   
   T = TypeVar('T')
   
   class SQLAlchemyRepository(Repository[T], Generic[T]):
       """SQLAlchemy implementation of Repository."""
       
       def __init__(self, session: Session, model_class: Type[T]):
           self.session = session
           self.model_class = model_class
       
       def add(self, entity: T) -> T:
           """Add an entity to the repository."""
           self.session.add(entity)
           return entity
       
       def get_by_id(self, id: Any) -> Optional[T]:
           """Get an entity by its ID."""
           return self.session.query(self.model_class).get(id)
       
       def get_all(self) -> List[T]:
           """Get all entities."""
           return self.session.query(self.model_class).all()
       
       def update(self, entity: T) -> T:
           """Update an entity."""
           self.session.merge(entity)
           return entity
       
       def delete(self, entity: T) -> None:
           """Delete an entity."""
           self.session.delete(entity)
   ```

### 3.4. Implement Unit of Work

1. Create `fca_dashboard/core/unit_of_work.py`:

   ```python
   from sqlalchemy import create_engine
   from sqlalchemy.orm import sessionmaker, Session
   
   from fca_dashboard.core.interfaces import UnitOfWork
   
   class SQLAlchemyUnitOfWork(UnitOfWork):
       """SQLAlchemy implementation of UnitOfWork."""
       
       def __init__(self, connection_string: str):
           self.connection_string = connection_string
           self.engine = create_engine(connection_string)
           self.session_factory = sessionmaker(bind=self.engine)
           self.session = None
       
       def __enter__(self):
           """Enter the context."""
           self.session = self.session_factory()
           return self
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           """Exit the context."""
           if exc_type is not None:
               self.rollback()
           self.session.close()
       
       def commit(self):
           """Commit the transaction."""
           self.session.commit()
       
       def rollback(self):
           """Rollback the transaction."""
           self.session.rollback()
   ```

## 4. Extraction Components

### 4.1. Create Extraction Strategy Interface

1. Create `fca_dashboard/extractors/strategies/base.py`:

   ```python
   from abc import ABC, abstractmethod
   import pandas as pd
   
   class ExtractionStrategy(ABC):
       """Base extraction strategy."""
       
       @abstractmethod
       def extract(self, source: str) -> pd.DataFrame:
           """Extract data from the source."""
           pass
       
       def _validate(self, data: pd.DataFrame) -> None:
           """Validate extracted data."""
           if data.empty:
               raise ValueError("Extracted data is empty.")
   ```

### 4.2. Implement Excel Extraction Strategy

1. Create `fca_dashboard/extractors/strategies/excel.py`:

   ```python
   import pandas as pd
   from fca_dashboard.extractors.strategies.base import ExtractionStrategy
   
   class ExcelExtractionStrategy(ExtractionStrategy):
       """Excel-specific extraction."""
       
       def __init__(self, sheet_name=0, header=0):
           self.sheet_name = sheet_name
           self.header = header
       
       def extract(self, source: str) -> pd.DataFrame:
           """Extract data from an Excel file."""
           try:
               data = pd.read_excel(
                   source, 
                   sheet_name=self.sheet_name, 
                   header=self.header
               )
               self._validate(data)
               return data
           except Exception as e:
               raise ValueError(f"Failed to extract data from {source}: {e}")
   ```

### 4.3. Create Excel Extractor

1. Create `fca_dashboard/extractors/excel_extractor.py`:

   ```python
   import pandas as pd
   import logging
   from typing import Dict, Any
   
   from fca_dashboard.extractors.strategies.base import ExtractionStrategy
   
   logger = logging.getLogger(__name__)
   
   class ExcelExtractor:
       """Excel data extractor."""
       
       def __init__(self, strategy: ExtractionStrategy):
           self.strategy = strategy
       
       def extract(self, source: str) -> pd.DataFrame:
           """Extract data from an Excel file."""
           logger.info(f"Extracting data from {source}")
           try:
               data = self.strategy.extract(source)
               logger.info(f"Extracted {len(data)} rows from {source}")
               return data
           except Exception as e:
               logger.error(f"Error extracting data from {source}: {e}")
               raise
   ```

## 5. Mapping Components

### 5.1. Create Mapping Strategy Interface

1. Create `fca_dashboard/mappers/mapping_strategy.py`:

   ```python
   from abc import ABC, abstractmethod
   import pandas as pd
   from typing import Dict, Any, Callable
   
   class MappingStrategy(ABC):
       """Base mapping strategy."""
       
       @abstractmethod
       def map_data(self, source_data: pd.DataFrame) -> pd.DataFrame:
           """Map source data to target format."""
           pass
   ```

### 5.2. Implement Direct Mapping Strategy

1. Create `fca_dashboard/mappers/strategies/direct_mapping.py`:

   ```python
   import pandas as pd
   from typing import Dict
   
   from fca_dashboard.mappers.mapping_strategy import MappingStrategy
   
   class DirectMappingStrategy(MappingStrategy):
       """Direct column mapping."""
       
       def __init__(self, column_mappings: Dict[str, str]):
           self.column_mappings = column_mappings
       
       def map_data(self, source_data: pd.DataFrame) -> pd.DataFrame:
           """Map source data using direct column mapping."""
           # Create a new DataFrame with mapped columns
           result = pd.DataFrame()
           
           for target_col, source_col in self.column_mappings.items():
               if source_col in source_data.columns:
                   result[target_col] = source_data[source_col]
               else:
                   raise ValueError(f"Source column '{source_col}' not found in data")
           
           return result
   ```

### 5.3. Implement Transform Mapping Strategy

1. Create `fca_dashboard/mappers/strategies/transform_mapping.py`:

   ```python
   import pandas as pd
   from typing import Dict, Callable
   
   from fca_dashboard.mappers.mapping_strategy import MappingStrategy
   
   class TransformMappingStrategy(MappingStrategy):
       """Transform-based mapping."""
       
       def __init__(self, transformations: Dict[str, Callable[[pd.DataFrame], pd.Series]]):
           self.transformations = transformations
       
       def map_data(self, source_data: pd.DataFrame) -> pd.DataFrame:
           """Map source data using transformations."""
           # Create a new DataFrame with transformed columns
           result = pd.DataFrame()
           
           for target_col, transform_func in self.transformations.items():
               try:
                   result[target_col] = transform_func(source_data)
               except Exception as e:
                   raise ValueError(f"Error applying transformation for column '{target_col}': {e}")
           
           return result
   ```

## 6. Loading Components

### 6.1. Create SQLite Loader

1. Create `fca_dashboard/loaders/sqlite_loader.py`:

   ```python
   import pandas as pd
   import logging
   from sqlalchemy import create_engine
   from typing import Optional
   
   logger = logging.getLogger(__name__)
   
   class SQLiteLoader:
       """SQLite data loader."""
       
       def __init__(self, db_url: str, batch_size: int = 1000):
           self.db_url = db_url
           self.batch_size = batch_size
           self.engine = create_engine(db_url)
       
       def load(self, data: pd.DataFrame, table_name: str, if_exists: str = 'replace', index: bool = False) -> None:
           """Load data into SQLite database."""
           logger.info(f"Loading {len(data)} rows into {table_name}")
           
           try:
               # Process in batches to avoid memory issues
               total_rows = len(data)
               for i in range(0, total_rows, self.batch_size):
                   batch = data.iloc[i:i + self.batch_size]
                   
                   # For the first batch, use the if_exists parameter
                   current_if_exists = if_exists if i == 0 else 'append'
                   
                   batch.to_sql(
                       name=table_name,
                       con=self.engine,
                       if_exists=current_if_exists,
                       index=index
                   )
                   
                   logger.info(f"Loaded batch {i // self.batch_size + 1}/{(total_rows + self.batch_size - 1) // self.batch_size}")
               
               logger.info(f"Successfully loaded data into {table_name}")
           except Exception as e:
               logger.error(f"Error loading data into {table_name}: {e}")
               raise
   ```

### 6.2. Create PostgreSQL Loader

1. Create `fca_dashboard/loaders/postgresql_loader.py`:

   ```python
   import pandas as pd
   import logging
   from sqlalchemy import create_engine
   from typing import Optional
   
   logger = logging.getLogger(__name__)
   
   class PostgreSQLLoader:
       """PostgreSQL data loader."""
       
       def __init__(self, db_url: str, batch_size: int = 1000):
           self.db_url = db_url
           self.batch_size = batch_size
           self.engine = create_engine(db_url)
       
       def load(self, data: pd.DataFrame, table_name: str, if_exists: str = 'replace', index: bool = False) -> None:
           """Load data into PostgreSQL database."""
           logger.info(f"Loading {len(data)} rows into {table_name}")
           
           try:
               # Process in batches to avoid memory issues
               total_rows = len(data)
               for i in range(0, total_rows, self.batch_size):
                   batch = data.iloc[i:i + self.batch_size]
                   
                   # For the first batch, use the if_exists parameter
                   current_if_exists = if_exists if i == 0 else 'append'
                   
                   batch.to_sql(
                       name=table_name,
                       con=self.engine,
                       if_exists=current_if_exists,
                       index=index,
                       method='multi'  # Use multi-row insert for better performance
                   )
                   
                   logger.info(f"Loaded batch {i // self.batch_size + 1}/{(total_rows + self.batch_size - 1) // self.batch_size}")
               
               logger.info(f"Successfully loaded data into {table_name}")
           except Exception as e:
               logger.error(f"Error loading data into {table_name}: {e}")
               raise
   ```

## 7. Pipeline Components

### 7.1. Create Base Pipeline

1. Create `fca_dashboard/pipelines/base_pipeline.py`:

   ```python
   from abc import ABC, abstractmethod
   import logging
   
   logger = logging.getLogger(__name__)
   
   class Pipeline(ABC):
       """Base pipeline with template method."""
       
       def run(self, *args, **kwargs):
           """Run the pipeline."""
           logger.info(f"Starting pipeline: {self.__class__.__name__}")
           try:
               # Template method defines the algorithm structure
               data = self.extract(*args, **kwargs)
               transformed = self.transform(data)
               self.load(transformed)
               result = self.verify()
               logger.info(f"Pipeline completed successfully: {self.__class__.__name__}")
               return result
           except Exception as e:
               logger.error(f"Pipeline failed: {e}")
               raise
       
       @abstractmethod
       def extract(self, *args, **kwargs):
           """Extract data from source."""
           pass
       
       @abstractmethod
       def transform(self, data):
           """Transform the data."""
           pass
       
       @abstractmethod
       def load(self, data):
           """Load data to target."""
           pass
       
       @abstractmethod
       def verify(self):
           """Verify the pipeline execution."""
           pass
   ```

### 7.2. Create Excel to SQLite Pipeline

1. Create `fca_dashboard/pipelines/excel_to_sqlite.py`:

   ```python
   import pandas as pd
   import logging
   
   from fca_dashboard.pipelines.base_pipeline import Pipeline
   from fca_dashboard.extractors.excel_extractor import ExcelExtractor
   from fca_dashboard.mappers.mapping_strategy import MappingStrategy
   from fca_dashboard.loaders.sqlite_loader import SQLiteLoader
   
   logger = logging.getLogger(__name__)
   
   class ExcelToSQLitePipeline(Pipeline):
       """Excel to SQLite pipeline implementation."""
       
       def __init__(self, extractor: ExcelExtractor, mapping_strategy: MappingStrategy, 
                    loader: SQLiteLoader, table_name: str):
           self.extractor = extractor
           self.mapping_strategy = mapping_strategy
           self.loader = loader
           self.table_name = table_name
       
       def extract(self, source_file: str) -> pd.DataFrame:
           """Extract data from Excel file."""
           logger.info(f"Extracting data from {source_file}")
           return self.extractor.extract(source_file)
       
       def transform(self, data: pd.DataFrame) -> pd.DataFrame:
           """Transform data using mapping strategy."""
           logger.info("Transforming data")
           return self.mapping_strategy.map_data(data)
       
       def load(self, data: pd.DataFrame) -> None:
           """Load data to SQLite database."""
           logger.info(f"Loading data to {self.table_name}")
           self.loader.load(data, self.table_name)
       
       def verify(self) -> bool:
           """Verify the pipeline execution."""
           logger.info("Verifying pipeline execution")
           # Simple verification - could be enhanced
           return True
   ```

### 7.3. Create SQLite to PostgreSQL Pipeline

1. Create `fca_dashboard/pipelines/sqlite_to_postgresql.py`:

   ```python
   import pandas as pd
   import logging
   from sqlalchemy import create_engine, MetaData, Table, select
   
   from fca_dashboard.pipelines.base_pipeline import Pipeline
   from fca_dashboard.loaders.postgresql_loader import PostgreSQLLoader
   
   logger = logging.getLogger(__name__)
   
   class SQLiteToPostgreSQLPipeline(Pipeline):
       """SQLite to PostgreSQL pipeline implementation."""
       
       def __init__(self, sqlite_url: str, loader: PostgreSQLLoader, 
                    table_name: str, batch_size: int = 1000):
           self.sqlite_url = sqlite_url
           self.loader = loader
           self.table_name = table_name
           self.batch_size = batch_size
           self.sqlite_engine = create_engine(sqlite_url)
       
       def extract(self) -> pd.DataFrame:
           """Extract data from SQLite database."""
           logger.info(f"Extracting data from SQLite table {self.table_name}")
           return pd.read_sql_table(self.table_name, self.sqlite_engine)
       
       def transform(self, data: pd.DataFrame) -> pd.DataFrame:
           """Transform data (identity transformation in this case)."""
           logger.info("Transforming data (identity transformation)")
           return data
       
       def load(self, data: pd.DataFrame) -> None:
           """Load data to PostgreSQL database."""
           logger.info(f"Loading data to PostgreSQL table {self.table_name}")
           self.loader.load(data, self.table_name)
       
       def verify(self) -> bool:
           """Verify the pipeline execution."""
           logger.info("Verifying pipeline execution")
           # Simple verification - could be enhanced
           return True
   ```

## 8. Utility Components

### 8.1. Create Logging Configuration

1. Create `fca_dashboard/utils/logging_config.py`:

   ```python
   import logging
   import sys
   from typing import Optional
   
   def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
       """Configure logging for the application."""
       # Convert string log level to logging constant
       numeric_level = getattr(logging, log_level.upper(), None)
       if not isinstance(numeric_level, int):
           raise ValueError(f"Invalid log level: {log_level}")
       
       # Create formatter
       formatter = logging.Formatter(
           '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
       )
       
       # Configure root logger
       root_logger = logging.getLogger()
       root_logger.setLevel(numeric_level)
       
       # Clear existing handlers
       for handler in root_logger.handlers[:]:
           root_logger.removeHandler(handler)
       
       # Add console handler
       console_handler = logging.StreamHandler(sys.stdout)
       console_handler.setFormatter(formatter)
       root_logger.addHandler(console_handler)
       
       # Add file handler if specified
       if log_file:
           file_handler = logging.FileHandler(log_file)
           file_handler.setFormatter(formatter)
           root_logger.addHandler(file_handler)
       
       return root_logger
   ```

### 8.2. Create Factory Functions

1. Create `fca_dashboard/core/factories.py`:

   ```python
   from typing import Dict, Any, Type
   
   from sqlalchemy.orm import Session
   
   from fca_dashboard.core.interfaces import Repository
   from fca_dashboard.core.repository import SQLAlchemyRepository
   from fca_dashboard.core.unit_of_work import SQLAlchemyUnitOfWork
   from fca_dashboard.extractors.strategies.excel import ExcelExtractionStrategy
   from fca_dashboard.extractors.excel_extractor import ExcelExtractor
   from fca_dashboard.mappers.strategies.direct_mapping import DirectMappingStrategy
   from fca_dashboard.mappers.strategies.transform_mapping import TransformMappingStrategy
   from fca_dashboard.loaders.sqlite_loader import SQLiteLoader
   from fca_dashboard.loaders.postgresql_loader import PostgreSQLLoader
   
   def create_repository(session: Session, model_class: Type) -> Repository:
       """Create a SQLAlchemy repository."""
       return SQLAlchemyRepository(session, model_class)
   
   def create_unit_of_work(connection_string: str) -> SQLAlchemyUnitOfWork:
       """Create a SQLAlchemy unit of work."""
       return SQLAlchemyUnitOfWork(connection_string)
   
   def create_excel_extractor(sheet_name=0, header=0) -> ExcelExtractor:
       """Create an Excel extractor."""
       strategy = ExcelExtractionStrategy(sheet_name=sheet_name, header=header)
       return ExcelExtractor(strategy)
   
   def create_mapping_strategy(config: Dict[str, Any]):
       """Create a mapping strategy based on configuration."""
       mapping_type = config.get("mapping_type", "direct")
       
       if mapping_type == "direct":
           column_mappings = config.get("column_mappings", {})
           return DirectMappingStrategy(column_mappings)
       elif mapping_type == "transform":
           # This would require more complex handling of transformation functions
           raise NotImplementedError("Transform mapping strategy not yet implemented")
       else:
           raise ValueError(f"Unknown mapping type: {mapping_type}")
   
   def create_sqlite_loader(db_url: str, batch_size: int = 1000) -> SQLiteLoader:
       """Create a SQLite loader."""
       return SQLiteLoader(db_url, batch_size)
   
   def create_postgresql_loader(db_url: str, batch_size: int = 1000) -> PostgreSQLLoader:
       """Create a PostgreSQL loader."""
       return PostgreSQLLoader(db_url, batch_size)
   ```

## 9. Main Application

### 9.1. Create Main Entry Point

1. Create `fca_dashboard/main.py`:

   ```python
   import argparse
   import logging
   import sys
   from pathlib import Path
   
   from fca_dashboard.config.settings import load_settings
   from fca_dashboard.core.factories import (
       create_excel_extractor,
       create_mapping_strategy,
       create_sqlite_loader,
       create_postgresql_loader
   )
   from fca_dashboard.pipelines.excel_to_sqlite import ExcelToSQLitePipeline
   from fca_dashboard.pipelines.sqlite_to_postgresql import SQLiteToPostgreSQLPipeline
   from fca_dashboard.utils.logging_config import configure_logging
   
   logger = logging.getLogger(__name__)
   
   def parse_args():
       """Parse command line arguments."""
       parser = argparse.ArgumentParser(description="ETL Pipeline")
       parser.add_argument("--config", help="Path to configuration file")
       parser.add_argument("--excel-file", help="Path to Excel file")
       parser.add_argument("--table-name", help="Table name")
       parser.add_argument("--log-level", default="INFO", help="Logging level")
       parser.add_argument("--log-file", help="Log file path")
       return parser.parse_args()
   
   def main():
       """Main entry point."""
       args = parse_args()
       
       # Configure logging
       configure_logging(args.log_level, args.log_file)
       
       try:
           # Load configuration
           config = load_settings(args.config)
           
           # Get database settings
           sqlite_url = config["databases"]["sqlite"]["url"]
           postgresql_url = config["databases"]["postgresql"]["url"]
           
           # Get pipeline settings
           batch_size = config["pipeline_settings"]["batch_size"]
           
           # Get table configuration
           table_name = args.table_name
           if not table_name:
               raise ValueError("Table name is required")
           
           table_config = config["tables"].get(table_name)
           if not table_config:
               raise ValueError(f"Configuration for table '{table_name}' not found")
           
           # Create components
           excel_extractor = create_excel_extractor()
           mapping_strategy = create_mapping_strategy(table_config)
           sqlite_loader = create_sqlite_loader(sqlite_url, batch_size)
           postgresql_loader = create_postgresql_loader(postgresql_url, batch_size)
           
           # Run Excel to SQLite pipeline
           excel_file = args.excel_file
           if not excel_file:
               raise ValueError("Excel file path is required")
           
           logger.info(f"Running Excel to SQLite pipeline for table '{table_name}'")
           excel_
