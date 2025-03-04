This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: tests/
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

## Additional Info

# Directory Structure
```
.repomixignore
config/__init__.py
config/settings.py
config/settings.yml
coverage_output.txt
coverage_report.txt
db/master/schema/backup/20250206_master_schema_update.sql
db/master/schema/backup/master_schema_table_schema.sql
db/master/schema/backup/simplified_schema.sql
db/master/schema/master_schema_uncoupled.sql
db/staging/procedures/staging_to_master_proc.sql
db/staging/schema/staging_schema.sql
examples/__init__.py
examples/excel_analyzer_example.py
examples/excel_export_example.py
examples/excel_extractor_example.py
examples/excel_unique_values_analysis_example.py
examples/excel_validation_example.py
extractors/base_extractor.py
extractors/excel_extractor.py
main.py
pipelines/pipeline_medtronics.py
pipelines/pipeline_wichita.py
repomix.config.json
utils/__init__.py
utils/database/__init__.py
utils/database/base.py
utils/database/postgres_utils.py
utils/database/sqlite_utils.py
utils/date_utils.py
utils/env_utils.py
utils/error_handler.py
utils/excel_utils.py
utils/excel/__init__.py
utils/excel/analysis_utils.py
utils/excel/base.py
utils/excel/column_utils.py
utils/excel/conversion_utils.py
utils/excel/extraction_utils.py
utils/excel/file_utils.py
utils/excel/sheet_utils.py
utils/excel/validation_utils.py
utils/json_utils.py
utils/logging_config.py
utils/loguru_stubs.pyi
utils/number_utils.py
utils/path_util.py
utils/string_utils.py
utils/upload_util.py
utils/validation_utils.py
utils/validation_utils.py,cover
```

# Files

## File: .repomixignore
```
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/
```

## File: config/__init__.py
```python
"""
Configuration package for the FCA Dashboard application.

This package contains modules for loading and managing application configuration.
"""
```

## File: config/settings.py
```python
"""
Configuration module for loading and accessing application settings.

This module provides functionality to load settings from YAML configuration files
and access them in a structured way throughout the application.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Settings:
    """
    Settings class for loading and accessing application configuration.

    This class provides methods to load settings from YAML files and access
    them through a simple interface.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Settings with optional config path.

        Args:
            config_path: Path to the YAML configuration file. If None, uses default.
        """
        self.config: Dict[str, Any] = {}
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "settings.yml"
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from the YAML file."""
        config_path = self.config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
            
        # Process environment variable substitutions
        self._process_env_vars(self.config)
        
    def _process_env_vars(self, config_section: Any) -> None:
        """
        Recursively process environment variable substitutions in the config.
        
        Args:
            config_section: A section of the configuration to process
        """
        if isinstance(config_section, dict):
            for key, value in config_section.items():
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str):
                    config_section[key] = self._substitute_env_vars(value)
        elif isinstance(config_section, list):
            for i, value in enumerate(config_section):
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str):
                    config_section[i] = self._substitute_env_vars(value)
    
    def _substitute_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: The string value to process
            
        Returns:
            The string with environment variables substituted
        """
        # Match ${VAR_NAME} pattern
        pattern = r'\${([A-Za-z0-9_]+)}'
        
        def replace_env_var(match):
            env_var_name = match.group(1)
            env_var_value = os.environ.get(env_var_name)
            if env_var_value is None:
                # If environment variable is not set, keep the original placeholder
                return match.group(0)
            return env_var_value
            
        return re.sub(pattern, replace_env_var, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found

        Returns:
            The configuration value or default if not found
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


# Cache for settings instances
# Consider thread-safety if accessed from multiple threads
_settings_cache: Dict[str, Settings] = {}
# Create a default settings instance
settings = Settings()


def get_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """
    Get a Settings instance, with caching for repeated calls.

    Args:
        config_path: Optional path to a configuration file

    Returns:
        A Settings instance
    """
    if config_path is None:
        return settings

    # Convert to string for dictionary key
    cache_key = str(config_path)

    # Return cached instance if available
    if cache_key in _settings_cache:
        return _settings_cache[cache_key]

    # Create new instance and cache it
    new_settings = Settings(config_path)
    _settings_cache[cache_key] = new_settings
    return new_settings
```

## File: config/settings.yml
```yaml
# Environment settings
env:
  ENVIRONMENT: "development"  # Default environment (can be overridden by OS environment variables)
  LOG_LEVEL: "INFO"
  DEBUG: true

# File paths
file_paths:
  uploads_dir: "uploads"  # Directory for uploaded files (relative to project root)
  extracts_dir: "extracts"  # Directory for extracted data (relative to project root)
  examples_dir: "examples"  # Directory for example files (relative to project root)
  
# Example settings
examples:
  excel:
    sample_filename: "sample_data.xlsx"
    columns_to_extract: ["ID", "Product", "Price"]
    price_threshold: 15

# Medtronics pipeline settings
medtronics:
  input_file: "uploads/Medtronics - Asset Log Uploader.xlsx"
  output_dir: "outputs/pipeline/medtronic"
  db_name: "medtronics_assets.db"
  sheet_name: "Asset Data"
  columns_to_extract: ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]
  drop_na_columns: ["asset name"]  # Drop rows where these columns have NaN values

# Wichita Animal Shelter pipeline settings
wichita:
  input_file: "uploads/Asset_List Wichita Animal Shelter (1).csv"
  output_dir: "outputs/pipeline/wichita"
  db_name: "wichita_assets.db"
  columns_to_extract: []  # Empty list means use all columns
  drop_na_columns: ["Asset Name", "Asset Category Name"]  # Drop rows where these columns have NaN values

# Excel utilities settings
excel_utils:
  extraction:
    default:
      header_row: null  # Auto-detect
      drop_empty_rows: true
      clean_column_names: true
      strip_whitespace: true
      convert_dtypes: true
    asset_data:
      header_row: 6  # We know the header starts at row 7 (index 6)
    eq_ids:
      header_row: 0  # Header is in the first row
    cobie:
      header_row: 2  # Header is in the third row
    dropdowns:
      header_row: 0  # Header is in the first row
  
  validation:
    default:
      missing_values:
        threshold: 0.5  # Allow up to 50% missing values
        columns: null  # Check all columns
      duplicate_rows:
        subset: null  # Check all columns for duplicates
      data_types:
        date_columns: ["date", "scheduled delivery date", "actual on-site date"]
        numeric_columns: ["motor hp", "size"]
        string_columns: ["equipment name", "equipment tag id", "manufacturer"]
        boolean_columns: ["o&m received", "attic stock"]
    asset_data:
      value_ranges:
        motor hp:
          min: 0
          max: 1000
        size:
          min: 0
          max: null  # No upper limit
    eq_ids:
      required_columns: ["Lookup (for Uploader)", "Trade", "Precon System"]
  
  analysis:
    default:
      unique_values:
        max_unique_values: 20  # Maximum number of unique values to include in the result
      column_statistics:
        include_outliers: true
      text_analysis:
        include_pattern_analysis: true

# Database settings
databases:
  sqlite:
    url: "sqlite:///fca_dashboard.db"
  postgresql:
    url: "${POSTGRES_URL}"
    
# Pipeline settings
pipeline_settings:
  batch_size: 5000
  log_level: "${LOG_LEVEL}"  # Uses the environment variable from env section
  
# Table mappings
tables:
  equipment:
    mapping_type: "direct"
    column_mappings:
      tag: "Tag"
      name: "Name"
      description: "Description"
```

## File: coverage_output.txt
```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.4, pluggy-1.5.0 -- C:\Repos\fca-dashboard4\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Repos\fca-dashboard4
configfile: pytest.ini
plugins: cov-4.1.0
collecting ... collected 18 items

tests\unit\test_validation_utils.py::TestEmailValidation::test_valid_emails PASSED [  5%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_invalid_emails PASSED [ 11%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_with_consecutive_dots PASSED [ 16%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_with_trailing_dot PASSED [ 22%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_with_spaces PASSED [ 27%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_domain_with_hyphens PASSED [ 33%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_domain_parts_with_hyphens PASSED [ 38%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_none_input PASSED [ 44%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_non_string_input PASSED [ 50%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_valid_phone_numbers PASSED [ 55%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_invalid_phone_numbers PASSED [ 61%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_international_phone_formats PASSED [ 66%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_none_input PASSED [ 72%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_non_string_input PASSED [ 77%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_valid_urls PASSED [ 83%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_invalid_urls PASSED [ 88%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_none_input PASSED [ 94%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_non_string_input PASSED [100%]

---------- coverage: platform win32, python 3.12.6-final-0 -----------
Name                        Stmts   Miss  Cover
-----------------------------------------------
utils\validation_utils.py      50      3    94%
-----------------------------------------------
TOTAL                          50      3    94%


============================= 18 passed in 0.48s ==============================
```

## File: coverage_report.txt
```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.4, pluggy-1.5.0 -- C:\Repos\fca-dashboard4\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Repos\fca-dashboard4
configfile: pytest.ini
plugins: cov-4.1.0
collecting ... collected 14 items

tests\unit\test_validation_utils.py::TestEmailValidation::test_valid_emails PASSED [  7%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_invalid_emails PASSED [ 14%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_domain_parts_with_hyphens PASSED [ 21%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_none_input PASSED [ 28%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_non_string_input PASSED [ 35%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_valid_phone_numbers PASSED [ 42%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_invalid_phone_numbers PASSED [ 50%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_international_phone_formats PASSED [ 57%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_none_input PASSED [ 64%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_non_string_input PASSED [ 71%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_valid_urls PASSED [ 78%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_invalid_urls PASSED [ 85%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_none_input PASSED [ 92%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_non_string_input PASSED [100%]

---------- coverage: platform win32, python 3.12.6-final-0 -----------
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
utils\validation_utils.py      50      3    94%   35, 37, 42
---------------------------------------------------------
TOTAL                          50      3    94%


============================= 14 passed in 0.48s ==============================
```

## File: db/master/schema/backup/20250206_master_schema_update.sql
```sql
-------------------------------
-- DROP EXISTING OBJECTS
-------------------------------
-- (Drop in reverse dependency order)
DROP TABLE IF EXISTS Quality_Control_Records CASCADE;
DROP TABLE IF EXISTS Quality_Control_Types CASCADE;
DROP TABLE IF EXISTS Control_Board_Images CASCADE;
DROP TABLE IF EXISTS Control_Board_Items CASCADE;
DROP TABLE IF EXISTS Maintenance_Costs CASCADE;
DROP TABLE IF EXISTS Maintenance CASCADE;
DROP TABLE IF EXISTS Equipment_Documents CASCADE;
DROP TABLE IF EXISTS Project_Documents CASCADE;
DROP TABLE IF EXISTS Document_Types CASCADE;
DROP TABLE IF EXISTS Equipment_Projects CASCADE;
DROP TABLE IF EXISTS Project_Phases CASCADE;
DROP TABLE IF EXISTS Projects CASCADE;
DROP TABLE IF EXISTS equipment_mappings CASCADE;
DROP TABLE IF EXISTS cost_mappings CASCADE;
DROP TABLE IF EXISTS attribute_mappings CASCADE;
DROP TABLE IF EXISTS classification_mappings CASCADE;
DROP TABLE IF EXISTS direct_mappings CASCADE;
DROP TABLE IF EXISTS pattern_rules CASCADE;
DROP TABLE IF EXISTS mapping_rules CASCADE;
DROP TABLE IF EXISTS Equipment_TCO CASCADE;
DROP TABLE IF EXISTS ASHRAE_Service_Life CASCADE;
DROP TABLE IF EXISTS Equipment_Costs CASCADE;
DROP TABLE IF EXISTS Equipment_Attributes CASCADE;
DROP TABLE IF EXISTS Attribute_Templates CASCADE;
DROP TABLE IF EXISTS Equipment CASCADE;
DROP TABLE IF EXISTS Locations CASCADE;
DROP TABLE IF EXISTS Equipment_Categories CASCADE;
DROP TABLE IF EXISTS MCAA_Classifications CASCADE;
DROP TABLE IF EXISTS CatalogSystem CASCADE;
DROP TABLE IF EXISTS MasterFormat CASCADE;
DROP TABLE IF EXISTS UniFormat CASCADE;
DROP TABLE IF EXISTS OmniClass CASCADE;

-------------------------------
-- 1. CLASSIFICATION TABLES
-------------------------------
CREATE TABLE OmniClass (
    OmniClassID SERIAL PRIMARY KEY,
    OmniClassCode VARCHAR(50) NOT NULL,
    OmniClassTitle VARCHAR(100) NOT NULL,
    OmniClassDescription TEXT
);

CREATE TABLE UniFormat (
    UniFormatID SERIAL PRIMARY KEY,
    UniFormatCode VARCHAR(50) NOT NULL,
    UniFormatTitle VARCHAR(100) NOT NULL,
    UniFormatDescription TEXT
);

CREATE TABLE MasterFormat (
    MasterFormatID SERIAL PRIMARY KEY,
    MasterFormatCode VARCHAR(50) NOT NULL,
    MasterFormatTitle VARCHAR(100) NOT NULL,
    MasterFormatDescription TEXT
);

CREATE TABLE CatalogSystem (
    CatalogID SERIAL PRIMARY KEY,
    CatalogCode VARCHAR(50) NOT NULL,
    CatalogTitle VARCHAR(100) NOT NULL,
    CatalogDescription TEXT,
    ExternalReference VARCHAR(100)  -- Reference to external catalog system
);

CREATE TABLE MCAA_Classifications (
    MCAAID SERIAL PRIMARY KEY,
    SystemCategory VARCHAR(100) NOT NULL,  -- e.g., HVAC Equipment
    SystemName VARCHAR(100) NOT NULL,        -- e.g., Boilers
    SubSystemType VARCHAR(100),              -- e.g., Hot Water
    SubSystemClassification VARCHAR(100),    -- e.g., Cast Iron Sectional
    EquipmentSize VARCHAR(50),               -- Size specification
    Notes TEXT
);

-------------------------------
-- 2. EQUIPMENT_CATEGORIES (Unified)
-------------------------------
CREATE TABLE Equipment_Categories (
    CategoryID SERIAL PRIMARY KEY,
    OmniClassID INT NOT NULL,
    UniFormatID INT NOT NULL,
    MasterFormatID INT NOT NULL,
    CatalogID INT NOT NULL,
    MCAAID INT NOT NULL,
    CategoryName VARCHAR(100) NOT NULL,
    CategoryDescription TEXT,
    CONSTRAINT fk_ec_omnaclass FOREIGN KEY (OmniClassID)
        REFERENCES OmniClass(OmniClassID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_ec_unifromat FOREIGN KEY (UniFormatID)
        REFERENCES UniFormat(UniFormatID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_ec_masterformat FOREIGN KEY (MasterFormatID)
        REFERENCES MasterFormat(MasterFormatID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_ec_catalogsystem FOREIGN KEY (CatalogID)
        REFERENCES CatalogSystem(CatalogID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_ec_mcaa FOREIGN KEY (MCAAID)
        REFERENCES MCAA_Classifications(MCAAID) ON UPDATE CASCADE ON DELETE RESTRICT
);

-- Indexes for performance on classification FKs
CREATE INDEX idx_ec_omnaclass ON Equipment_Categories(OmniClassID);
CREATE INDEX idx_ec_unifromat ON Equipment_Categories(UniFormatID);
CREATE INDEX idx_ec_masterformat ON Equipment_Categories(MasterFormatID);
CREATE INDEX idx_ec_catalog ON Equipment_Categories(CatalogID);
CREATE INDEX idx_ec_mcaa ON Equipment_Categories(MCAAID);

-------------------------------
-- 3. LOCATIONS
-------------------------------
CREATE TABLE Locations (
    LocationID SERIAL PRIMARY KEY,
    BuildingName VARCHAR(100),
    Floor VARCHAR(50),
    Room VARCHAR(50),
    OtherLocationInfo TEXT,
    XCoordinate DECIMAL(10,6),   -- Spatial coordinate
    YCoordinate DECIMAL(10,6)
);

-------------------------------
-- 4. MAPPING MODULE BASE TABLE
-------------------------------
-- mapping_rules is referenced by Equipment and later mapping tables.
CREATE TABLE mapping_rules (
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,   -- e.g., pattern, direct, classification, attribute, cost
    source_type VARCHAR(50),
    target_type VARCHAR(50),
    priority INT
);

-------------------------------
-- 5. EQUIPMENT
-------------------------------
CREATE TABLE Equipment (
    EquipmentID SERIAL PRIMARY KEY,
    CategoryID INT NOT NULL,
    LocationID INT NOT NULL,
    EquipmentTag VARCHAR(50) NOT NULL,
    Manufacturer VARCHAR(100),
    Model VARCHAR(100),
    SerialNumber VARCHAR(100),
    Capacity FLOAT,
    InstallDate DATE,
    Status VARCHAR(50),
    rule_id INT,  -- Reference to mapping_rules
    CONSTRAINT fk_equip_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_equip_location FOREIGN KEY (LocationID)
        REFERENCES Locations(LocationID) ON UPDATE CASCADE ON DELETE RESTRICT
);
-- Now add FK for rule_id
ALTER TABLE Equipment
  ADD CONSTRAINT fk_equip_mapping_rule FOREIGN KEY (rule_id)
    REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL;

-------------------------------
-- 6. EQUIPMENT ATTRIBUTES (Unified)
-------------------------------
CREATE TABLE Attribute_Templates (
    TemplateID SERIAL PRIMARY KEY,
    CategoryID INT NOT NULL,
    AttributeName VARCHAR(100) NOT NULL,
    Description TEXT,
    DefaultUnit VARCHAR(50),
    IsRequired BOOLEAN DEFAULT FALSE,
    ValidationRule TEXT,        -- Regex or range rule
    DataType VARCHAR(50),       -- e.g., numeric, string, date
    CONSTRAINT fk_at_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Attributes (
    EquipAttrID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    TemplateID INT,   -- Optional link to Attribute_Templates
    AttributeName VARCHAR(100) NOT NULL,
    AttributeValue TEXT,
    UnitOfMeasure VARCHAR(50),
    rule_id INT,
    CONSTRAINT fk_ea_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ea_template FOREIGN KEY (TemplateID)
        REFERENCES Attribute_Templates(TemplateID) ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT fk_ea_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL
);
CREATE INDEX idx_ea_equipment ON Equipment_Attributes(EquipmentID);

-------------------------------
-- 7. EQUIPMENT COSTS (Unified)
-------------------------------
CREATE TABLE Equipment_Costs (
    CostID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    CostDate DATE NOT NULL,
    CostType VARCHAR(50) NOT NULL,
    Amount NUMERIC(12,2) NOT NULL,
    Comments TEXT,
    rule_id INT,
    CONSTRAINT fk_ecosts_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecosts_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL
);
CREATE INDEX idx_ecosts_equipment ON Equipment_Costs(EquipmentID);

-------------------------------
-- 8. TOTAL COST OF OWNERSHIP
-------------------------------
CREATE TABLE ASHRAE_Service_Life (
    ServiceLifeID SERIAL PRIMARY KEY,
    EquipmentType VARCHAR(100) NOT NULL,
    MedianLifeExpectancy INT NOT NULL,
    ServiceTeamPriority VARCHAR(50),
    Notes TEXT
);

CREATE TABLE Equipment_TCO (
    TCOID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    ServiceLifeID INT NOT NULL,
    -- Condition Assessment
    AssetCondition INT CHECK (AssetCondition BETWEEN 1 AND 5),
    FailureLikelihood INT CHECK (FailureLikelihood BETWEEN 1 AND 5),
    AssetCriticality INT CHECK (AssetCriticality BETWEEN 1 AND 5),
    ConditionScore NUMERIC(5,2),
    RiskCategory VARCHAR(20),
    -- Service Life
    EstimatedServiceLifeYears INT,
    EstimatedReplacementDate DATE,
    LifecycleStatus VARCHAR(50),
    -- Cost Data
    FirstCost NUMERIC(12,2),
    AnnualMaintenanceCost NUMERIC(12,2),
    ReplacementCost NUMERIC(12,2),
    AnnualEnergyCost NUMERIC(12,2),
    LastAssessmentDate DATE,
    AssessedBy VARCHAR(100),
    Notes TEXT,
    CONSTRAINT fk_tco_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_tco_service_life FOREIGN KEY (ServiceLifeID)
        REFERENCES ASHRAE_Service_Life(ServiceLifeID) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE INDEX idx_tco_equipment ON Equipment_TCO(EquipmentID);

-------------------------------
-- 9. MAPPING MODULE DETAILS
-------------------------------
CREATE TABLE pattern_rules (
    pattern_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    pattern_regex TEXT NOT NULL,
    replacement_template TEXT,
    CONSTRAINT fk_pattern_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE direct_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_value VARCHAR(100) NOT NULL,
    target_value VARCHAR(100) NOT NULL,
    CONSTRAINT fk_direct_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE classification_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_classification VARCHAR(100) NOT NULL,
    target_classification VARCHAR(50) NOT NULL,
    classification_type VARCHAR(50) NOT NULL,  -- e.g., OmniClass, UniFormat, MasterFormat
    CONSTRAINT fk_classification_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE attribute_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_attribute VARCHAR(100) NOT NULL,
    target_attribute VARCHAR(100) NOT NULL,
    transformation_rule TEXT,
    CONSTRAINT fk_attribute_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE cost_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    cost_type VARCHAR(50) NOT NULL,  -- e.g., Purchase, Installation, Maintenance
    source_currency VARCHAR(10),
    target_currency VARCHAR(10),
    CONSTRAINT fk_cost_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE equipment_mappings (
    mapping_entry_id SERIAL PRIMARY KEY,  -- surrogate key added
    equipment_id INT NOT NULL,
    rule_id INT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2) NOT NULL,  -- value between 0 and 1
    CONSTRAINT fk_equipmap_equipment FOREIGN KEY (equipment_id)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_equipmap_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_equipmap_equipment ON equipment_mappings(equipment_id);
CREATE INDEX idx_equipmap_rule ON equipment_mappings(rule_id);

-------------------------------
-- 10. PROJECTS, PHASES, and EQUIPMENT_PROJECTS
-------------------------------
CREATE TABLE Projects (
    ProjectID SERIAL PRIMARY KEY,
    ProjectName VARCHAR(100) NOT NULL,
    ProjectStartDate DATE,
    ProjectEndDate DATE,
    ProjectDescription TEXT
);

CREATE TABLE Project_Phases (
    PhaseID SERIAL PRIMARY KEY,
    ProjectID INT NOT NULL,
    OmniClassPhaseCode VARCHAR(50),
    PhaseTitle VARCHAR(100),
    StartDate DATE,
    EndDate DATE,
    Description TEXT,
    CONSTRAINT fk_phase_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_phase_project ON Project_Phases(ProjectID);

CREATE TABLE Equipment_Projects (
    EquipmentID INT NOT NULL,
    ProjectID INT NOT NULL,
    RoleOrStatus VARCHAR(50),
    StartDate DATE,
    EndDate DATE,
    PRIMARY KEY (EquipmentID, ProjectID),
    CONSTRAINT fk_eproj_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_eproj_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE
);

-------------------------------
-- 11. DOCUMENTATION MANAGEMENT
-------------------------------
CREATE TABLE Document_Types (
    DocTypeID SERIAL PRIMARY KEY,
    TypeName VARCHAR(50) NOT NULL,
    Description TEXT,
    AllowedFileTypes VARCHAR(100)
);

CREATE TABLE Project_Documents (
    DocumentID SERIAL PRIMARY KEY,
    ProjectID INT NOT NULL,
    PhaseID INT,
    DocTypeID INT NOT NULL,
    DocumentName VARCHAR(100) NOT NULL,
    FilePath TEXT NOT NULL,
    FileType VARCHAR(50),
    UploadDate DATE,
    Version VARCHAR(20),
    UploadedBy VARCHAR(100),
    Description TEXT,
    CONSTRAINT fk_pd_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_pd_phase FOREIGN KEY (PhaseID)
        REFERENCES Project_Phases(PhaseID) ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT fk_pd_doctype FOREIGN KEY (DocTypeID)
        REFERENCES Document_Types(DocTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Documents (
    DocID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    DocTypeID INT NOT NULL,
    DocumentName VARCHAR(100) NOT NULL,
    FilePath TEXT NOT NULL,
    FileType VARCHAR(50),
    UploadDate DATE,
    Version VARCHAR(20),
    UploadedBy VARCHAR(100),
    Description TEXT,
    CONSTRAINT fk_ed_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ed_doctype FOREIGN KEY (DocTypeID)
        REFERENCES Document_Types(DocTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);

-------------------------------
-- 12. MAINTENANCE and MAINTENANCE_COSTS
-------------------------------
CREATE TABLE Maintenance (
    MaintenanceID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    MaintenanceDate DATE NOT NULL,
    WorkPerformed TEXT,
    Technician VARCHAR(100),
    NextDueDate DATE,
    Comments TEXT,
    CONSTRAINT fk_maintenance_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE Maintenance_Costs (
    MaintCostID SERIAL PRIMARY KEY,
    MaintenanceID INT NOT NULL,
    CostType VARCHAR(50) NOT NULL,
    Amount NUMERIC(12,2) NOT NULL,
    Comments TEXT,
    CONSTRAINT fk_maintcost_maintenance FOREIGN KEY (MaintenanceID)
        REFERENCES Maintenance(MaintenanceID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_maintcost_maintenance ON Maintenance_Costs(MaintenanceID);

-------------------------------
-- 13. CONTROL BOARD MANAGEMENT
-------------------------------
CREATE TABLE Control_Board_Items (
    ControlItemID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    ItemName VARCHAR(100) NOT NULL,
    Description TEXT,
    Location VARCHAR(100),
    SetPoint VARCHAR(50),
    NormalRange VARCHAR(50),
    Units VARCHAR(50),
    CONSTRAINT fk_cbi_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_cbi_equipment ON Control_Board_Items(EquipmentID);

CREATE TABLE Control_Board_Images (
    ImageID SERIAL PRIMARY KEY,
    ControlItemID INT NOT NULL,
    ImagePath TEXT NOT NULL,
    ImageType VARCHAR(50),
    CaptureDate DATE,
    Description TEXT,
    CONSTRAINT fk_cbi_images FOREIGN KEY (ControlItemID)
        REFERENCES Control_Board_Items(ControlItemID) ON UPDATE CASCADE ON DELETE CASCADE
);

-------------------------------
-- 14. QUALITY CONTROL MANAGEMENT
-------------------------------
CREATE TABLE Quality_Control_Types (
    QCTypeID SERIAL PRIMARY KEY,
    TypeName VARCHAR(100) NOT NULL,  -- e.g., Service Verification, Installation Check, Data Accuracy
    Description TEXT,
    Department VARCHAR(100),
    RequiresApproval BOOLEAN DEFAULT FALSE
);

CREATE TABLE Quality_Control_Records (
    QCID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    QCTypeID INT NOT NULL,
    Verified BOOLEAN,
    VerificationDate DATE,
    VerifiedBy VARCHAR(100),
    Notes TEXT,
    Status VARCHAR(50),  -- e.g., Pending, Approved, Failed
    ApprovedBy VARCHAR(100),
    ApprovalDate DATE,
    CONSTRAINT fk_qcr_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_qcr_qctype FOREIGN KEY (QCTypeID)
        REFERENCES Quality_Control_Types(QCTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE INDEX idx_qcr_equipment ON Quality_Control_Records(EquipmentID);
CREATE INDEX idx_qcr_qctype ON Quality_Control_Records(QCTypeID);

-------------------------------
-- INDEXES for Other Tables (if needed)
-------------------------------
CREATE INDEX idx_maintenance_equipment ON Maintenance(EquipmentID);

-------------------------------
-- VIEWS (Common Business Patterns)
-------------------------------

-- View joining Equipment with its full classification details and location.
CREATE OR REPLACE VIEW v_equipment_full_details AS
SELECT 
    e.EquipmentID,
    e.EquipmentTag,
    e.Manufacturer,
    e.Model,
    e.SerialNumber,
    e.Capacity,
    e.InstallDate,
    e.Status,
    ec.CategoryName,
    ec.CategoryDescription,
    oc.OmniClassCode,
    oc.OmniClassTitle,
    uf.UniFormatCode,
    uf.UniFormatTitle,
    mf.MasterFormatCode,
    mf.MasterFormatTitle,
    cs.CatalogCode,
    cs.CatalogTitle,
    mc.SystemCategory,
    mc.SystemName,
    mc.SubSystemType,
    mc.SubSystemClassification,
    mc.EquipmentSize,
    mc.Notes,
    l.BuildingName,
    l.Floor,
    l.Room,
    l.OtherLocationInfo,
    l.XCoordinate,
    l.YCoordinate
FROM Equipment e
JOIN Equipment_Categories ec ON e.CategoryID = ec.CategoryID
JOIN OmniClass oc ON ec.OmniClassID = oc.OmniClassID
JOIN UniFormat uf ON ec.UniFormatID = uf.UniFormatID
JOIN MasterFormat mf ON ec.MasterFormatID = mf.MasterFormatID
JOIN CatalogSystem cs ON ec.CatalogID = cs.CatalogID
JOIN MCAA_Classifications mc ON ec.MCAAID = mc.MCAAID
JOIN Locations l ON e.LocationID = l.LocationID;

-- View for equipment mapping details.
CREATE OR REPLACE VIEW v_equipment_mapping AS
SELECT 
    em.mapping_entry_id,
    e.EquipmentID,
    e.EquipmentTag,
    mr.rule_id,
    mr.rule_name,
    em.applied_at,
    em.confidence_score
FROM equipment_mappings em
JOIN Equipment e ON em.equipment_id = e.EquipmentID
JOIN mapping_rules mr ON em.rule_id = mr.rule_id;

-- View for project documents with related project and phase information.
CREATE OR REPLACE VIEW v_project_documents AS
SELECT 
    pd.DocumentID,
    p.ProjectID,
    p.ProjectName,
    pp.PhaseID,
    pp.PhaseTitle,
    pd.DocumentName,
    pd.FilePath,
    pd.FileType,
    pd.UploadDate,
    pd.Version,
    pd.UploadedBy,
    pd.Description
FROM Project_Documents pd
JOIN Projects p ON pd.ProjectID = p.ProjectID
LEFT JOIN Project_Phases pp ON pd.PhaseID = pp.PhaseID;

-- View for equipment documents.
CREATE OR REPLACE VIEW v_equipment_documents AS
SELECT 
    ed.DocID,
    e.EquipmentID,
    e.EquipmentTag,
    ed.DocumentName,
    ed.FilePath,
    ed.FileType,
    ed.UploadDate,
    ed.Version,
    ed.UploadedBy,
    ed.Description
FROM Equipment_Documents ed
JOIN Equipment e ON ed.EquipmentID = e.EquipmentID;

-- View for maintenance details.
CREATE OR REPLACE VIEW v_equipment_maintenance AS
SELECT 
    m.MaintenanceID,
    e.EquipmentID,
    e.EquipmentTag,
    m.MaintenanceDate,
    m.WorkPerformed,
    m.Technician,
    m.NextDueDate,
    m.Comments
FROM Maintenance m
JOIN Equipment e ON m.EquipmentID = e.EquipmentID;
```

## File: db/master/schema/backup/master_schema_table_schema.sql
```sql
-------------------------------
-- MASTER TABLE SCHEMA FOR FCA DASHBOARD
-------------------------------
-- This schema uses a denormalized master table approach
-- that contains the most commonly used equipment data in one place,
-- with supporting tables for classifications and other related data.

DROP SCHEMA IF EXISTS master CASCADE;
CREATE SCHEMA master;

-------------------------------
-- CLASSIFICATION REFERENCE TABLES
-------------------------------
-- These tables provide reference for standardized classification systems

CREATE TABLE master.classification_systems (
    system_id SERIAL PRIMARY KEY,
    system_name VARCHAR(50) NOT NULL,  -- e.g., OmniClass, UniFormat, MasterFormat, MCAA
    system_version VARCHAR(50),
    description TEXT
);

CREATE TABLE master.classification_values (
    value_id SERIAL PRIMARY KEY,
    system_id INTEGER NOT NULL REFERENCES master.classification_systems(system_id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    parent_code VARCHAR(50),  -- For hierarchical classifications
    level INTEGER,            -- Hierarchy level
    description TEXT,
    UNIQUE(system_id, code)
);

-------------------------------
-- MASTER EQUIPMENT TABLE
-------------------------------
-- Core consolidated equipment table that contains all essential fields
-- This provides a simpler query interface for most common operations

CREATE TABLE master.equipment (
    equipment_id SERIAL PRIMARY KEY,
    
    -- Equipment identifiers
    equipment_tag VARCHAR(50) NOT NULL UNIQUE,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    capacity FLOAT,
    install_date DATE,
    status VARCHAR(50),
    
    -- Category & classification
    category_name VARCHAR(100) NOT NULL,
    omniclass_code VARCHAR(50),
    omniclass_title VARCHAR(100),
    uniformat_code VARCHAR(50),
    uniformat_title VARCHAR(100),
    masterformat_code VARCHAR(50),
    masterformat_title VARCHAR(100),
    catalog_code VARCHAR(50),
    
    -- MCAA classification data
    mcaa_system_category VARCHAR(100),
    mcaa_system_name VARCHAR(100),
    mcaa_subsystem_type VARCHAR(100),
    mcaa_subsystem_classification VARCHAR(100),
    mcaa_equipment_size VARCHAR(50),
    
    -- Location data
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    other_location_info TEXT,
    x_coordinate DECIMAL(10,6),
    y_coordinate DECIMAL(10,6),
    
    -- Asset condition & lifecycle data
    condition_score NUMERIC(5,2),  -- Overall condition (1-5 scale)
    risk_category VARCHAR(20),     -- e.g., High, Medium, Low
    expected_life_years INTEGER,
    estimated_replacement_date DATE,
    lifecycle_status VARCHAR(50),  -- e.g., New, Mid-Life, End-of-Life
    
    -- Cost data
    initial_cost NUMERIC(12,2),
    installation_cost NUMERIC(12,2),
    annual_maintenance_cost NUMERIC(12,2),
    replacement_cost NUMERIC(12,2),
    annual_energy_cost NUMERIC(12,2),
    last_cost_update DATE,
    
    -- Assessment data
    last_assessment_date DATE,
    assessed_by VARCHAR(100),
    
    -- Extended data (stored as JSON)
    attributes JSONB,           -- Flexible equipment attributes
    maintenance_summary JSONB,  -- Summary of recent maintenance
    documents_summary JSONB,    -- List of associated documents
    projects_summary JSONB,     -- List of associated projects
    
    -- Metadata and tracking
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    data_source VARCHAR(100),
    source_record_id VARCHAR(100),
    mapping_rule_id INTEGER,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for common query patterns
CREATE INDEX idx_master_equipment_tag ON master.equipment(equipment_tag);
CREATE INDEX idx_master_category ON master.equipment(category_name);
CREATE INDEX idx_master_building ON master.equipment(building_name);
CREATE INDEX idx_master_status ON master.equipment(status);
CREATE INDEX idx_master_lifecycle ON master.equipment(lifecycle_status);
CREATE INDEX idx_master_attributes ON master.equipment USING GIN (attributes);
CREATE INDEX idx_master_maintenance ON master.equipment USING GIN (maintenance_summary);
CREATE INDEX idx_master_documents ON master.equipment USING GIN (documents_summary);
CREATE INDEX idx_master_projects ON master.equipment USING GIN (projects_summary);

-------------------------------
-- SUPPORTING DETAIL TABLES
-------------------------------
-- These tables store detailed information that doesn't fit
-- well in the master table or needs separate management

-- Equipment attributes (for structured attribute storage)
CREATE TABLE master.equipment_attributes (
    attribute_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    attribute_name VARCHAR(100) NOT NULL,
    attribute_value TEXT,
    data_type VARCHAR(50),  -- e.g., text, number, date, boolean
    unit_of_measure VARCHAR(50),
    source_system VARCHAR(100),
    is_verified BOOLEAN DEFAULT FALSE,
    UNIQUE(equipment_id, attribute_name)
);

-- Maintenance records
CREATE TABLE master.maintenance_records (
    maintenance_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    maintenance_date DATE NOT NULL,
    maintenance_type VARCHAR(50),  -- e.g., Preventive, Corrective, Emergency
    description TEXT,
    performed_by VARCHAR(100),
    cost NUMERIC(12,2),
    parts_used TEXT,
    next_due_date DATE,
    status VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

-- Documents
CREATE TABLE master.documents (
    document_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    document_type VARCHAR(50) NOT NULL,  -- e.g., Manual, Warranty, Drawing, Photo
    document_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50),
    upload_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Projects
CREATE TABLE master.projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    description TEXT,
    start_date DATE,
    end_date DATE,
    status VARCHAR(50),
    budget NUMERIC(12,2),
    project_manager VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Equipment-project relationships
CREATE TABLE master.equipment_projects (
    equipment_id INTEGER NOT NULL REFERENCES master.equipment(equipment_id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES master.projects(project_id) ON DELETE CASCADE,
    relationship_type VARCHAR(50),  -- e.g., Installation, Replacement, Repair
    start_date DATE,
    end_date DATE,
    notes TEXT,
    PRIMARY KEY (equipment_id, project_id)
);

-------------------------------
-- AUDIT AND HISTORY TABLES
-------------------------------
-- These tables keep track of changes to equipment data

-- Equipment change history
CREATE TABLE master.equipment_history (
    history_id SERIAL PRIMARY KEY,
    equipment_id INTEGER NOT NULL,
    change_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    change_type VARCHAR(20) NOT NULL, -- INSERT
```

## File: db/master/schema/backup/simplified_schema.sql
```sql
-------------------------------
-- SIMPLIFIED SCHEMA FOR FCA DASHBOARD
-------------------------------
-- Contains a simplified staging table and master table design

-- Create schemas
DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;

DROP SCHEMA IF EXISTS master CASCADE;
CREATE SCHEMA master;

-- Simplified staging table
CREATE TABLE staging.equipment_staging (
    -- Staging metadata fields
    staging_id SERIAL PRIMARY KEY,
    import_batch_id VARCHAR(100),
    import_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, PROCESSED, ERROR
    error_message TEXT,
    
    -- Core equipment fields
    equipment_tag VARCHAR(50),
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    status VARCHAR(50),
    
    -- Basic classification
    category_name VARCHAR(100),
    
    -- Simple location
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    
    -- Basic attributes
    attributes JSONB,
    
    -- Source data for troubleshooting
    source_system VARCHAR(100),
    raw_data JSONB
);

-- Indexes for staging table
CREATE INDEX idx_staging_status ON staging.equipment_staging(processing_status);
CREATE INDEX idx_staging_equipment_tag ON staging.equipment_staging(equipment_tag);

-- Master table for processed equipment data
CREATE TABLE master.equipment (
    equipment_id SERIAL PRIMARY KEY,
    equipment_tag VARCHAR(50) UNIQUE NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    status VARCHAR(50),
    category_name VARCHAR(100),
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    attributes JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    source_staging_id INT REFERENCES staging.equipment_staging(staging_id)
);

-- Indexes for master table
CREATE INDEX idx_master_equipment_tag ON master.equipment(equipment_tag);
CREATE INDEX idx_master_category ON master.equipment(category_name);
CREATE INDEX idx_master_building ON master.equipment(building_name);

-- Views
CREATE OR REPLACE VIEW staging.v_pending_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'PENDING';

CREATE OR REPLACE VIEW staging.v_error_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'ERROR';

-- Function to move data from staging to master
CREATE OR REPLACE FUNCTION staging.process_staging_data()
RETURNS INTEGER AS $$
DECLARE
    processed_count INTEGER := 0;
BEGIN
    -- Insert new equipment or update existing ones
    INSERT INTO master.equipment (
        equipment_tag, manufacturer, model, serial_number, 
        status, category_name, building_name, floor, room, 
        attributes, source_staging_id
    )
    SELECT 
        s.equipment_tag, s.manufacturer, s.model, s.serial_number,
        s.status, s.category_name, s.building_name, s.floor, s.room,
        s.attributes, s.staging_id
    FROM staging.equipment_staging s
    WHERE s.processing_status = 'PENDING'
    ON CONFLICT (equipment_tag) 
    DO UPDATE SET
        manufacturer = EXCLUDED.manufacturer,
        model = EXCLUDED.model,
        serial_number = EXCLUDED.serial_number,
        status = EXCLUDED.status,
        category_name = EXCLUDED.category_name,
        building_name = EXCLUDED.building_name,
        floor = EXCLUDED.floor,
        room = EXCLUDED.room,
        attributes = EXCLUDED.attributes,
        updated_at = CURRENT_TIMESTAMP,
        source_staging_id = EXCLUDED.source_staging_id;

    -- Mark records as processed
    UPDATE staging.equipment_staging
    SET processing_status = 'PROCESSED'
    WHERE processing_status = 'PENDING';
    
    GET DIAGNOSTICS processed_count = ROW_COUNT;
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clear processed records
CREATE OR REPLACE FUNCTION staging.clear_processed_items(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM staging.equipment_staging
    WHERE processing_status = 'PROCESSED' 
    AND import_timestamp < CURRENT_DATE - days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

## File: db/master/schema/master_schema_uncoupled.sql
```sql
-------------------------------
-- DROP EXISTING OBJECTS
-------------------------------
-- (Drop in reverse dependency order)
DROP TABLE IF EXISTS Quality_Control_Records CASCADE;
DROP TABLE IF EXISTS Quality_Control_Types CASCADE;
DROP TABLE IF EXISTS Control_Board_Images CASCADE;
DROP TABLE IF EXISTS Control_Board_Items CASCADE;
DROP TABLE IF EXISTS Maintenance_Costs CASCADE;
DROP TABLE IF EXISTS Maintenance CASCADE;
DROP TABLE IF EXISTS Equipment_Documents CASCADE;
DROP TABLE IF EXISTS Project_Documents CASCADE;
DROP TABLE IF EXISTS Document_Types CASCADE;
DROP TABLE IF EXISTS Equipment_Projects CASCADE;
DROP TABLE IF EXISTS Project_Phases CASCADE;
DROP TABLE IF EXISTS Projects CASCADE;
DROP TABLE IF EXISTS equipment_mappings CASCADE;
DROP TABLE IF EXISTS cost_mappings CASCADE;
DROP TABLE IF EXISTS attribute_mappings CASCADE;
DROP TABLE IF EXISTS classification_mappings CASCADE;
DROP TABLE IF EXISTS direct_mappings CASCADE;
DROP TABLE IF EXISTS pattern_rules CASCADE;
DROP TABLE IF EXISTS mapping_rules CASCADE;
DROP TABLE IF EXISTS Equipment_TCO CASCADE;
DROP TABLE IF EXISTS ASHRAE_Service_Life CASCADE;
DROP TABLE IF EXISTS Equipment_Costs CASCADE;
DROP TABLE IF EXISTS Equipment_Attributes CASCADE;
DROP TABLE IF EXISTS Attribute_Templates CASCADE;
DROP TABLE IF EXISTS Equipment CASCADE;
DROP TABLE IF EXISTS Locations CASCADE;
DROP TABLE IF EXISTS Equipment_Categories CASCADE;
DROP TABLE IF EXISTS MCAA_Classifications CASCADE;
DROP TABLE IF EXISTS CatalogSystem CASCADE;
DROP TABLE IF EXISTS MasterFormat CASCADE;
DROP TABLE IF EXISTS UniFormat CASCADE;
DROP TABLE IF EXISTS OmniClass CASCADE;

-------------------------------
-- 1. CLASSIFICATION TABLES
-------------------------------
CREATE TABLE OmniClass (
    OmniClassID SERIAL PRIMARY KEY,
    OmniClassCode VARCHAR(50) NOT NULL,
    OmniClassTitle VARCHAR(100) NOT NULL,
    OmniClassDescription TEXT
);

CREATE TABLE UniFormat (
    UniFormatID SERIAL PRIMARY KEY,
    UniFormatCode VARCHAR(50) NOT NULL,
    UniFormatTitle VARCHAR(100) NOT NULL,
    UniFormatDescription TEXT
);

CREATE TABLE MasterFormat (
    MasterFormatID SERIAL PRIMARY KEY,
    MasterFormatCode VARCHAR(50) NOT NULL,
    MasterFormatTitle VARCHAR(100) NOT NULL,
    MasterFormatDescription TEXT
);

CREATE TABLE CatalogSystem (
    CatalogID SERIAL PRIMARY KEY,
    CatalogCode VARCHAR(50) NOT NULL,
    CatalogTitle VARCHAR(100) NOT NULL,
    CatalogDescription TEXT,
    ExternalReference VARCHAR(100)  -- Reference to external catalog system
);

CREATE TABLE MCAA_Classifications (
    MCAAID SERIAL PRIMARY KEY,
    SystemCategory VARCHAR(100) NOT NULL,  -- e.g., HVAC Equipment
    SystemName VARCHAR(100) NOT NULL,        -- e.g., Boilers
    SubSystemType VARCHAR(100),              -- e.g., Hot Water
    SubSystemClassification VARCHAR(100),    -- e.g., Cast Iron Sectional
    EquipmentSize VARCHAR(50),               -- Size specification
    Notes TEXT
);

-------------------------------
-- 2. EQUIPMENT_CATEGORIES (Decoupled)
-------------------------------
CREATE TABLE Equipment_Categories (
    CategoryID SERIAL PRIMARY KEY,
    CategoryName VARCHAR(100) NOT NULL,
    CategoryDescription TEXT
);

-- Junction Tables for Classifications
CREATE TABLE Equipment_Categories_OmniClass (
    CategoryID INT NOT NULL,
    OmniClassID INT NOT NULL,
    PRIMARY KEY (CategoryID, OmniClassID),
    CONSTRAINT fk_eco_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_eco_omniclass FOREIGN KEY (OmniClassID)
        REFERENCES OmniClass(OmniClassID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_UniFormat (
    CategoryID INT NOT NULL,
    UniFormatID INT NOT NULL,
    PRIMARY KEY (CategoryID, UniFormatID),
    CONSTRAINT fk_ecu_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecu_unifornat FOREIGN KEY (UniFormatID)
        REFERENCES UniFormat(UniFormatID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_MasterFormat (
    CategoryID INT NOT NULL,
    MasterFormatID INT NOT NULL,
    PRIMARY KEY (CategoryID, MasterFormatID),
    CONSTRAINT fk_ecm_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecm_masterformat FOREIGN KEY (MasterFormatID)
        REFERENCES MasterFormat(MasterFormatID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_Catalog (
    CategoryID INT NOT NULL,
    CatalogID INT NOT NULL,
    PRIMARY KEY (CategoryID, CatalogID),
    CONSTRAINT fk_ecc_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecc_catalog FOREIGN KEY (CatalogID)
        REFERENCES CatalogSystem(CatalogID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Categories_MCAA (
    CategoryID INT NOT NULL,
    MCAAID INT NOT NULL,
    PRIMARY KEY (CategoryID, MCAAID),
    CONSTRAINT fk_ecmc_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecmc_mcaa FOREIGN KEY (MCAAID)
        REFERENCES MCAA_Classifications(MCAAID) ON UPDATE CASCADE ON DELETE RESTRICT
);

-- Create indexes for performance on junction tables
CREATE INDEX idx_eco_omniclass ON Equipment_Categories_OmniClass(OmniClassID);
CREATE INDEX idx_ecu_unifornat ON Equipment_Categories_UniFormat(UniFormatID);
CREATE INDEX idx_ecm_masterformat ON Equipment_Categories_MasterFormat(MasterFormatID);
CREATE INDEX idx_ecc_catalog ON Equipment_Categories_Catalog(CatalogID);
CREATE INDEX idx_ecmc_mcaa ON Equipment_Categories_MCAA(MCAAID);

-------------------------------
-- 3. LOCATIONS
-------------------------------
CREATE TABLE Locations (
    LocationID SERIAL PRIMARY KEY,
    BuildingName VARCHAR(100),
    Floor VARCHAR(50),
    Room VARCHAR(50),
    OtherLocationInfo TEXT,
    XCoordinate DECIMAL(10,6),   -- Spatial coordinate
    YCoordinate DECIMAL(10,6)
);

-------------------------------
-- 4. MAPPING MODULE BASE TABLE
-------------------------------
-- mapping_rules is referenced by Equipment and later mapping tables.
CREATE TABLE mapping_rules (
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,   -- e.g., pattern, direct, classification, attribute, cost
    source_type VARCHAR(50),
    target_type VARCHAR(50),
    priority INT
);

-------------------------------
-- 5. EQUIPMENT
-------------------------------
CREATE TABLE Equipment (
    EquipmentID SERIAL PRIMARY KEY,
    CategoryID INT NOT NULL,
    LocationID INT NOT NULL,
    EquipmentTag VARCHAR(50) NOT NULL,
    Manufacturer VARCHAR(100),
    Model VARCHAR(100),
    SerialNumber VARCHAR(100),
    Capacity FLOAT,
    InstallDate DATE,
    Status VARCHAR(50),
    rule_id INT,  -- Reference to mapping_rules
    CONSTRAINT fk_equip_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_equip_location FOREIGN KEY (LocationID)
        REFERENCES Locations(LocationID) ON UPDATE CASCADE ON DELETE RESTRICT
);
-- Now add FK for rule_id
ALTER TABLE Equipment
  ADD CONSTRAINT fk_equip_mapping_rule FOREIGN KEY (rule_id)
    REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL;

-------------------------------
-- 6. EQUIPMENT ATTRIBUTES (Unified)
-------------------------------
CREATE TABLE Attribute_Templates (
    TemplateID SERIAL PRIMARY KEY,
    CategoryID INT NOT NULL,
    AttributeName VARCHAR(100) NOT NULL,
    Description TEXT,
    DefaultUnit VARCHAR(50),
    IsRequired BOOLEAN DEFAULT FALSE,
    ValidationRule TEXT,        -- Regex or range rule
    DataType VARCHAR(50),       -- e.g., numeric, string, date
    CONSTRAINT fk_at_category FOREIGN KEY (CategoryID)
        REFERENCES Equipment_Categories(CategoryID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Attributes (
    EquipAttrID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    TemplateID INT,   -- Optional link to Attribute_Templates
    AttributeName VARCHAR(100) NOT NULL,
    AttributeValue TEXT,
    UnitOfMeasure VARCHAR(50),
    rule_id INT,
    CONSTRAINT fk_ea_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ea_template FOREIGN KEY (TemplateID)
        REFERENCES Attribute_Templates(TemplateID) ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT fk_ea_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL
);
CREATE INDEX idx_ea_equipment ON Equipment_Attributes(EquipmentID);

-------------------------------
-- 7. EQUIPMENT COSTS (Unified)
-------------------------------
CREATE TABLE Equipment_Costs (
    CostID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    CostDate DATE NOT NULL,
    CostType VARCHAR(50) NOT NULL,
    Amount NUMERIC(12,2) NOT NULL,
    Comments TEXT,
    rule_id INT,
    CONSTRAINT fk_ecosts_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ecosts_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE SET NULL
);
CREATE INDEX idx_ecosts_equipment ON Equipment_Costs(EquipmentID);

-------------------------------
-- 8. TOTAL COST OF OWNERSHIP
-------------------------------
CREATE TABLE ASHRAE_Service_Life (
    ServiceLifeID SERIAL PRIMARY KEY,
    EquipmentType VARCHAR(100) NOT NULL,
    MedianLifeExpectancy INT NOT NULL,
    ServiceTeamPriority VARCHAR(50),
    Notes TEXT
);

CREATE TABLE Equipment_TCO (
    TCOID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    ServiceLifeID INT NOT NULL,
    -- Condition Assessment
    AssetCondition INT CHECK (AssetCondition BETWEEN 1 AND 5),
    FailureLikelihood INT CHECK (FailureLikelihood BETWEEN 1 AND 5),
    AssetCriticality INT CHECK (AssetCriticality BETWEEN 1 AND 5),
    ConditionScore NUMERIC(5,2),
    RiskCategory VARCHAR(20),
    -- Service Life
    EstimatedServiceLifeYears INT,
    EstimatedReplacementDate DATE,
    LifecycleStatus VARCHAR(50),
    -- Cost Data
    FirstCost NUMERIC(12,2),
    AnnualMaintenanceCost NUMERIC(12,2),
    ReplacementCost NUMERIC(12,2),
    AnnualEnergyCost NUMERIC(12,2),
    LastAssessmentDate DATE,
    AssessedBy VARCHAR(100),
    Notes TEXT,
    CONSTRAINT fk_tco_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_tco_service_life FOREIGN KEY (ServiceLifeID)
        REFERENCES ASHRAE_Service_Life(ServiceLifeID) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE INDEX idx_tco_equipment ON Equipment_TCO(EquipmentID);

-------------------------------
-- 9. MAPPING MODULE DETAILS
-------------------------------
CREATE TABLE pattern_rules (
    pattern_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    pattern_regex TEXT NOT NULL,
    replacement_template TEXT,
    CONSTRAINT fk_pattern_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE direct_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_value VARCHAR(100) NOT NULL,
    target_value VARCHAR(100) NOT NULL,
    CONSTRAINT fk_direct_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE classification_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_classification VARCHAR(100) NOT NULL,
    target_classification VARCHAR(50) NOT NULL,
    classification_type VARCHAR(50) NOT NULL,  -- e.g., OmniClass, UniFormat, MasterFormat
    CONSTRAINT fk_classification_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE attribute_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    source_attribute VARCHAR(100) NOT NULL,
    target_attribute VARCHAR(100) NOT NULL,
    transformation_rule TEXT,
    CONSTRAINT fk_attribute_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE cost_mappings (
    mapping_id SERIAL PRIMARY KEY,
    rule_id INT NOT NULL,
    cost_type VARCHAR(50) NOT NULL,  -- e.g., Purchase, Installation, Maintenance
    source_currency VARCHAR(10),
    target_currency VARCHAR(10),
    CONSTRAINT fk_cost_mapping_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE equipment_mappings (
    mapping_entry_id SERIAL PRIMARY KEY,  -- surrogate key added
    equipment_id INT NOT NULL,
    rule_id INT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence_score DECIMAL(3,2) NOT NULL,  -- value between 0 and 1
    CONSTRAINT fk_equipmap_equipment FOREIGN KEY (equipment_id)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_equipmap_rule FOREIGN KEY (rule_id)
        REFERENCES mapping_rules(rule_id) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_equipmap_equipment ON equipment_mappings(equipment_id);
CREATE INDEX idx_equipmap_rule ON equipment_mappings(rule_id);

-------------------------------
-- 10. PROJECTS, PHASES, and EQUIPMENT_PROJECTS
-------------------------------
CREATE TABLE Projects (
    ProjectID SERIAL PRIMARY KEY,
    ProjectName VARCHAR(100) NOT NULL,
    ProjectStartDate DATE,
    ProjectEndDate DATE,
    ProjectDescription TEXT
);

CREATE TABLE Project_Phases (
    PhaseID SERIAL PRIMARY KEY,
    ProjectID INT NOT NULL,
    OmniClassPhaseCode VARCHAR(50),
    PhaseTitle VARCHAR(100),
    StartDate DATE,
    EndDate DATE,
    Description TEXT,
    CONSTRAINT fk_phase_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_phase_project ON Project_Phases(ProjectID);

CREATE TABLE Equipment_Projects (
    EquipmentID INT NOT NULL,
    ProjectID INT NOT NULL,
    RoleOrStatus VARCHAR(50),
    StartDate DATE,
    EndDate DATE,
    PRIMARY KEY (EquipmentID, ProjectID),
    CONSTRAINT fk_eproj_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_eproj_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE
);

-------------------------------
-- 11. DOCUMENTATION MANAGEMENT
-------------------------------
CREATE TABLE Document_Types (
    DocTypeID SERIAL PRIMARY KEY,
    TypeName VARCHAR(50) NOT NULL,
    Description TEXT,
    AllowedFileTypes VARCHAR(100)
);

CREATE TABLE Project_Documents (
    DocumentID SERIAL PRIMARY KEY,
    ProjectID INT NOT NULL,
    PhaseID INT,
    DocTypeID INT NOT NULL,
    DocumentName VARCHAR(100) NOT NULL,
    FilePath TEXT NOT NULL,
    FileType VARCHAR(50),
    UploadDate DATE,
    Version VARCHAR(20),
    UploadedBy VARCHAR(100),
    Description TEXT,
    CONSTRAINT fk_pd_project FOREIGN KEY (ProjectID)
        REFERENCES Projects(ProjectID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_pd_phase FOREIGN KEY (PhaseID)
        REFERENCES Project_Phases(PhaseID) ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT fk_pd_doctype FOREIGN KEY (DocTypeID)
        REFERENCES Document_Types(DocTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE Equipment_Documents (
    DocID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    DocTypeID INT NOT NULL,
    DocumentName VARCHAR(100) NOT NULL,
    FilePath TEXT NOT NULL,
    FileType VARCHAR(50),
    UploadDate DATE,
    Version VARCHAR(20),
    UploadedBy VARCHAR(100),
    Description TEXT,
    CONSTRAINT fk_ed_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_ed_doctype FOREIGN KEY (DocTypeID)
        REFERENCES Document_Types(DocTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);

-------------------------------
-- 12. MAINTENANCE and MAINTENANCE_COSTS
-------------------------------
CREATE TABLE Maintenance (
    MaintenanceID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    MaintenanceDate DATE NOT NULL,
    WorkPerformed TEXT,
    Technician VARCHAR(100),
    NextDueDate DATE,
    Comments TEXT,
    CONSTRAINT fk_maintenance_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE Maintenance_Costs (
    MaintCostID SERIAL PRIMARY KEY,
    MaintenanceID INT NOT NULL,
    CostType VARCHAR(50) NOT NULL,
    Amount NUMERIC(12,2) NOT NULL,
    Comments TEXT,
    CONSTRAINT fk_maintcost_maintenance FOREIGN KEY (MaintenanceID)
        REFERENCES Maintenance(MaintenanceID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_maintcost_maintenance ON Maintenance_Costs(MaintenanceID);

-------------------------------
-- 13. CONTROL BOARD MANAGEMENT
-------------------------------
CREATE TABLE Control_Board_Items (
    ControlItemID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    ItemName VARCHAR(100) NOT NULL,
    Description TEXT,
    Location VARCHAR(100),
    SetPoint VARCHAR(50),
    NormalRange VARCHAR(50),
    Units VARCHAR(50),
    CONSTRAINT fk_cbi_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_cbi_equipment ON Control_Board_Items(EquipmentID);

CREATE TABLE Control_Board_Images (
    ImageID SERIAL PRIMARY KEY,
    ControlItemID INT NOT NULL,
    ImagePath TEXT NOT NULL,
    ImageType VARCHAR(50),
    CaptureDate DATE,
    Description TEXT,
    CONSTRAINT fk_cbi_images FOREIGN KEY (ControlItemID)
        REFERENCES Control_Board_Items(ControlItemID) ON UPDATE CASCADE ON DELETE CASCADE
);

-------------------------------
-- 14. QUALITY CONTROL MANAGEMENT
-------------------------------
CREATE TABLE Quality_Control_Types (
    QCTypeID SERIAL PRIMARY KEY,
    TypeName VARCHAR(100) NOT NULL,  -- e.g., Service Verification, Installation Check, Data Accuracy
    Description TEXT,
    Department VARCHAR(100),
    RequiresApproval BOOLEAN DEFAULT FALSE
);

CREATE TABLE Quality_Control_Records (
    QCID SERIAL PRIMARY KEY,
    EquipmentID INT NOT NULL,
    QCTypeID INT NOT NULL,
    Verified BOOLEAN,
    VerificationDate DATE,
    VerifiedBy VARCHAR(100),
    Notes TEXT,
    Status VARCHAR(50),  -- e.g., Pending, Approved, Failed
    ApprovedBy VARCHAR(100),
    ApprovalDate DATE,
    CONSTRAINT fk_qcr_equipment FOREIGN KEY (EquipmentID)
        REFERENCES Equipment(EquipmentID) ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_qcr_qctype FOREIGN KEY (QCTypeID)
        REFERENCES Quality_Control_Types(QCTypeID) ON UPDATE CASCADE ON DELETE RESTRICT
);
CREATE INDEX idx_qcr_equipment ON Quality_Control_Records(EquipmentID);
CREATE INDEX idx_qcr_qctype ON Quality_Control_Records(QCTypeID);

-------------------------------
-- INDEXES for Other Tables (if needed)
-------------------------------
CREATE INDEX idx_maintenance_equipment ON Maintenance(EquipmentID);

-------------------------------
-- VIEWS (Common Business Patterns)
-------------------------------

-- View joining Equipment with its full classification details and location.
CREATE OR REPLACE VIEW v_equipment_full_details AS
SELECT 
    e.EquipmentID,
    e.EquipmentTag,
    e.Manufacturer,
    e.Model,
    e.SerialNumber,
    e.Capacity,
    e.InstallDate,
    e.Status,
    ec.CategoryName,
    ec.CategoryDescription,
    oc.OmniClassCode,
    oc.OmniClassTitle,
    uf.UniFormatCode,
    uf.UniFormatTitle,
    mf.MasterFormatCode,
    mf.MasterFormatTitle,
    cs.CatalogCode,
    cs.CatalogTitle,
    mc.SystemCategory,
    mc.SystemName,
    mc.SubSystemType,
    mc.SubSystemClassification,
    mc.EquipmentSize,
    mc.Notes,
    l.BuildingName,
    l.Floor,
    l.Room,
    l.OtherLocationInfo,
    l.XCoordinate,
    l.YCoordinate
FROM Equipment e
JOIN Equipment_Categories ec ON e.CategoryID = ec.CategoryID
LEFT JOIN Equipment_Categories_OmniClass eco ON ec.CategoryID = eco.CategoryID
LEFT JOIN OmniClass oc ON eco.OmniClassID = oc.OmniClassID
LEFT JOIN Equipment_Categories_UniFormat ecu ON ec.CategoryID = ecu.CategoryID
LEFT JOIN UniFormat uf ON ecu.UniFormatID = uf.UniFormatID
LEFT JOIN Equipment_Categories_MasterFormat ecm ON ec.CategoryID = ecm.CategoryID
LEFT JOIN MasterFormat mf ON ecm.MasterFormatID = mf.MasterFormatID
LEFT JOIN Equipment_Categories_Catalog ecc ON ec.CategoryID = ecc.CategoryID
LEFT JOIN CatalogSystem cs ON ecc.CatalogID = cs.CatalogID
LEFT JOIN Equipment_Categories_MCAA ecmc ON ec.CategoryID = ecmc.CategoryID
LEFT JOIN MCAA_Classifications mc ON ecmc.MCAAID = mc.MCAAID
JOIN Locations l ON e.LocationID = l.LocationID;

-- View for equipment mapping details.
CREATE OR REPLACE VIEW v_equipment_mapping AS
SELECT 
    em.mapping_entry_id,
    e.EquipmentID,
    e.EquipmentTag,
    mr.rule_id,
    mr.rule_name,
    em.applied_at,
    em.confidence_score
FROM equipment_mappings em
JOIN Equipment e ON em.equipment_id = e.EquipmentID
JOIN mapping_rules mr ON em.rule_id = mr.rule_id;

-- View for project documents with related project and phase information.
CREATE OR REPLACE VIEW v_project_documents AS
SELECT 
    pd.DocumentID,
    p.ProjectID,
    p.ProjectName,
    pp.PhaseID,
    pp.PhaseTitle,
    pd.DocumentName,
    pd.FilePath,
    pd.FileType,
    pd.UploadDate,
    pd.Version,
    pd.UploadedBy,
    pd.Description
FROM Project_Documents pd
JOIN Projects p ON pd.ProjectID = p.ProjectID
LEFT JOIN Project_Phases pp ON pd.PhaseID = pp.PhaseID;

-- View for equipment documents.
CREATE OR REPLACE VIEW v_equipment_documents AS
SELECT 
    ed.DocID,
    e.EquipmentID,
    e.EquipmentTag,
    ed.DocumentName,
    ed.FilePath,
    ed.FileType,
    ed.UploadDate,
    ed.Version,
    ed.UploadedBy,
    ed.Description
FROM Equipment_Documents ed
JOIN Equipment e ON ed.EquipmentID = e.EquipmentID;

-- View for maintenance details.
CREATE OR REPLACE VIEW v_equipment_maintenance AS
SELECT 
    m.MaintenanceID,
    e.EquipmentID,
    e.EquipmentTag,
    m.MaintenanceDate,
    m.WorkPerformed,
    m.Technician,
    m.NextDueDate,
    m.Comments
FROM Maintenance m
JOIN Equipment e ON m.EquipmentID = e.EquipmentID;
```

## File: db/staging/procedures/staging_to_master_proc.sql
```sql
-------------------------------
-- STAGING TO MASTER PROCEDURES
-------------------------------
-- These procedures handle the ETL process from staging to master tables

DROP SCHEMA IF EXISTS etl CASCADE;
CREATE SCHEMA etl;

CREATE OR REPLACE PROCEDURE etl.process_staging_data(
    batch_id VARCHAR DEFAULT NULL,
    limit_rows INTEGER DEFAULT 1000
)
LANGUAGE plpgsql
AS $$
DECLARE
    staging_rec RECORD;
    equipment_id INTEGER;
    category_id INTEGER;
    location_id INTEGER;
    cursor_count INTEGER := 0;
BEGIN
    -- Process a batch of records
    FOR staging_rec IN 
        SELECT * FROM staging.equipment_staging 
        WHERE (batch_id IS NULL OR import_batch_id = batch_id)
        AND processing_status = 'PENDING'
        LIMIT limit_rows
    LOOP
        BEGIN
            -- Mark as processing
            UPDATE staging.equipment_staging 
            SET processing_status = 'PROCESSING'
            WHERE staging_id = staging_rec.staging_id;
            
            -- Begin transaction for this record
            -- STEP 1: Process Equipment Category
            SELECT CategoryID INTO category_id
            FROM Equipment_Categories
            WHERE CategoryName = staging_rec.category_name;
            
            IF NOT FOUND THEN
                -- Create new category
                INSERT INTO Equipment_Categories(CategoryName, CategoryDescription)
                VALUES (staging_rec.category_name, 'Auto-created from staging')
                RETURNING CategoryID INTO category_id;
                -- Add classification mappings here
            END IF;
            
            -- STEP 2: Process Location
            SELECT LocationID INTO location_id
            FROM Locations
            WHERE BuildingName = staging_rec.building_name
            AND (Floor = staging_rec.floor OR (Floor IS NULL AND staging_rec.floor IS NULL))
            AND (Room = staging_rec.room OR (Room IS NULL AND staging_rec.room IS NULL));
            
            IF NOT FOUND THEN
                -- Create new location
                INSERT INTO Locations(BuildingName, Floor, Room, OtherLocationInfo, XCoordinate, YCoordinate)
                VALUES (
                    staging_rec.building_name,
                    staging_rec.floor,
                    staging_rec.room,
                    staging_rec.other_location_info,
                    staging_rec.x_coordinate,
                    staging_rec.y_coordinate
                )
                RETURNING LocationID INTO location_id;
            END IF;
            
            -- STEP 3: Process Equipment record
            SELECT EquipmentID INTO equipment_id
            FROM Equipment
            WHERE EquipmentTag = staging_rec.equipment_tag;
            
            IF NOT FOUND THEN
                -- Create new equipment
                INSERT INTO Equipment(
                    CategoryID,
                    LocationID,
                    EquipmentTag,
                    Manufacturer,
                    Model,
                    SerialNumber,
                    Capacity,
                    InstallDate,
                    Status,
                    rule_id
                )
                VALUES (
                    category_id,
                    location_id,
                    staging_rec.equipment_tag,
                    staging_rec.manufacturer,
                    staging_rec.model,
                    staging_rec.serial_number,
                    staging_rec.capacity,
                    staging_rec.install_date,
                    staging_rec.status,
                    staging_rec.mapping_rule_id
                )
                RETURNING EquipmentID INTO equipment_id;
            ELSE
                -- Update existing equipment
                UPDATE Equipment
                SET 
                    CategoryID = category_id,
                    LocationID = location_id,
                    Manufacturer = staging_rec.manufacturer,
                    Model = staging_rec.model,
                    SerialNumber = staging_rec.serial_number,
                    Capacity = staging_rec.capacity,
                    InstallDate = staging_rec.install_date,
                    Status = staging_rec.status,
                    rule_id = staging_rec.mapping_rule_id
                WHERE EquipmentID = equipment_id;
            END IF;
            
            -- STEP 4: Process attributes
            IF staging_rec.attributes IS NOT NULL THEN
                -- Process each attribute from JSON
                -- (simplified, would expand in real implementation)
            END IF;
            
            -- STEP 5: Process costs
            IF staging_rec.initial_cost IS NOT NULL THEN
                -- Insert or update costs
                -- (simplified, would expand in real implementation)
            END IF;
            
            -- Mark as processed
            UPDATE staging.equipment_staging 
            SET 
                processing_status = 'COMPLETED',
                is_processed = TRUE,
                processed_timestamp = CURRENT_TIMESTAMP
            WHERE staging_id = staging_rec.staging_id;
            
            cursor_count := cursor_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            -- Mark as error
            UPDATE staging.equipment_staging 
            SET 
                processing_status = 'ERROR',
                error_message = SQLERRM
            WHERE staging_id = staging_rec.staging_id;
        END;
    END LOOP;
    
    COMMIT;
    RAISE NOTICE 'Processed % records', cursor_count;
END;
$$;
```

## File: db/staging/schema/staging_schema.sql
```sql
-------------------------------
-- STAGING SCHEMA FOR FCA DASHBOARD
-------------------------------
-- This schema contains a single comprehensive staging table
-- to hold all incoming data before transformation and loading
-- into the master schema tables.

DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;

-- Central staging table for all incoming data
CREATE TABLE staging.equipment_staging (
    -- Staging metadata fields
    staging_id SERIAL PRIMARY KEY,
    source_system VARCHAR(100),
    import_batch_id VARCHAR(100),
    import_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, ERROR
    error_message TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    processed_timestamp TIMESTAMPTZ,
    
    -- Equipment fields
    equipment_tag VARCHAR(50),
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    serial_number VARCHAR(100),
    capacity FLOAT,
    install_date DATE,
    status VARCHAR(50),
    
    -- Classification fields
    category_name VARCHAR(100),
    omniclass_code VARCHAR(50),
    omniclass_title VARCHAR(100),
    uniformat_code VARCHAR(50),
    uniformat_title VARCHAR(100),
    masterformat_code VARCHAR(50),
    masterformat_title VARCHAR(100),
    catalog_code VARCHAR(50),
    catalog_title VARCHAR(100),
    mcaa_system_category VARCHAR(100),
    mcaa_system_name VARCHAR(100),
    mcaa_subsystem_type VARCHAR(100),
    mcaa_subsystem_classification VARCHAR(100),
    mcaa_equipment_size VARCHAR(50),
    
    -- Location fields
    building_name VARCHAR(100),
    floor VARCHAR(50),
    room VARCHAR(50),
    other_location_info TEXT,
    x_coordinate DECIMAL(10,6),
    y_coordinate DECIMAL(10,6),
    
    -- Attribute fields (flexible JSON structure for varying attributes)
    attributes JSONB,
    
    -- Cost fields
    initial_cost NUMERIC(12,2),
    installation_cost NUMERIC(12,2),
    annual_maintenance_cost NUMERIC(12,2),
    replacement_cost NUMERIC(12,2),
    annual_energy_cost NUMERIC(12,2),
    cost_date DATE,
    cost_comments TEXT,
    
    -- TCO fields
    asset_condition INT,
    failure_likelihood INT,
    asset_criticality INT,
    condition_score NUMERIC(5,2),
    risk_category VARCHAR(20),
    estimated_service_life_years INT,
    estimated_replacement_date DATE,
    lifecycle_status VARCHAR(50),
    last_assessment_date DATE,
    assessed_by VARCHAR(100),
    
    -- Service life info
    equipment_type VARCHAR(100),
    median_life_expectancy INT,
    service_team_priority VARCHAR(50),
    
    -- Maintenance fields
    maintenance_data JSONB, -- Flexible structure for maintenance records
    
    -- Project fields
    project_data JSONB, -- Flexible structure for project relationships
    
    -- Document fields
    document_data JSONB, -- Flexible structure for documents
    
    -- Quality control fields
    qc_data JSONB, -- Flexible structure for quality control records
    
    -- Source data fields (for troubleshooting)
    raw_source_data JSONB,
    source_file_name VARCHAR(255),
    source_record_id VARCHAR(100),
    
    -- Mapping fields
    mapping_rule_id INT,
    mapping_confidence_score DECIMAL(3,2)
);

-- Indexes for performance
CREATE INDEX idx_staging_equipment_tag ON staging.equipment_staging(equipment_tag);
CREATE INDEX idx_staging_status ON staging.equipment_staging(processing_status);
CREATE INDEX idx_staging_batch ON staging.equipment_staging(import_batch_id);
CREATE INDEX idx_staging_processed ON staging.equipment_staging(is_processed);
CREATE INDEX idx_staging_attributes ON staging.equipment_staging USING GIN (attributes);
CREATE INDEX idx_staging_maintenance_data ON staging.equipment_staging USING GIN (maintenance_data);
CREATE INDEX idx_staging_project_data ON staging.equipment_staging USING GIN (project_data);
CREATE INDEX idx_staging_document_data ON staging.equipment_staging USING GIN (document_data);
CREATE INDEX idx_staging_qc_data ON staging.equipment_staging USING GIN (qc_data);

-- View for pending items
CREATE OR REPLACE VIEW staging.v_pending_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'PENDING';

-- View for errored items
CREATE OR REPLACE VIEW staging.v_error_items AS
SELECT * FROM staging.equipment_staging 
WHERE processing_status = 'ERROR';

-- Function to reset status of errored items
CREATE OR REPLACE FUNCTION staging.reset_error_items()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE staging.equipment_staging
    SET processing_status = 'PENDING',
        error_message = NULL
    WHERE processing_status = 'ERROR';
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clear staging table
CREATE OR REPLACE FUNCTION staging.clear_processed_items(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM staging.equipment_staging
    WHERE is_processed = TRUE 
    AND processed_timestamp < CURRENT_DATE - days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

## File: examples/__init__.py
```python
"""
Examples package for the FCA Dashboard application.

This package contains example scripts demonstrating how to use
various components of the FCA Dashboard application.
"""
```

## File: examples/excel_analyzer_example.py
```python
"""
Example script for analyzing Excel files using the robust Excel utilities.

This script demonstrates how to use the advanced Excel utilities to extract data from Excel files
with complex structures, such as headers in non-standard positions, multiple sheets, etc.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_excel_structure,
    extract_excel_with_config,
    normalize_sheet_names,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def analyze_and_extract_medtronics_file(file_path):
    """
    Analyze and extract data from the Medtronics Asset Log Uploader Excel file using the robust Excel utilities.
    
    Args:
        file_path: Path to the Excel file to analyze.
        
    Returns:
        A dictionary containing the extracted data for each sheet.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Analyzing Medtronics Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get normalized sheet names
    sheet_name_mapping = normalize_sheet_names(file_path)
    print(f"Normalized sheet names: {sheet_name_mapping}")
    
    # Create a configuration for the Medtronics Excel file
    # This configuration is based on our analysis of the file structure
    config = {
        "default": {
            "header_row": None,  # Auto-detect for most sheets
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "Asset Data": {
            "header_row": 6,  # We know the header starts at row 7 (index 6)
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "EQ IDs": {
            "header_row": 0,  # Header is in the first row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "Cobie": {
            "header_row": 1,  # Header is in the second row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "Dropdowns": {
            "header_row": 0,  # Header is in the first row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        }
    }
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Print information about each extracted sheet
    for sheet_name, df in extracted_data.items():
        print(f"\nProcessed sheet: {sheet_name}")
        print(f"Extracted {len(df)} rows with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        print(f"First few rows:")
        print(df.head(3))
    
    return extracted_data


def save_config_example():
    """
    Example of how to save a configuration to a JSON file.
    """
    # Create a sample configuration
    config = {
        "default": {
            "header_row": None,  # Auto-detect
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
            "convert_dtypes": True,
            "date_columns": ["Date", "Scheduled Delivery Date", "Actual On-Site Date"],
            "numeric_columns": ["Motor HP", "Size"],
            "boolean_columns": ["O&M Received", "Attic Stock"],
        },
        "Asset Data": {
            "header_row": 6,  # We know the header starts at row 7 (index 6)
        },
        "Equipment Log": {
            "required_columns": ["Equipment Name", "Equipment Tag ID"],
        }
    }
    
    # Save the configuration to a JSON file
    config_path = Path("excel_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved configuration to {config_path}")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Analyze and extract data from the Medtronics Excel file
    extracted_data = analyze_and_extract_medtronics_file(file_path)
    
    # Print a summary of the extracted data
    print("\nExtraction Summary:")
    for sheet_name, df in extracted_data.items():
        print(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
    
    # Uncomment to save a sample configuration
    # save_config_example()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## File: examples/excel_export_example.py
```python
"""
Example script for exporting Excel data to different formats.

This script demonstrates how to use the Excel utilities to extract data from Excel files
and save it in different formats (CSV, Excel, SQLite database).
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_excel_structure,
    convert_excel_to_csv,
    extract_excel_with_config,
    save_excel_to_database,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def extract_and_export_data(file_path, output_dir):
    """
    Extract data from an Excel file and export it to different formats.
    
    Args:
        file_path: Path to the Excel file to analyze.
        output_dir: Directory to save the output files.
        
    Returns:
        A dictionary containing the paths to the exported files.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Extracting data from Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get extraction configuration from settings
    extraction_config = settings.get("excel_utils.extraction", {})
    
    # Convert sheet names to match the format in the settings
    # The settings use lowercase with underscores, but the Excel file uses spaces and title case
    config = {}
    
    # Add default settings
    if "default" in extraction_config:
        config["default"] = extraction_config["default"]
    
    # Add sheet-specific settings
    for sheet_name in analysis['sheet_names']:
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = sheet_name.lower().replace(" ", "_")
        
        # If there are settings for this sheet, add them to the config
        if settings_key in extraction_config:
            config[sheet_name] = extraction_config[settings_key]
    
    print(f"Using extraction configuration from settings: {config}")
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a dictionary to store the paths to the exported files
    exported_files = {}
    
    # Export each sheet to different formats
    for sheet_name, df in extracted_data.items():
        print(f"\nExporting sheet: {sheet_name}")
        
        # Skip empty sheets
        if len(df) == 0:
            print(f"  Skipping empty sheet: {sheet_name}")
            continue
        
        # 1. Export to CSV
        csv_path = os.path.join(output_dir, f"{sheet_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Exported to CSV: {csv_path}")
        exported_files[f"{sheet_name}_csv"] = csv_path
        
        # 2. Export to Excel
        excel_path = os.path.join(output_dir, f"{sheet_name}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"  Exported to Excel: {excel_path}")
        exported_files[f"{sheet_name}_excel"] = excel_path
        
        # 3. Export to SQLite database
        db_path = os.path.join(output_dir, "excel_data.db")
        connection_string = f"sqlite:///{db_path}"
        save_excel_to_database(
            df=df,
            table_name=sheet_name,
            connection_string=connection_string,
            if_exists="replace"
        )
        print(f"  Exported to SQLite database: {db_path}, table: {sheet_name}")
        exported_files["sqlite_db"] = db_path
    
    return exported_files


def verify_exports(exported_files):
    """
    Verify that the exported files were created correctly.
    
    Args:
        exported_files: Dictionary containing the paths to the exported files.
    """
    print("\nVerifying exported files:")
    
    # Verify CSV files
    for key, path in exported_files.items():
        if key.endswith("_csv"):
            if os.path.exists(path):
                # Read the CSV file to verify it contains data
                df = pd.read_csv(path)
                print(f"  CSV file {path}: {len(df)} rows, {len(df.columns)} columns")
            else:
                print(f"  Error: CSV file {path} not found")
        
        elif key.endswith("_excel"):
            if os.path.exists(path):
                # Read the Excel file to verify it contains data
                df = pd.read_excel(path)
                print(f"  Excel file {path}: {len(df)} rows, {len(df.columns)} columns")
            else:
                print(f"  Error: Excel file {path} not found")
    
    # Verify SQLite database
    if "sqlite_db" in exported_files:
        db_path = exported_files["sqlite_db"]
        if os.path.exists(db_path):
            # Connect to the database and list tables
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"  SQLite database {db_path} contains tables: {[table[0] for table in tables]}")
            
            # Query each table to verify it contains data
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                print(f"    Table {table_name}: {row_count} rows")
            
            conn.close()
        else:
            print(f"  Error: SQLite database {db_path} not found")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Get the output directory
    output_dir = os.path.join(get_root_dir(), "examples", "output")
    
    # Extract and export data
    exported_files = extract_and_export_data(file_path, output_dir)
    
    # Verify the exported files
    verify_exports(exported_files)
    
    print("\nExport completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## File: examples/excel_extractor_example.py
```python
"""
Example script demonstrating the use of the Excel extractor.

This script shows how to use the Excel extractor to load data from an Excel file
into a pandas DataFrame and perform basic operations on it.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.excel_extractor import extract_excel_to_dataframe
from fca_dashboard.utils.path_util import get_root_dir


def main():
    """
    Demonstrate the Excel extractor functionality.
    
    This function creates a sample Excel file, extracts data from it,
    and shows how to work with the resulting DataFrame.
    """
    try:
        # Import pandas here to avoid import errors in the module
        import pandas as pd
        
        # Get settings for examples
        examples_dir_name = settings.get("file_paths.examples_dir", "examples")
        examples_dir = get_root_dir() / examples_dir_name
        os.makedirs(examples_dir, exist_ok=True)
        
        # Get sample filename from settings
        sample_filename = settings.get("examples.excel.sample_filename", "sample_data.xlsx")
        
        # Create a sample Excel file
        sample_data = {
            "ID": [1, 2, 3, 4, 5],
            "Product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
            "Price": [10.99, 20.50, 15.75, 8.25, 30.00],
            "InStock": [True, False, True, True, False],
            "Category": ["Electronics", "Tools", "Electronics", "Office", "Tools"]
        }
        
        # Create a DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save the DataFrame to an Excel file
        sample_file = examples_dir / sample_filename
        df.to_excel(sample_file, index=False)
        print(f"Created sample Excel file: {sample_file}")
        
        # Extract data from the Excel file
        print("\nExtracting data from Excel file...")
        # Get uploads directory from settings for informational purposes
        uploads_dir_name = settings.get("file_paths.uploads_dir", "uploads")
        uploads_dir = get_root_dir() / uploads_dir_name
        print(f"File will be uploaded to: {uploads_dir}")
        extracted_df = extract_excel_to_dataframe(sample_file, upload=True)
        
        # Display the extracted data
        print("\nExtracted DataFrame:")
        print(extracted_df)
        
        # Demonstrate filtering columns
        print("\nExtracting only specific columns...")
        # Get columns to extract from settings
        columns_to_extract = settings.get("examples.excel.columns_to_extract", ["ID", "Product", "Price"])
        filtered_df = extract_excel_to_dataframe(sample_file, columns=columns_to_extract)
        print("\nFiltered DataFrame:")
        print(filtered_df)
        
        # Demonstrate basic DataFrame operations
        print("\nPerforming basic DataFrame operations:")
        
        # Filter rows based on a condition
        # Get price threshold from settings
        price_threshold = settings.get("examples.excel.price_threshold", 15)
        print(f"\n1. Filter products with price > {price_threshold}:")
        expensive_products = extracted_df[extracted_df["Price"] > price_threshold]
        print(expensive_products)
        
        # Group by a column and calculate statistics
        print("\n2. Group by Category and calculate average price:")
        category_stats = extracted_df.groupby("Category")["Price"].mean()
        print(category_stats)
        
        # Sort the DataFrame
        print("\n3. Sort by Price (descending):")
        sorted_df = extracted_df.sort_values("Price", ascending=False)
        print(sorted_df)
        
        print("\nExample completed successfully!")
        
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Please ensure pandas and openpyxl are installed.")
        return 1
    except Exception as e:
        print(f"Error in example: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## File: examples/excel_unique_values_analysis_example.py
```python
"""
Example script for analyzing Excel data.

This script demonstrates how to use the Excel analysis utilities to analyze data
from Excel files, including analyzing unique values, column statistics, and text patterns.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    extract_excel_with_config,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def extract_and_analyze_data(file_path, output_dir):
    """
    Extract data from an Excel file and analyze it.
    
    Args:
        file_path: Path to the Excel file to analyze.
        output_dir: Directory to save the analysis reports.
        
    Returns:
        A dictionary containing the analysis results.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Extracting and analyzing data from Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get extraction configuration from settings
    extraction_config = settings.get("excel_utils.extraction", {})
    
    # Convert sheet names to match the format in the settings
    # The settings use lowercase with underscores, but the Excel file uses spaces and title case
    config = {}
    
    # Add default settings
    if "default" in extraction_config:
        config["default"] = extraction_config["default"]
    
    # Add sheet-specific settings
    for sheet_name in analysis['sheet_names']:
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = sheet_name.lower().replace(" ", "_")
        
        # If there are settings for this sheet, add them to the config
        if settings_key in extraction_config:
            config[sheet_name] = extraction_config[settings_key]
    
    print(f"Using extraction configuration from settings: {config}")
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a dictionary to store the analysis results
    analysis_results = {}
    
    # Analyze each sheet
    for sheet_name, df in extracted_data.items():
        print(f"\nAnalyzing sheet: {sheet_name}")
        
        # Skip empty sheets
        if len(df) == 0:
            print(f"  Skipping empty sheet: {sheet_name}")
            continue
        
        # Initialize sheet results
        sheet_results = {}
        
        # Get analysis configuration from settings
        analysis_config = settings.get("excel_utils.analysis", {})
        
        # Get default analysis settings
        default_analysis = analysis_config.get("default", {})
        
        # 1. Analyze unique values
        print(f"  Analyzing unique values...")
        # Get unique values settings
        unique_values_settings = default_analysis.get("unique_values", {})
        max_unique_values = unique_values_settings.get("max_unique_values", 20)
        
        # Select a subset of columns for unique value analysis
        # For demonstration purposes, we'll analyze the first 5 columns
        unique_columns = list(df.columns[:5])
        unique_values_results = analyze_unique_values(
            df,
            columns=unique_columns,
            max_unique_values=max_unique_values
        )
        sheet_results['unique_values'] = unique_values_results
        
        # 2. Analyze column statistics for numeric columns
        print(f"  Analyzing column statistics...")
        # Get column statistics settings
        column_stats_settings = default_analysis.get("column_statistics", {})
        include_outliers = column_stats_settings.get("include_outliers", True)
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns[:5])
            sheet_results['column_statistics'] = column_stats_results
        
        # 3. Analyze text columns
        print(f"  Analyzing text columns...")
        # Get text analysis settings
        text_analysis_settings = default_analysis.get("text_analysis", {})
        include_pattern_analysis = text_analysis_settings.get("include_pattern_analysis", True)
        
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            text_analysis_results = analyze_text_columns(df, columns=text_columns[:5])
            sheet_results['text_analysis'] = text_analysis_results
        
        # Store the results
        analysis_results[sheet_name] = sheet_results
        
        # Save the analysis report
        save_analysis_report(sheet_name, df, sheet_results, output_dir)
    
    return analysis_results


def save_analysis_report(sheet_name, df, results, output_dir):
    """
    Save an analysis report for a sheet.
    
    Args:
        sheet_name: Name of the sheet.
        df: DataFrame that was analyzed.
        results: Analysis results.
        output_dir: Directory to save the report.
    """
    # Create a report file
    report_path = os.path.join(output_dir, f"{sheet_name}_analysis_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Analysis Report for Sheet: {sheet_name}\n")
        f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Write unique values report
        if 'unique_values' in results:
            f.write("Unique Values Analysis:\n")
            f.write("-" * 50 + "\n")
            
            unique_values = results['unique_values']
            for col, res in unique_values.items():
                f.write(f"  Column: {col}\n")
                f.write(f"    Unique value count: {res['count']}\n")
                f.write(f"    Null count: {res['null_count']} ({res['null_percentage'] * 100:.2f}%)\n")
                
                if 'values' in res:
                    f.write(f"    Unique values: {', '.join(res['values'][:10])}")
                    if len(res['values']) > 10:
                        f.write(f" ... and {len(res['values']) - 10} more")
                    f.write("\n")
                
                if 'value_counts' in res:
                    f.write(f"    Value counts (top 5):\n")
                    sorted_counts = sorted(res['value_counts'].items(), key=lambda x: x[1], reverse=True)
                    for val, count in sorted_counts[:5]:
                        f.write(f"      {val}: {count}\n")
                
                f.write("\n")
            
            f.write("\n")
        
        # Write column statistics report
        if 'column_statistics' in results:
            f.write("Column Statistics Analysis:\n")
            f.write("-" * 50 + "\n")
            
            column_stats = results['column_statistics']
            for col, stats in column_stats.items():
                f.write(f"  Column: {col}\n")
                f.write(f"    Min: {stats['min']}\n")
                f.write(f"    Max: {stats['max']}\n")
                f.write(f"    Mean: {stats['mean']}\n")
                f.write(f"    Median: {stats['median']}\n")
                f.write(f"    Standard deviation: {stats['std']}\n")
                f.write(f"    Q1 (25th percentile): {stats['q1']}\n")
                f.write(f"    Q3 (75th percentile): {stats['q3']}\n")
                f.write(f"    IQR: {stats['iqr']}\n")
                f.write(f"    Outliers count: {stats['outliers_count']}\n")
                f.write("\n")
            
            f.write("\n")
        
        # Write text analysis report
        if 'text_analysis' in results:
            f.write("Text Analysis:\n")
            f.write("-" * 50 + "\n")
            
            text_analysis = results['text_analysis']
            for col, analysis in text_analysis.items():
                f.write(f"  Column: {col}\n")
                f.write(f"    Min length: {analysis['min_length']}\n")
                f.write(f"    Max length: {analysis['max_length']}\n")
                f.write(f"    Average length: {analysis['avg_length']:.2f}\n")
                f.write(f"    Empty strings: {analysis['empty_count']}\n")
                
                if 'pattern_analysis' in analysis:
                    f.write(f"    Pattern analysis:\n")
                    for pattern, count in analysis['pattern_analysis'].items():
                        if count > 0:
                            f.write(f"      {pattern}: {count}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"  Saved analysis report to: {report_path}")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Get the output directory
    output_dir = os.path.join(get_root_dir(), "examples", "output")
    
    # Extract and analyze data
    analysis_results = extract_and_analyze_data(file_path, output_dir)
    
    print("\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## File: examples/excel_validation_example.py
```python
"""
Example script for validating Excel data.

This script demonstrates how to use the Excel validation utilities to validate data
from Excel files, including checking for missing values, duplicate rows, value ranges,
and data types.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_excel_structure,
    check_data_types,
    check_duplicate_rows,
    check_missing_values,
    check_value_ranges,
    extract_excel_with_config,
    validate_dataframe,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def extract_and_validate_data(file_path, output_dir):
    """
    Extract data from an Excel file and validate it.
    
    Args:
        file_path: Path to the Excel file to analyze.
        output_dir: Directory to save the validation reports.
        
    Returns:
        A dictionary containing the validation results.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Extracting and validating data from Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get extraction configuration from settings
    extraction_config = settings.get("excel_utils.extraction", {})
    
    # Convert sheet names to match the format in the settings
    # The settings use lowercase with underscores, but the Excel file uses spaces and title case
    config = {}
    
    # Add default settings
    if "default" in extraction_config:
        config["default"] = extraction_config["default"]
    
    # Add sheet-specific settings
    for sheet_name in analysis['sheet_names']:
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = sheet_name.lower().replace(" ", "_")
        
        # If there are settings for this sheet, add them to the config
        if settings_key in extraction_config:
            config[sheet_name] = extraction_config[settings_key]
    
    print(f"Using extraction configuration from settings: {config}")
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a dictionary to store the validation results
    validation_results = {}
    
    # Validate each sheet
    for sheet_name, df in extracted_data.items():
        print(f"\nValidating sheet: {sheet_name}")
        
        # Skip empty sheets
        if len(df) == 0:
            print(f"  Skipping empty sheet: {sheet_name}")
            continue
        
        # Create a validation configuration based on the sheet
        validation_config = create_validation_config(sheet_name, df)
        
        # Validate the DataFrame
        results = validate_dataframe(df, validation_config)
        validation_results[sheet_name] = results
        
        # Save the validation report
        save_validation_report(sheet_name, df, results, output_dir)
    
    return validation_results


def create_validation_config(sheet_name, df):
    """
    Create a validation configuration based on the sheet name and DataFrame.
    
    Args:
        sheet_name: Name of the sheet.
        df: DataFrame to validate.
        
    Returns:
        A dictionary containing the validation configuration.
    """
    # Get validation configuration from settings
    validation_settings = settings.get("excel_utils.validation", {})
    
    # Initialize the validation configuration with default settings
    validation_config = {}
    
    # Add default settings
    if "default" in validation_settings:
        default_settings = validation_settings["default"]
        
        # Add missing values check
        if "missing_values" in default_settings:
            missing_values_settings = default_settings["missing_values"]
            validation_config["missing_values"] = {
                "columns": missing_values_settings.get("columns") or list(df.columns),
                "threshold": missing_values_settings.get("threshold", 0.5)
            }
        
        # Add duplicate rows check
        if "duplicate_rows" in default_settings:
            duplicate_rows_settings = default_settings["duplicate_rows"]
            validation_config["duplicate_rows"] = {
                "subset": duplicate_rows_settings.get("subset")
            }
        
        # Add data types check
        if "data_types" in default_settings:
            data_types_settings = default_settings["data_types"]
            
            # Create type specifications based on settings
            type_specs = {}
            
            # Add date columns
            for col in data_types_settings.get("date_columns", []):
                if col in df.columns:
                    type_specs[col] = "date"
            
            # Add numeric columns
            for col in data_types_settings.get("numeric_columns", []):
                if col in df.columns:
                    type_specs[col] = "float"
            
            # Add string columns
            for col in data_types_settings.get("string_columns", []):
                if col in df.columns:
                    type_specs[col] = "str"
            
            # Add boolean columns
            for col in data_types_settings.get("boolean_columns", []):
                if col in df.columns:
                    type_specs[col] = "bool"
            
            if type_specs:
                validation_config["data_types"] = type_specs
    
    # Add sheet-specific validation
    # Convert sheet name to the format used in settings (lowercase with underscores)
    settings_key = sheet_name.lower()
    
    if settings_key in validation_settings:
        sheet_settings = validation_settings[settings_key]
        
        # Add value ranges check
        if "value_ranges" in sheet_settings:
            validation_config["value_ranges"] = sheet_settings["value_ranges"]
        
        # Add required columns check
        if "required_columns" in sheet_settings:
            validation_config["required_columns"] = sheet_settings["required_columns"]
    
    # If no validation config was created from settings, create a basic one
    if not validation_config:
        # Add missing values check for all sheets
        validation_config["missing_values"] = {
            "columns": list(df.columns),
            "threshold": 0.5  # Allow up to 50% missing values
        }
        
        # Add duplicate rows check for all sheets
        validation_config["duplicate_rows"] = {
            "subset": None  # Check all columns for duplicates
        }
    
    return validation_config


def save_validation_report(sheet_name, df, results, output_dir):
    """
    Save a validation report for a sheet.
    
    Args:
        sheet_name: Name of the sheet.
        df: DataFrame that was validated.
        results: Validation results.
        output_dir: Directory to save the report.
    """
    # Create a report file
    report_path = os.path.join(output_dir, f"{sheet_name}_validation_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Validation Report for Sheet: {sheet_name}\n")
        f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Write missing values report
        if "missing_values" in results:
            f.write("Missing Values Report:\n")
            f.write("-" * 50 + "\n")
            
            missing_values = results["missing_values"]
            for col, pct in missing_values.items():
                f.write(f"  {col}: {pct * 100:.2f}% missing\n")
            
            f.write("\n")
        
        # Write duplicate rows report
        if "duplicate_rows" in results:
            f.write("Duplicate Rows Report:\n")
            f.write("-" * 50 + "\n")
            
            duplicate_rows = results["duplicate_rows"]
            f.write(f"  Duplicate rows: {duplicate_rows['duplicate_count']}\n")
            
            if duplicate_rows["duplicate_count"] > 0:
                f.write(f"  Duplicate indices: {duplicate_rows['duplicate_indices'][:10]}")
                if len(duplicate_rows["duplicate_indices"]) > 10:
                    f.write(f" ... and {len(duplicate_rows['duplicate_indices']) - 10} more")
                f.write("\n")
            
            f.write("\n")
        
        # Write value ranges report
        if "value_ranges" in results:
            f.write("Value Ranges Report:\n")
            f.write("-" * 50 + "\n")
            
            value_ranges = results["value_ranges"]
            for col, res in value_ranges.items():
                f.write(f"  {col}:\n")
                f.write(f"    Below minimum: {res['below_min']}\n")
                f.write(f"    Above maximum: {res['above_max']}\n")
                f.write(f"    Total outside range: {res['total_outside_range']}\n")
            
            f.write("\n")
        
        # Write data types report
        if "data_types" in results:
            f.write("Data Types Report:\n")
            f.write("-" * 50 + "\n")
            
            data_types = results["data_types"]
            for col, res in data_types.items():
                f.write(f"  {col}:\n")
                f.write(f"    Expected type: {res['expected_type']}\n")
                f.write(f"    Current type: {res['current_type']}\n")
                f.write(f"    Error count: {res['error_count']}\n")
            
            f.write("\n")
    
    print(f"  Saved validation report to: {report_path}")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Get the output directory
    output_dir = os.path.join(get_root_dir(), "examples", "output")
    
    # Extract and validate data
    validation_results = extract_and_validate_data(file_path, output_dir)
    
    print("\nValidation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## File: extractors/base_extractor.py
```python
"""
Base extractor module for the FCA Dashboard application.

This module provides the base classes and interfaces for data extraction
from various file formats.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


class ExtractionError(FCADashboardError):
    """Exception raised for errors during data extraction."""
    pass


class DataExtractor(ABC):
    """Base class for all data extractors."""
    
    def __init__(self, logger=None):
        """
        Initialize the extractor with an optional logger.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
        """
        self.logger = logger or get_logger(self.__class__.__name__)
    
    @abstractmethod
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this extractor can handle the file, False otherwise.
        """
        pass
    
    @abstractmethod
    def extract(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from the file into a pandas DataFrame.
        
        Args:
            file_path: Path to the file to extract data from.
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If an error occurs during extraction.
        """
        pass
    
    def extract_and_save(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data and optionally save it to a file.
        
        Args:
            file_path: Path to the file to extract data from.
            output_path: Optional path to save the extracted data to.
                If None, the data is not saved.
            output_format: Format to save the data in (default: "csv").
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If an error occurs during extraction or saving.
        """
        # Extract the data
        df = self.extract(file_path, **kwargs)
        
        # Save the data if an output path is provided
        if output_path:
            output_path = resolve_path(output_path)
            
            # Create the directory if it doesn't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            try:
                if output_format.lower() == "csv":
                    df.to_csv(output_path, index=False)
                elif output_format.lower() in ["xlsx", "excel"]:
                    df.to_excel(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
                
                self.logger.info(f"Saved extracted data to {output_path}")
            except Exception as e:
                error_msg = f"Error saving data to {output_path}: {str(e)}"
                self.logger.error(error_msg)
                raise ExtractionError(error_msg) from e
        
        return df


class ExtractorFactory:
    """Factory for creating extractors based on file type."""
    
    def __init__(self):
        """Initialize the factory with an empty list of extractors."""
        self.extractors: List[DataExtractor] = []
    
    def register_extractor(self, extractor: DataExtractor) -> None:
        """
        Register an extractor with the factory.
        
        Args:
            extractor: The extractor to register.
        """
        self.extractors.append(extractor)
    
    def get_extractor(self, file_path: Union[str, Path]) -> Optional[DataExtractor]:
        """
        Get an appropriate extractor for the given file.
        
        Args:
            file_path: Path to the file to extract data from.
            
        Returns:
            An appropriate extractor, or None if no suitable extractor is found.
        """
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None
    
    def extract(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from the file using an appropriate extractor.
        
        Args:
            file_path: Path to the file to extract data from.
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If no suitable extractor is found or an error occurs during extraction.
        """
        extractor = self.get_extractor(file_path)
        if not extractor:
            raise ExtractionError(f"No suitable extractor found for {file_path}")
        
        return extractor.extract(file_path, **kwargs)
    
    def extract_and_save(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data and optionally save it to a file.
        
        Args:
            file_path: Path to the file to extract data from.
            output_path: Optional path to save the extracted data to.
                If None, the data is not saved.
            output_format: Format to save the data in (default: "csv").
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If no suitable extractor is found or an error occurs during extraction or saving.
        """
        extractor = self.get_extractor(file_path)
        if not extractor:
            raise ExtractionError(f"No suitable extractor found for {file_path}")
        
        return extractor.extract_and_save(file_path, output_path, output_format, **kwargs)


# Create a global factory instance
extractor_factory = ExtractorFactory()
```

## File: extractors/excel_extractor.py
```python
"""
Excel extractor module for the FCA Dashboard application.

This module provides functionality for extracting data from Excel files
and loading it into pandas DataFrames, with features like error handling,
logging, and optional file upload.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.base_extractor import DataExtractor, ExtractionError, extractor_factory

# Alias for backward compatibility
ExcelExtractionError = ExtractionError
from fca_dashboard.utils.excel import get_excel_file_type, is_excel_file
from fca_dashboard.utils.path_util import get_root_dir, resolve_path
from fca_dashboard.utils.upload_util import upload_file


class ExcelExtractor(DataExtractor):
    """
    Extractor for Excel files (XLSX, XLS, XLSM, XLSB) and CSV files.
    
    This extractor can handle various Excel formats and CSV files,
    with options for sheet selection, column filtering, and file uploading.
    """
    
    def __init__(self, upload_service=None, logger=None):
        """
        Initialize the Excel extractor.
        
        Args:
            upload_service: Optional service for uploading files. If None, uses the default upload_file function.
            logger: Optional logger instance. If None, a default logger will be created.
        """
        super().__init__(logger)
        self.upload_service = upload_service or upload_file
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this extractor can handle the file, False otherwise.
        """
        return is_excel_file(file_path)
    
    def extract(
        self,
        file_path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        columns: Optional[List[str]] = None,
        upload: bool = False,
        target_filename: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from an Excel file into a pandas DataFrame.
        
        Args:
            file_path: Path to the Excel file. Can be absolute or relative.
            sheet_name: Name or index of the sheet to extract (default: 0, first sheet).
            columns: Optional list of column names to extract. If None, extracts all columns.
            upload: Whether to upload the file to the uploads directory.
            target_filename: Optional filename to use when uploading. If None, uses the source filename.
            **kwargs: Additional extraction options passed to pandas.read_excel or pandas.read_csv.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            FileNotFoundError: If the source file does not exist.
            ExtractionError: If an error occurs during the extraction process.
        """
        # Resolve the file path
        source_path = resolve_path(file_path)
        
        # Validate source file
        if not source_path.is_file():
            self.logger.error(f"Source file not found: {source_path}")
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Log the extraction operation
        self.logger.info(f"Extracting data from Excel file: {source_path}")
        
        try:
            # Determine the file type
            file_type = get_excel_file_type(source_path)
            
            # Read the file into a DataFrame
            if file_type == "csv":
                df = pd.read_csv(source_path, **kwargs)
            else:
                df = pd.read_excel(source_path, sheet_name=sheet_name, **kwargs)
            
            # Filter columns if specified
            if columns:
                # Validate that all requested columns exist
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    error_msg = f"Columns not found in Excel file: {missing_columns}"
                    self.logger.error(error_msg)
                    raise ExtractionError(error_msg)
                
                # Select only the specified columns
                df = df[columns]
            
            # Upload the file if requested
            if upload:
                self._upload_file(source_path, target_filename)
            
            # Log success and return the DataFrame
            self.logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
            return df
        
        except (EmptyDataError, ParserError) as e:
            # Handle pandas-specific errors
            error_msg = f"Error parsing Excel file {source_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e
        
        except ExtractionError:
            # Re-raise ExtractionError
            raise
        
        except Exception as e:
            # Handle any other errors
            error_msg = f"Error extracting data from Excel file {source_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e
    
    def _upload_file(self, source_path: Path, target_filename: Optional[str] = None) -> None:
        """
        Upload a file to the uploads directory.
        
        Args:
            source_path: Path to the file to upload.
            target_filename: Optional filename to use when uploading.
                If None, uses the source filename.
        
        Raises:
            ExtractionError: If an error occurs during the upload process.
        """
        try:
            # Get the uploads directory from settings
            uploads_dir = get_root_dir() / settings.get("file_paths.uploads_dir", "uploads")
            
            # Ensure the uploads directory exists
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Upload the file
            self.logger.info(f"Uploading Excel file to: {uploads_dir}")
            self.upload_service(source_path, uploads_dir, target_filename)
        except Exception as e:
            error_msg = f"Error uploading file {source_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ExtractionError(error_msg) from e


# For backward compatibility
def extract_excel_to_dataframe(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    columns: Optional[List[str]] = None,
    upload: bool = False,
    target_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract data from an Excel file into a pandas DataFrame.
    
    This function is maintained for backward compatibility.
    New code should use the ExcelExtractor class directly.
    
    Args:
        file_path: Path to the Excel file. Can be absolute or relative.
        sheet_name: Name or index of the sheet to extract (default: 0, first sheet).
        columns: Optional list of column names to extract. If None, extracts all columns.
        upload: Whether to upload the file to the uploads directory.
        target_filename: Optional filename to use when uploading. If None, uses the source filename.
        
    Returns:
        pandas DataFrame containing the extracted data.
        
    Raises:
        FileNotFoundError: If the source file does not exist.
        ExtractionError: If an error occurs during the extraction process.
    """
    extractor = ExcelExtractor()
    return extractor.extract(
        file_path=file_path,
        sheet_name=sheet_name,
        columns=columns,
        upload=upload,
        target_filename=target_filename
    )


# Register the Excel extractor with the factory
excel_extractor = ExcelExtractor()
extractor_factory.register_extractor(excel_extractor)
```

## File: main.py
```python
"""
Main entry point for the FCA Dashboard ETL pipeline.

This module provides the main functionality to run the ETL pipeline,
including command-line argument parsing and pipeline execution.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from fca_dashboard.config.settings import get_settings
from fca_dashboard.utils.error_handler import (
    ConfigurationError,
    DataExtractionError,
    ErrorHandler,
)
from fca_dashboard.utils.logging_config import configure_logging, get_logger
from fca_dashboard.utils.path_util import get_logs_path, resolve_path


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="FCA Dashboard ETL Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config" / "settings.yml"),
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument("--excel-file", type=str, help="Path to Excel file to process")

    parser.add_argument("--table-name", type=str, help="Name of the table to process")

    return parser.parse_args()


def run_etl_pipeline(args: argparse.Namespace, log: Any) -> int:
    """
    Run the ETL pipeline with the given arguments.

    Args:
        args: Command line arguments
        log: Logger instance

    Returns:
        Exit code (0 for success, non-zero for failure)

    Raises:
        ConfigurationError: If there is an error in the configuration
        DataExtractionError: If there is an error extracting data
        FileNotFoundError: If a required file is not found
    """
    # Resolve the configuration file path
    config_path = resolve_path(args.config)
    log.info(f"Loading configuration from {config_path}")

    # Load settings
    try:
        settings = get_settings(str(config_path))
    except yaml.YAMLError as yaml_err:
        raise ConfigurationError(f"YAML configuration error: {yaml_err}") from yaml_err

    # Log startup information
    log.info("FCA Dashboard ETL Pipeline starting")
    log.info(f"Python version: {sys.version}")
    log.info(f"Current working directory: {Path.cwd()}")

    # TODO: Implement ETL pipeline execution (See GitHub issue #123)
    # Steps include:
    # 1. Extract data from Excel or database source (See GitHub issue #124)
    #    - Read source data using appropriate extractor strategy
    #    - Validate source data structure
    # 2. Transform data (cleaning, normalization, enrichment) (See GitHub issue #125)
    #    - Apply business rules and transformations
    #    - Map source fields to destination schema
    # 3. Load data into destination database or output format (See GitHub issue #126)
    #    - Batch insert/update operations
    #    - Validate data integrity after loading
    log.info("ETL Pipeline execution would start here")

    log.info(f"Database URL: {settings.get('databases.sqlite.url')}")

    if args.excel_file:
        try:
            excel_path = resolve_path(args.excel_file)
            log.info(f"Would process Excel file: {excel_path}")
        except FileNotFoundError:
            raise DataExtractionError(f"Excel file not found: {args.excel_file}") from None

    if args.table_name:
        log.info(f"Would process table: {args.table_name}")

    # Log successful completion
    log.info("ETL Pipeline completed successfully")
    return 0


def main() -> int:
    """
    Main entry point for the ETL pipeline.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse command line arguments
    args = parse_args()

    # Configure logging
    log_file = get_logs_path("fca_dashboard.log")
    configure_logging(level=args.log_level, log_file=str(log_file), rotation="10 MB", retention="1 month")

    # Get a logger for this module
    log = get_logger("main")

    # Create an error handler
    error_handler = ErrorHandler("main")

    # Run the ETL pipeline with error handling
    try:
        return run_etl_pipeline(args, log)
    except Exception as e:
        return error_handler.handle_error(e)


if __name__ == "__main__":
    sys.exit(main())
```

## File: pipelines/pipeline_medtronics.py
```python
"""
Medtronics Asset Data Pipeline.

This pipeline extracts data from the Medtronics Asset Log Uploader Excel file,
analyzes and validates it, and then exports it to a SQLite database.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.database import (
    get_table_schema,
    save_dataframe_to_database,
)
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    extract_excel_with_config,
    validate_dataframe,
)
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


class MedtronicsPipeline:
    """
    Pipeline for processing Medtronics Asset Data.
    
    This pipeline extracts data from the Medtronics Asset Log Uploader Excel file,
    analyzes and validates it, and then exports it to a SQLite database.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.logger = get_logger("medtronics_pipeline")
        
        # Get file paths from settings
        self.input_file = settings.get("medtronics.input_file", "uploads/Medtronics - Asset Log Uploader.xlsx")
        self.output_dir = settings.get("medtronics.output_dir", "outputs/pipeline/medtronic")
        self.db_name = settings.get("medtronics.db_name", "medtronics_assets.db")
        
        # Get sheet name from settings
        self.sheet_name = settings.get("medtronics.sheet_name", "Asset Data")
        
        # Get extraction configuration from settings
        self.extraction_config = settings.get("excel_utils.extraction", {})
        
        # Get validation configuration from settings
        self.validation_config = settings.get("excel_utils.validation", {})
        
        # Get analysis configuration from settings
        self.analysis_config = settings.get("excel_utils.analysis", {})
        
        # Get columns to extract from settings
        self.columns_to_extract = settings.get("medtronics.columns_to_extract", [])
        
        # Get columns to drop NaN values from settings
        self.drop_na_columns = settings.get("medtronics.drop_na_columns", [])
        
        # Initialize data storage
        self.extracted_data = None
        self.analysis_results = None
        self.validation_results = None
    
    def extract(self):
        """
        Extract data from the Medtronics Excel file.
        
        Returns:
            The extracted DataFrame for the Asset Data sheet.
        """
        self.logger.info(f"Extracting data from {self.input_file}")
        
        # Resolve the file path
        file_path = resolve_path(self.input_file)
        
        # Analyze the Excel file structure
        analysis = analyze_excel_structure(file_path)
        self.logger.info(f"File type: {analysis['file_type']}")
        self.logger.info(f"Sheet names: {analysis['sheet_names']}")
        
        # Create extraction configuration
        config = {}
        
        # Add default settings
        if "default" in self.extraction_config:
            config["default"] = self.extraction_config["default"]
        
        # Add sheet-specific settings
        for sheet_name in analysis['sheet_names']:
            # Convert sheet name to the format used in settings (lowercase with underscores)
            settings_key = sheet_name.lower().replace(" ", "_")
            
            # If there are settings for this sheet, add them to the config
            if settings_key in self.extraction_config:
                config[sheet_name] = self.extraction_config[settings_key]
        
        self.logger.info(f"Using extraction configuration: {config}")
        
        # Extract data from the Excel file
        extracted_data = extract_excel_with_config(file_path, config)
        
        # Store the extracted data
        self.extracted_data = extracted_data
        
        # Return the Asset Data sheet
        # The extraction process converts sheet names to lowercase with underscores
        normalized_sheet_name = self.sheet_name.lower().replace(" ", "_")
        
        df = None
        if normalized_sheet_name in extracted_data:
            df = extracted_data[normalized_sheet_name]
        elif self.sheet_name in extracted_data:
            df = extracted_data[self.sheet_name]
        else:
            self.logger.error(f"Sheet '{self.sheet_name}' not found in extracted data")
            self.logger.error(f"Available sheets: {list(extracted_data.keys())}")
            return None
        
        # Filter columns if columns_to_extract is specified
        if self.columns_to_extract:
            self.logger.info(f"Filtering columns to: {self.columns_to_extract}")
            
            # Convert Excel column letters to column indices
            if all(isinstance(col, str) and len(col) <= 3 and col.isalpha() for col in self.columns_to_extract):
                # Convert Excel column letters to 0-based indices
                def excel_col_to_index(col_str):
                    """Convert Excel column letter to 0-based index."""
                    col_str = col_str.upper()
                    result = 0
                    for c in col_str:
                        result = result * 26 + (ord(c) - ord('A') + 1)
                    return result - 1
                
                # Get column indices
                col_indices = [excel_col_to_index(col) for col in self.columns_to_extract]
                
                # Get column names from indices
                if len(df.columns) > max(col_indices):
                    col_names = [df.columns[i] for i in col_indices if i < len(df.columns)]
                    df = df[col_names]
                    self.logger.info(f"Filtered to {len(col_names)} columns: {col_names}")
                else:
                    self.logger.warning(f"Some column indices are out of range. Max index: {len(df.columns) - 1}")
            else:
                # Assume column names are provided
                existing_cols = [col for col in self.columns_to_extract if col in df.columns]
                if existing_cols:
                    df = df[existing_cols]
                    self.logger.info(f"Filtered to {len(existing_cols)} columns: {existing_cols}")
                else:
                    self.logger.warning(f"None of the specified columns exist in the DataFrame")
        
        # Normalize column names to lowercase
        if df is not None:
            self.logger.info("Normalizing column names to lowercase")
            df.columns = [col.lower() for col in df.columns]
            self.logger.info(f"Normalized column names: {list(df.columns)}")
        
        # Drop rows with NaN values in specified columns
        if self.drop_na_columns and df is not None:
            original_row_count = len(df)
            
            # Convert drop_na_columns to lowercase for consistency
            drop_na_columns_lower = [col.lower() for col in self.drop_na_columns]
            
            # Check if the specified columns exist in the DataFrame
            existing_cols = [col for col in drop_na_columns_lower if col in df.columns]
            
            if existing_cols:
                self.logger.info(f"Dropping rows with NaN values in columns: {existing_cols}")
                df = df.dropna(subset=existing_cols)
                self.logger.info(f"Dropped {original_row_count - len(df)} rows with NaN values")
            else:
                self.logger.warning(f"None of the specified columns for dropping NaN values exist in the DataFrame")
                self.logger.warning(f"Available columns: {list(df.columns)}")
        
        return df
    
    def analyze(self, df):
        """
        Analyze the extracted data.
        
        Args:
            df: The DataFrame to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info(f"Analyzing data from sheet '{self.sheet_name}'")
        
        # Get default analysis settings
        default_analysis = self.analysis_config.get("default", {})
        
        # Initialize results dictionary
        results = {}
        
        # 1. Analyze unique values
        self.logger.info("Analyzing unique values...")
        unique_values_settings = default_analysis.get("unique_values", {})
        max_unique_values = unique_values_settings.get("max_unique_values", 20)
        
        # Select columns for unique value analysis
        # For demonstration purposes, we'll analyze the first 5 columns
        unique_columns = list(df.columns[:5])
        unique_values_results = analyze_unique_values(
            df, 
            columns=unique_columns,
            max_unique_values=max_unique_values
        )
        results['unique_values'] = unique_values_results
        
        # 2. Analyze column statistics for numeric columns
        self.logger.info("Analyzing column statistics...")
        column_stats_settings = default_analysis.get("column_statistics", {})
        include_outliers = column_stats_settings.get("include_outliers", True)
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns[:5])
            results['column_statistics'] = column_stats_results
        
        # 3. Analyze text columns
        self.logger.info("Analyzing text columns...")
        text_analysis_settings = default_analysis.get("text_analysis", {})
        include_pattern_analysis = text_analysis_settings.get("include_pattern_analysis", True)
        
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            text_analysis_results = analyze_text_columns(df, columns=text_columns[:5])
            results['text_analysis'] = text_analysis_results
        
        # Store the analysis results
        self.analysis_results = results
        
        return results
    
    def validate(self, df):
        """
        Validate the extracted data.
        
        Args:
            df: The DataFrame to validate.
            
        Returns:
            A dictionary containing the validation results.
        """
        self.logger.info(f"Validating data from sheet '{self.sheet_name}'")
        
        # Create validation configuration
        validation_config = {}
        
        # Add default settings
        if "default" in self.validation_config:
            default_settings = self.validation_config["default"]
            
            # Add missing values check
            if "missing_values" in default_settings:
                missing_values_settings = default_settings["missing_values"]
                validation_config["missing_values"] = {
                    "columns": missing_values_settings.get("columns") or list(df.columns),
                    "threshold": missing_values_settings.get("threshold", 0.5)
                }
            
            # Add duplicate rows check
            if "duplicate_rows" in default_settings:
                duplicate_rows_settings = default_settings["duplicate_rows"]
                validation_config["duplicate_rows"] = {
                    "subset": duplicate_rows_settings.get("subset")
                }
            
            # Add data types check
            if "data_types" in default_settings:
                data_types_settings = default_settings["data_types"]
                
                # Create type specifications based on settings
                type_specs = {}
                
                # Add date columns
                for col in data_types_settings.get("date_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "date"
                
                # Add numeric columns
                for col in data_types_settings.get("numeric_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "float"
                
                # Add string columns
                for col in data_types_settings.get("string_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "str"
                
                # Add boolean columns
                for col in data_types_settings.get("boolean_columns", []):
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        type_specs[col_lower] = "bool"
                
                if type_specs:
                    validation_config["data_types"] = type_specs
        
        # Add sheet-specific validation
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = self.sheet_name.lower().replace(" ", "_")
        
        if settings_key in self.validation_config:
            sheet_settings = self.validation_config[settings_key]
            
            # Add value ranges check
            if "value_ranges" in sheet_settings:
                # Convert keys to lowercase
                value_ranges = {}
                for col, range_values in sheet_settings["value_ranges"].items():
                    value_ranges[col.lower()] = range_values
                
                validation_config["value_ranges"] = value_ranges
            
            # Add required columns check
            if "required_columns" in sheet_settings:
                validation_config["required_columns"] = sheet_settings["required_columns"]
        
        self.logger.info(f"Using validation configuration: {validation_config}")
        
        # Validate the DataFrame
        results = validate_dataframe(df, validation_config)
        
        # Store the validation results
        self.validation_results = results
        
        return results
    
    def export(self, df):
        """
        Export the data to a SQLite database.
        
        Args:
            df: The DataFrame to export.
            
        Returns:
            The path to the SQLite database file.
        """
        self.logger.info(f"Exporting data to SQLite database")
        
        # Create the output directory if it doesn't exist
        output_dir = resolve_path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the database path
        db_path = os.path.join(output_dir, self.db_name)
        
        # Create the connection string
        connection_string = f"sqlite:///{db_path}"
        
        # Export the data to the database
        table_name = self.sheet_name.lower().replace(" ", "_")
        save_dataframe_to_database(
            df=df,
            table_name=table_name,
            connection_string=connection_string,
            if_exists="replace"
        )
        
        self.logger.info(f"Data exported to {db_path}, table: {table_name}")
        
        # Get the schema of the table
        try:
            schema = get_table_schema(connection_string, table_name)
            self.logger.info(f"Database schema:\n{schema}")
            
            # Save the schema to a file
            schema_path = os.path.join(output_dir, f"{table_name}_schema.sql")
            with open(schema_path, "w") as f:
                f.write(schema)
            
            self.logger.info(f"Schema saved to {schema_path}")
        except Exception as e:
            self.logger.error(f"Error getting schema: {str(e)}")
        
        return db_path
    
    def save_reports(self, df):
        """
        Save analysis and validation reports.
        
        Args:
            df: The DataFrame that was analyzed and validated.
            
        Returns:
            A dictionary containing the paths to the report files.
        """
        self.logger.info(f"Saving analysis and validation reports")
        
        # Create the output directory if it doesn't exist
        output_dir = resolve_path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize a dictionary to store the report paths
        report_paths = {}
        
        # Save analysis report
        if self.analysis_results:
            analysis_report_path = os.path.join(output_dir, f"{self.sheet_name.lower().replace(' ', '_')}_analysis_report.txt")
            
            with open(analysis_report_path, "w") as f:
                f.write(f"Analysis Report for Sheet: {self.sheet_name}\n")
                f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                
                # Write unique values report
                if 'unique_values' in self.analysis_results:
                    f.write("Unique Values Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    unique_values = self.analysis_results['unique_values']
                    for col, res in unique_values.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Unique value count: {res['count']}\n")
                        f.write(f"    Null count: {res['null_count']} ({res['null_percentage'] * 100:.2f}%)\n")
                        
                        if 'values' in res:
                            f.write(f"    Unique values: {', '.join(res['values'][:10])}")
                            if len(res['values']) > 10:
                                f.write(f" ... and {len(res['values']) - 10} more")
                            f.write("\n")
                        
                        if 'value_counts' in res:
                            f.write(f"    Value counts (top 5):\n")
                            sorted_counts = sorted(res['value_counts'].items(), key=lambda x: x[1], reverse=True)
                            for val, count in sorted_counts[:5]:
                                f.write(f"      {val}: {count}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write column statistics report
                if 'column_statistics' in self.analysis_results:
                    f.write("Column Statistics Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    column_stats = self.analysis_results['column_statistics']
                    for col, stats in column_stats.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Min: {stats['min']}\n")
                        f.write(f"    Max: {stats['max']}\n")
                        f.write(f"    Mean: {stats['mean']}\n")
                        f.write(f"    Median: {stats['median']}\n")
                        f.write(f"    Standard deviation: {stats['std']}\n")
                        f.write(f"    Q1 (25th percentile): {stats['q1']}\n")
                        f.write(f"    Q3 (75th percentile): {stats['q3']}\n")
                        f.write(f"    IQR: {stats['iqr']}\n")
                        f.write(f"    Outliers count: {stats['outliers_count']}\n")
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write text analysis report
                if 'text_analysis' in self.analysis_results:
                    f.write("Text Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    text_analysis = self.analysis_results['text_analysis']
                    for col, analysis in text_analysis.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Min length: {analysis['min_length']}\n")
                        f.write(f"    Max length: {analysis['max_length']}\n")
                        f.write(f"    Average length: {analysis['avg_length']:.2f}\n")
                        f.write(f"    Empty strings: {analysis['empty_count']}\n")
                        
                        if 'pattern_analysis' in analysis:
                            f.write(f"    Pattern analysis:\n")
                            for pattern, count in analysis['pattern_analysis'].items():
                                if count > 0:
                                    f.write(f"      {pattern}: {count}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Analysis report saved to {analysis_report_path}")
            report_paths['analysis_report'] = analysis_report_path
        
        # Save validation report
        if self.validation_results:
            validation_report_path = os.path.join(output_dir, f"{self.sheet_name.lower().replace(' ', '_')}_validation_report.txt")
            
            with open(validation_report_path, "w") as f:
                f.write(f"Validation Report for Sheet: {self.sheet_name}\n")
                f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                
                # Write missing values report
                if "missing_values" in self.validation_results:
                    f.write("Missing Values Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    missing_values = self.validation_results["missing_values"]
                    for col, pct in missing_values.items():
                        f.write(f"  {col}: {pct * 100:.2f}% missing\n")
                    
                    f.write("\n")
                
                # Write duplicate rows report
                if "duplicate_rows" in self.validation_results:
                    f.write("Duplicate Rows Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    duplicate_rows = self.validation_results["duplicate_rows"]
                    f.write(f"  Duplicate rows: {duplicate_rows['duplicate_count']}\n")
                    
                    if duplicate_rows["duplicate_count"] > 0:
                        f.write(f"  Duplicate indices: {duplicate_rows['duplicate_indices'][:10]}")
                        if len(duplicate_rows["duplicate_indices"]) > 10:
                            f.write(f" ... and {len(duplicate_rows['duplicate_indices']) - 10} more")
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write value ranges report
                if "value_ranges" in self.validation_results:
                    f.write("Value Ranges Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    value_ranges = self.validation_results["value_ranges"]
                    for col, res in value_ranges.items():
                        f.write(f"  {col}:\n")
                        f.write(f"    Below minimum: {res['below_min']}\n")
                        f.write(f"    Above maximum: {res['above_max']}\n")
                        f.write(f"    Total outside range: {res['total_outside_range']}\n")
                    
                    f.write("\n")
                
                # Write data types report
                if "data_types" in self.validation_results:
                    f.write("Data Types Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    data_types = self.validation_results["data_types"]
                    for col, res in data_types.items():
                        f.write(f"  {col}:\n")
                        f.write(f"    Expected type: {res['expected_type']}\n")
                        f.write(f"    Current type: {res['current_type']}\n")
                        f.write(f"    Error count: {res['error_count']}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Validation report saved to {validation_report_path}")
            report_paths['validation_report'] = validation_report_path
        
        return report_paths
    
    def run(self):
        """
        Run the pipeline.
        
        Returns:
            A dictionary containing the results of the pipeline.
        """
        try:
            self.logger.info("Starting Medtronics Asset Data Pipeline")
            
            # Extract data
            try:
                df = self.extract()
                
                if df is None or len(df) == 0:
                    self.logger.error(f"No data extracted from sheet '{self.sheet_name}'")
                    return {
                        "status": "error",
                        "message": f"No data extracted from sheet '{self.sheet_name}'"
                    }
            except Exception as e:
                self.logger.error(f"Error extracting data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error extracting data: {str(e)}"
                }
            
            # Analyze data
            try:
                analysis_results = self.analyze(df)
            except Exception as e:
                self.logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error analyzing data: {str(e)}"
                }
            
            # Validate data
            try:
                validation_results = self.validate(df)
            except Exception as e:
                self.logger.error(f"Error validating data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error validating data: {str(e)}"
                }
            
            # Export data
            try:
                db_path = self.export(df)
            except Exception as e:
                self.logger.error(f"Error exporting data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error exporting data: {str(e)}"
                }
            
            # Save reports
            try:
                report_paths = self.save_reports(df)
            except Exception as e:
                self.logger.error(f"Error saving reports: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error saving reports: {str(e)}"
                }
            
            self.logger.info("Medtronics Asset Data Pipeline completed successfully")
            
            # Return results
            return {
                "status": "success",
                "message": "Pipeline completed successfully",
                "data": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "db_path": db_path,
                    "sheet_name": self.sheet_name.lower().replace(" ", "_"),
                    "report_paths": report_paths
                }
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error in pipeline: {str(e)}"
            }


def main():
    """Main function."""
    try:
        # Create and run the pipeline
        pipeline = MedtronicsPipeline()
        results = pipeline.run()
        
        # Print the results
        print("\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Message: {results['message']}")
        
        if results['status'] == 'success':
            print(f"\nData:")
            print(f"  Rows: {results['data']['rows']}")
            print(f"  Columns: {results['data']['columns']}")
            print(f"  Database: {results['data']['db_path']}")
            
            # Check if schema file exists
            schema_path = os.path.join(os.path.dirname(results['data']['db_path']), f"{results['data']['sheet_name']}_schema.sql")
            if os.path.exists(schema_path):
                print(f"  Schema: {schema_path}")
                
                # Print the schema
                print(f"\nDatabase Schema:")
                with open(schema_path, "r") as f:
                    schema = f.read()
                print(schema)
            
            print(f"\nReports:")
            for report_type, report_path in results['data']['report_paths'].items():
                print(f"  {report_type}: {report_path}")
            
            return 0
        else:
            # Return non-zero exit code for errors
            return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## File: pipelines/pipeline_wichita.py
```python
"""
Wichita Animal Shelter Asset Data Pipeline.

This pipeline extracts data from the Wichita Animal Shelter Asset List CSV file,
analyzes and validates it, and then exports it to a SQLite database.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.database import (
    get_table_schema,
    save_dataframe_to_database,
)
from fca_dashboard.utils.excel import (
    analyze_column_statistics,
    analyze_text_columns,
    analyze_unique_values,
    validate_dataframe,
)
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


class WichitaPipeline:
    """
    Pipeline for processing Wichita Animal Shelter Asset Data.
    
    This pipeline extracts data from the Wichita Animal Shelter Asset List CSV file,
    analyzes and validates it, and then exports it to a SQLite database.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.logger = get_logger("wichita_pipeline")
        
        # Get file paths from settings or use defaults
        self.input_file = settings.get(
            "wichita.input_file", 
            "uploads/Asset_List Wichita Animal Shelter (1).csv"
        )
        self.output_dir = settings.get(
            "wichita.output_dir", 
            "outputs/pipeline/wichita"
        )
        self.db_name = settings.get(
            "wichita.db_name", 
            "wichita_assets.db"
        )
        
        # Get validation configuration from settings
        self.validation_config = settings.get("excel_utils.validation", {})
        
        # Get analysis configuration from settings
        self.analysis_config = settings.get("excel_utils.analysis", {})
        
        # Get columns to extract from settings or use all columns
        self.columns_to_extract = settings.get("wichita.columns_to_extract", [])
        
        # Get columns to drop NaN values from settings
        self.drop_na_columns = settings.get(
            "wichita.drop_na_columns", 
            ["Asset Name", "Asset Category Name"]
        )
        
        # Initialize data storage
        self.extracted_data = None
        self.analysis_results = None
        self.validation_results = None
    
    def extract(self):
        """
        Extract data from the Wichita Animal Shelter CSV file.
        
        Returns:
            The extracted DataFrame.
        """
        self.logger.info(f"Extracting data from {self.input_file}")
        
        # Resolve the file path
        file_path = resolve_path(self.input_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns")
            
            # Filter columns if columns_to_extract is specified
            if self.columns_to_extract:
                self.logger.info(f"Filtering columns to: {self.columns_to_extract}")
                existing_cols = [col for col in self.columns_to_extract if col in df.columns]
                if existing_cols:
                    df = df[existing_cols]
                    self.logger.info(f"Filtered to {len(existing_cols)} columns: {existing_cols}")
                else:
                    self.logger.warning(f"None of the specified columns exist in the DataFrame")
            
            # Drop rows with NaN values in specified columns
            if self.drop_na_columns:
                original_row_count = len(df)
                
                # Check if the specified columns exist in the DataFrame
                existing_cols = [col for col in self.drop_na_columns if col in df.columns]
                
                if existing_cols:
                    self.logger.info(f"Dropping rows with NaN values in columns: {existing_cols}")
                    df = df.dropna(subset=existing_cols)
                    self.logger.info(f"Dropped {original_row_count - len(df)} rows with NaN values")
                else:
                    self.logger.warning(f"None of the specified columns for dropping NaN values exist in the DataFrame")
                    self.logger.warning(f"Available columns: {list(df.columns)}")
            
            # Store the extracted data
            self.extracted_data = df
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error extracting data from CSV file: {str(e)}", exc_info=True)
            raise
    
    def analyze(self, df):
        """
        Analyze the extracted data.
        
        Args:
            df: The DataFrame to analyze.
            
        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info(f"Analyzing data from Wichita Animal Shelter asset list")
        
        # Get default analysis settings
        default_analysis = self.analysis_config.get("default", {})
        
        # Initialize results dictionary
        results = {}
        
        # 1. Analyze unique values
        self.logger.info("Analyzing unique values...")
        unique_values_settings = default_analysis.get("unique_values", {})
        max_unique_values = unique_values_settings.get("max_unique_values", 20)
        
        # Select important columns for unique value analysis
        unique_columns = [
            "Building Name", "Asset Category Name", "Type", 
            "Floor", "Room Number", "Manufacturer"
        ]
        # Filter to only columns that exist in the DataFrame
        unique_columns = [col for col in unique_columns if col in df.columns]
        
        unique_values_results = analyze_unique_values(
            df, 
            columns=unique_columns,
            max_unique_values=max_unique_values
        )
        results['unique_values'] = unique_values_results
        
        # 2. Analyze column statistics for numeric columns
        self.logger.info("Analyzing column statistics...")
        
        # Find numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            column_stats_results = analyze_column_statistics(df, columns=numeric_columns)
            results['column_statistics'] = column_stats_results
        
        # 3. Analyze text columns
        self.logger.info("Analyzing text columns...")
        
        # Find text columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        # Select a subset of important text columns
        important_text_columns = [
            "Asset Name", "Asset Category Name", "Type", 
            "Manufacturer", "Model", "Description"
        ]
        # Filter to only columns that exist in the DataFrame
        text_columns = [col for col in important_text_columns if col in text_columns]
        
        if text_columns:
            text_analysis_results = analyze_text_columns(df, columns=text_columns)
            results['text_analysis'] = text_analysis_results
        
        # Store the analysis results
        self.analysis_results = results
        
        return results
    
    def validate(self, df):
        """
        Validate the extracted data.
        
        Args:
            df: The DataFrame to validate.
            
        Returns:
            A dictionary containing the validation results.
        """
        self.logger.info(f"Validating data from Wichita Animal Shelter asset list")
        
        # Create validation configuration
        validation_config = {}
        
        # Add default settings
        if "default" in self.validation_config:
            default_settings = self.validation_config["default"]
            
            # Add missing values check
            if "missing_values" in default_settings:
                missing_values_settings = default_settings["missing_values"]
                validation_config["missing_values"] = {
                    "columns": missing_values_settings.get("columns") or [
                        "Asset Name", "Asset Category Name", "Type", "Manufacturer"
                    ],
                    "threshold": missing_values_settings.get("threshold", 0.5)
                }
            
            # Add duplicate rows check
            if "duplicate_rows" in default_settings:
                duplicate_rows_settings = default_settings["duplicate_rows"]
                validation_config["duplicate_rows"] = {
                    "subset": duplicate_rows_settings.get("subset") or ["Asset Name", "ID"]
                }
            
            # Add data types check
            if "data_types" in default_settings:
                data_types_settings = default_settings["data_types"]
                
                # Create type specifications based on settings
                type_specs = {}
                
                # Add date columns
                date_columns = ["Installation Date", "Warranty Expiration Date", "Estimated Replacement Date"]
                for col in date_columns:
                    if col in df.columns:
                        type_specs[col] = "date"
                
                # Add numeric columns
                numeric_columns = ["Cost", "Service Life", "Quantity", "Square Feet"]
                for col in numeric_columns:
                    if col in df.columns:
                        type_specs[col] = "float"
                
                # Add string columns
                string_columns = ["Asset Name", "Asset Category Name", "Type", "Manufacturer", "Model"]
                for col in string_columns:
                    if col in df.columns:
                        type_specs[col] = "str"
                
                if type_specs:
                    validation_config["data_types"] = type_specs
        
        self.logger.info(f"Using validation configuration: {validation_config}")
        
        # Validate the DataFrame
        results = validate_dataframe(df, validation_config)
        
        # Store the validation results
        self.validation_results = results
        
        return results
    
    def export(self, df):
        """
        Export the data to a SQLite database.
        
        Args:
            df: The DataFrame to export.
            
        Returns:
            The path to the SQLite database file.
        """
        self.logger.info(f"Exporting data to SQLite database")
        
        # Create the output directory if it doesn't exist
        output_dir = resolve_path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the database path
        db_path = os.path.join(output_dir, self.db_name)
        
        # Create the connection string
        connection_string = f"sqlite:///{db_path}"
        
        # Export the data to the database
        table_name = "wichita_assets"
        save_dataframe_to_database(
            df=df,
            table_name=table_name,
            connection_string=connection_string,
            if_exists="replace"
        )
        
        self.logger.info(f"Data exported to {db_path}, table: {table_name}")
        
        # Get the schema of the table
        try:
            schema = get_table_schema(connection_string, table_name)
            self.logger.info(f"Database schema:\n{schema}")
            
            # Save the schema to a file
            schema_path = os.path.join(output_dir, f"{table_name}_schema.sql")
            with open(schema_path, "w") as f:
                f.write(schema)
            
            self.logger.info(f"Schema saved to {schema_path}")
        except Exception as e:
            self.logger.error(f"Error getting schema: {str(e)}")
        
        return db_path
    
    def save_reports(self, df):
        """
        Save analysis and validation reports.
        
        Args:
            df: The DataFrame that was analyzed and validated.
            
        Returns:
            A dictionary containing the paths to the report files.
        """
        self.logger.info(f"Saving analysis and validation reports")
        
        # Create the output directory if it doesn't exist
        output_dir = resolve_path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize a dictionary to store the report paths
        report_paths = {}
        
        # Save analysis report
        if self.analysis_results:
            analysis_report_path = os.path.join(output_dir, "wichita_assets_analysis_report.txt")
            
            with open(analysis_report_path, "w") as f:
                f.write(f"Analysis Report for Wichita Animal Shelter Asset List\n")
                f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                
                # Write unique values report
                if 'unique_values' in self.analysis_results:
                    f.write("Unique Values Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    unique_values = self.analysis_results['unique_values']
                    for col, res in unique_values.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Unique value count: {res['count']}\n")
                        f.write(f"    Null count: {res['null_count']} ({res['null_percentage'] * 100:.2f}%)\n")
                        
                        if 'values' in res:
                            f.write(f"    Unique values: {', '.join(str(v) for v in res['values'][:10])}")
                            if len(res['values']) > 10:
                                f.write(f" ... and {len(res['values']) - 10} more")
                            f.write("\n")
                        
                        if 'value_counts' in res:
                            f.write(f"    Value counts (top 5):\n")
                            sorted_counts = sorted(res['value_counts'].items(), key=lambda x: x[1], reverse=True)
                            for val, count in sorted_counts[:5]:
                                f.write(f"      {val}: {count}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write column statistics report
                if 'column_statistics' in self.analysis_results:
                    f.write("Column Statistics Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    column_stats = self.analysis_results['column_statistics']
                    for col, stats in column_stats.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Min: {stats['min']}\n")
                        f.write(f"    Max: {stats['max']}\n")
                        f.write(f"    Mean: {stats['mean']}\n")
                        f.write(f"    Median: {stats['median']}\n")
                        f.write(f"    Standard deviation: {stats['std']}\n")
                        f.write(f"    Q1 (25th percentile): {stats['q1']}\n")
                        f.write(f"    Q3 (75th percentile): {stats['q3']}\n")
                        f.write(f"    IQR: {stats['iqr']}\n")
                        f.write(f"    Outliers count: {stats['outliers_count']}\n")
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write text analysis report
                if 'text_analysis' in self.analysis_results:
                    f.write("Text Analysis:\n")
                    f.write("-" * 50 + "\n")
                    
                    text_analysis = self.analysis_results['text_analysis']
                    for col, analysis in text_analysis.items():
                        f.write(f"  Column: {col}\n")
                        f.write(f"    Min length: {analysis['min_length']}\n")
                        f.write(f"    Max length: {analysis['max_length']}\n")
                        f.write(f"    Average length: {analysis['avg_length']:.2f}\n")
                        f.write(f"    Empty strings: {analysis['empty_count']}\n")
                        
                        if 'pattern_analysis' in analysis:
                            f.write(f"    Pattern analysis:\n")
                            for pattern, count in analysis['pattern_analysis'].items():
                                if count > 0:
                                    f.write(f"      {pattern}: {count}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Analysis report saved to {analysis_report_path}")
            report_paths['analysis_report'] = analysis_report_path
        
        # Save validation report
        if self.validation_results:
            validation_report_path = os.path.join(output_dir, "wichita_assets_validation_report.txt")
            
            with open(validation_report_path, "w") as f:
                f.write(f"Validation Report for Wichita Animal Shelter Asset List\n")
                f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
                
                # Write missing values report
                if "missing_values" in self.validation_results:
                    f.write("Missing Values Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    missing_values = self.validation_results["missing_values"]
                    for col, pct in missing_values.items():
                        f.write(f"  {col}: {pct * 100:.2f}% missing\n")
                    
                    f.write("\n")
                
                # Write duplicate rows report
                if "duplicate_rows" in self.validation_results:
                    f.write("Duplicate Rows Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    duplicate_rows = self.validation_results["duplicate_rows"]
                    f.write(f"  Duplicate rows: {duplicate_rows['duplicate_count']}\n")
                    
                    if duplicate_rows["duplicate_count"] > 0:
                        f.write(f"  Duplicate indices: {duplicate_rows['duplicate_indices'][:10]}")
                        if len(duplicate_rows["duplicate_indices"]) > 10:
                            f.write(f" ... and {len(duplicate_rows['duplicate_indices']) - 10} more")
                        f.write("\n")
                    
                    f.write("\n")
                
                # Write value ranges report
                if "value_ranges" in self.validation_results:
                    f.write("Value Ranges Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    value_ranges = self.validation_results["value_ranges"]
                    for col, res in value_ranges.items():
                        f.write(f"  {col}:\n")
                        f.write(f"    Below minimum: {res['below_min']}\n")
                        f.write(f"    Above maximum: {res['above_max']}\n")
                        f.write(f"    Total outside range: {res['total_outside_range']}\n")
                    
                    f.write("\n")
                
                # Write data types report
                if "data_types" in self.validation_results:
                    f.write("Data Types Report:\n")
                    f.write("-" * 50 + "\n")
                    
                    data_types = self.validation_results["data_types"]
                    for col, res in data_types.items():
                        f.write(f"  {col}:\n")
                        f.write(f"    Expected type: {res['expected_type']}\n")
                        f.write(f"    Current type: {res['current_type']}\n")
                        f.write(f"    Error count: {res['error_count']}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Validation report saved to {validation_report_path}")
            report_paths['validation_report'] = validation_report_path
        
        return report_paths
    
    def run(self):
        """
        Run the pipeline.
        
        Returns:
            A dictionary containing the results of the pipeline.
        """
        try:
            self.logger.info("Starting Wichita Animal Shelter Asset Data Pipeline")
            
            # Extract data
            try:
                df = self.extract()
                
                if df is None or len(df) == 0:
                    self.logger.error(f"No data extracted from CSV file")
                    return {
                        "status": "error",
                        "message": f"No data extracted from CSV file"
                    }
            except Exception as e:
                self.logger.error(f"Error extracting data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error extracting data: {str(e)}"
                }
            
            # Analyze data
            try:
                analysis_results = self.analyze(df)
            except Exception as e:
                self.logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error analyzing data: {str(e)}"
                }
            
            # Validate data
            try:
                validation_results = self.validate(df)
            except Exception as e:
                self.logger.error(f"Error validating data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error validating data: {str(e)}"
                }
            
            # Export data
            try:
                db_path = self.export(df)
            except Exception as e:
                self.logger.error(f"Error exporting data: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error exporting data: {str(e)}"
                }
            
            # Save reports
            try:
                report_paths = self.save_reports(df)
            except Exception as e:
                self.logger.error(f"Error saving reports: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Error saving reports: {str(e)}"
                }
            
            self.logger.info("Wichita Animal Shelter Asset Data Pipeline completed successfully")
            
            # Return results
            return {
                "status": "success",
                "message": "Pipeline completed successfully",
                "data": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "db_path": db_path,
                    "table_name": "wichita_assets",
                    "report_paths": report_paths
                }
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error in pipeline: {str(e)}"
            }


def main():
    """Main function."""
    try:
        # Create and run the pipeline
        pipeline = WichitaPipeline()
        results = pipeline.run()
        
        # Print the results
        print("\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Message: {results['message']}")
        
        if results['status'] == 'success':
            print(f"\nData:")
            print(f"  Rows: {results['data']['rows']}")
            print(f"  Columns: {results['data']['columns']}")
            print(f"  Database: {results['data']['db_path']}")
            
            # Check if schema file exists
            schema_path = os.path.join(
                os.path.dirname(results['data']['db_path']), 
                f"{results['data']['table_name']}_schema.sql"
            )
            if os.path.exists(schema_path):
                print(f"  Schema: {schema_path}")
                
                # Print the schema
                print(f"\nDatabase Schema:")
                with open(schema_path, "r") as f:
                    schema = f.read()
                print(schema)
            
            print(f"\nReports:")
            for report_type, report_path in results['data']['report_paths'].items():
                print(f"  {report_type}: {report_path}")
            
            return 0
        else:
            # Return non-zero exit code for errors
            return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## File: repomix.config.json
```json
{
  "output": {
    "filePath": "repomix-output.md",
    "style": "markdown",
    "parsableStyle": false,
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "topFilesLength": 15,
    "showLineNumbers": false,
    "copyToClipboard": true
  },
  "include": [],
  "ignore": {
    "useGitignore": true,
    "useDefaultPatterns": true,
    "customPatterns": []
  },
  "security": {
    "enableSecurityCheck": true
  },
  "tokenCount": {
    "encoding": "o200k_base"
  }
}
```

## File: utils/__init__.py
```python
"""Utility modules for the FCA Dashboard application."""

from fca_dashboard.utils.date_utils import *  # noqa
from fca_dashboard.utils.error_handler import *  # noqa
from fca_dashboard.utils.json_utils import *  # noqa
from fca_dashboard.utils.logging_config import *  # noqa
from fca_dashboard.utils.number_utils import (  # noqa
    format_currency,
    random_number,
    round_to,
)
from fca_dashboard.utils.path_util import *  # noqa
from fca_dashboard.utils.string_utils import *  # noqa
from fca_dashboard.utils.validation_utils import (  # noqa
    is_valid_email,
    is_valid_phone,
    is_valid_url,
)
```

## File: utils/database/__init__.py
```python
"""
Database utilities for the FCA Dashboard application.

This module provides utilities for working with databases, including
connection management, schema operations, and data import/export.
"""

from fca_dashboard.utils.database.base import (
    DatabaseError,
    get_table_schema,
    save_dataframe_to_database,
)

__all__ = [
    "DatabaseError",
    "save_dataframe_to_database",
    "get_table_schema",
]
```

## File: utils/database/base.py
```python
"""
Base database utilities for the FCA Dashboard application.

This module provides base classes and utilities for database operations.
"""

from typing import Dict, Optional, Union

import pandas as pd

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger


class DatabaseError(FCADashboardError):
    """Base exception for database operations."""
    pass


def save_dataframe_to_database(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    schema: Optional[str] = None,
    if_exists: str = "replace",
    **kwargs
) -> None:
    """
    Save a DataFrame to a database table.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The database connection string.
        schema: The database schema (optional).
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        **kwargs: Additional arguments to pass to the database-specific implementation.
        
    Raises:
        DatabaseError: If an error occurs while saving to the database.
    """
    logger = get_logger("database_utils")
    
    try:
        # Determine the database type from the connection string
        if connection_string.startswith('sqlite'):
            from fca_dashboard.utils.database.sqlite_utils import save_dataframe_to_sqlite
            save_dataframe_to_sqlite(df, table_name, connection_string, if_exists, **kwargs)
        elif connection_string.startswith('postgresql'):
            from fca_dashboard.utils.database.postgres_utils import save_dataframe_to_postgres
            save_dataframe_to_postgres(df, table_name, connection_string, schema, if_exists, **kwargs)
        else:
            raise DatabaseError(f"Unsupported database type: {connection_string}")
            
        logger.info(f"Successfully saved {len(df)} rows to table {table_name}")
    except Exception as e:
        error_msg = f"Error saving DataFrame to database table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def get_table_schema(
    connection_string: str,
    table_name: str
) -> str:
    """
    Get the schema of a database table.
    
    Args:
        connection_string: The database connection string.
        table_name: The name of the table to get the schema for.
        
    Returns:
        A string containing the schema of the table.
        
    Raises:
        DatabaseError: If an error occurs while getting the schema.
    """
    logger = get_logger("database_utils")
    
    try:
        # Determine the database type from the connection string
        if connection_string.startswith('sqlite'):
            from fca_dashboard.utils.database.sqlite_utils import get_sqlite_table_schema
            return get_sqlite_table_schema(connection_string, table_name)
        elif connection_string.startswith('postgresql'):
            from fca_dashboard.utils.database.postgres_utils import get_postgres_table_schema
            return get_postgres_table_schema(connection_string, table_name)
        else:
            return f"Schema for {table_name} not available for this database type"
    except Exception as e:
        error_msg = f"Error getting schema for table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e
```

## File: utils/database/postgres_utils.py
```python
"""
PostgreSQL utilities for the FCA Dashboard application.

This module provides utilities for working with PostgreSQL databases.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.logging_config import get_logger


def save_dataframe_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    schema: str = None,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a DataFrame to a PostgreSQL table.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The PostgreSQL connection string.
        schema: The database schema.
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        index: Whether to include the index in the table.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
    """
    logger = get_logger("postgres_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Save the DataFrame to the database
    df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=index,
        **kwargs
    )
    
    logger.info(f"Successfully saved {len(df)} rows to PostgreSQL table {table_name}")


def get_postgres_table_schema(
    connection_string: str,
    table_name: str,
    schema: str = "public"
) -> str:
    """
    Get the schema of a PostgreSQL table.
    
    Args:
        connection_string: The PostgreSQL connection string.
        table_name: The name of the table to get the schema for.
        schema: The database schema (default: 'public').
        
    Returns:
        A string containing the schema of the table.
    """
    logger = get_logger("postgres_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Get the schema of the table
    with engine.connect() as conn:
        query = f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND table_schema = '{schema}'
            ORDER BY ordinal_position
        """
        result = conn.execute(text(query))
        columns = []
        for row in result:
            column_type = row[1]
            if row[2] is not None:
                column_type = f"{column_type}({row[2]})"
            columns.append(f"{row[0]} {column_type}")
        
        schema_str = f"CREATE TABLE {schema}.{table_name} (\n    " + ",\n    ".join(columns) + "\n);"
    
    logger.info(f"Successfully retrieved schema for PostgreSQL table {table_name}")
    return schema_str
```

## File: utils/database/sqlite_utils.py
```python
"""
SQLite utilities for the FCA Dashboard application.

This module provides utilities for working with SQLite databases.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.logging_config import get_logger


def save_dataframe_to_sqlite(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a DataFrame to a SQLite table.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The SQLite connection string.
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        index: Whether to include the index in the table.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
    """
    logger = get_logger("sqlite_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Save the DataFrame to the database
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=index,
        **kwargs
    )
    
    logger.info(f"Successfully saved {len(df)} rows to SQLite table {table_name}")


def get_sqlite_table_schema(
    connection_string: str,
    table_name: str
) -> str:
    """
    Get the schema of a SQLite table.
    
    Args:
        connection_string: The SQLite connection string.
        table_name: The name of the table to get the schema for.
        
    Returns:
        A string containing the schema of the table.
        
    Raises:
        Exception: If the table does not exist.
    """
    logger = get_logger("sqlite_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Check if the table exists
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
        if not result.fetchone():
            error_msg = f"Table '{table_name}' does not exist in the database"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Get the schema of the table
        result = conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = []
        for row in result:
            columns.append(f"{row[1]} {row[2]}")
        
        if not columns:
            error_msg = f"Table '{table_name}' exists but has no columns"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        schema = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
    
    logger.info(f"Successfully retrieved schema for SQLite table {table_name}")
    return schema
```

## File: utils/date_utils.py
```python
"""
Date and time utility functions for common operations.

This module provides a collection of utility functions for date and time manipulation
that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import datetime
from typing import Optional, Union

from dateutil import parser


def format_date(
    date: Optional[datetime.datetime], 
    format_str: str = "%b %d, %Y", 
    default: str = ""
) -> str:
    """
    Format a datetime object into a readable string.

    Args:
        date: The datetime object to format.
        format_str: The format string to use (default: "%b %d, %Y").
        default: The default value to return if date is None.

    Returns:
        A formatted date string or the default value if date is None.

    Examples:
        >>> format_date(datetime.datetime(2023, 5, 15, 14, 30, 0))
        'May 15, 2023'
        >>> format_date(datetime.datetime(2023, 5, 15, 14, 30, 0), "%Y-%m-%d")
        '2023-05-15'
    """
    if date is None:
        return default
    
    return date.strftime(format_str)


def time_since(date: Optional[datetime.datetime], default: str = "") -> str:
    """
    Calculate the relative time between the given date and now.

    Args:
        date: The datetime to calculate the time since.
        default: The default value to return if date is None.

    Returns:
        A human-readable string representing the time difference (e.g., "2 hours ago").

    Examples:
        >>> # Assuming current time is 2023-05-15 14:30:00
        >>> time_since(datetime.datetime(2023, 5, 15, 13, 30, 0))
        '1 hour ago'
        >>> time_since(datetime.datetime(2023, 5, 14, 14, 30, 0))
        '1 day ago'
    """
    if date is None:
        return default
    
    now = datetime.datetime.now()
    diff = now - date
    
    # Handle future dates
    if diff.total_seconds() < 0:
        diff = -diff
        is_future = True
    else:
        is_future = False
    
    seconds = int(diff.total_seconds())
    minutes = seconds // 60
    hours = minutes // 60
    days = diff.days
    months = days // 30  # Approximate
    years = days // 365  # Approximate
    
    if years > 0:
        time_str = f"{years} year{'s' if years != 1 else ''}"
    elif months > 0:
        time_str = f"{months} month{'s' if months != 1 else ''}"
    elif days > 0:
        time_str = f"{days} day{'s' if days != 1 else ''}"
    elif hours > 0:
        time_str = f"{hours} hour{'s' if hours != 1 else ''}"
    elif minutes > 0:
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
    
    return f"in {time_str}" if is_future else f"{time_str} ago"


def parse_date(
    date_str: Optional[Union[str, datetime.datetime]], 
    format: Optional[str] = None
) -> Optional[datetime.datetime]:
    """
    Convert a string into a datetime object.

    Args:
        date_str: The string to parse or a datetime object to return as-is.
        format: Optional format string for parsing (if None, tries to infer format).

    Returns:
        A datetime object or None if the input is None or empty.

    Raises:
        ValueError: If the string cannot be parsed as a date.

    Examples:
        >>> parse_date("2023-05-15")
        datetime.datetime(2023, 5, 15, 0, 0)
        >>> parse_date("15/05/2023", format="%d/%m/%Y")
        datetime.datetime(2023, 5, 15, 0, 0)
    """
    if date_str is None or (isinstance(date_str, str) and not date_str.strip()):
        return None
    
    if isinstance(date_str, datetime.datetime):
        return date_str
    
    if format:
        return datetime.datetime.strptime(date_str, format)
    
    # Handle common natural language date expressions
    if isinstance(date_str, str):
        date_str = date_str.lower().strip()
        now = datetime.datetime.now()
        
        if date_str == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "yesterday":
            return (now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "tomorrow":
            return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str.endswith(" days ago"):
            try:
                days = int(date_str.split(" ")[0])
                return (now - datetime.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            except (ValueError, IndexError):
                pass
    
    # Try to parse using dateutil's flexible parser
    try:
        return parser.parse(date_str)
    except (ValueError, parser.ParserError) as err:
        raise ValueError(f"Could not parse date string: {date_str}") from err
```

## File: utils/env_utils.py
```python
"""
Environment and configuration utilities.

This module provides functions to safely access environment variables
and check the current running environment. It integrates with the application's
settings module for consistent configuration access.
"""

import os
from typing import Any

from fca_dashboard.config.settings import settings

# The environment variable name used to determine the current environment
ENV_VAR_NAME = "ENVIRONMENT"


def get_env_var(key: str, fallback: Any = None) -> Any:
    """
    Safely access environment variables with an optional fallback value.
    
    This function first checks if the environment variable is set directly in
    the OS environment. If not found, it attempts to retrieve it from the
    application settings. If still not found, it returns the fallback value.
    
    Args:
        key: The name of the environment variable to retrieve
        fallback: The value to return if the environment variable is not set
        
    Returns:
        The value of the environment variable if it exists, otherwise the fallback value
    """
    # First check OS environment variables
    value = os.environ.get(key)
    
    # If not found in OS environment, check application settings
    if value is None:
        # Look for the key in the env section of settings
        value = settings.get(f"env.{key}")
        
        # If still not found, look for it at the top level
        if value is None:
            value = settings.get(key)
    
    # If still not found, return the fallback
    if value is None:
        return fallback
        
    return value


def is_dev() -> bool:
    """
    Check if the current environment is development.
    
    This function checks the environment variable specified by ENV_VAR_NAME
    to determine if the current environment is development.
    
    Returns:
        True if the current environment is development, False otherwise
    """
    env = str(get_env_var(ENV_VAR_NAME, "")).lower()
    return env in ["development", "dev"]


def is_prod() -> bool:
    """
    Check if the current environment is production.
    
    This function checks the environment variable specified by ENV_VAR_NAME
    to determine if the current environment is production.
    
    Returns:
        True if the current environment is production, False otherwise
    """
    env = str(get_env_var(ENV_VAR_NAME, "")).lower()
    return env in ["production", "prod"]


def get_environment() -> str:
    """
    Get the current environment name.
    
    Returns:
        The current environment name (e.g., 'development', 'production', 'staging')
        or 'unknown' if not set
    """
    return str(get_env_var(ENV_VAR_NAME, "unknown")).lower()
```

## File: utils/error_handler.py
```python
"""
Error handling module for the FCA Dashboard application.

This module provides a centralized error handling mechanism for the application,
including custom exceptions and an error handler class that integrates with
the logging system.
"""

import sys
from typing import Any, Callable, Dict, Type, TypeVar, cast

from fca_dashboard.utils.logging_config import get_logger

# Type variable for function return type
T = TypeVar("T")


class FCADashboardError(Exception):
    """Base exception class for all FCA Dashboard application errors."""

    def __init__(self, message: str, *args: Any) -> None:
        """
        Initialize the exception with a message and optional arguments.

        Args:
            message: Error message
            *args: Additional arguments to pass to the Exception constructor
        """
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(FCADashboardError):
    """Exception raised for errors in the configuration."""

    pass


class DataExtractionError(FCADashboardError):
    """Exception raised for errors during data extraction."""

    pass


class DataTransformationError(FCADashboardError):
    """Exception raised for errors during data transformation."""

    pass


class DataLoadingError(FCADashboardError):
    """Exception raised for errors during data loading."""

    pass


class ValidationError(FCADashboardError):
    """Exception raised for data validation errors."""

    pass


class ErrorHandler:
    """
    Centralized error handler for the FCA Dashboard application.

    This class provides methods for handling errors in a consistent way
    throughout the application, including logging and appropriate responses.
    """

    def __init__(self, logger_name: str = "error_handler") -> None:
        """
        Initialize the error handler with a logger.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = get_logger(logger_name)
        self.error_mapping: Dict[Type[Exception], Callable[[Exception], int]] = {
            FileNotFoundError: self._handle_file_not_found,
            ConfigurationError: self._handle_configuration_error,
            DataExtractionError: self._handle_data_extraction_error,
            DataTransformationError: self._handle_data_transformation_error,
            DataLoadingError: self._handle_data_loading_error,
            ValidationError: self._handle_validation_error,
        }

    def handle_error(self, error: Exception) -> int:
        """
        Handle an exception by logging it and returning an appropriate exit code.

        Args:
            error: The exception to handle

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Find the most specific handler for this error type
        for error_type, handler in self.error_mapping.items():
            if isinstance(error, error_type):
                return handler(error)

        # If no specific handler is found, use the generic handler
        return self._handle_generic_error(error)

    def _handle_file_not_found(self, error: FileNotFoundError) -> int:
        """
        Handle a FileNotFoundError.

        Args:
            error: The FileNotFoundError to handle

        Returns:
            Exit code (1 for file not found)
        """
        self.logger.error(f"File not found: {error}")
        return 1

    def _handle_configuration_error(self, error: ConfigurationError) -> int:
        """
        Handle a ConfigurationError.

        Args:
            error: The ConfigurationError to handle

        Returns:
            Exit code (2 for configuration error)
        """
        self.logger.error(f"Configuration error: {error.message}")
        return 2

    def _handle_data_extraction_error(self, error: DataExtractionError) -> int:
        """
        Handle a DataExtractionError.

        Args:
            error: The DataExtractionError to handle

        Returns:
            Exit code (3 for data extraction error)
        """
        self.logger.error(f"Data extraction error: {error.message}")
        return 3

    def _handle_data_transformation_error(self, error: DataTransformationError) -> int:
        """
        Handle a DataTransformationError.

        Args:
            error: The DataTransformationError to handle

        Returns:
            Exit code (4 for data transformation error)
        """
        self.logger.error(f"Data transformation error: {error.message}")
        return 4

    def _handle_data_loading_error(self, error: DataLoadingError) -> int:
        """
        Handle a DataLoadingError.

        Args:
            error: The DataLoadingError to handle

        Returns:
            Exit code (5 for data loading error)
        """
        self.logger.error(f"Data loading error: {error.message}")
        return 5

    def _handle_validation_error(self, error: ValidationError) -> int:
        """
        Handle a ValidationError.

        Args:
            error: The ValidationError to handle

        Returns:
            Exit code (6 for validation error)
        """
        self.logger.error(f"Validation error: {error.message}")
        return 6

    def _handle_generic_error(self, error: Exception) -> int:
        """
        Handle a generic exception.

        Args:
            error: The exception to handle

        Returns:
            Exit code (99 for generic error)
        """
        self.logger.exception(f"Unexpected error: {error}")
        return 99

    def with_error_handling(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to wrap a function with error handling.

        Args:
            func: The function to wrap

        Returns:
            Wrapped function with error handling
        """
        from typing import get_type_hints

        # Get the return type annotation of the function
        return_type = get_type_hints(func).get('return')
        returns_int = return_type is int

        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the function name where the error occurred for easier debugging
                self.logger.error(f"Error occurred in function '{func.__name__}'")
                # Call handle_error to get the exit code and log the error
                exit_code = self.handle_error(e)
                
                # For functions returning int, always return the exit code
                if returns_int:
                    return cast(T, exit_code)
                
                # For other return types, check if we're in a pytest environment
                if "pytest" in sys.modules and sys.modules["pytest"] is not None:
                    # In pytest environment, re-raise the exception for pytest to catch
                    raise
                
                # Not in pytest environment, exit the program
                sys.exit(exit_code)

        return wrapper
```

## File: utils/excel_utils.py
```python
"""
Excel utility module for the FCA Dashboard application.

This module provides backward compatibility for the refactored Excel utilities.
All functions and classes are re-exported from the new excel package.

DEPRECATED: Use the new excel package instead:
    from fca_dashboard.utils.excel import ...
"""

import warnings

# Show deprecation warning
warnings.warn(
    "The excel_utils module is deprecated. Use the new excel package instead: "
    "from fca_dashboard.utils.excel import ...",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all functions and classes from the excel package
from fca_dashboard.utils.excel import (
    # Base
    ExcelUtilError,
    # Analysis utils
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    # Validation utils
    check_data_types,
    check_duplicate_rows,
    check_missing_values,
    check_value_ranges,
    clean_sheet_name,
    # Conversion utils
    convert_excel_to_csv,
    detect_duplicate_columns,
    detect_empty_rows,
    detect_header_row,
    detect_unnamed_columns,
    extract_excel_with_config,
    # Column utils
    get_column_names,
    # File utils
    get_database_schema,
    get_excel_file_type,
    # Sheet utils
    get_sheet_names,
    is_excel_file,
    is_valid_excel_file,
    load_excel_config,
    merge_excel_files,
    normalize_sheet_names,
    # Extraction utils
    read_excel_with_header_detection,
    save_excel_to_database,
    validate_columns_exist,
    validate_dataframe,
)

# For backward compatibility
__all__ = [
    # Base
    'ExcelUtilError',
    
    # File utils
    'get_excel_file_type',
    'is_excel_file',
    'is_valid_excel_file',
    
    # Sheet utils
    'get_sheet_names',
    'clean_sheet_name',
    'normalize_sheet_names',
    
    # Column utils
    'get_column_names',
    'validate_columns_exist',
    
    # Conversion utils
    'convert_excel_to_csv',
    'get_database_schema',
    'merge_excel_files',
    'save_excel_to_database',
    
    # Analysis utils
    'analyze_excel_structure',
    'analyze_unique_values',
    'analyze_column_statistics',
    'analyze_text_columns',
    'detect_empty_rows',
    'detect_header_row',
    'detect_duplicate_columns',
    'detect_unnamed_columns',
    
    # Extraction utils
    'read_excel_with_header_detection',
    'extract_excel_with_config',
    'load_excel_config',
    
    # Validation utils
    'check_missing_values',
    'check_duplicate_rows',
    'check_value_ranges',
    'check_data_types',
    'validate_dataframe',
]
```

## File: utils/excel/__init__.py
```python
"""
Excel utilities package for the FCA Dashboard application.

This package provides utilities for working with Excel files,
including file type detection, validation, data extraction, and analysis.
"""

# Re-export all functions and classes from the modules
from fca_dashboard.utils.excel.analysis_utils import (
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    detect_duplicate_columns,
    detect_empty_rows,
    detect_header_row,
    detect_unnamed_columns,
)
from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.column_utils import (
    get_column_names,
    validate_columns_exist,
)
from fca_dashboard.utils.excel.conversion_utils import (
    convert_excel_to_csv,
    get_database_schema,
    merge_excel_files,
    save_excel_to_database,
)
from fca_dashboard.utils.excel.extraction_utils import (
    extract_excel_with_config,
    load_excel_config,
    read_excel_with_header_detection,
)
from fca_dashboard.utils.excel.file_utils import (
    get_excel_file_type,
    is_excel_file,
    is_valid_excel_file,
)
from fca_dashboard.utils.excel.sheet_utils import (
    clean_sheet_name,
    get_sheet_names,
    normalize_sheet_names,
)
from fca_dashboard.utils.excel.validation_utils import (
    check_data_types,
    check_duplicate_rows,
    check_missing_values,
    check_value_ranges,
    validate_dataframe,
)

# For backward compatibility
# This allows existing code to continue working with the old import paths
__all__ = [
    # Base
    'ExcelUtilError',
    
    # File utils
    'get_excel_file_type',
    'is_excel_file',
    'is_valid_excel_file',
    
    # Sheet utils
    'get_sheet_names',
    'clean_sheet_name',
    'normalize_sheet_names',
    
    # Column utils
    'get_column_names',
    'validate_columns_exist',
    
    # Conversion utils
    'convert_excel_to_csv',
    'merge_excel_files',
    'save_excel_to_database',
    'get_database_schema',
    
    # Analysis utils
    'analyze_excel_structure',
    'analyze_unique_values',
    'analyze_column_statistics',
    'analyze_text_columns',
    'detect_empty_rows',
    'detect_header_row',
    'detect_duplicate_columns',
    'detect_unnamed_columns',
    
    # Extraction utils
    'read_excel_with_header_detection',
    'extract_excel_with_config',
    'load_excel_config',
    
    # Validation utils
    'check_missing_values',
    'check_duplicate_rows',
    'check_value_ranges',
    'check_data_types',
    'validate_dataframe',
]
```

## File: utils/excel/analysis_utils.py
```python
"""
Analysis utility module for Excel operations.

This module provides utilities for analyzing Excel files,
including structure analysis, header detection, and column analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.file_utils import get_excel_file_type, is_excel_file
from fca_dashboard.utils.excel.sheet_utils import get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def analyze_excel_structure(
    file_path: Union[str, Path],
    max_rows: int = 20
) -> Dict[str, Any]:
    """
    Analyze the structure of an Excel file and return information about its sheets, headers, etc.
    
    Args:
        file_path: Path to the Excel file to analyze.
        max_rows: Maximum number of rows to read for analysis (default: 20).
        
    Returns:
        A dictionary containing information about the Excel file structure:
        {
            'file_type': str,
            'sheet_names': List[str],
            'sheets_info': Dict[str, Dict],  # Information about each sheet
        }
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs during analysis.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if it's an Excel file
    if not is_excel_file(path):
        error_msg = f"Not an Excel file: {path}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    try:
        # Get the file type
        file_type = get_excel_file_type(path)
        
        # Get the sheet names
        sheet_names = get_sheet_names(path)
        
        # Initialize the result dictionary
        result = {
            'file_type': file_type,
            'sheet_names': sheet_names,
            'sheets_info': {}
        }
        
        # Analyze each sheet
        for sheet_name in sheet_names:
            # Read a sample of the sheet
            if file_type == "csv":
                df_sample = pd.read_csv(path, nrows=max_rows)
            else:
                df_sample = pd.read_excel(path, sheet_name=sheet_name, nrows=max_rows)
            
            # Analyze the sheet structure
            sheet_info = {
                'shape': df_sample.shape,
                'columns': list(df_sample.columns),
                'empty_rows': detect_empty_rows(df_sample),
                'header_row': detect_header_row(df_sample),
                'duplicate_columns': detect_duplicate_columns(df_sample),
                'unnamed_columns': detect_unnamed_columns(df_sample)
            }
            
            result['sheets_info'][sheet_name] = sheet_info
        
        return result
    except Exception as e:
        error_msg = f"Error analyzing Excel file {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def detect_empty_rows(df: pd.DataFrame, max_rows: int = 10) -> List[int]:
    """
    Detect empty rows in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        max_rows: Maximum number of rows to check (default: 10).
        
    Returns:
        A list of indices of empty rows.
    """
    empty_rows = []
    for i in range(min(max_rows, len(df))):
        if df.iloc[i].isna().all():
            empty_rows.append(i)
    return empty_rows


def detect_header_row(df: pd.DataFrame, max_rows: int = 10) -> Optional[int]:
    """
    Detect the most likely header row in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        max_rows: Maximum number of rows to check (default: 10).
        
    Returns:
        The index of the most likely header row, or None if no header row is detected.
    """
    potential_header_rows = []
    
    for i in range(min(max_rows, len(df))):
        # Skip empty rows
        if df.iloc[i].isna().all():
            continue
        
        # Check if this row has string values that could be headers
        row = df.iloc[i]
        
        # Count string values that could be headers
        string_count = sum(1 for val in row if isinstance(val, str) and val.strip())
        
        # If most cells in this row are non-empty strings, it might be a header row
        if string_count / len(row) > 0.5:
            potential_header_rows.append((i, string_count))
    
    # Return the row with the most string values, or None if no potential header rows
    if potential_header_rows:
        # Sort by string count in descending order
        potential_header_rows.sort(key=lambda x: x[1], reverse=True)
        return potential_header_rows[0][0]
    
    return None


def detect_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect duplicate column names in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        
    Returns:
        A list of duplicate column names.
    """
    return df.columns[df.columns.duplicated()].tolist()


def detect_unnamed_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect unnamed columns in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        
    Returns:
        A list of unnamed column names.
    """
    return [str(col) for col in df.columns if "Unnamed" in str(col)]


def analyze_unique_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_unique_values: int = 20
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze unique values in specified columns of a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        columns: Optional list of column names to analyze. If None, analyzes all columns.
        max_unique_values: Maximum number of unique values to include in the result.
            If a column has more unique values than this, only the counts will be included.
        
    Returns:
        A dictionary mapping column names to dictionaries containing:
            - 'count': The number of unique values
            - 'values': The unique values (if count <= max_unique_values)
            - 'value_counts': Dictionary mapping values to their counts (if count <= max_unique_values)
            - 'null_count': The number of null values
            - 'null_percentage': The percentage of null values
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, analyze all columns
    if columns is None:
        columns = df.columns
    
    # Initialize result dictionary
    result = {}
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get unique values
        unique_values = df[col].dropna().unique()
        unique_count = len(unique_values)
        
        # Get null count and percentage
        null_count = df[col].isna().sum()
        null_percentage = null_count / len(df) if len(df) > 0 else 0.0
        
        # Initialize column result
        col_result = {
            'count': unique_count,
            'null_count': null_count,
            'null_percentage': null_percentage
        }
        
        # Include unique values and counts if there aren't too many
        if unique_count <= max_unique_values:
            # Convert values to strings for better display
            unique_values_str = [str(val) for val in unique_values]
            
            # Get value counts
            value_counts = df[col].value_counts().to_dict()
            
            # Convert keys to strings for better display
            value_counts_str = {str(k): v for k, v in value_counts.items() if pd.notna(k)}
            
            col_result['values'] = unique_values_str
            col_result['value_counts'] = value_counts_str
        
        # Add to result
        result[col] = col_result
    
    return result


def analyze_column_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for numeric columns in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        columns: Optional list of column names to analyze. If None, analyzes all numeric columns.
        
    Returns:
        A dictionary mapping column names to dictionaries containing statistics:
            - 'min': Minimum value
            - 'max': Maximum value
            - 'mean': Mean value
            - 'median': Median value
            - 'std': Standard deviation
            - 'q1': First quartile (25th percentile)
            - 'q3': Third quartile (75th percentile)
            - 'iqr': Interquartile range
            - 'outliers_count': Number of outliers (values outside 1.5*IQR from Q1 and Q3)
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Initialize result dictionary
    result = {}
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is not numeric, skipping statistics calculation")
            continue
        
        # Get numeric values (drop NaN)
        values = df[col].dropna()
        
        # Skip if no values
        if len(values) == 0:
            logger.warning(f"Column '{col}' has no non-null values, skipping statistics calculation")
            continue
        
        # Calculate statistics
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outliers_count = len(outliers)
        
        # Add to result
        result[col] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'outliers_count': outliers_count
        }
    
    return result


def analyze_text_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze text columns in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        columns: Optional list of column names to analyze. If None, analyzes all object columns.
        
    Returns:
        A dictionary mapping column names to dictionaries containing:
            - 'min_length': Minimum string length
            - 'max_length': Maximum string length
            - 'avg_length': Average string length
            - 'empty_count': Number of empty strings
            - 'pattern_analysis': Dictionary with counts of different patterns
                (e.g., 'numeric', 'alpha', 'alphanumeric', 'email', 'url', 'date', 'other')
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, use all object columns
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    # Initialize result dictionary
    result = {}
    
    # Compile regex patterns
    import re
    numeric_pattern = re.compile(r'^\d+$')
    alpha_pattern = re.compile(r'^[a-zA-Z]+$')
    alphanumeric_pattern = re.compile(r'^[a-zA-Z0-9]+$')
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    url_pattern = re.compile(r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$')
    date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Convert to string and drop NaN
        values = df[col].astype(str).replace('nan', '').replace('None', '')
        
        # Skip if no values
        if len(values) == 0:
            logger.warning(f"Column '{col}' has no values, skipping text analysis")
            continue
        
        # Calculate string lengths
        lengths = values.str.len()
        min_length = lengths.min()
        max_length = lengths.max()
        avg_length = lengths.mean()
        
        # Count empty strings
        empty_count = (lengths == 0).sum()
        
        # Analyze patterns
        pattern_counts = {
            'numeric': 0,
            'alpha': 0,
            'alphanumeric': 0,
            'email': 0,
            'url': 0,
            'date': 0,
            'other': 0
        }
        
        for val in values:
            if val == '':
                continue
            elif numeric_pattern.match(val):
                pattern_counts['numeric'] += 1
            elif alpha_pattern.match(val):
                pattern_counts['alpha'] += 1
            elif alphanumeric_pattern.match(val):
                pattern_counts['alphanumeric'] += 1
            elif email_pattern.match(val):
                pattern_counts['email'] += 1
            elif url_pattern.match(val):
                pattern_counts['url'] += 1
            elif date_pattern.match(val):
                pattern_counts['date'] += 1
            else:
                pattern_counts['other'] += 1
        
        # Add to result
        result[col] = {
            'min_length': min_length,
            'max_length': max_length,
            'avg_length': avg_length,
            'empty_count': empty_count,
            'pattern_analysis': pattern_counts
        }
    
    return result
```

## File: utils/excel/base.py
```python
"""
Base module for Excel utilities.

This module provides base classes and common utilities for Excel operations.
"""

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger


class ExcelUtilError(FCADashboardError):
    """Exception raised for errors in Excel utility functions."""
    pass
```

## File: utils/excel/column_utils.py
```python
"""
Column utility module for Excel operations.

This module provides utilities for working with Excel columns,
including column name retrieval and validation.
"""

from pathlib import Path
from typing import List, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.sheet_utils import get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def get_column_names(file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> List[str]:
    """
    Get the column names from an Excel file.
    
    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to read (default: 0, first sheet).
        
    Returns:
        A list of column names.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If the sheet does not exist or an error occurs while reading the file.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # Read the first row of the file to get column names
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=0)
        else:
            # Check if the sheet exists
            sheet_names = get_sheet_names(path)
            
            # Handle sheet name as index or name
            if isinstance(sheet_name, int) and sheet_name < len(sheet_names):
                # If sheet_name is an integer index, use it to get the actual sheet name
                actual_sheet_name = sheet_names[sheet_name]
            elif sheet_name in sheet_names:
                # If sheet_name is already a valid sheet name, use it directly
                actual_sheet_name = sheet_name
            else:
                # If sheet_name is neither a valid index nor a valid name, raise an error
                error_msg = f"Sheet '{sheet_name}' not found in {path}. Available sheets: {sheet_names}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            df = pd.read_excel(path, sheet_name=actual_sheet_name, nrows=0)
        
        return list(df.columns)
    except ExcelUtilError:
        # Re-raise ExcelUtilError
        raise
    except Exception as e:
        error_msg = f"Error getting column names from {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def validate_columns_exist(file_path: Union[str, Path], columns: List[str], sheet_name: Union[str, int] = 0) -> bool:
    """
    Validate that the specified columns exist in an Excel file.
    
    Args:
        file_path: Path to the Excel file.
        columns: List of column names to validate.
        sheet_name: Name or index of the sheet to read (default: 0, first sheet).
        
    Returns:
        True if all columns exist, False otherwise.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    # If no columns to validate, return True
    if not columns:
        return True
    
    # Get the column names from the file
    file_columns = get_column_names(file_path, sheet_name)
    
    # Check if all columns exist
    return all(column in file_columns for column in columns)
```

## File: utils/excel/conversion_utils.py
```python
"""
Conversion utility module for Excel operations.

This module provides utilities for converting Excel files to other formats
and merging Excel files.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.sheet_utils import get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def convert_excel_to_csv(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> None:
    """
    Convert an Excel file to CSV.
    
    Args:
        input_file: Path to the input Excel file.
        output_file: Path to the output CSV file.
        sheet_name: Name or index of the sheet to convert (default: 0, first sheet).
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_csv().
        
    Raises:
        FileNotFoundError: If the input file does not exist.
        ExcelUtilError: If the sheet does not exist or an error occurs during conversion.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file paths
    input_path = resolve_path(input_file)
    output_path = Path(output_file)  # Don't resolve output path as it may not exist yet
    
    # Check if input file exists
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Read the Excel file
        if input_path.suffix.lower() == ".csv":
            # If the input is already a CSV, just copy it
            df = pd.read_csv(input_path)
        else:
            # Check if the sheet exists
            sheet_names = get_sheet_names(input_path)
            
            # Handle sheet name as index or name
            if isinstance(sheet_name, int) and sheet_name < len(sheet_names):
                # If sheet_name is an integer index, use it to get the actual sheet name
                actual_sheet_name = sheet_names[sheet_name]
            elif sheet_name in sheet_names:
                # If sheet_name is already a valid sheet name, use it directly
                actual_sheet_name = sheet_name
            else:
                # If sheet_name is neither a valid index nor a valid name, raise an error
                error_msg = f"Sheet '{sheet_name}' not found in {input_path}. Available sheets: {sheet_names}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            # Read the Excel file
            df = pd.read_excel(input_path, sheet_name=actual_sheet_name)
        
        # Write to CSV
        df.to_csv(output_path, index=False, **kwargs)
        logger.info(f"Converted {input_path} to {output_path}")
    except ExcelUtilError:
        # Re-raise ExcelUtilError
        raise
    except Exception as e:
        error_msg = f"Error converting {input_path} to CSV: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def merge_excel_files(
    input_files: List[Union[str, Path]],
    output_file: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> None:
    """
    Merge multiple Excel files into a single Excel file.
    
    Args:
        input_files: List of paths to the input Excel files.
        output_file: Path to the output Excel file.
        sheet_name: Name or index of the sheet to read from each input file (default: 0, first sheet).
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_excel().
        
    Raises:
        ValueError: If the input file list is empty.
        FileNotFoundError: If any input file does not exist.
        ExcelUtilError: If the files have different columns or an error occurs during merging.
    """
    logger = get_logger("excel_utils")
    
    # Check if input file list is empty
    if not input_files:
        error_msg = "Input file list is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Resolve the file paths
    input_paths = [resolve_path(file) for file in input_files]
    output_path = Path(output_file)  # Don't resolve output path as it may not exist yet
    
    # Check if all input files exist
    for path in input_paths:
        if not path.is_file():
            logger.error(f"Input file not found: {path}")
            raise FileNotFoundError(f"Input file not found: {path}")
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Read the first file to get the columns
        if input_paths[0].suffix.lower() == ".csv":
            first_df = pd.read_csv(input_paths[0])
        else:
            # Handle sheet name for the first file
            sheet_names = get_sheet_names(input_paths[0])
            
            # Handle sheet name as index or name
            if isinstance(sheet_name, int) and sheet_name < len(sheet_names):
                # If sheet_name is an integer index, use it to get the actual sheet name
                actual_sheet_name = sheet_names[sheet_name]
            elif sheet_name in sheet_names:
                # If sheet_name is already a valid sheet name, use it directly
                actual_sheet_name = sheet_name
            else:
                # If sheet_name is neither a valid index nor a valid name, raise an error
                error_msg = f"Sheet '{sheet_name}' not found in {input_paths[0]}. Available sheets: {sheet_names}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            first_df = pd.read_excel(input_paths[0], sheet_name=actual_sheet_name)
        
        first_columns = list(first_df.columns)
        
        # Initialize the merged DataFrame with the first file
        merged_df = first_df.copy()
        
        # Merge the rest of the files
        for path in input_paths[1:]:
            # Read the file
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                # Handle sheet name for each file
                file_sheet_names = get_sheet_names(path)
                
                # Handle sheet name as index or name
                if isinstance(sheet_name, int) and sheet_name < len(file_sheet_names):
                    # If sheet_name is an integer index, use it to get the actual sheet name
                    file_actual_sheet_name = file_sheet_names[sheet_name]
                elif sheet_name in file_sheet_names:
                    # If sheet_name is already a valid sheet name, use it directly
                    file_actual_sheet_name = sheet_name
                else:
                    # If sheet_name is neither a valid index nor a valid name, raise an error
                    error_msg = f"Sheet '{sheet_name}' not found in {path}. Available sheets: {file_sheet_names}"
                    logger.error(error_msg)
                    raise ExcelUtilError(error_msg)
                
                df = pd.read_excel(path, sheet_name=file_actual_sheet_name)
            
            # Check if the columns match
            if list(df.columns) != first_columns:
                error_msg = f"Columns in {path} do not match the columns in {input_paths[0]}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            # Append the data
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        # Write to Excel
        merged_df.to_excel(output_path, index=False, **kwargs)
        logger.info(f"Merged {len(input_paths)} files into {output_path}")
    except Exception as e:
        error_msg = f"Error merging Excel files: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def save_excel_to_database(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    schema: Optional[str] = None,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a DataFrame to a database table.
    
    This function is a bridge to the database utilities module.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The database connection string.
        schema: The database schema (optional).
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        index: Whether to include the index in the table.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
        
    Raises:
        ExcelUtilError: If an error occurs while saving to the database.
    """
    logger = get_logger("excel_utils")
    
    try:
        from fca_dashboard.utils.database.base import save_dataframe_to_database
        save_dataframe_to_database(
            df=df,
            table_name=table_name,
            connection_string=connection_string,
            schema=schema,
            if_exists=if_exists,
            index=index,
            **kwargs
        )
    except Exception as e:
        error_msg = f"Error saving DataFrame to database table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def get_database_schema(
    connection_string: str,
    table_name: str
) -> str:
    """
    Get the schema of a database table.
    
    This function is a bridge to the database utilities module.
    
    Args:
        connection_string: The database connection string.
        table_name: The name of the table to get the schema for.
        
    Returns:
        A string containing the schema of the table.
        
    Raises:
        ExcelUtilError: If an error occurs while getting the schema.
    """
    logger = get_logger("excel_utils")
    
    try:
        from fca_dashboard.utils.database.base import get_table_schema
        return get_table_schema(connection_string, table_name)
    except Exception as e:
        error_msg = f"Error getting schema for table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e
```

## File: utils/excel/extraction_utils.py
```python
"""
Extraction utility module for Excel operations.

This module provides utilities for extracting data from Excel files,
including header detection, configuration-based extraction, and more.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.analysis_utils import detect_header_row
from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.file_utils import get_excel_file_type, is_excel_file
from fca_dashboard.utils.excel.sheet_utils import clean_sheet_name, get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def read_excel_with_header_detection(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    header_row: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read an Excel file with automatic header detection.
    
    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to read (default: 0, first sheet).
        header_row: Row index to use as the header (0-based). If None, will attempt to detect the header row.
        **kwargs: Additional arguments to pass to pandas.read_excel or pandas.read_csv.
        
    Returns:
        A pandas DataFrame with the detected headers.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # If header_row is not provided, detect it
        if header_row is None:
            # Read a sample of the file to detect the header row
            if path.suffix.lower() == ".csv":
                sample_df = pd.read_csv(path, nrows=20, header=None)
            else:
                sample_df = pd.read_excel(path, sheet_name=sheet_name, nrows=20, header=None)
            
            # Detect the header row
            detected_header_row = detect_header_row(sample_df)
            
            if detected_header_row is not None:
                header_row = detected_header_row
                logger.info(f"Detected header row at index {header_row}")
            else:
                # If no header row is detected, use the first row
                header_row = 0
                logger.info("No header row detected, using the first row")
        
        # Read the file with the detected header row
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, header=header_row, **kwargs)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, **kwargs)
        
        # Clean up column names
        df.columns = [str(col).strip() for col in df.columns]
        
        return df
    except Exception as e:
        error_msg = f"Error reading Excel file {path} with header detection: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def extract_excel_with_config(
    file_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Extract data from an Excel file using a configuration dictionary.
    
    This function provides a robust way to extract data from Excel files with complex structures,
    such as headers in non-standard positions, multiple sheets with different structures, etc.
    
    Args:
        file_path: Path to the Excel file.
        config: Configuration dictionary with the following structure:
            {
                "default": {  # Default settings for all sheets
                    "header_row": int or None,  # Row index to use as header (0-based)
                    "skip_rows": int or list,  # Rows to skip
                    "column_mapping": dict,  # Map original column names to new names
                    "required_columns": list,  # Columns that must exist
                    "drop_empty_rows": bool,  # Whether to drop rows that are all NaN
                    "drop_empty_columns": bool,  # Whether to drop columns that are all NaN
                    "clean_column_names": bool,  # Whether to clean column names
                    "strip_whitespace": bool,  # Whether to strip whitespace from string values
                    "convert_dtypes": bool,  # Whether to convert data types
                    "date_columns": list,  # Columns to convert to dates
                    "numeric_columns": list,  # Columns to convert to numeric
                    "boolean_columns": list,  # Columns to convert to boolean
                    "fillna_values": dict,  # Values to use for filling NaN values
                    "drop_columns": list,  # Columns to drop
                    "rename_columns": dict,  # Columns to rename
                    "sheet_name_mapping": dict,  # Map original sheet names to new names
                },
                "sheet_name1": {  # Settings specific to sheet_name1, overrides defaults
                    # Same structure as default
                },
                "sheet_name2": {
                    # Same structure as default
                }
            }
        **kwargs: Additional arguments to pass to pandas.read_excel or pandas.read_csv.
        
    Returns:
        A dictionary mapping sheet names to pandas DataFrames.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs during extraction.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if it's an Excel file
    if not is_excel_file(path):
        error_msg = f"Not an Excel file: {path}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    # Get the file type
    file_type = get_excel_file_type(path)
    
    # Get the sheet names
    sheet_names = get_sheet_names(path)
    
    # Initialize the default configuration
    default_config = {
        "header_row": None,  # Auto-detect
        "skip_rows": None,
        "column_mapping": {},
        "required_columns": [],
        "drop_empty_rows": True,
        "drop_empty_columns": False,
        "clean_column_names": True,
        "strip_whitespace": True,
        "convert_dtypes": True,
        "date_columns": [],
        "numeric_columns": [],
        "boolean_columns": [],
        "fillna_values": {},
        "drop_columns": [],
        "rename_columns": {},
        "sheet_name_mapping": {},
    }
    
    # Use provided config or empty dict
    config = config or {}
    
    # Get the default configuration from the provided config
    default_config.update(config.get("default", {}))
    
    # Initialize the result dictionary
    result = {}
    
    # Process each sheet
    for sheet_name in sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        
        try:
            # Get sheet-specific configuration, falling back to default
            sheet_config = default_config.copy()
            sheet_config.update(config.get(sheet_name, {}))
            
            # Determine the header row
            header_row = sheet_config["header_row"]
            
            # If header_row is None, try to detect it
            if header_row is None:
                # Read a sample of the sheet to detect the header row
                if file_type == "csv":
                    sample_df = pd.read_csv(path, nrows=20, header=None)
                else:
                    sample_df = pd.read_excel(path, sheet_name=sheet_name, nrows=20, header=None)
                
                # Detect the header row
                detected_header_row = detect_header_row(sample_df)
                
                if detected_header_row is not None:
                    header_row = detected_header_row
                    logger.info(f"Detected header row at index {header_row}")
                else:
                    # If no header row is detected, use the first row
                    header_row = 0
                    logger.info("No header row detected, using the first row")
            
            # Determine the rows to skip
            skip_rows = sheet_config["skip_rows"]
            
            # Read the sheet
            if file_type == "csv":
                df = pd.read_csv(path, header=header_row, skiprows=skip_rows, **kwargs)
            else:
                df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, skiprows=skip_rows, **kwargs)
            
            # Clean column names if requested
            if sheet_config["clean_column_names"]:
                df.columns = [str(col).strip() for col in df.columns]
            
            # Check for required columns
            required_columns = sheet_config["required_columns"]
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    error_msg = f"Required columns not found in sheet {sheet_name}: {missing_columns}"
                    logger.error(error_msg)
                    raise ExcelUtilError(error_msg)
            
            # Drop empty rows if requested
            if sheet_config["drop_empty_rows"]:
                df = df.dropna(how='all')
            
            # Drop empty columns if requested
            if sheet_config["drop_empty_columns"]:
                df = df.dropna(axis=1, how='all')
            
            # Strip whitespace from string values if requested
            if sheet_config["strip_whitespace"]:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
            
            # Convert data types if requested
            if sheet_config["convert_dtypes"]:
                # Convert date columns
                for col in sheet_config["date_columns"]:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error converting column {col} to date: {str(e)}")
                
                # Convert numeric columns
                for col in sheet_config["numeric_columns"]:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error converting column {col} to numeric: {str(e)}")
                
                # Convert boolean columns
                for col in sheet_config["boolean_columns"]:
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(str).str.lower().map({
                                'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                                'false': False, 'no': False, 'n': False, '0': False, 'f': False,
                                'nan': None, 'none': None, 'null': None, '': None
                            })
                        except Exception as e:
                            logger.warning(f"Error converting column {col} to boolean: {str(e)}")
            
            # Fill NaN values if requested
            fillna_values = sheet_config["fillna_values"]
            if fillna_values:
                df = df.fillna(fillna_values)
            
            # Drop columns if requested
            drop_columns = sheet_config["drop_columns"]
            if drop_columns:
                df = df.drop(columns=[col for col in drop_columns if col in df.columns])
            
            # Rename columns if requested
            rename_columns = sheet_config["rename_columns"]
            if rename_columns:
                df = df.rename(columns=rename_columns)
            
            # Apply column mapping if provided
            column_mapping = sheet_config["column_mapping"]
            if column_mapping:
                # Create a new DataFrame with mapped columns
                new_df = pd.DataFrame()
                for new_col, old_col in column_mapping.items():
                    if old_col in df.columns:
                        new_df[new_col] = df[old_col]
                
                # Replace the original DataFrame with the mapped one
                if not new_df.empty:
                    df = new_df
            
            # Get the normalized sheet name
            sheet_name_mapping = sheet_config["sheet_name_mapping"]
            if sheet_name_mapping and sheet_name in sheet_name_mapping:
                normalized_sheet_name = sheet_name_mapping[sheet_name]
            else:
                normalized_sheet_name = clean_sheet_name(sheet_name)
            
            # Store the processed DataFrame
            result[normalized_sheet_name] = df
            
            logger.info(f"Successfully processed sheet {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
        
        except Exception as e:
            error_msg = f"Error processing sheet {sheet_name}: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, ExcelUtilError):
                raise
            else:
                raise ExcelUtilError(error_msg) from e
    
    return result


def load_excel_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an Excel configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        A dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while loading the configuration.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(config_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        # Determine the file type
        if path.suffix.lower() == ".json":
            import json
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.suffix.lower() in [".yml", ".yaml"]:
            import yaml
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            error_msg = f"Unsupported configuration file format: {path.suffix}"
            logger.error(error_msg)
            raise ExcelUtilError(error_msg)
        
        return config
    except Exception as e:
        error_msg = f"Error loading configuration from {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e
```

## File: utils/excel/file_utils.py
```python
"""
File utility module for Excel operations.

This module provides utilities for working with Excel files,
including file type detection and validation.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def get_excel_file_type(file_path: Union[str, Path]) -> Optional[str]:
    """
    Get the file type of an Excel file.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        The file type as a string (e.g., "xlsx", "csv"), or None if not an Excel file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    # Get the file extension
    extension = path.suffix.lower().lstrip(".")
    
    # Check if it's an Excel file
    if extension in ["xlsx", "xls", "xlsm", "xlsb"]:
        return extension
    elif extension == "csv":
        return "csv"
    else:
        return None


def is_excel_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is an Excel file.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        True if the file is an Excel file, False otherwise.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_type = get_excel_file_type(file_path)
    return file_type is not None


def is_valid_excel_file(file_path: Union[str, Path]) -> bool:
    """
    Check if an Excel file is valid by attempting to read it.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        True if the file is a valid Excel file, False otherwise.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logger = get_logger("excel_utils")
    
    # Check if it's an Excel file
    if not is_excel_file(file_path):
        return False
    
    try:
        # Try to read the file
        if str(file_path).lower().endswith(".csv"):
            pd.read_csv(file_path, nrows=1)
        else:
            pd.read_excel(file_path, nrows=1)
        return True
    except Exception as e:
        logger.warning(f"Invalid Excel file {file_path}: {str(e)}")
        return False
```

## File: utils/excel/sheet_utils.py
```python
"""
Sheet utility module for Excel operations.

This module provides utilities for working with Excel sheets,
including sheet name retrieval, cleaning, and normalization.
"""

import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.file_utils import get_excel_file_type
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def get_sheet_names(file_path: Union[str, Path]) -> List[Union[str, int]]:
    """
    Get the sheet names from an Excel file.
    
    Args:
        file_path: Path to the Excel file.
        
    Returns:
        A list of sheet names. For CSV files, returns a list with a single element [0].
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # Handle CSV files
        if path.suffix.lower() == ".csv":
            return [0]  # CSV files have a single sheet with index 0
        
        # Get sheet names from Excel file
        excel_file = pd.ExcelFile(path)
        return excel_file.sheet_names
    except Exception as e:
        error_msg = f"Error getting sheet names from {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def clean_sheet_name(sheet_name: str) -> str:
    """
    Clean and normalize a sheet name.
    
    Args:
        sheet_name: The sheet name to clean.
        
    Returns:
        A cleaned and normalized sheet name.
    """
    # Convert to string if it's not already
    sheet_name = str(sheet_name)
    
    # Remove leading/trailing whitespace
    sheet_name = sheet_name.strip()
    
    # Replace special characters with underscores
    sheet_name = re.sub(r'[^\w\s]', '_', sheet_name)
    
    # Replace multiple spaces with a single underscore
    sheet_name = re.sub(r'\s+', '_', sheet_name)
    
    # Convert to lowercase
    sheet_name = sheet_name.lower()
    
    return sheet_name


def normalize_sheet_names(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Get a mapping of original sheet names to normalized sheet names.
    
    Args:
        file_path: Path to the Excel file.
        
    Returns:
        A dictionary mapping original sheet names to normalized sheet names.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    # Get the sheet names
    sheet_names = get_sheet_names(file_path)
    
    # Create a mapping of original to normalized sheet names
    sheet_name_mapping = {}
    for name in sheet_names:
        normalized_name = clean_sheet_name(name)
        sheet_name_mapping[name] = normalized_name
    
    return sheet_name_mapping
```

## File: utils/excel/validation_utils.py
```python
"""
Validation utility module for Excel operations.

This module provides utilities for validating Excel data,
including checking for NaN values, null values, and other validation checks.
"""

from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.logging_config import get_logger


def check_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.0,
    raise_error: bool = False
) -> Dict[str, float]:
    """
    Check for missing values (NaN, None) in a DataFrame.
    
    Args:
        df: The DataFrame to check.
        columns: Optional list of column names to check. If None, checks all columns.
        threshold: Maximum allowed percentage of missing values (0.0 to 1.0).
            If the percentage of missing values exceeds this threshold, an error is raised.
        raise_error: Whether to raise an error if the threshold is exceeded.
        
    Returns:
        A dictionary mapping column names to the percentage of missing values.
        
    Raises:
        ExcelUtilError: If raise_error is True and any column exceeds the threshold.
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, check all columns
    if columns is None:
        columns = df.columns
    
    # Calculate the percentage of missing values for each column
    missing_percentages = {}
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Count missing values (NaN, None)
        missing_count = df[col].isna().sum()
        total_count = len(df)
        
        # Calculate percentage
        if total_count > 0:
            missing_percentage = missing_count / total_count
        else:
            missing_percentage = 0.0
        
        missing_percentages[col] = missing_percentage
    
    # Check if any column exceeds the threshold
    if raise_error and any(pct > threshold for pct in missing_percentages.values()):
        # Get columns that exceed the threshold
        exceeding_columns = [col for col, pct in missing_percentages.items() if pct > threshold]
        
        # Format the error message
        error_msg = f"The following columns exceed the missing values threshold ({threshold * 100}%):\n"
        for col in exceeding_columns:
            error_msg += f"  - {col}: {missing_percentages[col] * 100:.2f}%\n"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return missing_percentages


def check_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    raise_error: bool = False
) -> Dict[str, int]:
    """
    Check for duplicate rows in a DataFrame.
    
    Args:
        df: The DataFrame to check.
        subset: Optional list of column names to consider when identifying duplicates.
            If None, uses all columns.
        raise_error: Whether to raise an error if duplicates are found.
        
    Returns:
        A dictionary with the count of duplicate rows and the indices of duplicate rows.
        
    Raises:
        ExcelUtilError: If raise_error is True and duplicates are found.
    """
    logger = get_logger("excel_utils")
    
    # Find duplicate rows
    duplicates = df.duplicated(subset=subset, keep='first')
    duplicate_indices = df[duplicates].index.tolist()
    duplicate_count = len(duplicate_indices)
    
    # Create result dictionary
    result = {
        "duplicate_count": duplicate_count,
        "duplicate_indices": duplicate_indices
    }
    
    # Check if duplicates were found
    if raise_error and duplicate_count > 0:
        # Format the error message
        if subset:
            error_msg = f"Found {duplicate_count} duplicate rows based on columns: {subset}"
        else:
            error_msg = f"Found {duplicate_count} duplicate rows"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return result


def check_value_ranges(
    df: pd.DataFrame,
    ranges: Dict[str, Dict[str, Union[int, float, None]]],
    raise_error: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Check if values in specified columns are within the given ranges.
    
    Args:
        df: The DataFrame to check.
        ranges: A dictionary mapping column names to range specifications.
            Each range specification is a dictionary with 'min' and 'max' keys.
            Example: {'age': {'min': 0, 'max': 120}, 'score': {'min': 0, 'max': 100}}
        raise_error: Whether to raise an error if values are outside the ranges.
        
    Returns:
        A dictionary mapping column names to dictionaries with counts of values outside the ranges.
        
    Raises:
        ExcelUtilError: If raise_error is True and values are outside the ranges.
    """
    logger = get_logger("excel_utils")
    
    # Initialize result dictionary
    result = {}
    
    # Check each column
    for col, range_spec in ranges.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get min and max values
        min_val = range_spec.get('min')
        max_val = range_spec.get('max')
        
        # Initialize counters
        below_min = 0
        above_max = 0
        
        # Check values
        if min_val is not None:
            below_min = df[df[col] < min_val].shape[0]
        
        if max_val is not None:
            above_max = df[df[col] > max_val].shape[0]
        
        # Store results
        result[col] = {
            'below_min': below_min,
            'above_max': above_max,
            'total_outside_range': below_min + above_max
        }
    
    # Check if any values are outside the ranges
    if raise_error and any(res['total_outside_range'] > 0 for res in result.values()):
        # Format the error message
        error_msg = "Values outside specified ranges found:\n"
        for col, res in result.items():
            if res['total_outside_range'] > 0:
                range_spec = ranges[col]
                min_val = range_spec.get('min', 'None')
                max_val = range_spec.get('max', 'None')
                error_msg += f"  - {col} (range: {min_val} to {max_val}):\n"
                error_msg += f"    - Below minimum: {res['below_min']}\n"
                error_msg += f"    - Above maximum: {res['above_max']}\n"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return result


def check_data_types(
    df: pd.DataFrame,
    type_specs: Dict[str, str],
    raise_error: bool = False
) -> Dict[str, Dict[str, Union[str, int]]]:
    """
    Check if values in specified columns have the expected data types.
    
    Args:
        df: The DataFrame to check.
        type_specs: A dictionary mapping column names to expected data types.
            Supported types: 'int', 'float', 'str', 'bool', 'date'.
            Example: {'age': 'int', 'name': 'str', 'score': 'float', 'is_active': 'bool', 'birth_date': 'date'}
        raise_error: Whether to raise an error if values have incorrect types.
        
    Returns:
        A dictionary mapping column names to dictionaries with type information and error counts.
        
    Raises:
        ExcelUtilError: If raise_error is True and values have incorrect types.
    """
    logger = get_logger("excel_utils")
    
    # Initialize result dictionary
    result = {}
    
    # Check each column
    for col, expected_type in type_specs.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get the current type
        current_type = str(df[col].dtype)
        
        # Initialize error count
        error_count = 0
        
        # Check type based on expected_type
        if expected_type == 'int':
            # Check if values can be converted to integers
            try:
                pd.to_numeric(df[col], errors='raise', downcast='integer')
            except (ValueError, TypeError):
                error_count = df[~df[col].isna() & ~df[col].astype(str).str.match(r'^-?\d+$')].shape[0]
        
        elif expected_type == 'float':
            # Check if values can be converted to floats
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                error_count = df[~df[col].isna() & ~df[col].astype(str).str.match(r'^-?\d+(\.\d+)?$')].shape[0]
        
        elif expected_type == 'str':
            # Check if values are strings
            error_count = df[~df[col].isna() & ~df[col].apply(lambda x: isinstance(x, str))].shape[0]
        
        elif expected_type == 'bool':
            # Check if values are booleans or can be interpreted as booleans
            valid_bool_values = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f', True, False, 1, 0}
            error_count = df[~df[col].isna() & ~df[col].astype(str).str.lower().isin(valid_bool_values)].shape[0]
        
        elif expected_type == 'date':
            # Check if values can be converted to dates
            try:
                pd.to_datetime(df[col], errors='raise')
            except (ValueError, TypeError):
                error_count = df[~df[col].isna()].shape[0]
        
        else:
            logger.warning(f"Unsupported type specification: {expected_type}")
            continue
        
        # Store results
        result[col] = {
            'expected_type': expected_type,
            'current_type': current_type,
            'error_count': error_count
        }
    
    # Check if any values have incorrect types
    if raise_error and any(res['error_count'] > 0 for res in result.values()):
        # Format the error message
        error_msg = "Values with incorrect data types found:\n"
        for col, res in result.items():
            if res['error_count'] > 0:
                error_msg += f"  - {col} (expected: {res['expected_type']}, current: {res['current_type']}):\n"
                error_msg += f"    - Error count: {res['error_count']}\n"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return result


def validate_dataframe(
    df: pd.DataFrame,
    validation_config: Dict[str, Dict],
    raise_error: bool = False
) -> Dict[str, Dict]:
    """
    Validate a DataFrame using multiple validation checks.
    
    Args:
        df: The DataFrame to validate.
        validation_config: A dictionary with validation configurations.
            Example:
            {
                'missing_values': {
                    'columns': ['col1', 'col2'],
                    'threshold': 0.1
                },
                'duplicate_rows': {
                    'subset': ['col1', 'col2']
                },
                'value_ranges': {
                    'age': {'min': 0, 'max': 120},
                    'score': {'min': 0, 'max': 100}
                },
                'data_types': {
                    'age': 'int',
                    'name': 'str',
                    'score': 'float',
                    'is_active': 'bool',
                    'birth_date': 'date'
                }
            }
        raise_error: Whether to raise an error if validation fails.
        
    Returns:
        A dictionary with validation results for each check.
        
    Raises:
        ExcelUtilError: If raise_error is True and validation fails.
    """
    logger = get_logger("excel_utils")
    
    # Initialize result dictionary
    result = {}
    
    # Perform validation checks
    try:
        # Check missing values
        if 'missing_values' in validation_config:
            config = validation_config['missing_values']
            columns = config.get('columns')
            threshold = config.get('threshold', 0.0)
            
            result['missing_values'] = check_missing_values(
                df=df,
                columns=columns,
                threshold=threshold,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check duplicate rows
        if 'duplicate_rows' in validation_config:
            config = validation_config['duplicate_rows']
            subset = config.get('subset')
            
            result['duplicate_rows'] = check_duplicate_rows(
                df=df,
                subset=subset,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check value ranges
        if 'value_ranges' in validation_config:
            ranges = validation_config['value_ranges']
            
            result['value_ranges'] = check_value_ranges(
                df=df,
                ranges=ranges,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check data types
        if 'data_types' in validation_config:
            type_specs = validation_config['data_types']
            
            result['data_types'] = check_data_types(
                df=df,
                type_specs=type_specs,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check if validation failed
        validation_failed = False
        error_msg = "Validation failed:\n"
        
        # Check missing values
        if 'missing_values' in result:
            missing_values = result['missing_values']
            threshold = validation_config['missing_values'].get('threshold', 0.0)
            
            if any(pct > threshold for pct in missing_values.values()):
                validation_failed = True
                error_msg += "Missing values check failed:\n"
                for col, pct in missing_values.items():
                    if pct > threshold:
                        error_msg += f"  - {col}: {pct * 100:.2f}%\n"
        
        # Check duplicate rows
        if 'duplicate_rows' in result:
            duplicate_rows = result['duplicate_rows']
            
            if duplicate_rows['duplicate_count'] > 0:
                validation_failed = True
                subset = validation_config['duplicate_rows'].get('subset')
                
                if subset:
                    error_msg += f"Duplicate rows check failed: Found {duplicate_rows['duplicate_count']} duplicate rows based on columns: {subset}\n"
                else:
                    error_msg += f"Duplicate rows check failed: Found {duplicate_rows['duplicate_count']} duplicate rows\n"
        
        # Check value ranges
        if 'value_ranges' in result:
            value_ranges = result['value_ranges']
            
            if any(res['total_outside_range'] > 0 for res in value_ranges.values()):
                validation_failed = True
                error_msg += "Value ranges check failed:\n"
                
                for col, res in value_ranges.items():
                    if res['total_outside_range'] > 0:
                        range_spec = validation_config['value_ranges'][col]
                        min_val = range_spec.get('min', 'None')
                        max_val = range_spec.get('max', 'None')
                        error_msg += f"  - {col} (range: {min_val} to {max_val}):\n"
                        error_msg += f"    - Below minimum: {res['below_min']}\n"
                        error_msg += f"    - Above maximum: {res['above_max']}\n"
        
        # Check data types
        if 'data_types' in result:
            data_types = result['data_types']
            
            if any(res['error_count'] > 0 for res in data_types.values()):
                validation_failed = True
                error_msg += "Data types check failed:\n"
                
                for col, res in data_types.items():
                    if res['error_count'] > 0:
                        error_msg += f"  - {col} (expected: {res['expected_type']}, current: {res['current_type']}):\n"
                        error_msg += f"    - Error count: {res['error_count']}\n"
        
        # Raise error if validation failed and raise_error is True
        if validation_failed and raise_error:
            logger.error(error_msg)
            raise ExcelUtilError(error_msg)
        
        return result
    
    except Exception as e:
        if not isinstance(e, ExcelUtilError):
            error_msg = f"Error during validation: {str(e)}"
            logger.error(error_msg)
            if raise_error:
                raise ExcelUtilError(error_msg) from e
        else:
            raise
        
        return result
```

## File: utils/json_utils.py
```python
"""
JSON utility functions for common JSON data operations.

This module provides utility functions for JSON serialization, deserialization,
validation, formatting, and safe access following CLEAN principles:
- Clear: Functions have descriptive names and clear docstrings.
- Logical: Each function has a single, well-defined purpose.
- Efficient: Optimized for typical JSON-related tasks.
- Adaptable: Allow optional parameters for flexibility.
"""

import json
from typing import Any, Dict, Optional, TypeVar, Union

T = TypeVar("T")


def json_load(file_path: str, encoding: str = "utf-8") -> Any:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.
        encoding: File encoding (default utf-8).

    Returns:
        Parsed JSON data.

    Raises:
        JSONDecodeError: if JSON is invalid.
        FileNotFoundError: if file does not exist.
    
    Example:
        >>> data = json_load("data.json")
        >>> print(data)
        {'name': 'Bob'}
    """
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def json_save(data: Any, file_path: str, encoding: str = "utf-8", indent: int = 2) -> None:
    """
    Save data as JSON to a file.

    Args:
        data: Data to serialize.
        file_path: Path to save the JSON file.
        encoding: File encoding (default utf-8).
        indent: Indentation spaces for formatting (default 2).

    Returns:
        None

    Raises:
        JSONDecodeError: if JSON is invalid.
        FileNotFoundError: if file does not exist.
    
    Example:
        >>> data = {"name": "Bob"}
        >>> json_save(data, "data.json")
    """
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def json_serialize(data: Any, indent: Optional[int] = None) -> str:
    """
    Serialize data to a JSON string.

    Args:
        data: Data to serialize.
        indent: Optional indentation for formatting.

    Returns:
        JSON-formatted string.

    Example:
        >>> json_serialize({"key": "value"})
        '{"key": "value"}'
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)


def json_deserialize(json_str: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Deserialize a JSON string into a Python object.

    Args:
        json_str: JSON-formatted string.
        default: Value to return if deserialization fails.

    Returns:
        Python data object or default.

    Example:
        >>> json_deserialize('{"name": "Bob"}')
        {'name': 'Bob'}
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default


def json_is_valid(json_str: str) -> bool:
    """
    Check if a string is valid JSON.

    Args:
        json_str: String to validate.

    Returns:
        True if valid JSON, False otherwise.

    Example:
        >>> json_is_valid('{"valid": true}')
        True
        >>> json_is_valid('{invalid json}')
        False
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def pretty_print_json(data: Any) -> str:
    """
    Pretty-print JSON data with indentation.

    Args:
        data: JSON data (Python object).

    Returns:
        Pretty-printed JSON string.

    Example:
        >>> pretty_print_json({"key": "value"})
        '{\n  "key": "value"\n}'
    """
    return json.dumps(data, ensure_ascii=False, indent=2)


def safe_get(data: Dict, key: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Safely get a value from a dictionary.

    Args:
        data: Dictionary to extract value from.
        key: Key to look up.
        default: Default value if key is missing.

    Returns:
        Value associated with key or default.

    Example:
        >>> safe_get({"a": 1}, "a")
        1
        >>> safe_get({"a": 1}, "b", 0)
        0
    """
    return data.get(key, default)


def safe_get_nested(data: Dict, *keys: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Safely retrieve a nested value from a dictionary.

    Args:
        data: Nested dictionary.
        *keys: Sequence of keys for nested lookup.
        default: Default value if key path is missing.

    Returns:
        Nested value or default.

    Example:
        >>> safe_get_nested({"a": {"b": 2}}, "a", "b")
        2
        >>> safe_get_nested({"a": {"b": 2}}, "a", "c", default="missing")
        'missing'
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
```

## File: utils/logging_config.py
```python
"""
Logging configuration module for the FCA Dashboard application.

This module provides functionality to configure logging for the application
using Loguru, which offers improved formatting, better exception handling,
and simplified configuration compared to the standard logging module.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from loguru import logger  # type: ignore

# Define a Record type for type hints
Record = Dict[str, Any]
# Define a FormatFunction type for type hints
FormatFunction = Callable[[Record], str]


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 month",
    format_string: Optional[Union[str, Callable[[Record], str]]] = None,
    simple_format: bool = False,
) -> None:
    """
    Configure application logging with console and optional file output using Loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is configured.
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 month", "1 year")
        format_string: Custom format string for log messages
        simple_format: Use a simplified format for production environments
    """
    # Remove default handlers
    logger.remove()

    # Default format string if none provided
    if format_string is None:
        if simple_format:
            # Simple format for production environments
            def simple_format_fn(record: Record) -> str:
                return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            
            format_string = simple_format_fn
        else:
            # Detailed format for development environments
            def safe_format(record: Record) -> str:
                # Add the name from extra if available, otherwise use empty string
                name = record["extra"].get("name", "")
                name_part = f"<cyan>{name}</cyan> | " if name else ""

                return (
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    f"{name_part}"
                    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                    "<level>{message}</level>"
                ).format_map(record)

            format_string = safe_format

    # Add console handler
    logger.add(sys.stderr, level=level.upper(), format=format_string, colorize=True)  # type: ignore

    # Add file handler if log_file is provided
    if log_file:
        # Create the log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add rotating file handler
        logger.add(  # type: ignore[arg-type]
            str(log_path),
            level=level.upper(),
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured with level: {level}")


def get_logger(name: str = "fca_dashboard") -> Any:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name, typically the module name

    Returns:
        Loguru logger instance
    """
    return logger.bind(name=name)
```

## File: utils/loguru_stubs.pyi
```
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

class Logger:
    def remove(self, handler_id: Optional[int] = None) -> None: ...
    def add(
        self,
        sink: Union[TextIO, str, Callable, Dict[str, Any]],
        *,
        level: Optional[Union[str, int]] = None,
        format: Optional[Union[str, Callable[[Dict[str, Any]], str]]] = None,
        filter: Optional[Union[str, Callable, Dict[str, Any]]] = None,
        colorize: Optional[bool] = None,
        serialize: Optional[bool] = None,
        backtrace: Optional[bool] = None,
        diagnose: Optional[bool] = None,
        enqueue: Optional[bool] = None,
        catch: Optional[bool] = None,
        rotation: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
        retention: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
        compression: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
        delay: Optional[bool] = None,
        mode: Optional[str] = None,
        buffering: Optional[int] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> int: ...
    def bind(self, **kwargs: Any) -> "Logger": ...
    def opt(
        self,
        *,
        exception: Optional[Union[bool, Tuple[Any, ...], Dict[str, Any]]] = None,
        record: Optional[bool] = None,
        lazy: Optional[bool] = None,
        colors: Optional[bool] = None,
        raw: Optional[bool] = None,
        capture: Optional[bool] = None,
        depth: Optional[int] = None,
        ansi: Optional[bool] = None,
    ) -> "Logger": ...
    def trace(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def info(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def success(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def error(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def log(self, level: Union[int, str], __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def level(self, name: str, no: int = 0, color: Optional[str] = None, icon: Optional[str] = None) -> "Logger": ...
    def disable(self, name: str) -> None: ...
    def enable(self, name: str) -> None: ...
    def configure(
        self,
        *,
        handlers: List[Dict[str, Any]] = [],
        levels: List[Dict[str, Any]] = [],
        extra: Dict[str, Any] = {},
        patcher: Optional[Callable] = None,
        activation: List[Tuple[str, bool]] = [],
    ) -> None: ...
    def patch(self, patcher: Callable) -> "Logger": ...
    def complete(self) -> None: ...
    @property
    def catch(self) -> Callable: ...

logger: Logger
```

## File: utils/number_utils.py
```python
"""
Number utility functions for common numeric operations.

This module provides a collection of utility functions for number formatting,
rounding, and random number generation that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import random
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Union, overload

# Type aliases for numeric types
NumericType = Union[int, float, Decimal]


def format_currency(
    value: Optional[NumericType],
    symbol: str = "$",
    decimal_places: int = 2,
    thousands_sep: str = ",",
    decimal_sep: str = ".",
    default: str = "",
) -> str:
    """
    Format a number as a currency string.

    Args:
        value: The numeric value to format.
        symbol: Currency symbol to prepend (default: "$").
        decimal_places: Number of decimal places to show (default: 2).
        thousands_sep: Character to use as thousands separator (default: ",").
        decimal_sep: Character to use as decimal separator (default: ".").
        default: Value to return if input is None (default: "").

    Returns:
        Formatted currency string.

    Raises:
        TypeError: If value is not a numeric type.

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234.56, symbol="€", decimal_sep=",")
        '€1,234,56'
    """
    if value is None:
        return default

    # Validate input type
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Handle negative values
    is_negative = value < 0
    abs_value = abs(value)

    # Round to specified decimal places
    if isinstance(value, Decimal):
        rounded_value = abs_value.quantize(Decimal(f"0.{'0' * decimal_places}"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = round(abs_value, decimal_places)

    # Convert to string and split into integer and decimal parts
    str_value = str(rounded_value)
    if "." in str_value:
        int_part, dec_part = str_value.split(".")
    else:
        int_part, dec_part = str_value, ""

    # Format integer part with thousands separator
    formatted_int = ""
    for i, char in enumerate(reversed(int_part)):
        if i > 0 and i % 3 == 0:
            formatted_int = thousands_sep + formatted_int
        formatted_int = char + formatted_int

    # Format decimal part
    if decimal_places > 0:
        # Pad with zeros if needed
        dec_part = dec_part.ljust(decimal_places, "0")
        # Truncate if too long
        dec_part = dec_part[:decimal_places]
        formatted_value = formatted_int + decimal_sep + dec_part
    else:
        formatted_value = formatted_int

    # Add currency symbol and handle negative values
    if is_negative:
        return f"-{symbol}{formatted_value}"
    else:
        return f"{symbol}{formatted_value}"


def round_to(value: NumericType, places: int = 0) -> NumericType:
    """
    Round a number to a specified number of decimal places with ROUND_HALF_UP rounding.

    This function handles both positive and negative decimal places:
    - Positive places round to that many decimal places
    - Zero places round to the nearest integer
    - Negative places round to tens, hundreds, etc.

    Args:
        value: The numeric value to round.
        places: Number of decimal places to round to (default: 0).

    Returns:
        Rounded value of the same type as the input.

    Raises:
        TypeError: If value is not a numeric type.

    Examples:
        >>> round_to(1.234, 2)
        1.23
        >>> round_to(1.235, 2)
        1.24
        >>> round_to(123, -1)
        120
        >>> round_to(125, -1)
        130
    """
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Preserve the original type
    original_type = type(value)
    
    # Convert to Decimal for consistent rounding behavior
    if not isinstance(value, Decimal):
        decimal_value = Decimal(str(value))
    else:
        decimal_value = value
    
    # Calculate the factor based on places
    factor = Decimal("10") ** places
    
    if places >= 0:
        # For positive places (decimal places)
        result = decimal_value.quantize(Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP)
    else:
        # For negative places (tens, hundreds, etc.)
        # First divide by factor, round to integer, then multiply back
        factor = Decimal("10") ** abs(places)
        result = (decimal_value / factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * factor
    
    # Return the result in the original type
    if original_type == int or (places == 0 and original_type == float):
        # Convert to int if original was int or if rounding to integer (places=0)
        return int(result)
    elif original_type == float:
        return float(result)
    else:
        return result  # Already a Decimal


def random_number(min_value: int, max_value: int) -> int:
    """
    Generate a random integer within a specified range.

    Args:
        min_value: The minimum value (inclusive).
        max_value: The maximum value (inclusive).

    Returns:
        A random integer between min_value and max_value (inclusive).

    Raises:
        ValueError: If min_value is greater than max_value.
        TypeError: If min_value or max_value is not an integer.

    Examples:
        >>> # Returns a random number between 1 and 10
        >>> random_number(1, 10)
        7
        >>> # Returns a random number between -10 and 10
        >>> random_number(-10, 10)
        -3
    """
    # Validate input types
    if not isinstance(min_value, int):
        raise TypeError(f"min_value must be an integer, got {type(min_value).__name__}")
    if not isinstance(max_value, int):
        raise TypeError(f"max_value must be an integer, got {type(max_value).__name__}")

    # Validate range
    if min_value > max_value:
        raise ValueError(f"min_value ({min_value}) must be less than or equal to max_value ({max_value})")

    return random.randint(min_value, max_value)
```

## File: utils/path_util.py
```python
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def get_root_dir() -> Path:
    """Return the project's root directory (assuming this module is within project)."""
    return Path(__file__).resolve().parents[2]


def get_config_path(filename: str = "settings.yml") -> Path:
    """Get absolute path to the config file, ensuring it exists."""
    config_path = get_root_dir() / "fca_dashboard" / "config" / filename
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
    return config_path


def get_logs_path(filename: str = "fca_dashboard.log") -> Path:
    """Get absolute path to the log file, ensuring the logs directory exists."""
    logs_dir = get_root_dir() / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    return logs_dir / filename


def resolve_path(path: Union[str, Path], base_dir: Union[Path, None] = None) -> Path:
    """
    Resolve a path relative to the base directory or project root.

    Args:
        path: The path to resolve.
        base_dir: Optional base directory; defaults to the project's root directory.

    Returns:
        Resolved Path object.
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    if path_obj.exists():
        return path_obj.resolve()

    if base_dir is None:
        base_dir = get_root_dir()

    candidate_paths = [
        base_dir / path_obj,
        base_dir / "fca_dashboard" / path_obj,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            logger.debug(f"Resolved path '{path}' to '{candidate.resolve()}'")
            return candidate.resolve()

    logger.warning(f"Failed to resolve path '{path}'. Returning as is.")
    return path_obj
```

## File: utils/string_utils.py
```python
"""
String utility functions for common text operations.

This module provides a collection of utility functions for string manipulation
that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import re
import unicodedata
from typing import Optional


def capitalize(text: str) -> str:
    """
    Capitalize the first letter of a string, preserving leading whitespace.

    Args:
        text: The string to capitalize.

    Returns:
        A string with the first non-space character capitalized.

    Examples:
        >>> capitalize("hello")
        'Hello'
        >>> capitalize("  hello")
        '  Hello'
        >>> capitalize("123abc")
        '123abc'
        >>> capitalize("")
        ''
    """
    if not text:
        return ""
    
    # If the string starts with non-alphabetic characters (except whitespace),
    # return it unchanged
    if text.strip() and not text.strip()[0].isalpha():
        return text
    
    # Preserve leading whitespace and capitalize the first non-space character
    leading_spaces = len(text) - len(text.lstrip())
    return text[:leading_spaces] + text[leading_spaces:].capitalize()


def slugify(text: str) -> str:
    """
    Convert text into a URL-friendly slug.

    This function:
    1. Converts to lowercase
    2. Removes accents/diacritics
    3. Replaces spaces and special characters with hyphens

    Args:
        text: The string to convert to a slug.

    Returns:
        URL-friendly slug.

    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Héllo, Wörld!")
        'hello-world'
    """
    if not text:
        return ""

    # Normalize and remove accents
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    
    # Replace multiple hyphens with a single hyphen
    text = re.sub(r"-+", "-", text)
    
    return text


def truncate(text: str, length: int, suffix: str = "...") -> str:
    """
    Limit the length of a string and add a suffix if truncated.

    Args:
        text: The string to truncate.
        length: Maximum allowed length before truncation.
        suffix: String appended after truncation (default "...").

    Returns:
        Truncated string.

    Examples:
        >>> truncate("Hello World", 5)
        'Hello...'
        >>> truncate("Hello", 10)
        'Hello'
    """
    if not text:
        return ""

    if length <= 0:
        return suffix

    return text if len(text) <= length else text[:length] + suffix


def is_empty(text: Optional[str]) -> bool:
    """
    Check if a string is empty or contains only whitespace.

    Args:
        text: The string to check.

    Returns:
        True if empty or whitespace, False otherwise.

    Raises:
        TypeError: if text is None.

    Examples:
        >>> is_empty("   ")
        True
        >>> is_empty("Hello")
        False
    """
    if text is None:
        raise TypeError("Cannot check emptiness of None")

    return not bool(text.strip())
```

## File: utils/upload_util.py
```python
"""
File upload utility module for the FCA Dashboard application.

This module provides functionality for uploading files to specified destinations,
with features like duplicate handling, error handling, and logging.
"""

import shutil
import time
from pathlib import Path
from typing import Union

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


class FileUploadError(FCADashboardError):
    """Exception raised for errors during file upload operations."""
    pass


def upload_file(source: Union[str, Path], destination_dir: Union[str, Path], target_filename: str = None) -> bool:
    """
    Upload a file by copying it to the destination directory.
    
    Args:
        source: Path to the file to upload. Can be absolute or relative.
        destination_dir: Directory where the file should be uploaded. Can be absolute or relative.
        target_filename: Optional filename to use in the destination. If None, uses the source filename.
        
    Returns:
        True if the file was successfully uploaded.
        
    Raises:
        FileNotFoundError: If the source file does not exist.
        FileNotFoundError: If the destination directory does not exist.
        FileUploadError: If an error occurs during the upload process.
    """
    logger = get_logger("upload_util")
    
    # Resolve paths to handle relative paths correctly
    source_path = resolve_path(source)
    dest_dir_path = resolve_path(destination_dir)
    
    # Validate source file
    if not source_path.is_file():
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Validate destination directory
    if not dest_dir_path.is_dir():
        logger.error(f"Destination directory not found: {dest_dir_path}")
        raise FileNotFoundError(f"Destination directory not found: {dest_dir_path}")
    
    # Get the filename
    filename = target_filename if target_filename else source_path.name
    dest_path = dest_dir_path / filename
    
    # Handle duplicate filenames
    if dest_path.exists():
        logger.info(f"File already exists at destination: {dest_path}")
        # Generate a new filename with timestamp
        base_name = dest_path.stem
        extension = dest_path.suffix
        timestamp = int(time.time())
        new_filename = f"{base_name}_{timestamp}{extension}"
        dest_path = dest_dir_path / new_filename
        logger.info(f"Renamed to: {dest_path}")
    
    try:
        # Copy the file to the destination
        logger.info(f"Uploading file {source_path} to {dest_path}")
        shutil.copy(source_path, dest_path)
        logger.info(f"Successfully uploaded file to {dest_path}")
        return True
    except Exception as e:
        error_msg = f"Error uploading file {source_path} to {dest_path}: {str(e)}"
        logger.error(error_msg)
        raise FileUploadError(error_msg) from e
```

## File: utils/validation_utils.py
```python
"""
Validation utilities for common data formats.

This module provides functions to validate common data formats such as
email addresses, phone numbers, and URLs.
"""
import re
from typing import Any


def is_valid_email(email: Any) -> bool:
    """
    Validate if the input is a properly formatted email address.

    Args:
        email: The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    # RFC 5322 compliant email regex pattern with additional validations
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
    
    # Basic pattern match
    if not re.match(pattern, email):
        return False
    
    # Additional validations
    if '..' in email:  # No consecutive dots
        return False
    if email.endswith('.'):  # No trailing dot
        return False
    if ' ' in email:  # No spaces
        return False
    
    # Check domain part
    domain = email.split('@')[1]
    if domain.startswith('-') or domain.endswith('-'):  # No leading/trailing hyphens in domain
        return False
    
    # Check for hyphens at the end of domain parts
    domain_parts = domain.split('.')
    return all(not part.endswith('-') for part in domain_parts)


def is_valid_phone(phone: Any) -> bool:
    """
    Validate if the input is a properly formatted phone number.

    Accepts various formats including:
    - 10 digits: 1234567890
    - Hyphenated: 123-456-7890
    - Parentheses: (123) 456-7890
    - International: +1 123-456-7890
    - Dots: 123.456.7890
    - Spaces: 123 456 7890

    Args:
        phone: The phone number to validate.

    Returns:
        bool: True if the phone number is valid, False otherwise.
    """
    if not isinstance(phone, str):
        return False

    # Check for specific invalid formats first
    if phone == "":
        return False
    
    # Check for spaces around hyphens
    if " - " in phone:
        return False
    
    # Check for missing space after parentheses in format like (123)456-7890
    if re.search(r'\)[0-9]', phone):
        return False
    
    # Remove all non-alphanumeric characters for normalization
    normalized = re.sub(r'[^0-9+]', '', phone)
    
    # Check for letters in the phone number
    if re.search(r'[a-zA-Z]', phone):
        return False
    
    # Check for international format (starting with +)
    if normalized.startswith('+'):
        # International numbers should have at least 8 digits after the country code
        return len(normalized) >= 9 and normalized[1:].isdigit()
    
    # For US/Canada numbers, expect 10 digits
    return len(normalized) == 10 and normalized.isdigit()


def is_valid_url(url: Any) -> bool:
    """
    Validate if the input is a properly formatted URL.

    Validates URLs with http or https protocols.

    Args:
        url: The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    if not isinstance(url, str):
        return False

    # Check for specific invalid formats first
    if url == "":
        return False
    
    # Check for spaces
    if ' ' in url:
        return False
    
    # Check for double dots
    if '..' in url:
        return False
    
    # Check for trailing dot
    if url.endswith('.'):
        return False
    
    # URL regex pattern that validates common URL formats
    pattern = r'^(https?:\/\/)' + \
              r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|' + \
              r'(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
              r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
              r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
              r'(#[-a-zA-Z0-9%_]+)?$'
    
    # Basic pattern match
    if not re.match(pattern, url):
        return False
    
    # Check for domain part
    domain_part = url.split('://')[1].split('/')[0].split(':')[0]
    return not (domain_part.startswith('-') or domain_part.endswith('-'))
```

## File: utils/validation_utils.py,cover
```
> """
> Validation utilities for common data formats.
  
> This module provides functions to validate common data formats such as
> email addresses, phone numbers, and URLs.
> """
> import re
> from typing import Any
  
  
> def is_valid_email(email: Any) -> bool:
>     """
>     Validate if the input is a properly formatted email address.
  
>     Args:
>         email: The email address to validate.
  
>     Returns:
>         bool: True if the email is valid, False otherwise.
>     """
>     if not isinstance(email, str):
>         return False
  
      # RFC 5322 compliant email regex pattern with additional validations
>     pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
      
      # Basic pattern match
>     if not re.match(pattern, email):
>         return False
      
      # Additional validations
>     if '..' in email:  # No consecutive dots
>         return False
>     if email.endswith('.'):  # No trailing dot
!         return False
>     if ' ' in email:  # No spaces
!         return False
      
      # Check domain part
>     domain = email.split('@')[1]
>     if domain.startswith('-') or domain.endswith('-'):  # No leading/trailing hyphens in domain
!         return False
      
      # Check for hyphens at the end of domain parts
>     domain_parts = domain.split('.')
>     return all(not part.endswith('-') for part in domain_parts)
  
  
> def is_valid_phone(phone: Any) -> bool:
>     """
>     Validate if the input is a properly formatted phone number.
  
>     Accepts various formats including:
>     - 10 digits: 1234567890
>     - Hyphenated: 123-456-7890
>     - Parentheses: (123) 456-7890
>     - International: +1 123-456-7890
>     - Dots: 123.456.7890
>     - Spaces: 123 456 7890
  
>     Args:
>         phone: The phone number to validate.
  
>     Returns:
>         bool: True if the phone number is valid, False otherwise.
>     """
>     if not isinstance(phone, str):
>         return False
  
      # Check for specific invalid formats first
>     if phone == "":
>         return False
      
      # Check for spaces around hyphens
>     if " - " in phone:
>         return False
      
      # Check for missing space after parentheses in format like (123)456-7890
>     if re.search(r'\)[0-9]', phone):
>         return False
      
      # Remove all non-alphanumeric characters for normalization
>     normalized = re.sub(r'[^0-9+]', '', phone)
      
      # Check for letters in the phone number
>     if re.search(r'[a-zA-Z]', phone):
>         return False
      
      # Check for international format (starting with +)
>     if normalized.startswith('+'):
          # International numbers should have at least 8 digits after the country code
>         return len(normalized) >= 9 and normalized[1:].isdigit()
      
      # For US/Canada numbers, expect 10 digits
>     return len(normalized) == 10 and normalized.isdigit()
  
  
> def is_valid_url(url: Any) -> bool:
>     """
>     Validate if the input is a properly formatted URL.
  
>     Validates URLs with http or https protocols.
  
>     Args:
>         url: The URL to validate.
  
>     Returns:
>         bool: True if the URL is valid, False otherwise.
>     """
>     if not isinstance(url, str):
>         return False
  
      # Check for specific invalid formats first
>     if url == "":
>         return False
      
      # Check for spaces
>     if ' ' in url:
>         return False
      
      # Check for double dots
>     if '..' in url:
>         return False
      
      # Check for trailing dot
>     if url.endswith('.'):
>         return False
      
      # URL regex pattern that validates common URL formats
>     pattern = r'^(https?:\/\/)' + \
>               r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
>               r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
>               r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
>               r'(#[-a-zA-Z0-9%_]+)?$'
      
      # Basic pattern match
>     if not re.match(pattern, url):
>         return False
      
      # Check for domain part
>     domain_part = url.split('://')[1].split('/')[0].split(':')[0]
>     return not (domain_part.startswith('-') or domain_part.endswith('-'))
```
