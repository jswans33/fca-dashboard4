This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been formatted for parsing in markdown style, content has been compressed (code blocks are separated by ⋮---- delimiter).

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
- Only files matching these patterns are included: nexusml/
- Files matching these patterns are excluded: nexusml/ingest/data/**, nexusml/docs, nexusml/output/**, nexusml/core/deprecated/**, nexusml/test/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style
- Content has been compressed - code blocks are separated by ⋮---- delimiter

## Additional Info

# Directory Structure
```
nexusml/__init__.py
nexusml/config/__init__.py
nexusml/config/.repomixignore
nexusml/config/data_config.yml
nexusml/config/eav/equipment_attributes.json
nexusml/config/feature_config.yml
nexusml/config/mappings/masterformat_equipment.json
nexusml/config/mappings/masterformat_primary.json
nexusml/config/repomix.config.json
nexusml/core/__init__.py
nexusml/core/data_mapper.py
nexusml/core/data_preprocessing.py
nexusml/core/eav_manager.py
nexusml/core/evaluation.py
nexusml/core/feature_engineering.py
nexusml/core/model_building.py
nexusml/core/model.py
nexusml/examples/__init__.py
nexusml/examples/advanced_example.py
nexusml/examples/common.py
nexusml/examples/feature_engineering_example.py
nexusml/examples/integrated_classifier_example.py
nexusml/examples/omniclass_generator_example.py
nexusml/examples/omniclass_hierarchy_example.py
nexusml/examples/simple_example.py
nexusml/examples/staging_data_example.py
nexusml/ingest/__init__.py
nexusml/ingest/generator/__init__.py
nexusml/ingest/generator/omniclass_description_generator.py
nexusml/ingest/generator/omniclass_example.py
nexusml/ingest/generator/omniclass_hierarchy.py
nexusml/ingest/generator/omniclass_tree.py
nexusml/ingest/generator/omniclass.py
nexusml/ingest/generator/README.md
nexusml/pyproject.toml
nexusml/README.md
nexusml/setup.py
nexusml/tests/__init__.py
nexusml/tests/conftest.py
nexusml/tests/integration/__init__.py
nexusml/tests/integration/test_integration.py
nexusml/tests/unit/__init__.py
nexusml/tests/unit/test_generator.py
nexusml/tests/unit/test_pipeline.py
nexusml/utils/__init__.py
nexusml/utils/csv_utils.py
nexusml/utils/excel_utils.py
nexusml/utils/logging.py
nexusml/utils/verification.py
```

# Files

## File: nexusml/__init__.py
````python
"""
NexusML - Modern machine learning classification engine
"""
⋮----
__version__ = "0.1.0"
⋮----
# Import key functionality to expose at the top level
⋮----
__all__ = [
````

## File: nexusml/config/__init__.py
````python
"""
Centralized Configuration Module for NexusML

This module provides a unified approach to configuration management,
handling both standalone usage and integration with fca_dashboard.
"""
⋮----
# Default paths
DEFAULT_PATHS = {
⋮----
# Try to load from fca_dashboard if available (only once at import time)
⋮----
FCA_DASHBOARD_AVAILABLE = True
# Store the imported functions to avoid "possibly unbound" errors
FCA_GET_CONFIG_PATH = get_config_path
FCA_RESOLVE_PATH = resolve_path
⋮----
FCA_DASHBOARD_AVAILABLE = False
# Define dummy functions that will never be called
FCA_GET_CONFIG_PATH = None
FCA_RESOLVE_PATH = None
⋮----
def get_project_root() -> Path
⋮----
"""Get the project root directory."""
⋮----
def get_data_path(path_key: str = "training_data") -> Union[str, Path]
⋮----
"""
    Get a data path from config or defaults.

    Args:
        path_key: Key for the path in the configuration

    Returns:
        Resolved path as string or Path object
    """
root = get_project_root()
⋮----
# Try to load settings
settings = load_settings()
⋮----
# Check in nexusml section first, then classifier section for backward compatibility
nexusml_settings = settings.get("nexusml", {})
classifier_settings = settings.get("classifier", {})
⋮----
# Merge settings, preferring nexusml if available
merged_settings = {**classifier_settings, **nexusml_settings}
⋮----
# Get path from settings
path = merged_settings.get("data_paths", {}).get(path_key)
⋮----
# Use default path
path = os.path.join(str(root), DEFAULT_PATHS.get(path_key, ""))
⋮----
# If running in fca_dashboard context and path is not absolute, resolve it
⋮----
# Fall back to local resolution
⋮----
# If path is not absolute, make it relative to project root
⋮----
def get_output_dir() -> Union[str, Path]
⋮----
"""
    Get the output directory path.

    Returns:
        Path to the output directory as string or Path object
    """
⋮----
def load_settings() -> Dict[str, Any]
⋮----
"""
    Load settings from the configuration file.

    Returns:
        Configuration settings as a dictionary
    """
# Try to find a settings file
⋮----
settings_path = cast(Union[str, Path], FCA_GET_CONFIG_PATH("settings.yml"))
⋮----
settings_path = None
⋮----
settings_path = get_project_root() / DEFAULT_PATHS["config_file"]
⋮----
# Check environment variable as fallback
⋮----
settings_path_str = os.environ.get("NEXUSML_CONFIG", "")
settings_path = Path(settings_path_str) if settings_path_str else None
⋮----
# Return default settings
⋮----
def get_config_value(key_path: str, default: Any = None) -> Any
⋮----
"""
    Get a configuration value using a dot-separated path.

    Args:
        key_path: Dot-separated path to the config value (e.g., 'nexusml.data_paths.training_data')
        default: Default value to return if the key is not found

    Returns:
        The configuration value or the default
    """
⋮----
keys = key_path.split(".")
⋮----
# Navigate through the nested dictionary
current = settings
⋮----
current = current[key]
````

## File: nexusml/config/.repomixignore
````
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/


.csv
````

## File: nexusml/config/data_config.yml
````yaml
# Data Preprocessing Configuration

# Required columns for the model
# If these columns are missing, they will be created with default values
required_columns:
  # Source columns (from raw data)
  - name: 'Asset Category'
    default_value: ''
    data_type: 'str'
  - name: 'Equip Name ID'
    default_value: ''
    data_type: 'str'
  - name: 'System Type ID'
    default_value: ''
    data_type: 'str'
  - name: 'Precon System'
    default_value: ''
    data_type: 'str'
  - name: 'Operations System'
    default_value: ''
    data_type: 'str'
  - name: 'Sub System Type'
    default_value: ''
    data_type: 'str'
  - name: 'Sub System ID'
    default_value: ''
    data_type: 'str'
  - name: 'Sub System Class'
    default_value: ''
    data_type: 'str'
  - name: 'Title'
    default_value: ''
    data_type: 'str'
  - name: 'Drawing Abbreviation'
    default_value: ''
    data_type: 'str'
  - name: 'Equipment Size'
    default_value: 0
    data_type: 'float'
  - name: 'Unit'
    default_value: ''
    data_type: 'str'
  - name: 'Service Life'
    default_value: 0
    data_type: 'float'

  # Target columns (created during feature engineering)
  - name: 'Equipment_Category'
    default_value: ''
    data_type: 'str'
  - name: 'Uniformat_Class'
    default_value: ''
    data_type: 'str'
  - name: 'System_Type'
    default_value: ''
    data_type: 'str'
  - name: 'Equipment_Subcategory'
    default_value: ''
    data_type: 'str'
  - name: 'Subsystem_Type'
    default_value: ''
    data_type: 'str'
  - name: 'Subsystem_ID'
    default_value: ''
    data_type: 'str'
  - name: 'combined_text'
    default_value: ''
    data_type: 'str'
  - name: 'size_feature'
    default_value: ''
    data_type: 'str'
  - name: 'service_life'
    default_value: 0
    data_type: 'float'
  - name: 'equipment_size'
    default_value: 0
    data_type: 'float'
  - name: 'Equipment_Type'
    default_value: ''
    data_type: 'str'
  - name: 'System_Subtype'
    default_value: ''
    data_type: 'str'
  - name: 'Full_Classification'
    default_value: ''
    data_type: 'str'

# Training data configuration
training_data:
  default_path: 'ingest/data/eq_ids.csv'
  encoding: 'utf-8'
  fallback_encoding: 'latin1'
````

## File: nexusml/config/eav/equipment_attributes.json
````json
{
  "Chiller": {
    "omniclass_id": "23-33 11 11 11",
    "masterformat_id": "23 64 00",
    "uniformat_id": "D3020",
    "required_attributes": [
      "cooling_capacity_tons",
      "efficiency_kw_per_ton",
      "refrigerant_type",
      "chiller_type"
    ],
    "optional_attributes": [
      "condenser_type",
      "compressor_type",
      "min_part_load_ratio",
      "evaporator_flow_rate_gpm",
      "condenser_flow_rate_gpm"
    ],
    "units": {
      "cooling_capacity_tons": "tons",
      "efficiency_kw_per_ton": "kW/ton",
      "evaporator_flow_rate_gpm": "GPM",
      "condenser_flow_rate_gpm": "GPM"
    },
    "performance_fields": {
      "service_life": {
        "default": 20,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 3,
        "unit": "months"
      }
    }
  },
  "Air Handler": {
    "omniclass_id": "23-33 11 13 11",
    "masterformat_id": "23 73 00",
    "uniformat_id": "D3040",
    "required_attributes": [
      "airflow_cfm",
      "static_pressure_in_wg",
      "cooling_capacity_mbh",
      "fan_type"
    ],
    "optional_attributes": [
      "heating_capacity_mbh",
      "filter_type",
      "filter_efficiency",
      "motor_hp",
      "motor_voltage"
    ],
    "units": {
      "airflow_cfm": "CFM",
      "static_pressure_in_wg": "in WG",
      "cooling_capacity_mbh": "MBH",
      "heating_capacity_mbh": "MBH",
      "motor_hp": "HP",
      "motor_voltage": "V"
    },
    "performance_fields": {
      "service_life": {
        "default": 15,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 1,
        "unit": "months"
      }
    }
  },
  "Boiler": {
    "omniclass_id": "23-33 11 21 11",
    "masterformat_id": "23 52 00",
    "uniformat_id": "D3020",
    "required_attributes": [
      "heating_capacity_mbh",
      "efficiency_percent",
      "fuel_type",
      "boiler_type"
    ],
    "optional_attributes": [
      "pressure_rating_psig",
      "flow_rate_gpm",
      "temperature_rise_f",
      "burner_type",
      "nox_emissions"
    ],
    "units": {
      "heating_capacity_mbh": "MBH",
      "efficiency_percent": "%",
      "pressure_rating_psig": "PSIG",
      "flow_rate_gpm": "GPM",
      "temperature_rise_f": "°F"
    },
    "performance_fields": {
      "service_life": {
        "default": 25,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 6,
        "unit": "months"
      }
    }
  },
  "Pump": {
    "omniclass_id": "23-27 11 11 11",
    "masterformat_id": "22 11 23",
    "uniformat_id": "D2010",
    "required_attributes": [
      "flow_rate_gpm",
      "head_pressure_ft",
      "pump_type",
      "motor_hp"
    ],
    "optional_attributes": [
      "impeller_diameter",
      "efficiency_percent",
      "motor_speed_rpm",
      "motor_voltage",
      "motor_phase"
    ],
    "units": {
      "flow_rate_gpm": "GPM",
      "head_pressure_ft": "ft",
      "impeller_diameter": "in",
      "efficiency_percent": "%",
      "motor_speed_rpm": "RPM",
      "motor_hp": "HP",
      "motor_voltage": "V"
    },
    "performance_fields": {
      "service_life": {
        "default": 15,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 6,
        "unit": "months"
      }
    }
  },
  "Heat Exchanger": {
    "omniclass_id": "23-33 11 31 11",
    "masterformat_id": "23 57 00",
    "uniformat_id": "D3020",
    "required_attributes": [
      "heat_transfer_capacity_mbh",
      "primary_fluid",
      "secondary_fluid",
      "exchanger_type"
    ],
    "optional_attributes": [
      "primary_flow_rate_gpm",
      "secondary_flow_rate_gpm",
      "primary_pressure_drop_ft",
      "secondary_pressure_drop_ft",
      "material"
    ],
    "units": {
      "heat_transfer_capacity_mbh": "MBH",
      "primary_flow_rate_gpm": "GPM",
      "secondary_flow_rate_gpm": "GPM",
      "primary_pressure_drop_ft": "ft",
      "secondary_pressure_drop_ft": "ft"
    },
    "performance_fields": {
      "service_life": {
        "default": 20,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 12,
        "unit": "months"
      }
    }
  },
  "Cooling Tower": {
    "omniclass_id": "23-33 11 14 11",
    "masterformat_id": "23 65 00",
    "uniformat_id": "D3030",
    "required_attributes": [
      "cooling_capacity_tons",
      "flow_rate_gpm",
      "tower_type",
      "fan_type"
    ],
    "optional_attributes": [
      "fan_hp",
      "design_wet_bulb",
      "design_range",
      "design_approach",
      "basin_capacity_gal"
    ],
    "units": {
      "cooling_capacity_tons": "tons",
      "flow_rate_gpm": "GPM",
      "fan_hp": "HP",
      "design_wet_bulb": "°F",
      "design_range": "°F",
      "design_approach": "°F",
      "basin_capacity_gal": "gal"
    },
    "performance_fields": {
      "service_life": {
        "default": 15,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 1,
        "unit": "months"
      }
    }
  },
  "Fan": {
    "omniclass_id": "23-33 11 13 17",
    "masterformat_id": "23 34 00",
    "uniformat_id": "D3040",
    "required_attributes": [
      "airflow_cfm",
      "static_pressure_in_wg",
      "fan_type",
      "motor_hp"
    ],
    "optional_attributes": [
      "motor_voltage",
      "motor_phase",
      "motor_speed_rpm",
      "drive_type",
      "wheel_diameter"
    ],
    "units": {
      "airflow_cfm": "CFM",
      "static_pressure_in_wg": "in WG",
      "motor_hp": "HP",
      "motor_voltage": "V",
      "motor_speed_rpm": "RPM",
      "wheel_diameter": "in"
    },
    "performance_fields": {
      "service_life": {
        "default": 15,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 3,
        "unit": "months"
      }
    }
  },
  "VAV Box": {
    "omniclass_id": "23-33 11 13 23",
    "masterformat_id": "23 36 00",
    "uniformat_id": "D3040",
    "required_attributes": [
      "airflow_cfm",
      "heating_capacity_mbh",
      "box_type",
      "inlet_size"
    ],
    "optional_attributes": [
      "pressure_drop_in_wg",
      "control_type",
      "sound_rating",
      "reheat_coil_type",
      "damper_type"
    ],
    "units": {
      "airflow_cfm": "CFM",
      "heating_capacity_mbh": "MBH",
      "inlet_size": "in",
      "pressure_drop_in_wg": "in WG"
    },
    "performance_fields": {
      "service_life": {
        "default": 15,
        "unit": "years"
      },
      "maintenance_interval": {
        "default": 6,
        "unit": "months"
      }
    }
  }
}
````

## File: nexusml/config/feature_config.yml
````yaml
text_combinations:
  - name: 'combined_text'
    columns: [
        'Asset Name', # From staging table
        'Manufacturer', # From staging table
        'Model Number', # From staging table
        'System Category', # From staging table
        'Sub System Type', # From staging table
        'Sub System Classification', # From staging table
      ]
    separator: ' '

  - name: 'size_feature'
    columns: ['Size', 'Unit']
    separator: ' '

numeric_columns:
  - name: 'Service Life'
    new_name: 'service_life'
    fill_value: 20
    dtype: 'float'

  - name: 'Motor HP'
    new_name: 'motor_hp'
    fill_value: 0
    dtype: 'float'

hierarchies:
  - new_col: 'Equipment_Type'
    parents: ['System Category', 'Asset Name']
    separator: '-'

  - new_col: 'System_Subtype'
    parents: ['System Category', 'Sub System Type']
    separator: '-'

column_mappings:
  - source: 'Asset Name'
    target: 'Equipment_Category'

  - source: 'Trade'
    target: 'Uniformat_Class'

  - source: 'System Category'
    target: 'System_Type'

classification_systems:
  - name: 'OmniClass'
    source_column: 'Equipment_Category'
    target_column: 'OmniClass_ID'
    mapping_type: 'eav'

  - name: 'MasterFormat'
    source_columns:
      [
        'Uniformat_Class',
        'System_Type',
        'Equipment_Category',
        'Equipment_Subcategory',
      ]
    target_column: 'MasterFormat_ID'
    mapping_function: 'enhanced_masterformat_mapping'

  - name: 'Uniformat'
    source_column: 'Uniformat_Class'
    target_column: 'Uniformat_ID'
    mapping_type: 'eav'

eav_integration:
  enabled: true
  equipment_type_column: 'Equipment_Category'
  description_column: 'combined_text'
  service_life_column: 'service_life'
  add_classification_ids: true
  add_performance_fields: true
````

## File: nexusml/config/mappings/masterformat_equipment.json
````json
{
  "Heat Exchanger": "23 57 00",
  "Water Softener": "22 31 00",
  "Humidifier": "23 84 13",
  "Radiant Panel": "23 83 16",
  "Make-up Air Unit": "23 74 23",
  "Energy Recovery Ventilator": "23 72 00",
  "DI/RO Equipment": "22 31 16",
  "Bypass Filter Feeder": "23 25 00",
  "Grease Interceptor": "22 13 23",
  "Heat Trace": "23 05 33",
  "Dust Collector": "23 35 16",
  "Venturi VAV Box": "23 36 00",
  "Water Treatment Controller": "23 25 13",
  "Polishing System": "23 25 00",
  "Ozone Generator": "22 67 00"
}
````

## File: nexusml/config/mappings/masterformat_primary.json
````json
{
  "H": {
    "Chiller Plant": "23 64 00",
    "Cooling Tower Plant": "23 65 00",
    "Heating Water Boiler Plant": "23 52 00",
    "Steam Boiler Plant": "23 52 33",
    "Air Handling Units": "23 73 00"
  },
  "P": {
    "Domestic Water Plant": "22 11 00",
    "Medical/Lab Gas Plant": "22 63 00",
    "Sanitary Equipment": "22 13 00"
  },
  "SM": {
    "Air Handling Units": "23 74 00",
    "SM Accessories": "23 33 00",
    "SM Equipment": "23 30 00"
  }
}
````

## File: nexusml/config/repomix.config.json
````json
{
  "output": {
    "filePath": "nexusml-repomix-output.md",
    "style": "markdown",
    "parsableStyle": true,
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": true,
    "topFilesLength": 5,
    "showLineNumbers": false,
    "copyToClipboard": true
  },
  "include": ["nexusml/"],
  "ignore": {
    "useGitignore": true,
    "useDefaultPatterns": true,
    "customPatterns": [
      "nexusml/ingest/data/**",
      "nexusml/docs",
      "nexusml/output/**",
      "nexusml/core/deprecated/**",
      "nexusml/test/**"
    ]
  },
  "security": {
    "enableSecurityCheck": true
  },
  "tokenCount": {
    "encoding": "o200k_base"
  }
}
````

## File: nexusml/core/__init__.py
````python
"""
Core functionality for NexusML classification engine.
"""
⋮----
# Import main functions to expose at the package level
⋮----
__all__ = [
````

## File: nexusml/core/data_mapper.py
````python
"""
Data Mapper Module

This module handles mapping between staging data and the ML model input format.
"""
⋮----
class DataMapper
⋮----
"""
    Maps data between different formats, specifically from staging data to ML model input
    and from ML model output to master database fields.
    """
⋮----
def __init__(self, column_mapping: Optional[Dict[str, str]] = None)
⋮----
"""
        Initialize the data mapper with an optional column mapping.

        Args:
            column_mapping: Dictionary mapping staging columns to model input columns
        """
# Default mapping from staging columns to model input columns
⋮----
# Required fields with default values
⋮----
"Trade": "H",  # Default to HVAC
⋮----
# Numeric fields with default values
⋮----
def map_staging_to_model_input(self, staging_df: pd.DataFrame) -> pd.DataFrame
⋮----
"""
        Maps staging data columns to the format expected by the ML model.

        Args:
            staging_df: DataFrame from staging table

        Returns:
            DataFrame with columns mapped to what the ML model expects
        """
# Create a new DataFrame for the model input
model_df = pd.DataFrame()
⋮----
# Map columns
⋮----
# Use empty values for missing columns
⋮----
# Fill required fields with defaults if missing
⋮----
# Handle numeric fields - convert to proper numeric values
⋮----
# Convert to numeric, coercing errors to NaN
⋮----
# Fill NaN values with default
⋮----
# Create the field with default value if missing
⋮----
# Create required columns for the ML model
⋮----
"""
        Maps model predictions to master database fields.

        Args:
            predictions: Dictionary of predictions from the ML model

        Returns:
            Dictionary with fields mapped to master DB structure
        """
# Map to Equipment and Equipment_Categories tables
equipment_data = {
⋮----
"EquipmentTag": predictions.get("Asset Tag", ""),  # Required NOT NULL field
⋮----
# Add classification IDs from EAV
⋮----
# Map CategoryID (foreign key to Equipment_Categories)
⋮----
# Default LocationID if not provided
equipment_data["LocationID"] = 1  # Default location ID
⋮----
def _map_to_category_id(self, equipment_category: str) -> int
⋮----
"""
        Maps an equipment category name to a CategoryID for the master database.
        In a real implementation, this would query the Equipment_Categories table.

        Args:
            equipment_category: The equipment category name

        Returns:
            CategoryID as an integer
        """
# This is a placeholder. In a real implementation, this would query
# the Equipment_Categories table or use a mapping dictionary.
# For now, we'll use a simple hash function to generate a positive integer
category_hash = hash(equipment_category) % 10000
return abs(category_hash) + 1  # Ensure positive and non-zero
⋮----
def map_staging_to_model_input(staging_df: pd.DataFrame) -> pd.DataFrame
⋮----
"""
    Maps staging data columns to the format expected by the ML model.

    Args:
        staging_df: DataFrame from staging table

    Returns:
        DataFrame with columns mapped to what the ML model expects
    """
mapper = DataMapper()
⋮----
def map_predictions_to_master_db(predictions: Dict[str, Any]) -> Dict[str, Any]
⋮----
"""
    Maps model predictions to master database fields.

    Args:
        predictions: Dictionary of predictions from the ML model

    Returns:
        Dictionary with fields mapped to master DB structure
    """
````

## File: nexusml/core/data_preprocessing.py
````python
"""
Data Preprocessing Module

This module handles loading and preprocessing data for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on data loading and cleaning.
"""
⋮----
def load_data_config() -> Dict
⋮----
"""
    Load the data preprocessing configuration from YAML file.

    Returns:
        Dict: Configuration dictionary
    """
⋮----
# Get the path to the configuration file
root = Path(__file__).resolve().parent.parent
config_path = root / "config" / "data_config.yml"
⋮----
# Load the configuration
⋮----
config = yaml.safe_load(f)
⋮----
# Return a minimal default configuration
⋮----
"""
    Verify that all required columns exist in the DataFrame and create them if they don't.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (Dict, optional): Configuration dictionary. If None, loads from file.

    Returns:
        pd.DataFrame: DataFrame with all required columns
    """
⋮----
config = load_data_config()
⋮----
required_columns = config.get("required_columns", [])
⋮----
# Create a copy of the DataFrame to avoid modifying the original
df = df.copy()
⋮----
# Check each required column
⋮----
column_name = column_info["name"]
default_value = column_info["default_value"]
data_type = column_info["data_type"]
⋮----
# Check if the column exists
⋮----
# Create the column with the default value
⋮----
# Default to string if type is unknown
⋮----
def load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame
⋮----
"""
    Load and preprocess data from a CSV file

    Args:
        data_path (str, optional): Path to the CSV file. Defaults to the standard location.

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
# Load the configuration
⋮----
training_data_config = config.get("training_data", {})
⋮----
# Use default path if none provided
⋮----
# Try to load from settings if available
⋮----
# Check if we're running within the fca_dashboard context
⋮----
settings_path = get_config_path("settings.yml")
⋮----
settings = yaml.safe_load(file)
⋮----
data_path = (
⋮----
# Resolve the path to ensure it exists
data_path = str(resolve_path(data_path))
⋮----
# Not running in fca_dashboard context
data_path = None
⋮----
# If still no data_path, use the default in nexusml
⋮----
# Use the default path from config
default_path = training_data_config.get(
data_path = str(Path(__file__).resolve().parent.parent / default_path)
⋮----
# Use default path from config as fallback
⋮----
# Read CSV file using pandas
encoding = training_data_config.get("encoding", "utf-8")
fallback_encoding = training_data_config.get("fallback_encoding", "latin1")
⋮----
df = pd.read_csv(data_path, encoding=encoding)
⋮----
# Try with a different encoding if the primary one fails
⋮----
df = pd.read_csv(data_path, encoding=fallback_encoding)
⋮----
# Clean up column names (remove any leading/trailing whitespace)
⋮----
# Fill NaN values with empty strings for text columns
⋮----
# Verify and create required columns
df = verify_required_columns(df, config)
````

## File: nexusml/core/eav_manager.py
````python
"""
Equipment Attribute-Value (EAV) Manager

This module manages the EAV structure for equipment attributes, providing functionality to:
1. Load attribute templates for different equipment types
2. Validate equipment attributes against templates
3. Generate attribute templates for equipment based on ML predictions
4. Fill in missing attributes using ML predictions and rules
"""
⋮----
class EAVManager
⋮----
"""
    Manages the Entity-Attribute-Value (EAV) structure for equipment attributes.
    """
⋮----
def __init__(self, templates_path: Optional[str] = None)
⋮----
"""
        Initialize the EAV Manager with templates.

        Args:
            templates_path: Path to the JSON file containing attribute templates.
                           If None, uses the default path.
        """
⋮----
def load_templates(self) -> None
⋮----
"""Load attribute templates from the JSON file."""
templates_path = self.templates_path
⋮----
# Use default path
root = get_project_root()
templates_path = root / "config" / "eav" / "equipment_attributes.json"
⋮----
def get_equipment_template(self, equipment_type: str) -> Dict[str, Any]
⋮----
"""
        Get the attribute template for a specific equipment type.

        Args:
            equipment_type: The type of equipment (e.g., "Chiller", "Air Handler")

        Returns:
            Dict containing the attribute template, or an empty dict if not found
        """
# Try exact match first
⋮----
# Try case-insensitive match
⋮----
# Try partial match (e.g., "Centrifugal Chiller" should match "Chiller")
⋮----
# Return empty template if no match found
⋮----
def get_required_attributes(self, equipment_type: str) -> List[str]
⋮----
"""
        Get required attributes for a given equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            List of required attribute names
        """
template = self.get_equipment_template(equipment_type)
⋮----
def get_optional_attributes(self, equipment_type: str) -> List[str]
⋮----
"""
        Get optional attributes for a given equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            List of optional attribute names
        """
⋮----
def get_all_attributes(self, equipment_type: str) -> List[str]
⋮----
"""
        Get all attributes (required and optional) for a given equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            List of all attribute names
        """
⋮----
required = template.get("required_attributes", [])
optional = template.get("optional_attributes", [])
⋮----
def get_attribute_unit(self, equipment_type: str, attribute: str) -> str
⋮----
"""
        Get the unit for a specific attribute of an equipment type.

        Args:
            equipment_type: The type of equipment
            attribute: The attribute name

        Returns:
            Unit string, or empty string if not found
        """
⋮----
units = template.get("units", {})
⋮----
def get_classification_ids(self, equipment_type: str) -> Dict[str, str]
⋮----
"""
        Get the classification IDs (OmniClass, MasterFormat, Uniformat) for an equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            Dictionary with classification IDs
        """
⋮----
def get_performance_fields(self, equipment_type: str) -> Dict[str, Dict[str, Any]]
⋮----
"""
        Get the performance fields for an equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            Dictionary with performance fields
        """
⋮----
"""
        Validate attributes against the template for an equipment type.

        Args:
            equipment_type: The type of equipment
            attributes: Dictionary of attribute name-value pairs

        Returns:
            Dictionary with validation results:
            {
                "missing_required": List of missing required attributes,
                "unknown": List of attributes not in the template
            }
        """
⋮----
required = set(template.get("required_attributes", []))
optional = set(template.get("optional_attributes", []))
all_valid = required.union(optional)
⋮----
# Check for missing required attributes
provided = set(attributes.keys())
missing_required = required - provided
⋮----
# Check for unknown attributes
unknown = provided - all_valid
⋮----
def generate_attribute_template(self, equipment_type: str) -> Dict[str, Any]
⋮----
"""
        Generate an attribute template for an equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            Dictionary with attribute template
        """
⋮----
result = {
⋮----
# Add required attributes with units
⋮----
unit = self.get_attribute_unit(equipment_type, attr)
⋮----
# Add optional attributes with units
⋮----
"""
        Fill in missing attributes using ML predictions and rules.

        Args:
            equipment_type: The type of equipment
            attributes: Dictionary of existing attribute name-value pairs
            description: Text description of the equipment
            model: Optional ML model to use for predictions

        Returns:
            Dictionary with filled attributes
        """
result = attributes.copy()
⋮----
# Get all attributes that should be present
all_attrs = self.get_all_attributes(equipment_type)
⋮----
# Identify missing attributes
missing_attrs = [
⋮----
return result  # No missing attributes to fill
⋮----
# Fill in performance fields from template defaults
perf_fields = self.get_performance_fields(equipment_type)
⋮----
# If we have a model, use it to predict missing attributes
⋮----
predictions = model.predict_attributes(equipment_type, description)
⋮----
class EAVTransformer(BaseEstimator, TransformerMixin)
⋮----
"""
    Transformer that adds EAV attributes to the feature set.
    """
⋮----
def __init__(self, eav_manager: Optional[EAVManager] = None)
⋮----
"""
        Initialize the EAV Transformer.

        Args:
            eav_manager: EAVManager instance. If None, creates a new one.
        """
⋮----
def fit(self, X, y=None)
⋮----
"""Fit method (does nothing but is required for the transformer interface)."""
⋮----
def transform(self, X)
⋮----
"""
        Transform the input DataFrame by adding EAV attributes.

        Args:
            X: Input DataFrame with at least 'Equipment_Category' column

        Returns:
            Transformed DataFrame with EAV attributes
        """
X = X.copy()
⋮----
# Check if Equipment_Category column exists
⋮----
# Add empty columns for EAV attributes
⋮----
# Add classification IDs
⋮----
# Add performance fields
⋮----
# Create a feature indicating how many required attributes are typically needed
⋮----
def get_eav_manager() -> EAVManager
⋮----
"""
    Get an instance of the EAVManager.

    Returns:
        EAVManager instance
    """
````

## File: nexusml/core/evaluation.py
````python
"""
Evaluation Module

This module handles model evaluation and analysis of "Other" categories.
It follows the Single Responsibility Principle by focusing solely on model evaluation.
"""
⋮----
def enhanced_evaluation(model: Pipeline, X_test: Union[pd.Series, pd.DataFrame], y_test: pd.DataFrame) -> pd.DataFrame
⋮----
"""
    Evaluate the model with focus on "Other" categories performance
    
    This function has been updated to handle both Series and DataFrame inputs for X_test,
    to support the new pipeline structure that uses both text and numeric features.
    
    Args:
        model (Pipeline): Trained model pipeline
        X_test: Test features
        y_test (pd.DataFrame): Test targets
        
    Returns:
        pd.DataFrame: Predictions dataframe
    """
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
⋮----
# Print overall evaluation metrics
⋮----
# Specifically examine "Other" category performance
⋮----
other_indices = y_test[col] == "Other"
other_accuracy = accuracy_score(
⋮----
# Calculate confusion metrics for "Other" category
tp = ((y_test[col] == "Other") & (y_pred_df[col] == "Other")).sum()
fp = ((y_test[col] != "Other") & (y_pred_df[col] == "Other")).sum()
fn = ((y_test[col] == "Other") & (y_pred_df[col] != "Other")).sum()
⋮----
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
⋮----
def analyze_other_category_features(model: Pipeline, X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None
⋮----
"""
    Analyze what features are most important for classifying items as "Other"
    
    This function has been updated to work with the new pipeline structure that uses
    a ColumnTransformer to combine text and numeric features.
    
    Args:
        model (Pipeline): Trained model pipeline
        X_test (pd.Series): Test features
        y_test (pd.DataFrame): Test targets
        y_pred_df (pd.DataFrame): Predictions dataframe
    """
# Extract the Random Forest model from the pipeline
rf_model = model.named_steps['clf'].estimators_[0]
⋮----
# Get feature names from the TF-IDF vectorizer (now nested in preprocessor)
# Access the text transformer from the ColumnTransformer, then the TF-IDF vectorizer
tfidf_vectorizer = model.named_steps['preprocessor'].transformers_[0][1].named_steps['tfidf']
text_feature_names = tfidf_vectorizer.get_feature_names_out()
⋮----
# Also include numeric features for a complete analysis
numeric_feature_names = ['service_life']
all_feature_names = list(text_feature_names) + numeric_feature_names
⋮----
# For each classification column
⋮----
# Find examples predicted as "Other"
other_indices = y_pred_df[col] == "Other"
⋮----
# Create a DataFrame with the required structure for the preprocessor
⋮----
X_test_df = pd.DataFrame({
⋮----
'service_life': np.zeros(other_indices.sum())  # Placeholder
⋮----
# Transform using the full preprocessor
transformed_features = model.named_steps['preprocessor'].transform(X_test_df)
⋮----
# Extract just the text features (first part of the transformed features)
text_feature_count = len(text_feature_names)
text_features = transformed_features[:, :text_feature_count]
⋮----
# Get the average feature values for text features
avg_features = text_features.mean(axis=0)
if hasattr(avg_features, 'A1'):  # If it's a sparse matrix
avg_features = avg_features.A1
⋮----
# Get the top text features
top_indices = np.argsort(avg_features)[-20:]
⋮----
# Also analyze feature importance from the Random Forest model
# This will show the importance of both text and numeric features
⋮----
# Get feature importances for this specific estimator (for the current target column)
# Find the index of the current column in the target columns
col_idx = list(y_test.columns).index(col)
rf_estimator = model.named_steps['clf'].estimators_[col_idx]
⋮----
# Get feature importances
importances = rf_estimator.feature_importances_
⋮----
# Create a DataFrame to sort and display importances
importance_df = pd.DataFrame({
⋮----
# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)
⋮----
# Display top 10 features
⋮----
# Check if service_life is important
service_life_importance = importance_df[importance_df['feature'] == 'service_life']
⋮----
def analyze_other_misclassifications(X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None
⋮----
"""
    Analyze cases where "Other" was incorrectly predicted or missed
    
    Args:
        X_test (pd.Series): Test features
        y_test (pd.DataFrame): Test targets
        y_pred_df (pd.DataFrame): Predictions dataframe
    """
⋮----
# False positives: Predicted as "Other" but actually something else
fp_indices = (y_test[col] != "Other") & (y_pred_df[col] == "Other")
⋮----
fp_examples = X_test[fp_indices].values[:5]  # Show first 5
fp_actual = y_test[col][fp_indices].values[:5]
⋮----
print(f"  Text: {example[:100]}...")  # Show first 100 chars
⋮----
# False negatives: Actually "Other" but predicted as something else
fn_indices = (y_test[col] == "Other") & (y_pred_df[col] != "Other")
⋮----
fn_examples = X_test[fn_indices].values[:5]  # Show first 5
fn_predicted = y_pred_df[col][fn_indices].values[:5]
````

## File: nexusml/core/feature_engineering.py
````python
"""
Feature Engineering Module

This module handles feature engineering for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on feature transformations.
"""
⋮----
class TextCombiner(BaseEstimator, TransformerMixin)
⋮----
"""
    Combines multiple text columns into one column.

    Config example: {"columns": ["Asset Category","Equip Name ID"], "separator": " "}
    """
⋮----
def fit(self, X, y=None)
⋮----
def transform(self, X)
⋮----
X = X.copy()
# Check if all columns exist
missing_columns = [col for col in self.columns if col not in X.columns]
⋮----
available_columns = [col for col in self.columns if col in X.columns]
⋮----
# Create a single text column from available columns
⋮----
# Create a single text column from all specified columns
⋮----
class NumericCleaner(BaseEstimator, TransformerMixin)
⋮----
"""
    Cleans and transforms numeric columns.

    Config example: {"name": "Service Life", "new_name": "service_life", "fill_value": 0, "dtype": "float"}
    """
⋮----
# Check if the column exists
⋮----
# Clean and transform the numeric column
⋮----
class HierarchyBuilder(BaseEstimator, TransformerMixin)
⋮----
"""
    Creates hierarchical category columns by combining parent columns.

    Config example: {"new_col": "Equipment_Type", "parents": ["Asset Category", "Equip Name ID"], "separator": "-"}
    """
⋮----
# Check if all parent columns exist
missing_columns = [col for col in self.parent_columns if col not in X.columns]
⋮----
available_columns = [col for col in self.parent_columns if col in X.columns]
⋮----
# Create hierarchical column from available parent columns
⋮----
# Create hierarchical column from all parent columns
⋮----
class ColumnMapper(BaseEstimator, TransformerMixin)
⋮----
"""
    Maps source columns to target columns.

    Config example: {"source": "Asset Category", "target": "Equipment_Category"}
    """
⋮----
def __init__(self, mappings: List[Dict[str, str]])
⋮----
# Map source columns to target columns
⋮----
source = mapping["source"]
target = mapping["target"]
⋮----
class ClassificationSystemMapper(BaseEstimator, TransformerMixin)
⋮----
"""
    Maps equipment categories to classification system IDs (OmniClass, MasterFormat, Uniformat).

    Config example: {
        "name": "OmniClass",
        "source_column": "Equipment_Category",
        "target_column": "OmniClass_ID",
        "mapping_type": "eav"
    }
    """
⋮----
# Handle different mapping types
⋮----
# Use EAV manager to get classification IDs
⋮----
# Check if the source column exists
⋮----
# Single source column
⋮----
# Multiple source columns not supported for EAV mapping
⋮----
# Use the enhanced_masterformat_mapping function
⋮----
# Extract the required columns
uniformat_col = self.source_column[0]
system_type_col = self.source_column[1]
equipment_category_col = self.source_column[2]
equipment_subcategory_col = (
⋮----
# Check if all required columns exist
missing_columns = [
⋮----
# Apply the mapping function
⋮----
class GenericFeatureEngineer(BaseEstimator, TransformerMixin)
⋮----
"""
    A generic feature engineering transformer that applies multiple transformations
    based on a configuration file.
    """
⋮----
"""
        Initialize the transformer with a configuration file path.

        Args:
            config_path: Path to the YAML configuration file. If None, uses the default path.
            eav_manager: EAVManager instance. If None, creates a new one.
        """
⋮----
def _load_config(self)
⋮----
"""Load the configuration from the YAML file."""
config_path = self.config_path
⋮----
# Use default path
root = get_project_root()
config_path = root / "config" / "feature_config.yml"
⋮----
# Load the configuration
⋮----
"""
        Transform the input DataFrame based on the configuration.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
⋮----
# 1. Apply column mappings
⋮----
mapper = ColumnMapper(self.config["column_mappings"])
X = mapper.transform(X)
⋮----
# 2. Apply text combinations
⋮----
combiner = TextCombiner(
X = combiner.transform(X)
⋮----
# 3. Apply numeric column cleaning
⋮----
cleaner = NumericCleaner(
X = cleaner.transform(X)
⋮----
# 4. Apply hierarchical categories
⋮----
builder = HierarchyBuilder(
X = builder.transform(X)
⋮----
# 5. Apply classification system mappings
⋮----
mapper = ClassificationSystemMapper(
⋮----
# 6. Apply EAV integration if enabled
⋮----
eav_config = self.config["eav_integration"]
eav_transformer = EAVTransformer(eav_manager=self.eav_manager)
X = eav_transformer.transform(X)
⋮----
def enhance_features(df: pd.DataFrame) -> pd.DataFrame
⋮----
"""
    Enhanced feature engineering with hierarchical structure and more granular categories

    This function now uses the GenericFeatureEngineer transformer to apply
    transformations based on the configuration file.

    Args:
        df (pd.DataFrame): Input dataframe with raw features

    Returns:
        pd.DataFrame: DataFrame with enhanced features
    """
# Use the GenericFeatureEngineer to apply transformations
engineer = GenericFeatureEngineer()
⋮----
def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame
⋮----
"""
    Create hierarchical category structure to better handle "Other" categories

    This function is kept for backward compatibility but now simply returns the
    input DataFrame as the hierarchical categories are created by the GenericFeatureEngineer.

    Args:
        df (pd.DataFrame): Input dataframe with basic features

    Returns:
        pd.DataFrame: DataFrame with hierarchical category features
    """
# This function is kept for backward compatibility
# The hierarchical categories are now created by the GenericFeatureEngineer
⋮----
def load_masterformat_mappings() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]
⋮----
"""
    Load MasterFormat mappings from JSON files.

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: Primary and equipment-specific mappings
    """
⋮----
primary_mapping = json.load(f)
⋮----
equipment_specific_mapping = json.load(f)
⋮----
# Return empty mappings if files cannot be loaded
⋮----
"""
    Enhanced mapping with better handling of specialty equipment types

    Args:
        uniformat_class (str): Uniformat classification
        system_type (str): System type
        equipment_category (str): Equipment category
        equipment_subcategory (Optional[str]): Equipment subcategory

    Returns:
        str: MasterFormat classification code
    """
# Load mappings from JSON files
⋮----
# Try equipment-specific mapping first
⋮----
# Then try primary mapping
⋮----
# Try EAV-based mapping
⋮----
eav_manager = EAVManager()
masterformat_id = eav_manager.get_classification_ids(equipment_category).get(
⋮----
# Refined fallback mappings by Uniformat class
fallbacks = {
⋮----
"H": "23 00 00",  # Heating, Ventilating, and Air Conditioning (HVAC)
"P": "22 00 00",  # Plumbing
"SM": "23 00 00",  # HVAC
"R": "11 40 00",  # Foodservice Equipment (Refrigeration)
⋮----
return fallbacks.get(uniformat_class, "00 00 00")  # Return unknown if no match
````

## File: nexusml/core/model_building.py
````python
"""
Model Building Module

This module defines the core model architecture for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on model definition.
"""
⋮----
"""
    Build an enhanced model with configurable sampling strategy

    This model incorporates both text features (via TF-IDF) and numeric features
    (like service_life) using a ColumnTransformer to create a more comprehensive
    feature representation. It now includes a GenericFeatureEngineer step for
    more flexible feature engineering.

    Args:
        sampling_strategy: Sampling strategy to use ("direct" is the only supported option for now)
        feature_config_path: Path to the feature configuration file. If None, uses the default path.
        **kwargs: Additional parameters for the model

    Returns:
        Pipeline: Scikit-learn pipeline with feature engineering, preprocessor and classifier
    """
# Try to load settings from configuration file
⋮----
# First try to load from fca_dashboard if available
⋮----
settings_path = get_config_path("settings.yml")
⋮----
# If not running in fca_dashboard context, look for settings in nexusml
settings_path = (
⋮----
# Fallback to environment variable
settings_path_str = os.environ.get("NEXUSML_CONFIG", "")
⋮----
settings = yaml.safe_load(file)
⋮----
# Get TF-IDF settings
tfidf_settings = settings.get("classifier", {}).get("tfidf", {})
max_features = tfidf_settings.get("max_features", 5000)
ngram_range = tuple(tfidf_settings.get("ngram_range", [1, 3]))
min_df = tfidf_settings.get("min_df", 2)
max_df = tfidf_settings.get("max_df", 0.9)
use_idf = tfidf_settings.get("use_idf", True)
sublinear_tf = tfidf_settings.get("sublinear_tf", True)
⋮----
# Get Random Forest settings
rf_settings = (
n_estimators = rf_settings.get("n_estimators", 200)
max_depth = rf_settings.get("max_depth", None)
min_samples_split = rf_settings.get("min_samples_split", 2)
min_samples_leaf = rf_settings.get("min_samples_leaf", 1)
class_weight = rf_settings.get("class_weight", "balanced_subsample")
random_state = rf_settings.get("random_state", 42)
⋮----
# Use default values if settings cannot be loaded
max_features = 5000
ngram_range = (1, 3)
min_df = 2
max_df = 0.9
use_idf = True
sublinear_tf = True
⋮----
n_estimators = 200
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
class_weight = "balanced_subsample"
random_state = 42
⋮----
# Text feature processing
text_features = Pipeline(
⋮----
ngram_range=ngram_range,  # Include more n-grams for better feature extraction
min_df=min_df,  # Ignore very rare terms
max_df=max_df,  # Ignore very common terms
⋮----
sublinear_tf=sublinear_tf,  # Apply sublinear scaling to term frequencies
⋮----
# Numeric feature processing - simplified to just use StandardScaler
# The ColumnTransformer handles column selection
numeric_features = Pipeline(
⋮----
[("scaler", StandardScaler())]  # Scale numeric features
⋮----
# Combine text and numeric features
# Use a list for numeric features to ensure it's treated as a column name, not a Series
preprocessor = ColumnTransformer(
⋮----
),  # Use a list to specify column
⋮----
remainder="drop",  # Drop any other columns
⋮----
# Complete pipeline with feature engineering, feature processing and classifier
# Use class_weight='balanced_subsample' for handling imbalanced classes
pipeline = Pipeline(
⋮----
# Optional feature engineering step - only used if called directly, not through train_enhanced_model
# In train_enhanced_model, this is applied separately before the pipeline
⋮----
n_estimators=n_estimators,  # More trees for better generalization
max_depth=max_depth,  # Allow trees to grow deeply
min_samples_split=min_samples_split,  # Default value
min_samples_leaf=min_samples_leaf,  # Default value
class_weight=class_weight,  # Protection against imbalance
⋮----
def optimize_hyperparameters(pipeline: Pipeline, x_train, y_train) -> Pipeline
⋮----
"""
    Optimize hyperparameters for better handling of all classes including "Other"

    This function uses GridSearchCV to find the best hyperparameters for the model.
    It optimizes both the text processing parameters and the classifier parameters.
    The scoring metric has been changed to f1_macro to better handle imbalanced classes.

    Args:
        pipeline (Pipeline): Model pipeline to optimize
        x_train: Training features
        y_train: Training targets

    Returns:
        Pipeline: Optimized pipeline
    """
⋮----
# Param grid for optimization with updated paths for the new pipeline structure
param_grid = {
⋮----
# Use GridSearchCV for hyperparameter optimization
# Changed scoring from 'accuracy' to 'f1_macro' for better handling of imbalanced classes
grid_search = GridSearchCV(
⋮----
scoring="f1_macro",  # Better for imbalanced classes than accuracy
⋮----
# Fit the grid search to the data
# Note: x_train must now be a DataFrame with both 'combined_features' and 'service_life' columns
````

## File: nexusml/core/model.py
````python
"""
Enhanced Equipment Classification Model with EAV Integration

This module implements a machine learning pipeline for classifying equipment based on text descriptions
and numeric features, with integrated EAV (Entity-Attribute-Value) structure. Key features include:

1. Combined Text and Numeric Features:
   - Uses a ColumnTransformer to incorporate both text features (via TF-IDF) and numeric features
     (like service_life) into a single model.

2. Improved Handling of Imbalanced Classes:
   - Uses class_weight='balanced_subsample' in the RandomForestClassifier for handling imbalanced classes.

3. Better Evaluation Metrics:
   - Uses f1_macro scoring for hyperparameter optimization, which is more appropriate for
     imbalanced classes than accuracy.
   - Provides detailed analysis of "Other" category performance.

4. Feature Importance Analysis:
   - Analyzes the importance of both text and numeric features in classifying equipment.

5. EAV Integration:
   - Incorporates EAV structure for flexible equipment attributes
   - Uses classification systems (OmniClass, MasterFormat, Uniformat) for comprehensive taxonomy
   - Includes performance fields (service life, maintenance intervals) in feature engineering
   - Can predict missing attribute values based on equipment descriptions
"""
⋮----
# Standard library imports
⋮----
# Third-party imports
⋮----
# Local imports
⋮----
class EquipmentClassifier
⋮----
"""
    Comprehensive equipment classifier with EAV integration.
    """
⋮----
"""
        Initialize the equipment classifier.

        Args:
            model: Trained ML model (if None, needs to be trained)
            feature_engineer: Feature engineering transformer
            eav_manager: EAV manager for attribute templates
            sampling_strategy: Strategy for handling class imbalance
        """
⋮----
"""
        Train the equipment classifier.

        Args:
            data_path: Path to the training data
            feature_config_path: Path to the feature configuration
            **kwargs: Additional parameters for training
        """
# Train the model using the train_enhanced_model function
⋮----
"""
        Predict equipment classifications from a description.

        Args:
            description: Text description of the equipment
            service_life: Service life value (optional)
            asset_tag: Asset tag for equipment (optional)

        Returns:
            Dictionary with classification results and master DB mappings
        """
⋮----
# Use the predict_with_enhanced_model function
result = predict_with_enhanced_model(
⋮----
# Add EAV template for the predicted equipment type
equipment_type = result["Equipment_Category"]
⋮----
"""
        Predict attribute values for a given equipment type and description.

        Args:
            equipment_type: Type of equipment
            description: Text description of the equipment

        Returns:
            Dictionary with predicted attribute values
        """
# This is a placeholder for attribute prediction
# In a real implementation, this would use ML to predict attribute values
# based on the description and equipment type
template = self.eav_manager.get_equipment_template(equipment_type)
required_attrs = template.get("required_attributes", [])
⋮----
# Simple rule-based attribute prediction based on keywords in description
predictions = {}
⋮----
# Extract numeric values with units from description
⋮----
# Look for patterns like "100 tons", "5 HP", etc.
capacity_pattern = r"(\d+(?:\.\d+)?)\s*(?:ton|tons)"
flow_pattern = r"(\d+(?:\.\d+)?)\s*(?:gpm|GPM)"
pressure_pattern = r"(\d+(?:\.\d+)?)\s*(?:psi|PSI|psig|PSIG)"
temp_pattern = r"(\d+(?:\.\d+)?)\s*(?:°F|F|deg F)"
airflow_pattern = r"(\d+(?:\.\d+)?)\s*(?:cfm|CFM)"
⋮----
# Check for cooling capacity
⋮----
match = re.search(capacity_pattern, description)
⋮----
# Check for flow rate
⋮----
match = re.search(flow_pattern, description)
⋮----
# Check for pressure
pressure_attrs = [attr for attr in required_attrs if "pressure" in attr]
⋮----
match = re.search(pressure_pattern, description)
⋮----
# Check for temperature
temp_attrs = [attr for attr in required_attrs if "temp" in attr]
⋮----
match = re.search(temp_pattern, description)
⋮----
# Check for airflow
⋮----
match = re.search(airflow_pattern, description)
⋮----
# Check for equipment types
⋮----
# Add more attribute predictions as needed
⋮----
"""
        Fill in missing attributes using ML predictions and rules.

        Args:
            equipment_type: Type of equipment
            attributes: Dictionary of existing attribute name-value pairs
            description: Text description of the equipment

        Returns:
            Dictionary with filled attributes
        """
⋮----
"""
        Validate attributes against the template for an equipment type.

        Args:
            equipment_type: Type of equipment
            attributes: Dictionary of attribute name-value pairs

        Returns:
            Dictionary with validation results
        """
⋮----
"""
    Train and evaluate the enhanced model with EAV integration

    Args:
        data_path: Path to the CSV file. Defaults to None, which uses the standard location.
        sampling_strategy: Strategy for handling class imbalance ("direct" is the only supported option for now)
        feature_config_path: Path to the feature configuration file. Defaults to None, which uses the standard location.
        eav_manager: EAVManager instance. If None, creates a new one.
        **kwargs: Additional parameters for the model

    Returns:
        tuple: (trained model, preprocessed dataframe)
    """
# 1. Load and preprocess data
⋮----
df = load_and_preprocess_data(data_path)
⋮----
# 1.5. Map staging data columns to model input format
⋮----
df = map_staging_to_model_input(df)
⋮----
# 2. Apply Generic Feature Engineering with EAV integration
⋮----
eav_manager = eav_manager or EAVManager()
feature_engineer = GenericFeatureEngineer(
df = feature_engineer.transform(df)
⋮----
# 3. Prepare training data - now including both text and numeric features
# Create a DataFrame with both text and numeric features
x = pd.DataFrame(
⋮----
"combined_features": df["combined_text"],  # Using the name from config
⋮----
# Use hierarchical classification targets
y = df[
⋮----
# 4. Split the data
⋮----
# 5. Print class distribution information
⋮----
# 6. Build enhanced model with class_weight='balanced_subsample'
⋮----
model = build_enhanced_model(sampling_strategy=sampling_strategy, **kwargs)
⋮----
# 7. Train the model
⋮----
# 8. Evaluate with focus on "Other" categories
⋮----
y_pred_df = enhanced_evaluation(model, x_test, y_test)
⋮----
# 9. Analyze "Other" category features
⋮----
# 10. Analyze misclassifications for "Other" categories
⋮----
"""
    Make predictions with enhanced detail for "Other" categories

    This function has been updated to work with the new pipeline structure that uses
    both text and numeric features.

    Args:
        model: Trained model pipeline
        description (str): Text description to classify
        service_life (float, optional): Service life value. Defaults to 0.0.
        asset_tag (str, optional): Asset tag for equipment. Defaults to "".

    Returns:
        dict: Prediction results with classifications and master DB mappings
    """
# Create a DataFrame with the required structure for the pipeline
input_data = pd.DataFrame(
⋮----
# Predict using the trained pipeline
pred = model.predict(input_data)[0]
⋮----
# Extract predictions
result = {
⋮----
"Asset Tag": asset_tag,  # Add asset tag for master DB mapping
⋮----
# Add MasterFormat prediction with enhanced mapping
⋮----
# Extract equipment subcategory if available
⋮----
# Add EAV template information
⋮----
eav_manager = EAVManager()
⋮----
# Get classification IDs
classification_ids = eav_manager.get_classification_ids(equipment_type)
⋮----
# Get performance fields
performance_fields = eav_manager.get_performance_fields(equipment_type)
⋮----
# Get required attributes
⋮----
# Map predictions to master database fields
⋮----
"""
    Visualize the distribution of categories in the dataset

    Args:
        df (pd.DataFrame): DataFrame with category columns
        output_dir (str, optional): Directory to save visualizations. Defaults to "outputs".

    Returns:
        Tuple[str, str]: Paths to the saved visualization files
    """
# Create output directory if it doesn't exist
⋮----
# Define output file paths
equipment_category_file = f"{output_dir}/equipment_category_distribution.png"
system_type_file = f"{output_dir}/system_type_distribution.png"
⋮----
# Generate visualizations
⋮----
# Example usage
⋮----
# Create and train the equipment classifier
classifier = EquipmentClassifier()
⋮----
# Example prediction with service life
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0  # Example service life in years
prediction = classifier.predict(description, service_life)
⋮----
if key != "attribute_template":  # Skip printing the full template
⋮----
template = prediction["attribute_template"]
⋮----
# Visualize category distribution
````

## File: nexusml/examples/__init__.py
````python
"""
Example scripts for NexusML.
"""
````

## File: nexusml/examples/advanced_example.py
````python
"""
Advanced Example Usage of NexusML

This script demonstrates how to use the NexusML package with visualization components.
It shows the complete workflow from data loading to model training, prediction, and visualization.
"""
⋮----
# Type aliases for better readability
ModelType = Any  # Replace with actual model type when known
PredictionDict = Dict[str, str]  # Dictionary with string keys and values
DataFrameType = Any  # Replace with actual DataFrame type when known
⋮----
# Import and add type annotation for predict_with_enhanced_model
from nexusml.core.model import predict_with_enhanced_model as _predict_with_enhanced_model  # type: ignore
⋮----
# Import from the nexusml package
⋮----
# Add type annotation for the imported function
def predict_with_enhanced_model(model: ModelType, description: str, service_life: float = 0) -> PredictionDict
⋮----
"""
    Wrapper with type annotation for the imported predict_with_enhanced_model function

    This wrapper ensures proper type annotations for the function.

    Args:
        model: The trained model
        description: Equipment description
        service_life: Service life in years

    Returns:
        PredictionDict: Dictionary with prediction results
    """
# Call the original function and convert the result to the expected type
result = _predict_with_enhanced_model(model, description, service_life)  # type: ignore
# We know the result is a dictionary with string keys and values
return {str(k): str(v) for k, v in result.items()}  # type: ignore
⋮----
# Constants
DEFAULT_TRAINING_DATA_PATH = "ingest/data/eq_ids.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_PREDICTION_FILENAME = "example_prediction.txt"
TARGET_CLASSES = ["Equipment_Category", "Uniformat_Class", "System_Type", "Equipment_Type", "System_Subtype"]
⋮----
def get_default_settings() -> Dict[str, Any]
⋮----
"""
    Return default settings when configuration file is not found

    Returns:
        Dict[str, Any]: Default configuration settings
    """
⋮----
def load_settings() -> Dict[str, Any]
⋮----
"""
    Load settings from the configuration file

    Returns:
        Dict[str, Any]: Configuration settings
    """
# Try to find a settings file
settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
⋮----
# Check if we're running in the context of fca_dashboard
⋮----
settings_path = get_config_path("settings.yml")
⋮----
# Not running in fca_dashboard context, use default settings
⋮----
# Return default settings
⋮----
def get_merged_settings(settings: Dict[str, Any]) -> Dict[str, Any]
⋮----
"""
    Merge settings from different sections for compatibility

    Args:
        settings: The loaded settings dictionary

    Returns:
        Dict[str, Any]: Merged settings
    """
# Try to get settings from both nexusml and classifier sections (for compatibility)
nexusml_settings = settings.get("nexusml", {})
classifier_settings = settings.get("classifier", {})
⋮----
# Merge settings, preferring nexusml if available
⋮----
def get_paths_from_settings(merged_settings: Dict[str, Any]) -> Tuple[str, str, str, str, str]
⋮----
"""
    Extract paths from settings

    Args:
        merged_settings: The merged settings dictionary

    Returns:
        Tuple[str, str, str, str, str]: data_path, output_dir, equipment_category_file, system_type_file, prediction_file
    """
# Get data path from settings
data_path = merged_settings.get("data_paths", {}).get("training_data")
⋮----
data_path = str(Path(__file__).resolve().parent.parent / DEFAULT_TRAINING_DATA_PATH)
⋮----
# Get output paths from settings
example_settings = merged_settings.get("examples", {})
output_dir = example_settings.get("output_dir", str(Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIR))
⋮----
equipment_category_file = example_settings.get(
⋮----
system_type_file = example_settings.get(
⋮----
prediction_file = example_settings.get("prediction_file", os.path.join(output_dir, DEFAULT_PREDICTION_FILENAME))
⋮----
def make_prediction(model: ModelType, description: str, service_life: float) -> PredictionDict
⋮----
"""
    Make a prediction using the trained model

    Args:
        model: The trained model
        description: Equipment description
        service_life: Service life in years

    Returns:
        Dict[str, str]: Prediction results
    """
⋮----
prediction = predict_with_enhanced_model(model, description, service_life)
⋮----
"""
    Save prediction results to a file

    Args:
        prediction_file: Path to save the prediction results
        prediction: Prediction results dictionary
        description: Equipment description
        service_life: Service life in years
        equipment_category_file: Path to equipment category visualization
        system_type_file: Path to system type visualization
    """
⋮----
# Add placeholder for model performance metrics
⋮----
target_index = list(prediction.keys()).index(target)
precision = 0.80 + 0.03 * (5 - target_index)
recall = 0.78 + 0.03 * (5 - target_index)
f1_score = 0.79 + 0.03 * (5 - target_index)
accuracy = 0.82 + 0.03 * (5 - target_index)
⋮----
def generate_visualizations(df: DataFrameType, output_dir: str) -> Tuple[str, str]
⋮----
"""
    Generate visualizations for the data

    Args:
        df: DataFrame with the data
        output_dir: Directory to save visualizations

    Returns:
        Tuple[str, str]: Paths to the saved visualization files
    """
⋮----
# Use the visualize_category_distribution function from the model module
⋮----
def main() -> None
⋮----
"""
    Main function demonstrating the usage of the NexusML package
    """
# Load and process settings
settings = load_settings()
merged_settings = get_merged_settings(settings)
⋮----
# Create output directory if it doesn't exist
⋮----
# Train enhanced model using the CSV file
⋮----
# Example prediction with service life
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0  # Example service life in years
⋮----
# Make prediction
prediction = make_prediction(model, description, service_life)
⋮----
# Save prediction results
⋮----
# Generate visualizations
````

## File: nexusml/examples/common.py
````python
"""
Common Utilities for NexusML Examples

This module provides shared functionality for example scripts to reduce code duplication
and ensure consistent behavior across examples.
"""
⋮----
# Initialize logger
logger = get_logger(__name__)
⋮----
"""
    Run a standard training and prediction workflow.

    Args:
        data_path: Path to training data CSV file (if None, uses default from config)
        description: Equipment description for prediction
        service_life: Service life value for prediction (in years)
        output_dir: Directory to save outputs (if None, uses default from config)
        save_results: Whether to save results to file

    Returns:
        Tuple: (trained model, training dataframe, prediction results)
    """
# Use config for default paths
⋮----
data_path = get_data_path("training_data")
⋮----
output_dir = get_output_dir()
⋮----
# Convert Path objects to strings
⋮----
data_path = str(data_path)
⋮----
output_dir = str(output_dir)
⋮----
# Create output directory if it doesn't exist
⋮----
# Training
⋮----
# Prediction
⋮----
prediction = predict_with_enhanced_model(model, description, service_life)
⋮----
# Save results if requested
⋮----
prediction_file = os.path.join(output_dir, "example_prediction.txt")
⋮----
"""
    Generate visualizations for model results.

    Args:
        df: Training dataframe
        model: Trained model
        output_dir: Directory to save visualizations (if None, uses default from config)
        show_plots: Whether to display plots (in addition to saving them)

    Returns:
        Dict[str, str]: Paths to generated visualization files
    """
⋮----
# Convert Path object to string if needed
⋮----
# If output_dir is still None, return empty dict
⋮----
# Define output file paths
visualization_files = {}
⋮----
# Equipment Category Distribution
equipment_category_file = os.path.join(
⋮----
# System Type Distribution
system_type_file = os.path.join(output_dir, "system_type_distribution.png")
````

## File: nexusml/examples/feature_engineering_example.py
````python
"""
Feature Engineering Example

This example demonstrates how to use the new config-driven feature engineering approach.
"""
⋮----
# Add the parent directory to the path so we can import nexusml
⋮----
def demonstrate_generic_feature_engineering()
⋮----
"""
    Demonstrate how to use the GenericFeatureEngineer class directly.
    """
⋮----
# Load sample data
⋮----
df = load_and_preprocess_data()
⋮----
# Print original columns
⋮----
# Apply generic feature engineering
⋮----
engineer = GenericFeatureEngineer()
df_transformed = engineer.transform(df)
⋮----
# Print new columns
⋮----
# Print sample of combined text
⋮----
# Print sample of hierarchical categories
⋮----
def demonstrate_model_training_with_config()
⋮----
"""
    Demonstrate how to train a model using the config-driven approach.
    """
⋮----
# Train model with config-driven feature engineering
⋮----
)  # Use direct to speed up example
⋮----
# Make a prediction
⋮----
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0
prediction = predict_with_enhanced_model(model, description, service_life)
⋮----
# Demonstrate using GenericFeatureEngineer directly
df_transformed = demonstrate_generic_feature_engineering()
⋮----
# Demonstrate training a model with config-driven feature engineering
````

## File: nexusml/examples/integrated_classifier_example.py
````python
"""
Integrated Equipment Classifier Example

This example demonstrates the comprehensive equipment classification model that integrates:
1. Multiple classification systems (OmniClass, MasterFormat, Uniformat)
2. EAV (Entity-Attribute-Value) structure for flexible equipment attributes
3. ML capabilities to fill in missing attribute data
"""
⋮----
# Add the parent directory to the path to import nexusml modules
⋮----
def main()
⋮----
"""Run the integrated equipment classifier example."""
⋮----
# Create output directory
output_dir = Path(__file__).resolve().parent.parent / "output"
⋮----
# Initialize the equipment classifier
⋮----
classifier = EquipmentClassifier()
⋮----
# Train the model
⋮----
# Example equipment descriptions
examples = [
⋮----
# Make predictions for each example
⋮----
results = []
⋮----
# Make prediction
prediction = classifier.predict(example["description"], example["service_life"])
⋮----
# Extract basic classification results
basic_result = {
⋮----
# Get the attribute template
template = prediction.get("attribute_template", {})
⋮----
# Try to extract attributes from the description
equipment_type = prediction["Equipment_Category"]
extracted_attributes = {}
⋮----
extracted_attributes = classifier.predict_attributes(
⋮----
# Add results to the list
⋮----
# Save results to JSON file
results_file = output_dir / "integrated_classifier_results.json"
⋮----
# Generate a complete EAV template example
⋮----
eav_manager = EAVManager()
⋮----
# Get templates for different equipment types
equipment_types = ["Chiller", "Air Handler", "Boiler", "Pump", "Cooling Tower"]
templates = {}
⋮----
# Save templates to JSON file
templates_file = output_dir / "equipment_templates.json"
⋮----
# Demonstrate attribute validation
⋮----
# Example: Valid attributes for a chiller
valid_attributes = {
⋮----
# Example: Invalid attributes for a chiller (missing required, has unknown)
invalid_attributes = {
⋮----
# Validate attributes
valid_result = eav_manager.validate_attributes("Chiller", valid_attributes)
invalid_result = eav_manager.validate_attributes("Chiller", invalid_attributes)
⋮----
# Demonstrate filling missing attributes
⋮----
# Example: Partial attributes for a chiller
partial_attributes = {"cooling_capacity_tons": 500, "chiller_type": "Centrifugal"}
⋮----
# Description with additional information
description = "Centrifugal chiller with 500 tons cooling capacity, 0.6 kW/ton efficiency, using R-134a refrigerant"
⋮----
# Fill missing attributes
filled_attributes = eav_manager.fill_missing_attributes(
````

## File: nexusml/examples/omniclass_generator_example.py
````python
"""
Example script demonstrating how to use the OmniClass generator in NexusML.

This script shows how to extract OmniClass data from Excel files and generate
descriptions using the Claude API.
"""
⋮----
def main()
⋮----
"""Run the OmniClass generator example."""
# Set up paths
input_dir = "files/omniclass_tables"
output_csv = "nexusml/ingest/generator/data/omniclass.csv"
output_with_descriptions = "nexusml/ingest/generator/data/omniclass_with_descriptions.csv"
⋮----
# Extract OmniClass data from Excel files
⋮----
df = extract_omniclass_data(input_dir=input_dir, output_file=output_csv, file_pattern="*.xlsx")
⋮----
# Check if ANTHROPIC_API_KEY is set
⋮----
# Generate descriptions for a small subset of the data
⋮----
result_df = generate_descriptions(
⋮----
end_index=5,  # Only process 5 rows for this example
⋮----
# Display sample results
````

## File: nexusml/examples/omniclass_hierarchy_example.py
````python
"""
OmniClass Hierarchy Visualization Example

This example demonstrates how to use the OmniClass hierarchy visualization tools
to display OmniClass data in a hierarchical tree structure.
"""
⋮----
# Add path to allow importing from nexusml package
⋮----
# Path to the data directory
DATA_DIR = os.path.dirname(data_file)
logger = get_logger(__name__)
⋮----
def main()
⋮----
"""
    Main function to demonstrate OmniClass hierarchy visualization.
    """
# Default output directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
⋮----
# Create output directory if it doesn't exist
⋮----
# Load the OmniClass data
omniclass_file = os.path.join(DATA_DIR, "omniclass.csv")
⋮----
# Try to read the CSV file safely
⋮----
df = read_csv_safe(omniclass_file)
⋮----
# Clean the CSV file
cleaned_file = clean_omniclass_csv(omniclass_file)
⋮----
# Read the cleaned file
df = read_csv_safe(cleaned_file)
⋮----
# Display available columns
⋮----
# Set column names
code_col = "OmniClass_Code"
title_col = "OmniClass_Title"
desc_col = "Description"
⋮----
# Filter data (optional)
# For example, filter to only show Table 23 (Products) entries
filter_value = "23-"
⋮----
filtered_df = df[df[code_col].str.contains(filter_value, na=False)]
⋮----
# Further filter to limit the number of entries for the example
# For example, only show entries related to HVAC
hvac_filter = "HVAC|mechanical|boiler|pump|chiller"
⋮----
hvac_df = filtered_df[
⋮----
# Build the tree
⋮----
tree = build_tree(hvac_df, code_col, title_col, desc_col)
⋮----
# Display the tree in terminal format
⋮----
# Display the tree in markdown format
⋮----
markdown_lines = print_tree_markdown(tree)
⋮----
# Save to file
output_file = os.path.join(output_dir, "omniclass_hvac_hierarchy.md")
⋮----
# Save terminal output to file as well
terminal_output_file = os.path.join(output_dir, "omniclass_hvac_hierarchy.txt")
⋮----
# Redirect stdout to file temporarily
````

## File: nexusml/examples/simple_example.py
````python
"""
Simplified Example Usage of NexusML

This script demonstrates the core functionality of the NexusML package
without the visualization components. It shows the workflow from data loading to model
training and prediction.
"""
⋮----
# Import from the nexusml package
⋮----
def load_settings()
⋮----
"""
    Load settings from the configuration file
    
    Returns:
        dict: Configuration settings
    """
# Try to find a settings file
settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
⋮----
# Check if we're running in the context of fca_dashboard
⋮----
settings_path = get_config_path("settings.yml")
⋮----
# Not running in fca_dashboard context, use default settings
⋮----
# Return default settings
⋮----
def main()
⋮----
"""
    Main function demonstrating the usage of the NexusML package
    """
# Load settings
settings = load_settings()
⋮----
# Try to get settings from both nexusml and classifier sections (for compatibility)
nexusml_settings = settings.get('nexusml', {})
classifier_settings = settings.get('classifier', {})
⋮----
# Merge settings, preferring nexusml if available
merged_settings = {**classifier_settings, **nexusml_settings}
⋮----
# Get data path from settings
data_path = merged_settings.get('data_paths', {}).get('training_data')
⋮----
data_path = str(Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv")
⋮----
# Get output paths from settings
example_settings = merged_settings.get('examples', {})
output_dir = example_settings.get('output_dir', str(Path(__file__).resolve().parent / "outputs"))
prediction_file = example_settings.get('prediction_file',
⋮----
# Create output directory if it doesn't exist
⋮----
# Train enhanced model using the CSV file
⋮----
# Example prediction with service life
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0  # Example service life in years
⋮----
prediction = predict_with_enhanced_model(model, description, service_life)
⋮----
# Save prediction results to file
⋮----
# Add placeholder for model performance metrics
````

## File: nexusml/examples/staging_data_example.py
````python
"""
Staging Data Classification Example

This example demonstrates how to use the ML model with staging data that has different column names.
It shows the complete workflow from staging data to master database field mapping.
"""
⋮----
# Add the parent directory to the path to import nexusml modules
⋮----
def create_test_staging_data()
⋮----
"""Create a sample staging data CSV file for testing."""
data = [
⋮----
# Create output directory
output_dir = Path(__file__).resolve().parent.parent / "output"
⋮----
# Save to CSV
df = pd.DataFrame(data)
csv_path = output_dir / "test_staging_data.csv"
⋮----
def main()
⋮----
"""Run the staging data classification example."""
⋮----
# Create test staging data
⋮----
staging_data_path = create_test_staging_data()
⋮----
# Load staging data
⋮----
staging_df = pd.read_csv(staging_data_path)
⋮----
# Initialize and train the equipment classifier
⋮----
classifier = EquipmentClassifier()
⋮----
# Process each equipment record
⋮----
results = []
⋮----
# Create description from relevant fields
description_parts = []
⋮----
description = " ".join(description_parts)
service_life = (
asset_tag = str(row.get("Asset Tag", ""))
⋮----
# Get prediction with master DB mapping
prediction = classifier.predict(description, service_life, asset_tag)
⋮----
# Print key results
⋮----
# Print master DB mapping
⋮----
# Add to results
⋮----
# Save results to JSON
results_file = (
````

## File: nexusml/ingest/__init__.py
````python
"""
Data ingestion functionality for NexusML.
"""
⋮----
# Import ingest functions to expose at the package level
# These will be populated as we migrate the ingest functionality
⋮----
__all__ = [
````

## File: nexusml/ingest/generator/__init__.py
````python
"""
Generator module for NexusML.

This module provides utilities for generating data for the NexusML module,
including OmniClass data extraction and description generation.
"""
⋮----
__all__ = [
````

## File: nexusml/ingest/generator/omniclass_description_generator.py
````python
"""
Utility for generating descriptions for OmniClass codes using the Claude API.

This module provides functions to generate plain-English descriptions for OmniClass codes
using the Claude API. It processes the data in batches to manage API rate limits and costs.
"""
⋮----
# Load environment variables from .env file
⋮----
# Define custom error classes
class NexusMLError(Exception)
⋮----
"""Base exception for NexusML errors."""
⋮----
class ApiClientError(NexusMLError)
⋮----
"""Exception raised for API client errors."""
⋮----
class DescriptionGeneratorError(NexusMLError)
⋮----
"""Exception raised for description generator errors."""
⋮----
# Load settings from config file if available
def load_settings()
⋮----
"""
    Load settings from the config file.

    Returns:
        dict: Settings dictionary
    """
⋮----
# Try to load from fca_dashboard settings if available
⋮----
# Not running in fca_dashboard context, load from local config
config_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
⋮----
# Initialize settings
settings = load_settings()
⋮----
# Import utilities if available, otherwise define minimal versions
⋮----
# Define minimal versions of required functions
def get_logger(name)
⋮----
"""Simple logger function."""
⋮----
def resolve_path(path)
⋮----
"""Resolve a path to an absolute path."""
⋮----
path = Path(path)
⋮----
# Load settings from config file
config = settings.get("generator", {}).get("omniclass_description_generator", {})
api_config = config.get("api", {})
⋮----
# Constants with defaults from config
BATCH_SIZE = config.get("batch_size", 50)
MODEL = api_config.get("model", "claude-3-haiku-20240307")
MAX_RETRIES = api_config.get("retries", 3)
RETRY_DELAY = api_config.get("delay", 5)
DEFAULT_INPUT_FILE = config.get("input_file", "nexusml/ingest/generator/data/omniclass.csv")
DEFAULT_OUTPUT_FILE = config.get("output_file", "nexusml/ingest/generator/data/omniclass_with_descriptions.csv")
DEFAULT_DESCRIPTION_COLUMN = config.get("description_column", "Description")
⋮----
# System prompt for Claude
SYSTEM_PROMPT = config.get(
⋮----
# Initialize logger
logger = get_logger("omniclass_description_generator")
⋮----
class ApiClient(ABC)
⋮----
"""Abstract base class for API clients."""
⋮----
@abstractmethod
    def call(self, prompt: str, system_prompt: str, **kwargs) -> Optional[str]
⋮----
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
⋮----
class AnthropicClient(ApiClient)
⋮----
"""Client for the Anthropic Claude API."""
⋮----
def __init__(self, api_key: Optional[str] = None)
⋮----
"""
        Initialize the Anthropic client.

        Args:
            api_key: The API key to use. If None, uses the ANTHROPIC_API_KEY environment variable.

        Raises:
            ApiClientError: If the API key is not provided and not found in environment variables
        """
# Get API key from environment variables if not provided
⋮----
# Create the client
⋮----
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
⋮----
response = self.client.messages.create(
⋮----
class DescriptionGenerator(ABC)
⋮----
"""Abstract base class for description generators."""
⋮----
@abstractmethod
    def generate(self, data: pd.DataFrame) -> List[Optional[str]]
⋮----
"""
        Generate descriptions for the given data.

        Args:
            data: DataFrame containing data to generate descriptions for

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If description generation fails
        """
⋮----
class OmniClassDescriptionGenerator(DescriptionGenerator)
⋮----
"""Generator for OmniClass descriptions using Claude API."""
⋮----
def __init__(self, api_client: Optional[ApiClient] = None, system_prompt: Optional[str] = None)
⋮----
"""
        Initialize the OmniClass description generator.

        Args:
            api_client: The API client to use. If None, creates a new AnthropicClient.
            system_prompt: The system prompt to use. If None, uses the default SYSTEM_PROMPT.
        """
⋮----
def generate_prompt(self, data: pd.DataFrame) -> str
⋮----
"""
        Generate a prompt for the API based on the data.

        Args:
            data: DataFrame containing OmniClass codes and titles

        Returns:
            Formatted prompt for the API
        """
prompt_items = []
⋮----
prompt = f"""
⋮----
def parse_response(self, response_text: str) -> List[Optional[str]]
⋮----
"""
        Parse the response from the API.

        Args:
            response_text: Response text from the API

        Returns:
            List of descriptions
        """
⋮----
# Extract JSON array from response
json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
⋮----
def generate(self, data: pd.DataFrame) -> List[Optional[str]]
⋮----
"""
        Generate descriptions for OmniClass codes.

        Args:
            data: DataFrame containing OmniClass codes and titles

        Returns:
            List of descriptions

        Raises:
            DescriptionGeneratorError: If description generation fails
        """
⋮----
# Check required columns
required_columns = ["OmniClass_Code", "OmniClass_Title"]
missing_columns = [col for col in required_columns if col not in data.columns]
⋮----
# Generate prompt
prompt = self.generate_prompt(data)
⋮----
# Call API
⋮----
response_text = self.api_client.call(prompt=prompt, system_prompt=self.system_prompt)
⋮----
# Parse response
descriptions = self.parse_response(response_text)
⋮----
# If we got fewer descriptions than expected, pad with None
⋮----
class BatchProcessor
⋮----
"""Processor for batch processing data."""
⋮----
def __init__(self, generator: DescriptionGenerator, batch_size: int = BATCH_SIZE)
⋮----
"""
        Initialize the batch processor.

        Args:
            generator: The description generator to use
            batch_size: The size of batches to process
        """
⋮----
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
⋮----
batch = result_df.iloc[i : min(i + self.batch_size, end_index)].copy()
⋮----
# Process all rows regardless of existing descriptions
batch_to_process = batch
⋮----
# Process batch
⋮----
descriptions = self.generator.generate(batch_to_process)
⋮----
# Update the dataframe
⋮----
# Convert column to string type if needed to avoid dtype warning
⋮----
# Continue with next batch
⋮----
# Save progress if callback provided
⋮----
# No rate limiting for Tier 4 API access
# time.sleep(1)
⋮----
# Convenience functions for backward compatibility and ease of use
⋮----
def create_client() -> anthropic.Anthropic
⋮----
"""Create and return an Anthropic client."""
⋮----
def generate_prompt(batch_data: pd.DataFrame) -> str
⋮----
"""
    Generate a prompt for the Claude API based on the batch data.

    Args:
        batch_data: DataFrame containing OmniClass codes and titles

    Returns:
        str: Formatted prompt for the Claude API
    """
⋮----
def call_claude_api(client: anthropic.Anthropic, prompt: str) -> Optional[str]
⋮----
"""
    Call the Claude API with retry logic.

    Args:
        client: Anthropic client
        prompt: Prompt for the Claude API

    Returns:
        str: Response from the Claude API
    """
api_client = AnthropicClient(api_key=client.api_key)
⋮----
def parse_response(response_text: str) -> List[Optional[str]]
⋮----
"""
    Parse the response from the Claude API.

    Args:
        response_text: Response text from the Claude API

    Returns:
        list: List of descriptions
    """
⋮----
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
⋮----
# Resolve paths
input_path = resolve_path(input_file)
⋮----
# Set default output file if not provided
⋮----
output_path = (
⋮----
output_path = resolve_path(output_file)
⋮----
# Create the output directory if it doesn't exist
output_dir = output_path.parent
⋮----
# Load the CSV
⋮----
df = pd.read_csv(input_path)
total_rows = len(df)
⋮----
# Create generator and processor
generator = OmniClassDescriptionGenerator()
processor = BatchProcessor(generator, batch_size=batch_size)
⋮----
# Define save callback
def save_progress(current_df: pd.DataFrame) -> None
⋮----
# Process data
⋮----
result_df = processor.process(
⋮----
# Save final result
⋮----
def main()
⋮----
"""Main function."""
⋮----
parser = argparse.ArgumentParser(description="Generate descriptions for OmniClass codes")
⋮----
args = parser.parse_args()
⋮----
# Use max-rows as end_index if provided
end_index = args.max_rows if args.max_rows is not None else args.end
````

## File: nexusml/ingest/generator/omniclass_example.py
````python
"""
OmniClass Example Visualization Tool

This module provides a simple example of visualizing OmniClass data in a hierarchical tree structure.
It uses a hardcoded example dataset of medical equipment (dialysis products) to demonstrate the
hierarchy visualization capabilities.
"""
⋮----
# Add path to allow importing from nexusml package
⋮----
logger = get_logger(__name__)
⋮----
def parse_omniclass_code(code)
⋮----
"""
    Parse an OmniClass code into its hierarchical components.
    Format: xx-yy yy yy-zz where:
    - xx: OmniClass table
    - yy yy yy: hierarchy
    - zz: detail number
    """
# Remove any whitespace and split by hyphens and spaces
parts = re.split(r"[-\s]+", code.strip())
⋮----
# Return the parsed components
if len(parts) >= 4:  # Full format with detail number
⋮----
else:  # Partial format without detail number
⋮----
def build_tree(data_lines)
⋮----
"""
    Build a hierarchical tree from OmniClass data lines.

    Args:
        data_lines: List of strings in format "code,title,description"

    Returns:
        A nested dictionary representing the tree structure
    """
tree = {}
⋮----
# Split the line into code, title, and description
parts = line.split(",", 2)
⋮----
code = parts[0].strip()
title = parts[1].strip()
description = parts[2].strip() if len(parts) > 2 else ""
⋮----
# Remove quotes from description if present
⋮----
description = description[1:-1]
⋮----
# Parse the code
parsed = parse_omniclass_code(code)
⋮----
# Navigate to the correct position in the tree
current = tree
path = [parsed["table"]] + parsed["hierarchy"]
⋮----
# Create the node if it doesn't exist
⋮----
if i == len(path) - 1:  # Last part (leaf node)
# Update the leaf node with actual data
⋮----
# Ensure children dictionary exists for intermediate nodes
⋮----
# Move to the next level
current = current[part]["children"]
⋮----
def print_tree_terminal(tree, indent=0, prefix="")
⋮----
"""
    Print the tree structure to the terminal.
    """
⋮----
title = node["title"]
description = node["description"]
⋮----
# Print the current node
⋮----
# Print children
⋮----
# Handle case where node is not properly formatted
⋮----
def print_tree_markdown(tree, indent=0, prefix="")
⋮----
"""
    Print the tree structure in Markdown format.
    """
markdown = []
⋮----
# Create the current node line
⋮----
line = f"{'  ' * indent}{prefix}**{key}**: {title} - *{description}*"
⋮----
line = f"{'  ' * indent}{prefix}**{key}**: {title}"
⋮----
# Add children
⋮----
child_md = print_tree_markdown(node["children"], indent + 1, "- ")
⋮----
line = f"{'  ' * indent}{prefix}**{key}**"
⋮----
# Example data of medical equipment (dialysis products)
example_data = """
⋮----
def main()
⋮----
# Default output directory
output_dir = os.path.abspath(
⋮----
# Create output directory if it doesn't exist
⋮----
# Process the example data
⋮----
data_lines = [
⋮----
# Build the tree
⋮----
tree = build_tree(data_lines)
⋮----
# Ask for output directory
output_dir_input = input(f"Enter output directory (default: {output_dir}): ")
⋮----
output_dir = output_dir_input
⋮----
# Output format
output_format = (
⋮----
markdown_lines = print_tree_markdown(tree)
⋮----
# Option to save to file
save_option = input("Save to file? (y/n, default: y): ").lower() or "y"
⋮----
filename = (
output_file = os.path.join(output_dir, filename)
⋮----
# Save terminal output to file as well
⋮----
# Redirect stdout to file temporarily
````

## File: nexusml/ingest/generator/omniclass_hierarchy.py
````python
"""
OmniClass Hierarchy Visualization Tool

This module provides functionality to visualize OmniClass data in a hierarchical tree structure.
It can parse OmniClass codes in the format xx-yy yy yy-zz and display the hierarchy in terminal
or markdown format.
"""
⋮----
# Add path to allow importing from nexusml package
⋮----
# Path to the data directory
DATA_DIR = os.path.dirname(data_file)
logger = get_logger(__name__)
⋮----
def parse_omniclass_code(code)
⋮----
"""
    Parse an OmniClass code into its hierarchical components.
    Format: xx-yy yy yy-zz where:
    - xx: OmniClass table
    - yy yy yy: hierarchy
    - zz: detail number
    """
# Remove any whitespace and split by hyphens
parts = re.split(r"[-\s]+", code.strip())
⋮----
# Return the parsed components
if len(parts) >= 4:  # Full format with detail number
⋮----
else:  # Partial format without detail number
⋮----
def build_tree(df, code_column, title_column, description_column=None)
⋮----
"""
    Build a hierarchical tree from OmniClass data.

    Args:
        df: DataFrame containing OmniClass data
        code_column: Name of the column containing OmniClass codes
        title_column: Name of the column containing titles
        description_column: Optional name of the column containing descriptions

    Returns:
        A nested dictionary representing the tree structure
    """
tree = {}
⋮----
# Sort by code to ensure parent nodes are processed before children
df_sorted = df.sort_values(by=code_column)
⋮----
code = row[code_column]
title = row[title_column]
description = row[description_column] if description_column else ""
⋮----
# Parse the code
parsed = parse_omniclass_code(code)
⋮----
# Navigate to the correct position in the tree
current = tree
path = [parsed["table"]] + parsed["hierarchy"]
⋮----
# Create the node if it doesn't exist
⋮----
if i == len(path) - 1:  # Last part (leaf node)
# Update the leaf node with actual data
⋮----
# Ensure children dictionary exists for intermediate nodes
⋮----
# Move to the next level
current = current[part]["children"]
⋮----
def print_tree_terminal(tree, indent=0, prefix="")
⋮----
"""
    Print the tree structure to the terminal.
    """
⋮----
title = node["title"]
description = node["description"]
⋮----
# Print the current node
⋮----
# Print children
⋮----
# Handle case where node is not properly formatted
⋮----
def print_tree_markdown(tree, indent=0, prefix="")
⋮----
"""
    Print the tree structure in Markdown format.
    """
markdown = []
⋮----
# Create the current node line
⋮----
line = f"{'  ' * indent}{prefix}**{key}**: {title} - *{description}*"
⋮----
line = f"{'  ' * indent}{prefix}**{key}**: {title}"
⋮----
# Add children
⋮----
child_md = print_tree_markdown(node["children"], indent + 1, "- ")
⋮----
line = f"{'  ' * indent}{prefix}**{key}**"
⋮----
def main()
⋮----
# Default values
output_dir = os.path.abspath(
⋮----
# Create output directory if it doesn't exist
⋮----
# Load the OmniClass data
⋮----
default_file = os.path.join(DATA_DIR, "omniclass.csv")
file_path = (
⋮----
# Ask for output directory
output_dir_input = input(f"Enter output directory (default: {output_dir}): ")
⋮----
output_dir = output_dir_input
⋮----
# Check if the file needs cleaning
⋮----
df = read_csv_safe(file_path)
⋮----
# Ask user if they want to clean the file
clean_option = (
⋮----
cleaned_file = clean_omniclass_csv(file_path)
⋮----
file_path = cleaned_file
⋮----
# Determine column names
⋮----
code_col = (
title_col = (
desc_col = input(
⋮----
# Optional filtering
filter_option = (
⋮----
filter_column = (
filter_value = input(
df = df[df[filter_column].str.contains(filter_value, na=False)]
⋮----
# Build the tree
⋮----
tree = build_tree(df, code_col, title_col, desc_col if desc_col else None)
⋮----
# Output format
output_format = (
⋮----
markdown_lines = print_tree_markdown(tree)
⋮----
# Option to save to file
save_option = input("\nSave to file? (y/n, default: y): ").lower() or "y"
⋮----
filename = (
output_file = os.path.join(output_dir, filename)
⋮----
# Save terminal output to file as well
⋮----
# Redirect stdout to file temporarily
````

## File: nexusml/ingest/generator/omniclass_tree.py
````python
"""
OmniClass Tree Visualization Tool

This module provides a simplified command-line tool to visualize OmniClass data in a hierarchical tree structure.
It can parse OmniClass codes in the format xx-yy yy yy-zz and display the hierarchy in terminal or markdown format.
"""
⋮----
# Add path to allow importing from nexusml package
⋮----
# Path to the data directory
DATA_DIR = os.path.dirname(data_file)
logger = get_logger(__name__)
⋮----
def parse_omniclass_code(code)
⋮----
"""
    Parse an OmniClass code into its hierarchical components.
    Format: xx-yy yy yy-zz where:
    - xx: OmniClass table
    - yy yy yy: hierarchy
    - zz: detail number
    """
# Remove any whitespace and split by hyphens and spaces
parts = re.split(r"[-\s]+", code.strip())
⋮----
# Return the parsed components
if len(parts) >= 4:  # Full format with detail number
⋮----
else:  # Partial format without detail number
⋮----
def build_tree(df, code_column, title_column, description_column=None)
⋮----
"""
    Build a hierarchical tree from OmniClass data.
    """
tree = {}
⋮----
# Sort by code to ensure parent nodes are processed before children
df_sorted = df.sort_values(by=code_column)
⋮----
code = row[code_column]
title = row[title_column]
description = (
⋮----
# Parse the code
parsed = parse_omniclass_code(code)
⋮----
# Navigate to the correct position in the tree
current = tree
path = [parsed["table"]] + parsed["hierarchy"]
⋮----
# Create the node if it doesn't exist
⋮----
if i == len(path) - 1:  # Last part (leaf node)
# Update the leaf node with actual data
⋮----
# Ensure children dictionary exists for intermediate nodes
⋮----
# Move to the next level
current = current[part]["children"]
⋮----
def print_tree_terminal(tree, indent=0, prefix="")
⋮----
"""
    Print the tree structure to the terminal.
    """
⋮----
title = node["title"]
description = node["description"]
⋮----
# Print the current node
⋮----
# Print children
⋮----
# Handle case where node is not properly formatted
⋮----
def print_tree_markdown(tree, indent=0, prefix="")
⋮----
"""
    Print the tree structure in Markdown format.
    """
markdown = []
⋮----
# Create the current node line
⋮----
line = f"{'  ' * indent}{prefix}**{key}**: {title} - *{description}*"
⋮----
line = f"{'  ' * indent}{prefix}**{key}**: {title}"
⋮----
# Add children
⋮----
child_md = print_tree_markdown(node["children"], indent + 1, "- ")
⋮----
line = f"{'  ' * indent}{prefix}**{key}**"
⋮----
def main()
⋮----
# Default values
default_file = os.path.join(DATA_DIR, "omniclass.csv")
file_path = default_file
code_column = "OmniClass_Code"
title_column = "OmniClass_Title"
description_column = "Description"
output_format = "terminal"
filter_value = None
clean_csv = False
output_dir = os.path.abspath(
⋮----
# Create output directory if it doesn't exist
⋮----
# Check for command line arguments
⋮----
file_path = sys.argv[1]
⋮----
# Additional command line arguments
⋮----
filter_value = sys.argv[2]
⋮----
output_format = sys.argv[3]
⋮----
clean_csv = True
⋮----
output_dir = sys.argv[4]
⋮----
# Load the OmniClass data
⋮----
file_path = clean_omniclass_csv(file_path)
⋮----
df = read_csv_safe(file_path)
⋮----
# Apply filter if specified
⋮----
df = df[df[code_column].str.contains(filter_value, na=False)]
⋮----
# Build the tree
⋮----
tree = build_tree(df, code_column, title_column, description_column)
⋮----
# Output the tree
⋮----
markdown_lines = print_tree_markdown(tree)
⋮----
# Save to file
output_file = os.path.join(
⋮----
# Save terminal output to file as well
⋮----
# Redirect stdout to file temporarily
````

## File: nexusml/ingest/generator/omniclass.py
````python
"""
OmniClass data extraction module for the NexusML application.

This module provides utilities for extracting OmniClass data from Excel files
and generating a unified CSV file for classifier training.
"""
⋮----
# Import utility functions from nexusml.utils
⋮----
# Load settings from config file if available
def load_settings()
⋮----
"""
    Load settings from the config file.

    Returns:
        dict: Settings dictionary
    """
⋮----
# Try to load from fca_dashboard settings if available
⋮----
# Not running in fca_dashboard context, load from local config
config_path = (
⋮----
# Initialize settings
settings = load_settings()
⋮----
"""
    Extract OmniClass data from Excel files and save to a CSV file.

    Args:
        input_dir: Directory containing OmniClass Excel files.
            If None, uses the directory from settings.
        output_file: Path to save the output CSV file.
            If None, uses the path from settings.
        file_pattern: Pattern to match Excel files (default: "*.xlsx").

    Returns:
        DataFrame containing the combined OmniClass data.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If no OmniClass files are found or if no FLAT sheets are found.
    """
logger = get_logger("generator")
⋮----
# Use settings if parameters are not provided
⋮----
input_dir = (
⋮----
output_file = (
⋮----
# Resolve paths
⋮----
input_dir = resolve_path(input_dir)
⋮----
input_dir = Path("files/omniclass_tables").resolve()
⋮----
output_file = resolve_path(output_file)
⋮----
output_file = Path("nexusml/ingest/generator/data/omniclass.csv").resolve()
⋮----
# Check if input directory exists
⋮----
error_msg = f"Input directory does not exist: {input_dir}"
⋮----
# Find all Excel files in the input directory
file_paths = list(input_dir.glob(file_pattern))
⋮----
error_msg = (
⋮----
# Create the output directory if it doesn't exist
output_dir = output_file.parent
⋮----
# Process each Excel file
all_data = []
⋮----
# Get sheet names
sheet_names = get_sheet_names(file_path)
⋮----
# Find the FLAT sheet
flat_sheet = find_flat_sheet(sheet_names)
⋮----
# Create extraction config
config = {
⋮----
"header_row": 0,  # Assume header is in the first row
⋮----
# Extract data from the FLAT sheet
extracted_data = extract_excel_with_config(file_path, config)
⋮----
# Find the sheet in the extracted data
# The sheet name might have been normalized
df = None
⋮----
df = extracted_data[flat_sheet]
⋮----
# Try to find a sheet with a similar name
normalized_sheet_names = normalize_sheet_names(file_path)
normalized_flat_sheet = None
⋮----
normalized_flat_sheet = normalized
⋮----
df = extracted_data[normalized_flat_sheet]
⋮----
# Just use the first sheet as a fallback
⋮----
sheet_name = list(extracted_data.keys())[0]
df = extracted_data[sheet_name]
⋮----
# Clean the DataFrame using our data cleaning utilities
# Set is_omniclass=True to enable special handling for OmniClass headers
cleaned_df = clean_dataframe(df, is_omniclass=True)
⋮----
# Add file name as a column for tracking
⋮----
# Add table number from filename (e.g., OmniClass_22_2020-08-15_2022.xlsx -> 22)
table_number = (
⋮----
# Append to the list of dataframes
⋮----
error_msg = "No data extracted from any OmniClass files"
⋮----
# Combine all dataframes
combined_df = pd.concat(all_data, ignore_index=True)
⋮----
# Get column mapping from settings
column_mapping = (
⋮----
# Standardize column names if needed
combined_df = standardize_column_names(combined_df, column_mapping=column_mapping)
⋮----
# Save to CSV if output_file is not None
⋮----
def main()
⋮----
"""
    Main function to run the OmniClass data extraction as a standalone script.
    """
⋮----
parser = argparse.ArgumentParser(
⋮----
args = parser.parse_args()
⋮----
# Extract OmniClass data
````

## File: nexusml/ingest/generator/README.md
````markdown
# NexusML Generator Module

This module provides utilities for generating data for the NexusML module,
including OmniClass data extraction, description generation, and hierarchy
visualization.

## Components

### OmniClass Data Extraction

The `omniclass.py` module provides functionality to extract OmniClass data from
Excel files and generate a unified CSV file for classifier training.

Key functions:

- `extract_omniclass_data`: Extract OmniClass data from Excel files and save to
  a CSV file.

### OmniClass Description Generator

The `omniclass_description_generator.py` module provides functionality to
generate plain-English descriptions for OmniClass codes using the Claude API.

Key components:

- `OmniClassDescriptionGenerator`: Generator for OmniClass descriptions using
  Claude API.
- `BatchProcessor`: Processor for batch processing data.
- `AnthropicClient`: Client for the Anthropic Claude API.
- `generate_descriptions`: Generate descriptions for OmniClass codes.

### OmniClass Hierarchy Visualization

The module includes several tools for visualizing OmniClass data in a
hierarchical tree structure:

- `omniclass_hierarchy.py`: Interactive tool for visualizing OmniClass data in a
  hierarchical tree structure.
- `omniclass_tree.py`: Command-line tool for quickly generating OmniClass
  hierarchy trees.
- `omniclass_example.py`: Example tool with hardcoded medical equipment data to
  demonstrate hierarchy visualization.

Key functions:

- `parse_omniclass_code`: Parse OmniClass codes in the format xx-yy yy yy-zz.
- `build_tree`: Build a hierarchical tree from OmniClass data.
- `print_tree_terminal`: Display the hierarchy tree in terminal format.
- `print_tree_markdown`: Generate a markdown representation of the hierarchy
  tree.

## Usage

### OmniClass Data Extraction

```python
from nexusml import extract_omniclass_data

# Extract OmniClass data from Excel files
df = extract_omniclass_data(
    input_dir="files/omniclass_tables",
    output_file="nexusml/ingest/generator/data/omniclass.csv",
    file_pattern="*.xlsx"
)
```

### OmniClass Description Generation

```python
from nexusml import generate_descriptions

# Generate descriptions for OmniClass codes
result_df = generate_descriptions(
    input_file="nexusml/ingest/generator/data/omniclass.csv",
    output_file="nexusml/ingest/generator/data/omniclass_with_descriptions.csv",
    start_index=0,
    end_index=None,  # Process all rows
    batch_size=50,
    description_column="Description"
)
```

### OmniClass Hierarchy Visualization

```python
from nexusml.ingest.generator.omniclass_hierarchy import build_tree, print_tree_terminal

# Load OmniClass data
import pandas as pd
df = pd.read_csv("nexusml/ingest/data/omniclass.csv")

# Filter data (optional)
filtered_df = df[df["OmniClass_Code"].str.contains("23-", na=False)]

# Build the hierarchy tree
tree = build_tree(filtered_df, "OmniClass_Code", "OmniClass_Title", "Description")

# Display the tree in terminal format
print_tree_terminal(tree)
```

## Requirements

- Python 3.8+
- pandas
- anthropic
- dotenv
- tqdm
- re

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for the Anthropic Claude API (only needed for
  description generation).

## Examples

- See `nexusml/examples/omniclass_generator_example.py` for a complete example
  of how to use the generator module.
- See `nexusml/examples/omniclass_hierarchy_example.py` for an example of how to
  use the hierarchy visualization tools.
````

## File: nexusml/pyproject.toml
````toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["core", "utils", "ingest", "examples", "config"]
package-dir = {"" = "."}

[project]
name = "nexusml"
version = "0.1.0"
description = "Modern machine learning classification engine"
readme = "README.md"
authors = [
    {name = "FCA Dashboard Team"}
]
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.8"
dependencies = [
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "imbalanced-learn>=0.8.0",
    "pyyaml>=6.0",
    "setuptools>=57.0.0",
    "wheel>=0.36.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.9.0",
]

[tool.black]
line-length = 88
target-version = ["py38"]

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Changed to false for more flexibility with ML code
disallow_incomplete_defs = false  # Changed to false for more flexibility with ML code
check_untyped_defs = true  # Added to check functions without requiring annotations
ignore_missing_imports = true  # Added to handle third-party libraries

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
````

## File: nexusml/README.md
````markdown
# NexusML

A modern machine learning classification engine for equipment classification.

## Overview

NexusML is a standalone Python package that provides machine learning
capabilities for classifying equipment based on descriptions and other features.
It was extracted from the FCA Dashboard project to enable independent
development and reuse.

## Features

- Data preprocessing and cleaning
- Feature engineering for text data
- Hierarchical classification models
- Model evaluation and validation
- Visualization of results
- Easy-to-use API for predictions
- OmniClass data extraction and description generation

## Installation

### From Source

```bash
# Install with pip
pip install -e .

# Or install with uv (recommended)
uv pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

Note: The package is named 'core' in the current monorepo structure, so imports
should use:

```python
from core.model import ...
```

rather than:

```python
from nexusml.core.model import ...
```

## Usage

### Basic Example

```python
from core.model import train_enhanced_model, predict_with_enhanced_model

# Train a model
model, df = train_enhanced_model("path/to/training_data.csv")

# Make a prediction
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0  # Example service life in years

prediction = predict_with_enhanced_model(model, description, service_life)
print(prediction)
```

### OmniClass Generator Usage

```python
from nexusml import extract_omniclass_data, generate_descriptions

# Extract OmniClass data from Excel files
df = extract_omniclass_data(
    input_dir="files/omniclass_tables",
    output_file="nexusml/ingest/generator/data/omniclass.csv",
    file_pattern="*.xlsx"
)

# Generate descriptions for OmniClass codes
result_df = generate_descriptions(
    input_file="nexusml/ingest/generator/data/omniclass.csv",
    output_file="nexusml/ingest/generator/data/omniclass_with_descriptions.csv",
    batch_size=50,
    description_column="Description"
)
```

### Advanced Usage

See the examples directory for more detailed usage examples:

- `simple_example.py`: Basic usage without visualizations
- `advanced_example.py`: Complete workflow with visualizations
- `omniclass_generator_example.py`: Example of using the OmniClass generator
- `advanced_example.py`: Complete workflow with visualizations

## Development

### Setup Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nexusml
```

## License

MIT
````

## File: nexusml/setup.py
````python
"""
Setup script for NexusML.

This is a minimal setup.py file that defers to pyproject.toml for configuration.
"""
⋮----
from setuptools import setup  # type: ignore
````

## File: nexusml/tests/__init__.py
````python
"""
Test suite for NexusML.
"""
````

## File: nexusml/tests/conftest.py
````python
"""
Pytest configuration for NexusML tests.
"""
⋮----
# Add the parent directory to sys.path to allow importing nexusml
⋮----
@pytest.fixture
def sample_data_path()
⋮----
"""
    Fixture that provides the path to sample data for testing.
    
    Returns:
        str: Path to sample data file
    """
⋮----
@pytest.fixture
def sample_description()
⋮----
"""
    Fixture that provides a sample equipment description for testing.
    
    Returns:
        str: Sample equipment description
    """
⋮----
@pytest.fixture
def sample_service_life()
⋮----
"""
    Fixture that provides a sample service life value for testing.
    
    Returns:
        float: Sample service life value
    """
````

## File: nexusml/tests/integration/__init__.py
````python
"""
Integration tests for NexusML.
"""
````

## File: nexusml/tests/integration/test_integration.py
````python
"""
Integration tests for NexusML.

These tests verify that the different components of NexusML work together correctly.
"""
⋮----
@pytest.mark.skip(reason="This test requires a full pipeline run which takes time")
def test_full_pipeline(sample_data_path, sample_description, sample_service_life, tmp_path)
⋮----
"""
    Test the full NexusML pipeline from data loading to prediction.
    
    This test is marked as skip by default because it can take a long time to run.
    """
# Load and preprocess data
df = load_and_preprocess_data(sample_data_path)
⋮----
# Enhance features
df = enhance_features(df)
⋮----
# Create hierarchical categories
df = create_hierarchical_categories(df)
⋮----
# Prepare training data
X = pd.DataFrame({
⋮----
y = df[['Equipment_Category', 'Uniformat_Class', 'System_Type', 'Equipment_Type', 'System_Subtype']]
⋮----
# Build model
model = build_enhanced_model()
⋮----
# Train model (this would take time)
⋮----
# Make a prediction
prediction = predict_with_enhanced_model(model, sample_description, sample_service_life)
⋮----
# Check the prediction
⋮----
# Test visualization (optional)
output_dir = str(tmp_path)
⋮----
@pytest.mark.skip(reason="This test requires FCA Dashboard integration")
def test_fca_dashboard_integration()
⋮----
"""
    Test integration with FCA Dashboard.
    
    This test is marked as skip by default because it requires FCA Dashboard to be available.
    """
⋮----
# Try to import from FCA Dashboard
⋮----
# If imports succeed, test the integration
# This would be a more complex test that verifies the integration works
````

## File: nexusml/tests/unit/__init__.py
````python
"""
Unit tests for NexusML.
"""
````

## File: nexusml/tests/unit/test_generator.py
````python
"""
Unit tests for the generator module.
"""
⋮----
class TestOmniClassGenerator
⋮----
"""Tests for the OmniClass generator module."""
⋮----
def test_find_flat_sheet(self)
⋮----
"""Test the find_flat_sheet function."""
# Test with a sheet name containing 'FLAT'
sheet_names = ["Sheet1", "FLAT_VIEW", "Sheet3"]
⋮----
# Test with no sheet name containing 'FLAT'
sheet_names = ["Sheet1", "Sheet2", "Sheet3"]
⋮----
@patch("nexusml.ingest.generator.omniclass_description_generator.AnthropicClient")
    def test_omniclass_description_generator(self, mock_client)
⋮----
"""Test the OmniClassDescriptionGenerator class."""
# Create a mock API client
mock_client_instance = MagicMock()
⋮----
# Create test data
data = pd.DataFrame({"OmniClass_Code": ["23-13 11 11"], "OmniClass_Title": ["Boilers"]})
⋮----
# Create generator
generator = OmniClassDescriptionGenerator(api_client=mock_client_instance)
⋮----
# Test generate_prompt
prompt = generator.generate_prompt(data)
⋮----
# Test parse_response
response = '[{"description": "Test description"}]'
descriptions = generator.parse_response(response)
⋮----
# Test generate
descriptions = generator.generate(data)
⋮----
@patch("nexusml.ingest.generator.omniclass_description_generator.OmniClassDescriptionGenerator")
    def test_batch_processor(self, mock_generator_class)
⋮----
"""Test the BatchProcessor class."""
# Create a mock generator
mock_generator = MagicMock()
⋮----
data = pd.DataFrame({"OmniClass_Code": ["23-13 11 11"], "OmniClass_Title": ["Boilers"], "Description": [""]})
⋮----
# Create processor
processor = BatchProcessor(generator=mock_generator, batch_size=1)
⋮----
# Test process
result_df = processor.process(data)
````

## File: nexusml/tests/unit/test_pipeline.py
````python
"""
Unit tests for the NexusML pipeline.
"""
⋮----
def test_load_and_preprocess_data(sample_data_path)
⋮----
"""Test that data can be loaded and preprocessed."""
df = load_and_preprocess_data(sample_data_path)
⋮----
def test_enhance_features()
⋮----
"""Test that features can be enhanced."""
# Create a minimal test dataframe
df = pd.DataFrame({
⋮----
enhanced_df = enhance_features(df)
⋮----
# Check that new columns were added
⋮----
def test_create_hierarchical_categories()
⋮----
"""Test that hierarchical categories can be created."""
# Create a minimal test dataframe with the required columns
⋮----
hierarchical_df = create_hierarchical_categories(df)
⋮----
# Check the values
⋮----
def test_build_enhanced_model()
⋮----
"""Test that the model can be built."""
model = build_enhanced_model()
⋮----
# Check that the model has the expected structure
⋮----
@pytest.mark.skip(reason="This test requires a trained model which takes time to create")
def test_predict_with_enhanced_model(sample_description, sample_service_life)
⋮----
"""Test that predictions can be made with the model."""
# This is a more complex test that requires a trained model
# In a real test suite, you might use a pre-trained model or mock the model
⋮----
# For now, we'll skip this test, but here's how it would look
⋮----
# Train a model (this would take time)
⋮----
# Make a prediction
prediction = predict_with_enhanced_model(model, sample_description, sample_service_life)
⋮----
# Check the prediction
````

## File: nexusml/utils/__init__.py
````python
"""
Utility functions for NexusML.
"""
⋮----
# Import utility functions to expose at the package level
⋮----
__all__: List[str] = [
````

## File: nexusml/utils/csv_utils.py
````python
"""
CSV Utilities for NexusML

This module provides utilities for working with CSV files, including cleaning,
verification, and safe reading of potentially malformed CSV files.
"""
⋮----
logger = get_logger(__name__)
⋮----
"""
    Verify a CSV file for common issues and optionally fix them.

    Args:
        filepath: Path to the CSV file
        expected_columns: List of column names that should be present
        expected_field_count: Expected number of fields per row
        fix_issues: Whether to attempt to fix issues
        output_filepath: Path to save the fixed file (if fix_issues is True)
                         If None, will use the original filename with "_fixed" appended

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if the file is valid or was fixed successfully
        - error_message: Description of the issue if not valid, or None if valid
    """
filepath = Path(filepath)
⋮----
output_filepath = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"
⋮----
# First, try to read the file with pandas to check for basic validity
⋮----
df = pd.read_csv(filepath)
⋮----
# Check expected columns if provided
⋮----
missing_columns = [
⋮----
# If we got here without errors and don't need to check field count, the file is valid
⋮----
# If pandas can't read it, we'll try a more manual approach
⋮----
# Manually check each row for the correct number of fields
issues = []
⋮----
reader = csv.reader(f)
⋮----
# Get the header row
⋮----
header = next(reader)
field_count = len(header)
⋮----
# If expected_field_count wasn't provided, use the header length
⋮----
expected_field_count = field_count
⋮----
# Check if header has the expected number of fields
⋮----
# Check each data row
⋮----
# If there are no issues, the file is valid
⋮----
# Log the issues found
⋮----
for line_num, row, msg in issues[:5]:  # Log first 5 issues
⋮----
# If we're not fixing issues, return False
⋮----
# Fix the issues by reading and writing the file manually
⋮----
reader = csv.reader(f_in)
writer = csv.writer(f_out)
⋮----
# Write the header
⋮----
# Fix header if needed
⋮----
# Add empty columns if there are too few
⋮----
# Combine extra columns if there are too many
header = header[: expected_field_count - 1] + [
⋮----
# Write each data row, fixing as needed
⋮----
# Add empty columns if there are too few
⋮----
# Combine extra columns if there are too many
row = row[: expected_field_count - 1] + [
⋮----
"""
    Safely read a CSV file, handling common issues.

    Args:
        filepath: Path to the CSV file
        expected_columns: List of column names that should be present
        expected_field_count: Expected number of fields per row
        fix_issues: Whether to attempt to fix issues
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame containing the CSV data

    Raises:
        ValueError: If the file is invalid and couldn't be fixed
    """
⋮----
# First, verify the file
⋮----
# If the file is valid, read it directly
⋮----
# If we tried to fix issues but still got an error, try reading the fixed file
fixed_filepath = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"
⋮----
# If we got here, the file is invalid and couldn't be fixed
⋮----
"""
    Clean the OmniClass CSV file, handling specific issues with this format.

    Args:
        input_filepath: Path to the input OmniClass CSV file
        output_filepath: Path to save the cleaned file (if None, will use input_filepath with "_cleaned" appended)
        expected_columns: List of expected column names

    Returns:
        Path to the cleaned CSV file

    Raises:
        ValueError: If the file couldn't be cleaned
    """
input_filepath = Path(input_filepath)
⋮----
output_filepath = (
⋮----
output_filepath = Path(output_filepath)
⋮----
# Determine the expected field count
⋮----
expected_field_count = len(expected_columns)
⋮----
# Try to determine from the header row
⋮----
expected_field_count = len(header)
⋮----
# Default to 3 fields for OmniClass CSV (code, title, description)
expected_field_count = 3
⋮----
# Verify and fix the CSV file
⋮----
# If run as a script, clean the OmniClass CSV file
⋮----
parser = argparse.ArgumentParser(description="Clean and verify CSV files")
⋮----
args = parser.parse_args()
⋮----
# Configure logging
⋮----
output_file = clean_omniclass_csv(
````

## File: nexusml/utils/excel_utils.py
````python
"""
Excel utility module for the NexusML application.

This module provides utilities for working with Excel files,
particularly for data extraction and cleaning.
"""
⋮----
class DataCleaningError(Exception)
⋮----
"""Exception raised for errors in the data cleaning process."""
⋮----
def get_logger(name: str)
⋮----
"""Simple logger function."""
⋮----
def resolve_path(path: Union[str, Path, None]) -> Path
⋮----
"""
    Resolve a path to an absolute path.

    Args:
        path: The path to resolve. If None, returns the current working directory.

    Returns:
        The resolved path as a Path object.
    """
⋮----
path = Path(path)
⋮----
def get_sheet_names(file_path: Union[str, Path]) -> List[str]
⋮----
"""
    Get sheet names from an Excel file.

    Args:
        file_path: Path to the Excel file.

    Returns:
        List of sheet names as strings.
    """
# Convert all sheet names to strings to ensure type safety
⋮----
"""
    Extract data from Excel file using a configuration.

    Args:
        file_path: Path to the Excel file.
        config: Configuration dictionary with sheet names as keys and sheet configs as values.
            Each sheet config can have the following keys:
            - header_row: Row index to use as header (default: 0)
            - drop_empty_rows: Whether to drop empty rows (default: False)
            - strip_whitespace: Whether to strip whitespace from string columns (default: False)

    Returns:
        Dictionary with sheet names as keys and DataFrames as values.
    """
result = {}
⋮----
header_row = sheet_config.get("header_row", 0)
df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
⋮----
df = df.dropna(how="all")
⋮----
def normalize_sheet_names(file_path: Union[str, Path]) -> Dict[str, str]
⋮----
"""
    Normalize sheet names in an Excel file.

    Args:
        file_path: Path to the Excel file.

    Returns:
        Dictionary mapping original sheet names to normalized names.
    """
sheet_names = get_sheet_names(file_path)
⋮----
def find_flat_sheet(sheet_names: List[str]) -> Optional[str]
⋮----
"""
    Find the sheet name that contains 'FLAT' in it.

    Args:
        sheet_names: List of sheet names to search through.

    Returns:
        The name of the sheet containing 'FLAT', or None if not found.
    """
⋮----
"""
    Clean a DataFrame.

    Args:
        df: The DataFrame to clean.
        header_patterns: List of patterns to identify the header row.
        copyright_patterns: List of patterns to identify copyright rows.
        column_mapping: Dictionary mapping original column names to standardized names.
        is_omniclass: Whether the DataFrame contains OmniClass data, which requires special handling.

    Returns:
        A cleaned DataFrame.
    """
# Basic cleaning
df = df.copy()
⋮----
# Drop completely empty rows
⋮----
# Handle OmniClass specific cleaning
⋮----
# Look for common OmniClass column names
⋮----
col_str = str(col).lower()
⋮----
"""
    Standardize column names in a DataFrame.

    Args:
        df: The DataFrame to standardize.
        column_mapping: Dictionary mapping original column names to standardized names.
            If None, uses default mapping.

    Returns:
        A new DataFrame with standardized column names.
    """
⋮----
df = df.rename(columns={v: k for k, v in column_mapping.items()})
````

## File: nexusml/utils/logging.py
````python
"""
Unified Logging Module for NexusML

This module provides a consistent logging interface that works both
standalone and when integrated with fca_dashboard.
"""
⋮----
# Try to use fca_dashboard logging if available
⋮----
FCA_LOGGING_AVAILABLE = True
FCA_CONFIGURE_LOGGING = fca_configure_logging
⋮----
FCA_LOGGING_AVAILABLE = False
FCA_CONFIGURE_LOGGING = None
⋮----
):  # Type checker will still warn about this, but it's the best we can do
"""
    Configure application logging.

    Args:
        level: Logging level (e.g., "INFO", "DEBUG", etc.)
        log_file: Path to log file (if None, logs to console only)
        simple_format: Whether to use a simplified log format

    Returns:
        logging.Logger: Configured root logger
    """
⋮----
# Convert level to string if it's an int to match fca_dashboard's API
⋮----
level = logging.getLevelName(level)
⋮----
# Use cast to tell the type checker that this will return a Logger
⋮----
# Fallback to standard logging
⋮----
level = getattr(logging, level.upper(), logging.INFO)
⋮----
# Create logs directory if it doesn't exist and log_file is specified
⋮----
log_dir = os.path.dirname(log_file)
⋮----
# Configure root logger
root_logger = logging.getLogger()
⋮----
# Remove existing handlers to avoid duplicates
⋮----
# Create formatters
⋮----
formatter = logging.Formatter("%(message)s")
⋮----
formatter = logging.Formatter(
⋮----
# Console handler
console_handler = logging.StreamHandler(sys.stdout)
⋮----
# File handler if log_file is specified
⋮----
file_handler = logging.FileHandler(log_file)
⋮----
def get_logger(name: str = "nexusml") -> logging.Logger
⋮----
"""
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
````

## File: nexusml/utils/verification.py
````python
"""
Classifier Verification Script

This script verifies that all necessary components are in place to run the NexusML examples.
It checks for required packages, data files, and module imports.
"""
⋮----
def get_package_version(package_name: str) -> str
⋮----
"""Get the version of a package in a type-safe way.

    Args:
        package_name: Name of the package

    Returns:
        Version string or "unknown" if version cannot be determined
    """
⋮----
# Try to get version directly from the module
module = importlib.import_module(package_name)
⋮----
# Fall back to importlib.metadata
⋮----
# For Python < 3.8
⋮----
def read_csv_safe(filepath: Union[str, Path]) -> DataFrame
⋮----
"""Type-safe wrapper for pd.read_csv.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing the CSV data
    """
# Use type ignore to suppress Pylance warning about complex type
return pd.read_csv(filepath)  # type: ignore
⋮----
def check_package_versions()
⋮----
"""Check if all required packages are installed and compatible."""
⋮----
all_ok = True
⋮----
# Check numpy
⋮----
version = get_package_version("numpy")
⋮----
all_ok = False
⋮----
# Check pandas
⋮----
version = get_package_version("pandas")
⋮----
# Check scikit-learn
⋮----
version = get_package_version("sklearn")
⋮----
# Check matplotlib
⋮----
version = get_package_version("matplotlib")
⋮----
# Check seaborn
⋮----
version = get_package_version("seaborn")
⋮----
# Check imbalanced-learn
⋮----
version = get_package_version("imblearn")
⋮----
def check_data_file()
⋮----
"""Check if the training data file exists."""
# Initialize data_path to None
data_path = None
⋮----
# Try to load from settings
⋮----
# Check if we're running in the context of fca_dashboard
⋮----
settings_path = get_config_path("settings.yml")
⋮----
settings = yaml.safe_load(file)
⋮----
data_path = settings.get("classifier", {}).get("data_paths", {}).get("training_data")
⋮----
# Fallback to default path in nexusml
data_path = "nexusml/ingest/data/eq_ids.csv"
⋮----
# Resolve the path to ensure it exists
data_path = str(resolve_path(data_path))
⋮----
# Not running in fca_dashboard context, use nexusml paths
# Look for a config file in the nexusml directory
settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
⋮----
data_path = settings.get("data_paths", {}).get("training_data")
⋮----
# Use default path in nexusml
data_path = str(Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv")
⋮----
# Use absolute path as fallback
⋮----
df = read_csv_safe(data_path)
⋮----
def check_module_imports()
⋮----
"""Check if all required module imports work correctly."""
⋮----
modules_to_check = [
⋮----
module = importlib.import_module(module_name)
attr = getattr(module, attr_name, None)
⋮----
def main()
⋮----
"""Run all verification checks."""
⋮----
packages_ok = check_package_versions()
data_ok = check_data_file()
imports_ok = check_module_imports()
````
