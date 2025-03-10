# Mechanical Equipment Classification Project

This project implements a production-ready pipeline for classifying mechanical equipment using NexusML, based on the production configuration in `production_nexusml_config.yml`.

## Project Structure

```
mech_equipment_classifier/
├── config/                                # Configuration files
│   ├── mech_equipment_config.yml          # Main configuration file
│   ├── feature_config.yml                 # Feature engineering configuration
│   ├── classification_config.yml          # Classification configuration
│   ├── data_config.yml                    # Data configuration
│   ├── reference_config.yml               # Reference data configuration
│   └── model_config.yml                   # Model configuration
├── data/                                  # Data directory
│   ├── raw/                               # Raw data files
│   ├── processed/                         # Processed data files
│   └── reference/                         # Reference data
│       ├── omniclass/                     # OmniClass reference data
│       ├── uniformat/                     # Uniformat reference data
│       ├── masterformat/                  # MasterFormat reference data
│       ├── mcaa-glossary/                 # MCAA glossary and abbreviations
│       ├── smacna-manufacturers/          # SMACNA manufacturers list
│       ├── service-life/                  # Service life reference data
│       │   ├── ashrae/                    # ASHRAE service life data
│       │   └── energize-denver/           # Energize Denver service life data
│       └── equipment-taxonomy/            # Equipment taxonomy reference data
├── models/                                # Trained models
│   ├── baseline/                          # Baseline models
│   └── production/                        # Production models
├── notebooks/                             # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb             # Data exploration notebook
│   ├── feature_engineering.ipynb          # Feature engineering notebook
│   ├── model_training.ipynb               # Model training notebook
│   └── model_evaluation.ipynb             # Model evaluation notebook
├── outputs/                               # Output files
│   ├── predictions/                       # Prediction results
│   ├── evaluation/                        # Evaluation results
│   └── model_cards/                       # Model cards
├── scripts/                               # Utility scripts
│   ├── data_preparation.py                # Data preparation script
│   ├── train_model.py                     # Model training script
│   └── make_predictions.py                # Prediction script
├── src/                                   # Source code
│   ├── data/                              # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loader.py                      # Data loader implementation
│   │   └── validator.py                   # Data validator implementation
│   ├── features/                          # Feature engineering
│   │   ├── __init__.py
│   │   ├── text_features.py               # Text feature engineering
│   │   ├── numeric_features.py            # Numeric feature engineering
│   │   ├── categorical_features.py        # Categorical feature engineering
│   │   └── hierarchical_features.py       # Hierarchical feature engineering
│   ├── models/                            # Model implementations
│   │   ├── __init__.py
│   │   ├── baseline.py                    # Baseline model implementation
│   │   └── classifier.py                  # Main classifier implementation
│   ├── pipeline/                          # Pipeline components
│   │   ├── __init__.py
│   │   ├── stages.py                      # Pipeline stages
│   │   └── orchestrator.py                # Pipeline orchestrator
│   └── utils/                             # Utility functions
│       ├── __init__.py
│       ├── logging.py                     # Logging utilities
│       └── evaluation.py                  # Evaluation utilities
├── tests/                                 # Tests
│   ├── __init__.py
│   ├── test_data.py                       # Data tests
│   ├── test_features.py                   # Feature tests
│   ├── test_models.py                     # Model tests
│   └── test_pipeline.py                   # Pipeline tests
├── .env.example                           # Example environment variables
├── .gitignore                             # Git ignore file
├── requirements.txt                       # Project dependencies
├── setup.py                               # Package setup file
└── README.md                              # Project README
```

## Data Schema

Based on the `production_nexusml_config.yml` file, the mechanical equipment classification pipeline uses the following data schema:

### Input Data Schema

| Column Name | Data Type | Description | Required | Default Value |
|-------------|-----------|-------------|----------|---------------|
| equipment_tag | string | Equipment tag or identifier | Yes | '' |
| manufacturer | string | Equipment manufacturer | Yes | '' |
| model | string | Equipment model number | Yes | '' |
| category_name | string | Equipment category name | Yes | '' |
| omniclass_code | string | OmniClass classification code | Yes | '' |
| uniformat_code | string | Uniformat classification code | Yes | '' |
| masterformat_code | string | MasterFormat classification code | Yes | '' |
| mcaa_system_category | string | MCAA system category | Yes | '' |
| CategoryID | integer | Category ID | Yes | 0 |
| OmniClassID | integer | OmniClass ID | Yes | 0 |
| UniFormatID | integer | Uniformat ID | Yes | 0 |
| MasterFormatID | integer | MasterFormat ID | Yes | 0 |
| MCAAID | string | MCAA ID | Yes | '' |
| LocationID | integer | Location ID | Yes | 0 |
| service_life | float | Expected service life in years | Yes | 0.0 |
| Precon_System | string | Precon system | Yes | '' |
| Drawing_Abbreviation | string | Drawing abbreviation | Yes | '' |
| Precon_Tag | string | Precon tag | Yes | '' |
| System_Type_ID | string | System type ID | Yes | '' |
| Equip_Name_ID | string | Equipment name ID | Yes | '' |
| Sub_System_ID | string | Sub-system ID | Yes | '' |
| Sub_System_Class | string | Sub-system class | Yes | '' |
| Class_ID | string | Class ID | Yes | '' |
| Unit | string | Unit of measurement | Yes | '' |

### Derived Features

| Column Name | Data Type | Description | Source |
|-------------|-----------|-------------|--------|
| combined_text | string | Combined text from multiple fields | Text combination of equipment_tag, manufacturer, model, category_name, mcaa_system_category |
| Equipment_Type | string | Hierarchical equipment type | Hierarchy of mcaa_system_category and category_name |
| System_Subtype | string | Hierarchical system subtype | Hierarchy of mcaa_system_category and category_name |
| Equipment_Category | string | Equipment category | Mapped from category_name |
| Uniformat_Class | string | Uniformat classification | Mapped from uniformat_code |
| System_Type | string | System type | Mapped from mcaa_system_category |
| Equipment_Subcategory | string | Equipment subcategory | Mapped from CategoryID |
| OmniClass_ID | string | OmniClass ID | Mapped from OmniClassID |
| Uniformat_ID | string | Uniformat ID | Mapped from UniFormatID |
| MasterFormat_ID | string | MasterFormat ID | Mapped from MasterFormatID |
| MCAA_ID | string | MCAA ID | Mapped from MCAAID |
| Location_ID | integer | Location ID | Mapped from LocationID |

### Output Data Schema (Classification Targets)

| Column Name | Data Type | Description | Required |
|-------------|-----------|-------------|----------|
| Equipment_Category | string | Primary equipment type (e.g., Chiller, Pump, Air Handler) | Yes |
| Uniformat_Class | string | Uniformat classification code (e.g., D3040, D2010) | Yes |
| System_Type | string | System type (e.g., HVAC, Plumbing) | Yes |
| Equipment_Type | string | Hierarchical equipment type | Yes |
| System_Subtype | string | Hierarchical system subtype | Yes |
| MasterFormat_Class | string | MasterFormat classification code | No |

## Feature Engineering

The pipeline uses the following feature engineering techniques based on the configuration:

### Text Features

- **Combined Text**: Combines equipment_tag, manufacturer, model, category_name, and mcaa_system_category into a single text field for text-based classification.
- **Text Vectorization**: Uses TF-IDF vectorization with n-gram range of 1-3.

### Hierarchical Features

- **Equipment_Type**: Hierarchical combination of mcaa_system_category and category_name with '-' separator.
- **System_Subtype**: Hierarchical combination of mcaa_system_category and category_name with '-' separator.

### Direct Mappings

- category_name → Equipment_Category
- uniformat_code → Uniformat_Class
- mcaa_system_category → System_Type
- CategoryID → Equipment_Subcategory
- OmniClassID → OmniClass_ID
- UniFormatID → Uniformat_ID
- MasterFormatID → MasterFormat_ID
- MCAAID → MCAA_ID
- LocationID → Location_ID

### Classification Systems

- **OmniClass**: Direct mapping from omniclass_code to OmniClass_ID
- **MasterFormat**: Direct mapping from masterformat_code to MasterFormat_ID
- **Uniformat**: Direct mapping from uniformat_code to Uniformat_ID

## Model Configuration

The model configuration is based on the production configuration:

### Architecture

- **Type**: RandomForestClassifier
- **Text Vectorizer**: TfidfVectorizer
- **N-gram Range**: [1, 3]

### Hyperparameters

- **n_estimators**: 100
- **max_depth**: 20
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **class_weight**: 'balanced'

### Features

- **Text Features**: combined_text
- **Numeric Features**: service_life
- **Categorical Features**: Equipment_Category, System_Type

### Classification Targets

- Equipment_Category
- Uniformat_Class
- System_Type
- Equipment_Type
- System_Subtype
- MasterFormat_Class

## Reference Data

The pipeline uses the following reference data sources:

- **OmniClass**: Classification system for the built environment
- **Uniformat**: Classification system for building elements
- **MasterFormat**: Classification system for construction specifications
- **MCAA Glossary**: Mechanical Contractors Association of America glossary
- **SMACNA Manufacturers**: Sheet Metal and Air Conditioning Contractors' National Association manufacturers list
- **ASHRAE Service Life**: American Society of Heating, Refrigerating and Air-Conditioning Engineers service life data
- **Energize Denver**: Energize Denver service life data
- **Equipment Taxonomy**: Equipment taxonomy reference data

## Getting Started

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure environment variables
6. Run the data preparation script: `python scripts/data_preparation.py`
7. Train the model: `python scripts/train_model.py`
8. Make predictions: `python scripts/make_predictions.py --input path/to/input.csv --output path/to/output.csv`

## Configuration

The project uses a modular configuration approach with separate configuration files for different aspects of the pipeline:

- **mech_equipment_config.yml**: Main configuration file that imports other configuration files
- **feature_config.yml**: Feature engineering configuration
- **classification_config.yml**: Classification configuration
- **data_config.yml**: Data configuration
- **reference_config.yml**: Reference data configuration
- **model_config.yml**: Model configuration

## Example Configuration

Here's an example of the main configuration file based on the production configuration:

```yaml
# Mechanical Equipment Classification Configuration

feature_engineering:
  # Text combinations
  text_combinations:
    - name: 'combined_text'
      columns:
        [
          'equipment_tag',
          'manufacturer',
          'model',
          'category_name',
          'mcaa_system_category',
        ]
      separator: ' '
  
  # Hierarchies
  hierarchies:
    - new_col: 'Equipment_Type'
      parents: ['mcaa_system_category', 'category_name']
      separator: '-'
    
    - new_col: 'System_Subtype'
      parents: ['mcaa_system_category', 'category_name']
      separator: '-'
  
  # Column mappings
  column_mappings:
    - source: 'category_name'
      target: 'Equipment_Category'
    
    - source: 'uniformat_code'
      target: 'Uniformat_Class'
    
    - source: 'mcaa_system_category'
      target: 'System_Type'
  
  # Classification systems
  classification_systems:
    - name: 'OmniClass'
      source_column: 'omniclass_code'
      target_column: 'OmniClass_ID'
      mapping_type: 'direct'
    
    - name: 'MasterFormat'
      source_column: 'masterformat_code'
      target_column: 'MasterFormat_ID'
      mapping_type: 'direct'
    
    - name: 'Uniformat'
      source_column: 'uniformat_code'
      target_column: 'Uniformat_ID'
      mapping_type: 'direct'

classification:
  # Classification targets
  classification_targets:
    - name: 'Equipment_Category'
      description: 'Primary equipment type (e.g., Chiller, Pump, Air Handler)'
      required: true
    
    - name: 'Uniformat_Class'
      description: 'Uniformat classification code (e.g., D3040, D2010)'
      required: true
    
    - name: 'System_Type'
      description: 'System type (e.g., HVAC, Plumbing)'
      required: true
    
    - name: 'MasterFormat_Class'
      description: 'MasterFormat classification code'
      required: false

model:
  # Model architecture
  architecture:
    type: 'RandomForestClassifier'
    text_vectorizer: 'TfidfVectorizer'
    ngram_range: [1, 3]
    hyperparameters:
      n_estimators: 100
      max_depth: 20
      min_samples_split: 2
      min_samples_leaf: 1
      class_weight: 'balanced'
  
  # Features
  features:
    text_features:
      - 'combined_text'
    numeric_features:
      - 'service_life'
    categorical_features:
      - 'Equipment_Category'
      - 'System_Type'
  
  # Classification targets
  targets:
    - 'Equipment_Category'
    - 'Uniformat_Class'
    - 'System_Type'
    - 'Equipment_Type'
    - 'System_Subtype'
    - 'MasterFormat_Class'