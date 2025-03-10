# Mechanical Equipment Classifier Setup Guide

This guide provides instructions for setting up the Mechanical Equipment Classifier project using the NexusML framework.

## Project Overview

The Mechanical Equipment Classifier is a specialized application of the NexusML framework for classifying mechanical equipment based on descriptions and metadata. It leverages the core components of NexusML while implementing custom components specific to mechanical equipment classification.

## Prerequisites

- Python 3.8 or higher
- NexusML package installed
- Access to reference data (OmniClass, Uniformat, MasterFormat, etc.)
- Training data for mechanical equipment

## Project Structure

The project follows this structure:

```
mech_equipment_classifier/
├── config/                  # Configuration files
├── data/                    # Data directory
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks
├── outputs/                 # Output files
├── scripts/                 # Utility scripts
├── src/                     # Source code
├── tests/                   # Tests
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup file
└── README.md                # Project README
```

## Components Breakdown

### Components to Import from NexusML

The project leverages these existing components from the NexusML package:

1. **Core Pipeline Architecture**
   - `nexusml.core.pipeline.orchestrator.PipelineOrchestrator`
   - `nexusml.core.pipeline.factory.PipelineFactory`
   - `nexusml.core.pipeline.registry.ComponentRegistry`
   - `nexusml.core.pipeline.context.PipelineContext`
   - `nexusml.core.pipeline.interfaces.*`
   - `nexusml.core.pipeline.stages.*`

2. **Dependency Injection System**
   - `nexusml.core.di.container.DIContainer`
   - `nexusml.core.di.decorators.inject`

3. **Base Configuration System**
   - `nexusml.config.manager.ConfigurationManager`
   - `nexusml.config.provider.ConfigProvider`

4. **Feature Engineering Base Classes**
   - `nexusml.core.feature_engineering.BaseFeatureTransformer`
   - `nexusml.core.feature_engineering.BaseColumnTransformer`
   - `nexusml.core.feature_engineering.BaseFeatureEngineer`

5. **Model Building Base Classes**
   - `nexusml.core.model_building.base.BaseModelBuilder`
   - `nexusml.core.model_building.base.BaseModelTrainer`
   - `nexusml.core.model_building.base.BaseModelEvaluator`
   - `nexusml.core.model_building.base.BaseModelSerializer`

6. **Validation Framework**
   - `nexusml.core.validation.*`

7. **Reference Data Management**
   - `nexusml.core.reference.manager.ReferenceManager`

### Components to Create in the Project

The project implements these custom components:

1. **Project-Specific Configuration Files**
   - `config/mech_equipment_config.yml`: Main configuration file
   - `config/feature_config.yml`: Feature engineering configuration
   - `config/classification_config.yml`: Classification configuration
   - `config/data_config.yml`: Data configuration
   - `config/reference_config.yml`: Reference data configuration
   - `config/model_config.yml`: Model configuration

2. **Custom Data Loaders**
   - `src/data/loader.py`: Custom data loader for mechanical equipment data

3. **Custom Feature Engineering**
   - `src/features/text_features.py`: Text feature engineering for equipment descriptions
   - `src/features/numeric_features.py`: Numeric feature engineering for equipment attributes
   - `src/features/categorical_features.py`: Categorical feature engineering for equipment types
   - `src/features/hierarchical_features.py`: Hierarchical feature engineering for equipment taxonomy

4. **Custom Model Implementations**
   - `src/models/classifier.py`: Custom classifier for mechanical equipment

5. **Project-Specific Scripts**
   - `scripts/data_preparation.py`: Data preparation script
   - `scripts/train_model.py`: Model training script
   - `scripts/make_predictions.py`: Prediction script

6. **Domain-Specific Utilities**
   - `src/utils/equipment_utils.py`: Utilities specific to mechanical equipment

## Setup Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-org/nexusml.git
   cd nexusml/projects/mech_equipment_classifier
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**

   Copy the example environment file and update it with your settings:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to set:
   - API keys (if needed)
   - Database connection strings
   - File paths

6. **Prepare Reference Data**

   Place reference data files in the appropriate directories:

   ```bash
   mkdir -p data/reference/omniclass
   mkdir -p data/reference/uniformat
   mkdir -p data/reference/masterformat
   mkdir -p data/reference/mcaa-glossary
   mkdir -p data/reference/smacna-manufacturers
   mkdir -p data/reference/service-life/ashrae
   mkdir -p data/reference/service-life/energize-denver
   mkdir -p data/reference/equipment-taxonomy
   ```

   Copy reference data files to these directories.

7. **Prepare Training Data**

   Place training data in the data directory:

   ```bash
   mkdir -p data/raw
   mkdir -p data/processed
   ```

   Copy training data files to the `data/raw` directory.

8. **Run Data Preparation**

   ```bash
   python scripts/data_preparation.py
   ```

9. **Train the Model**

   ```bash
   python scripts/train_model.py
   ```

10. **Make Predictions**

    ```bash
    python scripts/make_predictions.py --input path/to/input.csv --output path/to/output.csv
    ```

## Development Workflow

1. **Update Configuration**

   Modify the configuration files in the `config` directory to adjust:
   - Feature engineering settings
   - Classification targets
   - Data requirements
   - Model parameters

2. **Implement Custom Components**

   Implement custom components in the `src` directory:
   - Custom data loaders in `src/data`
   - Custom feature engineering in `src/features`
   - Custom models in `src/models`
   - Custom utilities in `src/utils`

3. **Create Scripts**

   Create scripts in the `scripts` directory for common tasks:
   - Data preparation
   - Model training
   - Prediction
   - Evaluation

4. **Write Tests**

   Write tests in the `tests` directory to verify:
   - Data loading and validation
   - Feature engineering
   - Model training and evaluation
   - End-to-end pipeline

5. **Document**

   Update documentation to reflect changes:
   - Update README.md
   - Update configuration examples
   - Add usage examples

## Integration with NexusML

The project integrates with NexusML by:

1. **Importing Core Components**

   ```python
   from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
   from nexusml.core.pipeline.factory import PipelineFactory
   from nexusml.core.pipeline.registry import ComponentRegistry
   from nexusml.core.pipeline.context import PipelineContext
   ```

2. **Registering Custom Components**

   ```python
   registry = ComponentRegistry()
   registry.register(DataLoader, "mech_equipment", MechEquipmentDataLoader)
   registry.set_default_implementation(DataLoader, "mech_equipment")
   ```

3. **Creating a Pipeline**

   ```python
   factory = PipelineFactory(registry, container)
   context = PipelineContext()
   orchestrator = PipelineOrchestrator(factory, context)
   ```

4. **Using the Pipeline**

   ```python
   model, metrics = orchestrator.train_model(
       data_path="data/processed/training_data.csv",
       feature_config_path="config/feature_config.yml",
       output_dir="models/production",
       model_name="equipment_classifier",
   )
   ```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   If you encounter missing dependencies, install them with:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration Errors**

   If you encounter configuration errors, check:
   - YAML syntax in configuration files
   - Required fields in configuration files
   - File paths in configuration files

3. **Data Loading Errors**

   If you encounter data loading errors, check:
   - File paths
   - File formats
   - Required columns
   - Data types

4. **Model Training Errors**

   If you encounter model training errors, check:
   - Feature engineering configuration
   - Model parameters
   - Training data quality
   - Memory usage

### Getting Help

If you need help, check:
- NexusML documentation
- Project README
- Issue tracker
- Community forums

## Next Steps

After setting up the project, you can:

1. **Explore the Data**
   - Use the notebooks in the `notebooks` directory to explore the data
   - Analyze the distribution of equipment types
   - Identify patterns in the data

2. **Experiment with Features**
   - Try different feature engineering techniques
   - Add new features
   - Remove irrelevant features

3. **Optimize the Model**
   - Try different model architectures
   - Tune hyperparameters
   - Use cross-validation

4. **Deploy the Model**
   - Create a deployment script
   - Set up a web service
   - Integrate with existing systems