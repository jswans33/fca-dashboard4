# NexusML Core Modules Documentation

This directory contains documentation for the core modules of the NexusML package. These modules provide the fundamental functionality for equipment classification and machine learning.

## Overview

The core modules in NexusML handle various aspects of the machine learning pipeline, from data preprocessing to model building and evaluation. Each module is designed to be modular, extensible, and well-documented.

## Core Modules

- [data_mapper.py](data_mapper.md): Maps data between different formats and schemas
- [data_preprocessing.py](data_preprocessing.py): Cleans and prepares data for machine learning
- [dynamic_mapper.py](dynamic_mapper.md): Provides dynamic mapping capabilities for flexible data transformation
- [eav_manager.py](eav_manager.md): Manages Entity-Attribute-Value (EAV) data structures for equipment attributes
- [evaluation.py](evaluation.md): Evaluates model performance with comprehensive metrics
- [feature_engineering.py](feature_engineering.md): Transforms raw data into features suitable for machine learning
- [model_building.py](model_building.md): Creates and configures machine learning models
- [model.py](model.md): Provides the core model functionality for equipment classification
- [reference_manager.py](reference_manager.md): Manages reference data for classification systems

## Key Concepts

### Data Mapping

Data mapping is the process of transforming data from one format or schema to another. NexusML provides several mapping mechanisms:

- Static mapping using predefined rules
- Dynamic mapping using runtime configuration
- EAV mapping for flexible attribute structures

### Data Preprocessing

Data preprocessing involves cleaning and preparing data for machine learning. NexusML provides preprocessing capabilities for:

- Handling missing values
- Removing duplicates
- Standardizing formats
- Validating data quality

### Feature Engineering

Feature engineering is the process of transforming raw data into features suitable for machine learning. NexusML provides feature engineering capabilities for:

- Text data (TF-IDF, word embeddings)
- Numerical data (scaling, normalization)
- Categorical data (one-hot encoding, target encoding)
- Custom feature transformations

### Model Building

Model building involves creating and configuring machine learning models. NexusML supports various model types:

- Random Forest
- Gradient Boosting
- Support Vector Machines
- Neural Networks
- Ensemble models

### Evaluation

Evaluation involves assessing model performance using various metrics. NexusML provides comprehensive evaluation capabilities:

- Accuracy, precision, recall, F1 score
- Confusion matrix
- ROC curve and AUC
- Cross-validation
- Feature importance analysis

## Integration with Other Components

The core modules integrate with other components in NexusML:

- They use the configuration system for flexible configuration
- They leverage the dependency injection system for component management
- They are orchestrated by the pipeline system for end-to-end workflows
- They are exposed through the CLI tools for command-line usage

## Next Steps

After reviewing the core modules documentation, you might want to:

1. Explore the [Examples](../examples/README.md) for practical usage examples
2. Check the [API Reference](../api_reference.md) for detailed information on classes and methods
3. Read the [Architecture Documentation](../architecture/README.md) for a deeper understanding of the system design