# NexusML Model Pipeline Architecture

This document provides an overview of the NexusML model pipeline architecture,
which is used for equipment classification in the FCA Dashboard project.

## Overview

NexusML is a standalone Python package that provides machine learning
capabilities for classifying equipment based on descriptions and other features.
It was extracted from the FCA Dashboard project to enable independent
development and reuse.

The model pipeline consists of several key components:

1. **Data Ingestion**: Entry points for loading and processing data
2. **Data Preprocessing**: Cleaning and preparing data for the model
3. **Feature Engineering**: Transforming raw data into features for the model
4. **Model Building**: Creating and configuring the machine learning model
5. **Training**: Training the model on the prepared data
6. **Evaluation**: Assessing model performance
7. **Prediction**: Using the trained model to make predictions
8. **EAV Integration**: Managing equipment attributes using an
   Entity-Attribute-Value structure

## Architecture Diagrams

### Model Pipeline Architecture

![Model Pipeline Architecture](output/nexusml/model_pipeline.png)

This diagram shows the main components of the NexusML model pipeline and their
relationships. The core components include:

- **model.py**: Central class that orchestrates the entire classification
  process
- **data_mapper.py**: Maps data between different formats
- **dynamic_mapper.py**: Flexibly maps input fields to expected format
- **eav_manager.py**: Manages the Entity-Attribute-Value structure
- **feature_engineering.py**: Transforms raw data into features
- **model_building.py**: Creates and configures the machine learning model
- **evaluation.py**: Assesses model performance

Entry points include:

- **classify_equipment.py**: For classifying equipment from any input file
- **predict.py**: For making predictions with a trained model
- **train_model_pipeline.py**: For training a new model

### Class Diagram

![Class Diagram](output/nexusml/class_diagram.png)

This diagram shows the classes in the NexusML model pipeline and their
relationships. Key classes include:

- **EquipmentClassifier**: Main class for equipment classification
- **DataMapper**: Maps data between different formats
- **DynamicFieldMapper**: Flexibly maps input fields to expected format
- **EAVManager**: Manages the Entity-Attribute-Value structure
- **GenericFeatureEngineer**: Applies feature engineering transformations
- Various transformer classes for specific feature engineering tasks

### Training Sequence

![Training Sequence](output/nexusml/training_sequence.png)

This diagram shows the sequence of operations during model training:

1. Load and preprocess data
2. Map staging data to model input format
3. Apply feature engineering
4. Split data into training and test sets
5. Build the model pipeline
6. Train the model
7. Evaluate the model
8. Save the model and metadata

### Prediction Sequence

![Prediction Sequence](output/nexusml/prediction_sequence.png)

This diagram shows the sequence of operations during prediction:

1. Load the trained model
2. Read input data
3. Map data to model input format
4. Apply feature engineering
5. For each row, make a prediction
6. Get classification IDs, performance fields, and required attributes
7. Map predictions to master database fields
8. Generate attribute templates
9. Save prediction results

### Data Flow

![Data Flow](output/nexusml/data_flow.png)

This diagram shows the flow of data through the model pipeline:

1. Input data (CSV, Excel, etc.)
2. Data preprocessing
3. Data mapping
4. Feature engineering
5. Data splitting
6. Model building
7. Model training
8. Model evaluation
9. Feature analysis
10. Misclassification analysis
11. Model saving

### EAV Structure

![EAV Structure](output/nexusml/eav_structure.png)

This diagram shows the Entity-Attribute-Value (EAV) structure used for equipment
attributes:

- **Equipment**: Represents a physical piece of equipment
- **EquipmentCategory**: Represents a type of equipment
- **EquipmentAttribute**: Stores attribute values for specific equipment
- **AttributeTemplate**: Defines expected attributes for each equipment category
- **ClassificationSystem**: Represents hierarchical classification systems
- **EAVManager**: Provides an interface to work with the EAV structure

## Key Components

### Data Mapping

The data mapping components (DataMapper and DynamicFieldMapper) handle mapping
between different data formats:

- **DataMapper**: Maps between staging data and ML model input, and from ML
  model output to master database fields
- **DynamicFieldMapper**: Flexibly maps input fields to expected format using
  pattern matching

### Feature Engineering

The feature engineering components transform raw data into features for the
model:

- **GenericFeatureEngineer**: Applies multiple transformations based on a
  configuration file
- **TextCombiner**: Combines multiple text columns into one
- **NumericCleaner**: Cleans and transforms numeric columns
- **HierarchyBuilder**: Creates hierarchical category columns
- **ColumnMapper**: Maps source columns to target columns
- **KeywordClassificationMapper**: Maps descriptions to classification system
  IDs using keywords
- **ClassificationSystemMapper**: Maps equipment categories to classification
  system IDs

### Model Building

The model building components create and configure the machine learning model:

- **build_enhanced_model()**: Creates a pipeline with feature engineering,
  preprocessing, and classification
- **optimize_hyperparameters()**: Optimizes model hyperparameters using grid
  search

### EAV Integration

The EAV integration components manage equipment attributes:

- **EAVManager**: Manages attribute templates, validation, and generation
- **EAVTransformer**: Adds EAV attributes to the feature set

## Conclusion

The NexusML model pipeline provides a comprehensive solution for equipment
classification, with flexible data mapping, powerful feature engineering, and
integrated EAV structure for equipment attributes. The modular architecture
allows for easy extension and customization.
