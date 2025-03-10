#!/bin/bash
# Script to reorganize the NexusML directory structure
# This script copies files from the old structure to the new structure

# Create necessary directories if they don't exist
mkdir -p nexusml/src/models/builders
mkdir -p nexusml/src/models/training
mkdir -p nexusml/src/models/cards
mkdir -p nexusml/src/features/transformers
mkdir -p nexusml/src/pipeline
mkdir -p nexusml/src/utils
mkdir -p nexusml/src/utils/cli
mkdir -p nexusml/src/utils/config
mkdir -p nexusml/src/utils/di
mkdir -p nexusml/src/utils/reference
mkdir -p nexusml/src/utils/validation

# Copy model building files
echo "Copying model building files..."
cp -r nexusml/core/model_building/builders/* nexusml/src/models/builders/
cp nexusml/core/model_building/base.py nexusml/src/models/
cp nexusml/core/model_building/interfaces.py nexusml/src/models/
cp nexusml/core/model_building/compatibility.py nexusml/src/models/
cp nexusml/core/model_building.py nexusml/src/models/model_building.py

# Copy feature engineering files
echo "Copying feature engineering files..."
cp -r nexusml/core/feature_engineering/transformers/* nexusml/src/features/transformers/
cp nexusml/core/feature_engineering/base.py nexusml/src/features/
cp nexusml/core/feature_engineering/interfaces.py nexusml/src/features/
cp nexusml/core/feature_engineering/config_driven.py nexusml/src/features/
cp nexusml/core/feature_engineering/registry.py nexusml/src/features/
cp nexusml/core/feature_engineering/compatibility.py nexusml/src/features/
cp nexusml/core/feature_engineering.py nexusml/src/features/feature_engineering.py

# Copy model training files
echo "Copying model training files..."
cp -r nexusml/core/model_training/trainers/* nexusml/src/models/training/
cp nexusml/core/model_training/base.py nexusml/src/models/training/
cp nexusml/core/model_training/scoring.py nexusml/src/models/training/

# Copy pipeline files
echo "Copying pipeline files..."
cp -r nexusml/core/pipeline/* nexusml/src/pipeline/

# Copy model card files
echo "Copying model card files..."
cp nexusml/core/model_card/generator.py nexusml/src/models/cards/
cp nexusml/core/model_card/model_card.py nexusml/src/models/cards/
cp nexusml/core/model_card/viewer.py nexusml/src/models/cards/
cp nexusml/core/model_card/__init__.py nexusml/src/models/cards/

# Copy CLI files
echo "Copying CLI files..."
cp nexusml/core/cli/prediction_args.py nexusml/src/utils/cli/
cp nexusml/core/cli/training_args.py nexusml/src/utils/cli/
cp nexusml/core/cli/__init__.py nexusml/src/utils/cli/

# Copy config files
echo "Copying config files..."
cp nexusml/core/config/configuration.py nexusml/src/utils/config/
cp nexusml/core/config/migration.py nexusml/src/utils/config/
cp nexusml/core/config/provider.py nexusml/src/utils/config/
cp nexusml/core/config/__init__.py nexusml/src/utils/config/

# Copy DI files
echo "Copying DI files..."
cp nexusml/core/di/container.py nexusml/src/utils/di/
cp nexusml/core/di/decorators.py nexusml/src/utils/di/
cp nexusml/core/di/pipeline_registration.py nexusml/src/utils/di/
cp nexusml/core/di/provider.py nexusml/src/utils/di/
cp nexusml/core/di/registration.py nexusml/src/utils/di/
cp nexusml/core/di/__init__.py nexusml/src/utils/di/

# Copy reference files
echo "Copying reference files..."
cp nexusml/core/reference/base.py nexusml/src/utils/reference/
cp nexusml/core/reference/classification.py nexusml/src/utils/reference/
cp nexusml/core/reference/equipment.py nexusml/src/utils/reference/
cp nexusml/core/reference/glossary.py nexusml/src/utils/reference/
cp nexusml/core/reference/manager.py nexusml/src/utils/reference/
cp nexusml/core/reference/manufacturer.py nexusml/src/utils/reference/
cp nexusml/core/reference/service_life.py nexusml/src/utils/reference/
cp nexusml/core/reference/validation.py nexusml/src/utils/reference/
cp nexusml/core/reference/__init__.py nexusml/src/utils/reference/

# Copy validation files
echo "Copying validation files..."
cp nexusml/core/validation/adapters.py nexusml/src/utils/validation/
cp nexusml/core/validation/interfaces.py nexusml/src/utils/validation/
cp nexusml/core/validation/rules.py nexusml/src/utils/validation/
cp nexusml/core/validation/validators.py nexusml/src/utils/validation/
cp nexusml/core/validation/__init__.py nexusml/src/utils/validation/

# Copy utility files
echo "Copying utility files..."
cp nexusml/core/data_mapper.py nexusml/src/data/mapper.py
cp nexusml/core/data_preprocessing.py nexusml/src/data/preprocessor.py
cp nexusml/core/reference_manager.py nexusml/src/utils/
cp nexusml/core/evaluation.py nexusml/src/utils/
cp nexusml/core/dynamic_mapper.py nexusml/src/utils/
cp nexusml/core/eav_manager.py nexusml/src/utils/
cp nexusml/core/model.py nexusml/src/models/model.py

echo "File reorganization complete!"