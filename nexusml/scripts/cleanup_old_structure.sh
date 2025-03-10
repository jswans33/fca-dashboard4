#!/bin/bash
# Script to clean up the old directory structure after reorganization
# This script removes files from the old structure that have been moved to the new structure

# Confirm before proceeding
echo "This script will remove files from the old directory structure."
echo "Make sure you have verified that the new structure is working correctly."
read -p "Are you sure you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

# Remove model building files
echo "Removing model building files..."
rm -rf nexusml/core/model_building/builders
rm -f nexusml/core/model_building/base.py
rm -f nexusml/core/model_building/interfaces.py
rm -f nexusml/core/model_building/compatibility.py
rm -f nexusml/core/model_building.py

# Remove feature engineering files
echo "Removing feature engineering files..."
rm -rf nexusml/core/feature_engineering/transformers
rm -f nexusml/core/feature_engineering/base.py
rm -f nexusml/core/feature_engineering/interfaces.py
rm -f nexusml/core/feature_engineering/config_driven.py
rm -f nexusml/core/feature_engineering/registry.py
rm -f nexusml/core/feature_engineering/compatibility.py
rm -f nexusml/core/feature_engineering.py

# Remove model training files
echo "Removing model training files..."
rm -rf nexusml/core/model_training/trainers
rm -f nexusml/core/model_training/base.py
rm -f nexusml/core/model_training/scoring.py

# Remove pipeline files
echo "Removing pipeline files..."
rm -rf nexusml/core/pipeline/adapters
rm -rf nexusml/core/pipeline/components
rm -rf nexusml/core/pipeline/pipelines
rm -rf nexusml/core/pipeline/stages
rm -f nexusml/core/pipeline/adapters.py
rm -f nexusml/core/pipeline/base.py
rm -f nexusml/core/pipeline/context.py
rm -f nexusml/core/pipeline/factory.py
rm -f nexusml/core/pipeline/interfaces.py
rm -f nexusml/core/pipeline/orchestrator.py
rm -f nexusml/core/pipeline/plugins.py
rm -f nexusml/core/pipeline/registry.py
rm -f nexusml/core/pipeline/README.md

# Remove model card files
echo "Removing model card files..."
rm -f nexusml/core/model_card/generator.py
rm -f nexusml/core/model_card/model_card.py
rm -f nexusml/core/model_card/viewer.py
rm -f nexusml/core/model_card/__init__.py
rm -rf nexusml/core/model_card

# Remove CLI files
echo "Removing CLI files..."
rm -f nexusml/core/cli/prediction_args.py
rm -f nexusml/core/cli/training_args.py
rm -f nexusml/core/cli/__init__.py
rm -rf nexusml/core/cli

# Remove config files
echo "Removing config files..."
rm -f nexusml/core/config/configuration.py
rm -f nexusml/core/config/migration.py
rm -f nexusml/core/config/provider.py
rm -f nexusml/core/config/__init__.py
rm -rf nexusml/core/config

# Remove DI files
echo "Removing DI files..."
rm -f nexusml/core/di/container.py
rm -f nexusml/core/di/decorators.py
rm -f nexusml/core/di/pipeline_registration.py
rm -f nexusml/core/di/provider.py
rm -f nexusml/core/di/registration.py
rm -f nexusml/core/di/__init__.py
rm -rf nexusml/core/di

# Remove reference files
echo "Removing reference files..."
rm -f nexusml/core/reference/base.py
rm -f nexusml/core/reference/classification.py
rm -f nexusml/core/reference/equipment.py
rm -f nexusml/core/reference/glossary.py
rm -f nexusml/core/reference/manager.py
rm -f nexusml/core/reference/manufacturer.py
rm -f nexusml/core/reference/service_life.py
rm -f nexusml/core/reference/validation.py
rm -f nexusml/core/reference/__init__.py
rm -rf nexusml/core/reference

# Remove validation files
echo "Removing validation files..."
rm -f nexusml/core/validation/adapters.py
rm -f nexusml/core/validation/interfaces.py
rm -f nexusml/core/validation/rules.py
rm -f nexusml/core/validation/validators.py
rm -f nexusml/core/validation/__init__.py
rm -rf nexusml/core/validation

# Remove utility files
echo "Removing utility files..."
rm -f nexusml/core/data_mapper.py
rm -f nexusml/core/data_preprocessing.py
rm -f nexusml/core/reference_manager.py
rm -f nexusml/core/evaluation.py
rm -f nexusml/core/dynamic_mapper.py
rm -f nexusml/core/eav_manager.py
rm -f nexusml/core/model.py
rm -f nexusml/core/__init__.py

# Remove empty directories
echo "Removing empty directories..."
find nexusml/core -type d -empty -delete

echo "Cleanup complete!"