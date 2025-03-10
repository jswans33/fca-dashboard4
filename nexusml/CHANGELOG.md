# Changelog

All notable changes to the NexusML project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Completed the immediate improvements document with detailed implementation
  plan
- Added comprehensive project template guidelines for new projects
- Created migration scripts for consolidating output directories and organizing
  examples (excluding fca_dashboard)
- Added script to identify and remove duplicate files (examples and outputs)
- Added script to update references to old output directories in Python files
- Added comprehensive README.md to the scripts directory documenting all utility
  scripts with `remove_duplicates.py`
- Added default output configurations to nexusml_config.yml
- Added JSON schema validation for configuration files
- Implemented environment variable override functionality for all configuration
  options
- Added validation for environment variables used for configuration overrides
- Created comprehensive configuration guide with documentation of all options
- Added test scripts for environment variable overrides and validation

### Fixed

- Corrected section numbering inconsistencies in the immediate improvements
  document
- Improved document structure and organization for better readability
- Fixed confusion matrix warnings by adding explicit labels parameter
- Standardized output directory paths across all scripts to use nexusml/output/
- Updated train_model_pipeline.py and predict.py to use standardized paths
