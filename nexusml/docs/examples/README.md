# NexusML Examples

This directory contains documentation for the various examples provided in the NexusML package. These examples demonstrate different aspects of the package and how to use them effectively.

## Overview

NexusML provides a comprehensive set of examples to help users understand how to use the package for various tasks related to equipment classification and machine learning. The examples are organized into the following categories:

1. **Basic Examples**: Simple examples demonstrating core functionality
2. **Data Loading Examples**: Examples showing how to load data from various sources
3. **Feature Engineering Examples**: Examples demonstrating feature transformation techniques
4. **Model Building Examples**: Examples showing how to build and train models
5. **Pipeline Examples**: Examples demonstrating the pipeline architecture
6. **Domain-Specific Examples**: Examples focused on specific domains like OmniClass and Uniformat

## Example Categories

### Basic Examples

- [Simple Example](simple_example.md): Basic usage of NexusML
- [Advanced Example](advanced_example.md): Advanced usage with custom components
- [Random Guessing](random_guessing.md): Simple baseline model using random guessing

### Data Loading Examples

- [Data Loader Example](data_loading_examples.md#basic-data-loader-example): Finding and loading data files from different locations
- [Enhanced Data Loader Example](data_loading_examples.md#enhanced-data-loader-example): Using the StandardDataLoader for automatic discovery
- [Staging Data Example](data_loading_examples.md#staging-data-example): Working with staging data that has different column names

### Feature Engineering Examples

- [Feature Engineering Example](feature_engineering_examples.md): Transforming raw data into features suitable for machine learning

### Model Building Examples

- [Model Building Example](model_building_examples.md#model-building-example): Building, training, and evaluating machine learning models
- [Training Pipeline Example](model_building_examples.md#training-pipeline-example): Using the training pipeline with various configurations

### Pipeline Examples

- [Pipeline Factory Example](pipeline_examples.md#pipeline-factory-example): Creating and configuring pipeline components
- [Pipeline Orchestrator Example](pipeline_examples.md#pipeline-orchestrator-example): Orchestrating complete machine learning workflows
- [Pipeline Stages Example](pipeline_examples.md#pipeline-stages-example): Using pipeline stages for a complete ML pipeline
- [Integrated Classifier Example](pipeline_examples.md#integrated-classifier-example): Comprehensive equipment classification model

### Domain-Specific Examples

- [OmniClass Generator Example](domain_specific_examples.md#omniclass-generator-example): Extracting OmniClass data and generating descriptions
- [OmniClass Hierarchy Example](domain_specific_examples.md#omniclass-hierarchy-example): Visualizing OmniClass data in a hierarchical structure
- [Uniformat Keywords Example](domain_specific_examples.md#uniformat-keywords-example): Finding Uniformat codes by keyword
- [Validation Example](domain_specific_examples.md#validation-example): Validating data quality and integrity

## Running the Examples

Most examples can be run directly using Python:

```bash
# Run a simple example
python -m nexusml.examples.simple_example

# Run a data loading example
python -m nexusml.examples.data_loader_example

# Run a feature engineering example
python -m nexusml.examples.feature_engineering_example
```

Some examples may require additional setup or data files. Please refer to the specific example documentation for details.

## Example Structure

Each example typically follows this structure:

1. **Imports**: Import necessary modules and packages
2. **Setup**: Set up any required data or configuration
3. **Main Logic**: Demonstrate the core functionality
4. **Output**: Display or save the results
5. **Cleanup**: Clean up any temporary files or resources

## Best Practices

When using these examples as a reference for your own code:

1. **Start Simple**: Begin with the basic examples and gradually move to more complex ones
2. **Understand the Components**: Take time to understand how different components work together
3. **Customize Gradually**: Make small changes to the examples to fit your needs
4. **Check Documentation**: Refer to the API documentation for detailed information on classes and methods
5. **Error Handling**: Add proper error handling in your production code

## Contributing New Examples

If you'd like to contribute new examples to NexusML:

1. Follow the existing example structure and coding style
2. Include comprehensive comments explaining what the code does
3. Add proper error handling and input validation
4. Create tests to verify the example works correctly
5. Update the documentation to include your new example

## Next Steps

After exploring these examples, you might want to:

1. Check the [API Reference](../api_reference.md) for detailed information on classes and methods
2. Read the [Usage Guide](../usage_guide.md) for comprehensive usage documentation
3. Explore the [Architecture Overview](../architecture/overview.md) for a deeper understanding of the system design