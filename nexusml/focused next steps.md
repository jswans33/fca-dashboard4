# Focused next steps

# NexusML Refactoring Phase 4 Progress Note

## Completed

1. Fixed the ModelEvaluator registration in the DI container
2. Fixed the ClassificationSystemMapper issue in config_driven.py by properly handling the 'name' parameter

## Current Status

- Component resolution verification is passing
- Pipeline factory verification is passing
- Feature engineering verification is passing
- Model building verification is passing

## Relevant Files

- C:\Repos\fca-dashboard4\docs\code_review\nexusml_refactoring_phase4_testing.md
- C:\Repos\fca-dashboard4\docs\code_review\nexusml_refactoring_detailed_tasks.md

## Remaining Issues

1. End-to-end verification is failing with error: "columns are missing: {'combined_text'}"
2. Prediction pipeline verification is failing with error: "DefaultTransformerRegistry.create_transformer() got multiple values for argument 'name'"

## Next Steps

1. Fix the remaining ClassificationSystemMapper issue in other parts of the codebase
2. Investigate and fix the missing 'combined_text' column issue in the end-to-end test
3. Complete the remaining verification tests
4. Update documentation based on the verification results

## Files to Focus On

- nexusml/core/feature_engineering/transformers/categorical.py (ClassificationSystemMapper class)
- nexusml/core/pipeline/orchestrator.py (end-to-end pipeline execution)
- nexusml/tests/verification_script.py (verification tests)

## COmmand we are testing

```python
python -m nexusml.tests.verification_script
```

## Current Errors for fixing

```text
python -m nexusml.tests.verification_script
$ python -m nexusml.tests.verification_script
2025-03-09 12:17:55,827 - verification_script - INFO - Starting NexusML Refactoring Verification
2025-03-09 12:17:55,827 - verification_script - INFO - Verifying component resolution...
2025-03-09 12:17:55,827 - nexusml.core.di.registration - INFO - Registered EAVManager with DI container
2025-03-09 12:17:55,827 - nexusml.core.di.registration - INFO - Registered FeatureEngineer implementations with DI container
2025-03-09 12:17:55,827 - nexusml.core.di.registration - INFO - Registered EquipmentClassifier with DI container
2025-03-09 12:17:55,827 - nexusml.core.di.registration - INFO - Registered DataLoader with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.registration - INFO - Registered DataPreprocessor with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.registration - INFO - Registered ModelBuilder with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.registration - INFO - Registered ModelTrainer with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.registration - INFO - Registered ModelEvaluator with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:55,842 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:55,847 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:55,849 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:55,849 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:55,850 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:55,850 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:55,853 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:55,853 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:55,853 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:55,853 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:55,853 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:55,853 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:55,860 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:55,860 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:55,862 - nexusml.core.di.registration - INFO - Registered pipeline components with DI container
2025-03-09 12:17:55,862 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:55,862 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:55,865 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:55,868 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:55,870 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:55,870 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:55,870 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:55,872 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:55,872 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:55,872 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:55,874 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:55,874 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:55,874 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:55,876 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:55,876 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Configuration manager (ConfigurationManager)'
Arguments: ()
2025-03-09 12:17:55,879 - verification_script - INFO - ✅ Successfully resolved Configuration manager (ConfigurationManager)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Data loading stage (ConfigurableDataLoadingStage)'
Arguments: ()
2025-03-09 12:17:55,891 - verification_script - INFO - ✅ Successfully resolved Data loading stage (ConfigurableDataLoadingStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Validation stage (ConfigDrivenValidationStage)'
Arguments: ()
2025-03-09 12:17:55,898 - verification_script - INFO - ✅ Successfully resolved Validation stage (ConfigDrivenValidationStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Feature engineering stage (SimpleFeatureEngineeringStage)'
Arguments: ()
2025-03-09 12:17:55,902 - verification_script - INFO - ✅ Successfully resolved Feature engineering stage (SimpleFeatureEngineeringStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Data splitting stage (RandomSplittingStage)'
Arguments: ()
2025-03-09 12:17:55,912 - verification_script - INFO - ✅ Successfully resolved Data splitting stage (RandomSplittingStage)
2025-03-09 12:17:55,933 - nexusml.core.model_building.base - INFO - Using default model configuration
2025-03-09 12:17:55,933 - nexusml.core.model_building.builders.random_forest - INFO - Initialized RandomForestBuilder
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Model building stage (ConfigDrivenModelBuildingStage)'
Arguments: ()
2025-03-09 12:17:55,936 - verification_script - INFO - ✅ Successfully resolved Model building stage (ConfigDrivenModelBuildingStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Model training stage (StandardModelTrainingStage)'
Arguments: ()
2025-03-09 12:17:55,947 - verification_script - INFO - ✅ Successfully resolved Model training stage (StandardModelTrainingStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Model evaluation stage (ClassificationEvaluationStage)'
Arguments: ()
2025-03-09 12:17:55,954 - verification_script - INFO - ✅ Successfully resolved Model evaluation stage (ClassificationEvaluationStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Model saving stage (ModelCardSavingStage)'
Arguments: ()
2025-03-09 12:17:55,961 - verification_script - INFO - ✅ Successfully resolved Model saving stage (ModelCardSavingStage)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 359, in main
    component_resolution_success = verify_component_resolution(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 113, in verify_component_resolution
    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
Message: '✅ Successfully resolved Prediction stage (StandardPredictionStage)'
Arguments: ()
2025-03-09 12:17:55,969 - verification_script - INFO - ✅ Successfully resolved Prediction stage (StandardPredictionStage)
2025-03-09 12:17:55,974 - verification_script - INFO - Verifying pipeline factory...
2025-03-09 12:17:55,979 - nexusml.core.pipeline.registry - INFO - ComponentRegistry initialized
2025-03-09 12:17:55,979 - nexusml.core.di.registration - INFO - Registered EAVManager with DI container
2025-03-09 12:17:55,979 - nexusml.core.di.registration - INFO - Registered FeatureEngineer implementations with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered EquipmentClassifier with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered DataLoader with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered DataPreprocessor with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered ModelBuilder with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered ModelTrainer with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered ModelEvaluator with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:55,983 - nexusml.core.di.registration - INFO - Registered pipeline components with DI container
2025-03-09 12:17:55,983 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:55,999 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:56,000 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:56,000 - nexusml.core.pipeline.factory - INFO - PipelineFactory initialized with built-in pipeline types
2025-03-09 12:17:56,000 - nexusml.core.pipeline.pipelines.training - INFO - Initializing training pipeline stages
2025-03-09 12:17:56,000 - nexusml.core.model_building.base - INFO - Using default model configuration
2025-03-09 12:17:56,000 - nexusml.core.model_building.builders.random_forest - INFO - Initialized RandomForestBuilder
2025-03-09 12:17:56,000 - nexusml.core.pipeline.pipelines.training - WARNING - No output_dir specified in config, skipping model saving stage
2025-03-09 12:17:56,014 - nexusml.core.pipeline.pipelines.training - INFO - Training pipeline initialized with 7 stages
2025-03-09 12:17:56,014 - nexusml.core.pipeline.factory - INFO - Created pipeline of type training
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 362, in main
    pipeline_factory_success = verify_pipeline_factory(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 168, in verify_pipeline_factory
    logger.info(f"✅ Successfully created {pipeline_type} pipeline")
Message: '✅ Successfully created training pipeline'
Arguments: ()
2025-03-09 12:17:56,015 - verification_script - INFO - ✅ Successfully created training pipeline
2025-03-09 12:17:56,020 - nexusml.core.pipeline.pipelines.prediction - INFO - Initializing prediction pipeline stages
2025-03-09 12:17:56,021 - nexusml.core.pipeline.pipelines.prediction - WARNING - No output_path specified in config, results will not be saved
2025-03-09 12:17:56,021 - nexusml.core.pipeline.pipelines.prediction - INFO - Prediction pipeline initialized with 5 stages
2025-03-09 12:17:56,021 - nexusml.core.pipeline.factory - INFO - Created pipeline of type prediction
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 362, in main
    pipeline_factory_success = verify_pipeline_factory(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 168, in verify_pipeline_factory
    logger.info(f"✅ Successfully created {pipeline_type} pipeline")
Message: '✅ Successfully created prediction pipeline'
Arguments: ()
2025-03-09 12:17:56,023 - verification_script - INFO - ✅ Successfully created prediction pipeline
2025-03-09 12:17:56,025 - nexusml.core.pipeline.pipelines.evaluation - INFO - Initializing evaluation pipeline stages
2025-03-09 12:17:56,027 - nexusml.core.pipeline.pipelines.evaluation - WARNING - No output_path specified in config, evaluation results will not be saved
2025-03-09 12:17:56,027 - nexusml.core.pipeline.pipelines.evaluation - INFO - Evaluation pipeline initialized with 5 stages
2025-03-09 12:17:56,027 - nexusml.core.pipeline.factory - INFO - Created pipeline of type evaluation
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 362, in main
    pipeline_factory_success = verify_pipeline_factory(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 168, in verify_pipeline_factory
    logger.info(f"✅ Successfully created {pipeline_type} pipeline")
Message: '✅ Successfully created evaluation pipeline'
Arguments: ()
2025-03-09 12:17:56,027 - verification_script - INFO - ✅ Successfully created evaluation pipeline
2025-03-09 12:17:56,033 - verification_script - INFO - Verifying feature engineering components...
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 365, in main
    feature_engineering_success = verify_feature_engineering(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 289, in verify_feature_engineering
    logger.info(f"✅ Feature engineering verification successful. Found normalized columns: {normalized_columns} and scaled columns: {scaled_columns}")
Message: "✅ Feature engineering verification successful. Found normalized columns: ['description_normalized'] and scaled columns: ['service_life_scaled']"
Arguments: ()
2025-03-09 12:17:56,038 - verification_script - INFO - ✅ Feature engineering verification successful. Found normalized columns: ['description_normalized'] and scaled columns: ['service_life_scaled']      
2025-03-09 12:17:56,042 - verification_script - INFO - Verifying model building components...
2025-03-09 12:17:56,042 - nexusml.core.model_building.base - INFO - Using default model configuration
2025-03-09 12:17:56,042 - nexusml.core.model_building.builders.random_forest - INFO - Initialized RandomForestBuilder
2025-03-09 12:17:56,042 - nexusml.core.model_building.builders.random_forest - INFO - Building Random Forest model
2025-03-09 12:17:56,042 - nexusml.core.model_building.builders.random_forest - INFO - Random Forest model built successfully
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 55: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 368, in main
    model_building_success = verify_model_building(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 340, in verify_model_building
    logger.info("✅ Model building verification successful")
Message: '✅ Model building verification successful'
Arguments: ()
2025-03-09 12:17:56,117 - verification_script - INFO - ✅ Model building verification successful
2025-03-09 12:17:56,128 - verification_script - INFO - Verifying end-to-end functionality...
2025-03-09 12:17:56,133 - nexusml.core.di.registration - INFO - Registered EAVManager with DI container
2025-03-09 12:17:56,135 - nexusml.core.di.registration - INFO - Registered FeatureEngineer implementations with DI container
2025-03-09 12:17:56,135 - nexusml.core.di.registration - INFO - Registered EquipmentClassifier with DI container
2025-03-09 12:17:56,136 - nexusml.core.di.registration - INFO - Registered DataLoader with DI container
2025-03-09 12:17:56,136 - nexusml.core.di.registration - INFO - Registered DataPreprocessor with DI container
2025-03-09 12:17:56,136 - nexusml.core.di.registration - INFO - Registered ModelBuilder with DI container
2025-03-09 12:17:56,136 - nexusml.core.di.registration - INFO - Registered ModelTrainer with DI container
2025-03-09 12:17:56,136 - nexusml.core.di.registration - INFO - Registered ModelEvaluator with DI container
2025-03-09 12:17:56,139 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:56,139 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:56,139 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:56,139 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:56,141 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:56,141 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:56,143 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:56,143 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:56,143 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:56,145 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:56,145 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:56,145 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:56,145 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:56,147 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:56,147 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:56,147 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:56,149 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:56,149 - nexusml.core.di.registration - INFO - Registered pipeline components with DI container
2025-03-09 12:17:56,150 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:56,150 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:56,150 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:56,150 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:56,150 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:56,154 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:56,154 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:56,155 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:56,155 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:56,156 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:56,156 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:56,157 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:56,158 - nexusml.core.pipeline.registry - INFO - ComponentRegistry initialized
2025-03-09 12:17:56,158 - nexusml.core.di.registration - INFO - Registered EAVManager with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.registration - INFO - Registered FeatureEngineer implementations with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.registration - INFO - Registered EquipmentClassifier with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.registration - INFO - Registered DataLoader with DI container
2025-03-09 12:17:56,158 - nexusml.core.di.registration - INFO - Registered DataPreprocessor with DI container
2025-03-09 12:17:56,163 - nexusml.core.di.registration - INFO - Registered ModelBuilder with DI container
2025-03-09 12:17:56,163 - nexusml.core.di.registration - INFO - Registered ModelTrainer with DI container
2025-03-09 12:17:56,163 - nexusml.core.di.registration - INFO - Registered ModelEvaluator with DI container
2025-03-09 12:17:56,163 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:56,163 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:17:56,167 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:56,167 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:56,167 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:56,167 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:56,167 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:56,167 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:56,171 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:56,171 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:56,171 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:56,173 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:56,173 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:56,174 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:56,174 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:56,174 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:56,174 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:56,174 - nexusml.core.di.registration - INFO - Registered pipeline components with DI container
2025-03-09 12:17:56,174 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:17:56,174 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:17:56,178 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:17:56,178 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:17:56,179 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:17:56,179 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:17:56,179 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:17:56,179 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:17:56,181 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:17:56,181 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:17:56,181 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:17:56,183 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:17:56,183 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:17:56,183 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:17:56,183 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:17:56,186 - nexusml.core.pipeline.factory - INFO - PipelineFactory initialized with built-in pipeline types
2025-03-09 12:17:56,186 - verification_script - INFO - Training model example
2025-03-09 12:17:56,187 - nexusml.core.pipeline.context - INFO - Pipeline execution started
2025-03-09 12:17:56,189 - nexusml.core.pipeline.context - INFO - Starting component: data_loading
2025-03-09 12:17:56,189 - pipeline_orchestrator_example - INFO - Loading data from examples/sample_data.xlsx
2025-03-09 12:17:56,509 - nexusml.core.pipeline.context - INFO - Component data_loading completed in 0.32 seconds
2025-03-09 12:17:56,509 - nexusml.core.pipeline.context - INFO - Starting component: data_preprocessing
2025-03-09 12:17:56,509 - nexusml.core.di.registration - INFO - Preprocessing data
2025-03-09 12:17:56,511 - nexusml.core.pipeline.context - INFO - Component data_preprocessing completed in 0.00 seconds
2025-03-09 12:17:56,511 - nexusml.core.pipeline.context - INFO - Starting component: feature_engineering
2025-03-09 12:17:56,511 - nexusml.core.di.registration - INFO - Fitting feature engineer
2025-03-09 12:17:56,511 - nexusml.core.di.registration - INFO - Transforming data with feature engineer
2025-03-09 12:17:56,511 - nexusml.core.pipeline.context - INFO - Component feature_engineering completed in 0.00 seconds
2025-03-09 12:17:56,511 - nexusml.core.pipeline.context - INFO - Starting component: data_splitting
2025-03-09 12:17:56,520 - nexusml.core.pipeline.context - INFO - Component data_splitting completed in 0.01 seconds
2025-03-09 12:17:56,520 - nexusml.core.pipeline.context - INFO - Starting component: model_building
2025-03-09 12:17:56,522 - nexusml.core.model_building.base - INFO - Using default model configuration
2025-03-09 12:17:56,522 - nexusml.core.model_building.builders.random_forest - INFO - Initialized RandomForestBuilder
2025-03-09 12:17:56,522 - nexusml.core.model_building.builders.random_forest - INFO - Building Random Forest model
2025-03-09 12:17:56,522 - nexusml.core.model_building.builders.random_forest - INFO - Random Forest model built successfully
2025-03-09 12:17:56,522 - nexusml.core.pipeline.orchestrator - INFO - Optimizing hyperparameters...
2025-03-09 12:17:56,524 - nexusml.core.pipeline.context - INFO - Component model_building completed in 0.00 seconds
2025-03-09 12:17:56,524 - nexusml.core.pipeline.context - INFO - Starting component: model_training
2025-03-09 12:17:56,527 - nexusml.core.model_building.base - INFO - Training model BaseModelTrainer
2025-03-09 12:17:58,336 - nexusml.core.model_building.base - INFO - Model BaseModelTrainer trained successfully
2025-03-09 12:17:58,338 - nexusml.core.model_building.base - INFO - Performing cross-validation for model BaseModelTrainer
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py:960: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Deta
ils:                                                                                                                                                                                                         Traceback (most recent call last):
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 949, in _score
    scores = scorer(estimator, X_test, y_test, **score_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 288, in __call__
    return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_scorer.py", line 388, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 227, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\metrics\_classification.py", line 118, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported

  warnings.warn(
2025-03-09 12:18:08,009 - nexusml.core.model_building.base - INFO - Cross-validation completed for model BaseModelTrainer
2025-03-09 12:18:08,021 - nexusml.core.pipeline.context - INFO - Component model_training completed in 11.50 seconds
2025-03-09 12:18:08,021 - nexusml.core.pipeline.context - INFO - Starting component: model_evaluation
2025-03-09 12:18:08,021 - nexusml.core.model_building.base - INFO - Using default evaluation configuration
2025-03-09 12:18:08,021 - nexusml.core.model_building.base - INFO - Evaluating model BaseModelEvaluator
2025-03-09 12:18:08,124 - nexusml.core.model_building.base - INFO - Model BaseModelEvaluator evaluation completed
2025-03-09 12:18:08,124 - nexusml.core.pipeline.orchestrator - INFO - x_test columns: Index(['combined_text', 'service_life'], dtype='object')
2025-03-09 12:18:08,124 - nexusml.core.pipeline.orchestrator - INFO - y_test columns: Index(['category_name', 'uniformat_code', 'mcaa_system_category',
       'Equipment_Type', 'System_Subtype'],
      dtype='object')
2025-03-09 12:18:08,124 - nexusml.core.pipeline.orchestrator - INFO - Making predictions with features shape: (2, 1)
2025-03-09 12:18:08,124 - nexusml.core.pipeline.orchestrator - ERROR - Error in pipeline execution: columns are missing: {'combined_text'}
2025-03-09 12:18:08,134 - nexusml.core.pipeline.orchestrator - ERROR - Traceback (most recent call last):
  File "C:\Repos\fca-dashboard4\nexusml\core\pipeline\orchestrator.py", line 216, in train_model
    y_pred = trained_model.predict(features_for_prediction)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\pipeline.py", line 787, in predict
    Xt = transform.transform(Xt)
         ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\utils\_set_output.py", line 319, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\repos\fca-dashboard4\.venv\Lib\site-packages\sklearn\compose\_column_transformer.py", line 1090, in transform
    raise ValueError(f"columns are missing: {diff}")
ValueError: columns are missing: {'combined_text'}

2025-03-09 12:18:08,135 - nexusml.core.pipeline.context - ERROR - Pipeline execution failed: columns are missing: {'combined_text'}
2025-03-09 12:18:08,135 - nexusml.core.pipeline.context - INFO - Pipeline execution failed in 11.95 seconds
2025-03-09 12:18:08,137 - verification_script - ERROR - Error training model: Error in pipeline execution: columns are missing: {'combined_text'}
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u274c' in position 56: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 371, in main
    end_to_end_success = verify_end_to_end(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 203, in verify_end_to_end
    logger.error("❌ End-to-end verification failed: model training failed")
Message: '❌ End-to-end verification failed: model training failed'
Arguments: ()
2025-03-09 12:18:08,137 - verification_script - ERROR - ❌ End-to-end verification failed: model training failed
2025-03-09 12:18:08,142 - verification_script - INFO - Verifying prediction pipeline...
2025-03-09 12:18:08,149 - nexusml.core.di.registration - INFO - Registered EAVManager with DI container
2025-03-09 12:18:08,150 - nexusml.core.di.registration - INFO - Registered FeatureEngineer implementations with DI container
2025-03-09 12:18:08,150 - nexusml.core.di.registration - INFO - Registered EquipmentClassifier with DI container
2025-03-09 12:18:08,150 - nexusml.core.di.registration - INFO - Registered DataLoader with DI container
2025-03-09 12:18:08,153 - nexusml.core.di.registration - INFO - Registered DataPreprocessor with DI container
2025-03-09 12:18:08,153 - nexusml.core.di.registration - INFO - Registered ModelBuilder with DI container
2025-03-09 12:18:08,153 - nexusml.core.di.registration - INFO - Registered ModelTrainer with DI container
2025-03-09 12:18:08,153 - nexusml.core.di.registration - INFO - Registered ModelEvaluator with DI container
2025-03-09 12:18:08,153 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:18:08,153 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:18:08,156 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:18:08,156 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:18:08,156 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:18:08,156 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:18:08,160 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:18:08,164 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:18:08,164 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:18:08,164 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:18:08,165 - nexusml.core.di.registration - INFO - Registered pipeline components with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:18:08,165 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:18:08,170 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:18:08,170 - verification_script - INFO - Starting prediction pipeline example
2025-03-09 12:18:08,175 - nexusml.core.pipeline.registry - INFO - ComponentRegistry initialized
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered EAVManager with DI container
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered FeatureEngineer implementations with DI container
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered EquipmentClassifier with DI container
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered DataLoader with DI container
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered DataPreprocessor with DI container
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered ModelBuilder with DI container
2025-03-09 12:18:08,175 - nexusml.core.di.registration - INFO - Registered ModelTrainer with DI container
2025-03-09 12:18:08,180 - nexusml.core.di.registration - INFO - Registered ModelEvaluator with DI container
2025-03-09 12:18:08,180 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:18:08,180 - nexusml.core.di.registration - INFO - Registered ModelSerializer with DI container
2025-03-09 12:18:08,180 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:18:08,182 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:18:08,182 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:18:08,182 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:18:08,184 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:18:08,189 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:18:08,189 - nexusml.core.di.registration - INFO - Registered pipeline components with DI container
2025-03-09 12:18:08,189 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurableDataLoadingStage with DI container
2025-03-09 12:18:08,189 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigDrivenValidationStage with DI container
2025-03-09 12:18:08,192 - nexusml.core.di.pipeline_registration - INFO - Registered feature engineering stages with DI container
2025-03-09 12:18:08,193 - nexusml.core.di.pipeline_registration - INFO - Registered RandomSplittingStage with DI container
2025-03-09 12:18:08,193 - nexusml.core.di.pipeline_registration - INFO - Registered model building stages with DI container
2025-03-09 12:18:08,193 - nexusml.core.di.pipeline_registration - INFO - Registered StandardModelTrainingStage with DI container
2025-03-09 12:18:08,193 - nexusml.core.di.pipeline_registration - INFO - Registered ClassificationEvaluationStage with DI container
2025-03-09 12:18:08,193 - nexusml.core.di.pipeline_registration - INFO - Registered ModelCardSavingStage with DI container
2025-03-09 12:18:08,193 - nexusml.core.di.pipeline_registration - INFO - Registered ModelLoadingStage with DI container
2025-03-09 12:18:08,197 - nexusml.core.di.pipeline_registration - INFO - Registered output saving stages with DI container
2025-03-09 12:18:08,197 - nexusml.core.di.pipeline_registration - INFO - Registered prediction stages with DI container
2025-03-09 12:18:08,198 - nexusml.core.di.pipeline_registration - INFO - Registered ConfigurationManager with DI container
2025-03-09 12:18:08,198 - nexusml.core.di.pipeline_registration - INFO - Registered ComponentRegistry with DI container
2025-03-09 12:18:08,198 - nexusml.core.di.pipeline_registration - INFO - Registered PipelineFactory with DI container
2025-03-09 12:18:08,198 - nexusml.core.di.pipeline_registration - INFO - All pipeline components registered successfully
2025-03-09 12:18:08,201 - nexusml.core.pipeline.factory - INFO - PipelineFactory initialized with built-in pipeline types
2025-03-09 12:18:08,201 - prediction_pipeline_example - INFO - Loading model from outputs/models/equipment_classifier.pkl
2025-03-09 12:18:08,201 - nexusml.core.pipeline.context - INFO - Pipeline execution started
2025-03-09 12:18:08,201 - nexusml.core.pipeline.context - INFO - Starting component: model_loading
2025-03-09 12:18:08,201 - pipeline_orchestrator_example - INFO - Loading model from outputs/models/equipment_classifier.pkl
2025-03-09 12:18:08,201 - nexusml.core.pipeline.context - INFO - Component model_loading completed in 0.00 seconds
2025-03-09 12:18:08,204 - nexusml.core.pipeline.context - INFO - Pipeline execution completed in 0.00 seconds
2025-03-09 12:18:08,205 - prediction_pipeline_example - INFO - Model loaded successfully
2025-03-09 12:18:08,205 - prediction_pipeline_example - INFO - Loading prediction data from examples/data/sample_prediction_data.csv
2025-03-09 12:18:08,209 - prediction_pipeline_example - INFO - Loaded 10 records for prediction
2025-03-09 12:18:08,211 - nexusml.core.pipeline.context - INFO - Pipeline execution started
2025-03-09 12:18:08,211 - nexusml.core.pipeline.context - INFO - Starting component: data_preprocessing
2025-03-09 12:18:08,213 - nexusml.core.di.registration - INFO - Preprocessing data
2025-03-09 12:18:08,213 - nexusml.core.pipeline.context - INFO - Component data_preprocessing completed in 0.00 seconds
2025-03-09 12:18:08,214 - nexusml.core.pipeline.context - INFO - Starting component: feature_engineering
2025-03-09 12:18:08,214 - nexusml.core.di.registration - INFO - Fitting feature engineer
2025-03-09 12:18:08,214 - nexusml.core.di.registration - INFO - Transforming data with feature engineer
2025-03-09 12:18:08,214 - nexusml.core.pipeline.context - INFO - Component feature_engineering completed in 0.00 seconds
2025-03-09 12:18:08,217 - nexusml.core.pipeline.context - INFO - Starting component: prediction
2025-03-09 12:18:08,227 - nexusml.core.pipeline.orchestrator - ERROR - Error in prediction pipeline: Error creating predictor: DefaultTransformerRegistry.create_transformer() got multiple values for argume
nt 'name'                                                                                                                                                                                                    2025-03-09 12:18:08,228 - nexusml.core.pipeline.orchestrator - ERROR - Traceback (most recent call last):
  File "C:\Repos\fca-dashboard4\nexusml\core\pipeline\factory.py", line 387, in create_predictor
    predictor = self.container.resolve(EquipmentClassifier)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\di\container.py", line 185, in resolve
    return factory(self)
           ^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\di\container.py", line 95, in factory
    kwargs[param_name] = container.resolve(args[0])
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\di\container.py", line 185, in resolve
    return factory(self)
           ^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\di\container.py", line 110, in factory
    return implementation_type(**kwargs)  # type: ignore
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\feature_engineering\compatibility.py", line 41, in __init__
    self.feature_engineer = ConfigDrivenFeatureEngineer(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\feature_engineering\config_driven.py", line 58, in __init__
    super().__init__(config, name)
  File "C:\Repos\fca-dashboard4\nexusml\core\feature_engineering\base.py", line 412, in __init__
    self.transformers = self.create_transformers_from_config()
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\feature_engineering\config_driven.py", line 246, in create_transformers_from_config
    transformer = _default_registry.create_transformer("classification_system_mapper", **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: DefaultTransformerRegistry.create_transformer() got multiple values for argument 'name'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Repos\fca-dashboard4\nexusml\core\pipeline\orchestrator.py", line 431, in predict
    predictor = self.factory.create_predictor()
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repos\fca-dashboard4\nexusml\core\pipeline\factory.py", line 396, in create_predictor
    raise PipelineFactoryError(f"Error creating predictor: {str(e)}") from e
nexusml.core.pipeline.factory.PipelineFactoryError: Error creating predictor: DefaultTransformerRegistry.create_transformer() got multiple values for argument 'name'

2025-03-09 12:18:08,228 - nexusml.core.pipeline.context - ERROR - Prediction pipeline failed: Error creating predictor: DefaultTransformerRegistry.create_transformer() got multiple values for argument 'nam
e'                                                                                                                                                                                                           2025-03-09 12:18:08,228 - nexusml.core.pipeline.context - INFO - Pipeline execution failed in 0.02 seconds
2025-03-09 12:18:08,228 - verification_script - ERROR - Error making predictions: Error in prediction pipeline: Error creating predictor: DefaultTransformerRegistry.create_transformer() got multiple values
 for argument 'name'                                                                                                                                                                                         --- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u274c' in position 56: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 374, in main
    prediction_pipeline_success = verify_prediction_pipeline(logger)
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 235, in verify_prediction_pipeline
    logger.error("❌ Prediction pipeline verification failed: output file not created")
Message: '❌ Prediction pipeline verification failed: output file not created'
Arguments: ()
2025-03-09 12:18:08,234 - verification_script - ERROR - ❌ Prediction pipeline verification failed: output file not created
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\logging\__init__.py", line 1163, in emit
    stream.write(msg + self.terminator)
  File "C:\Users\jswan\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u274c' in position 56: character maps to <undefined>
Call stack:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 407, in <module>
    main()
  File "C:\Repos\fca-dashboard4\nexusml\tests\verification_script.py", line 389, in main
    logger.error("❌ Some verification tests failed")
Message: '❌ Some verification tests failed'
Arguments: ()
2025-03-09 12:18:08,240 - verification_script - ERROR - ❌ Some verification tests failed
2025-03-09 12:18:08,246 - verification_script - ERROR -   - End-to-end verification failed
2025-03-09 12:18:08,246 - verification_script - ERROR -   - Prediction pipeline verification failed
2025-03-09 12:18:08,246 - verification_script - INFO - NexusML Refactoring Verification completed
```
