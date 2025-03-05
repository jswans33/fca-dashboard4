# SOP-003: Troubleshooting the Equipment Classifier Example

## Purpose

This procedure documents the steps to troubleshoot and resolve common issues when running the equipment classifier example in the FCA Dashboard project. It provides comprehensive guidance for diagnosing and fixing issues related to the classifier implementation, dependencies, and execution environment.

## Scope

This SOP covers:

- Diagnosing and resolving module import errors
- Fixing package compatibility issues
- Verifying data file availability
- Creating alternative solutions for visualization issues
- Setting up a clean environment for the classifier
- Interpreting model output and warnings
- Optimizing model performance
- Troubleshooting environment-specific issues

## Prerequisites

- Python 3.9+ installed
- Access to the project repository
- Basic understanding of Python virtual environments
- Familiarity with the project structure
- Understanding of machine learning concepts (for advanced troubleshooting)

## Quick Start

- Check installation status:
  - $ python -m fca_dashboard.utils.verify_classifier
  
- Install missing dependency:
  - $ python -m pip install scikit-learn matplotlib seaborn imbalanced-learn
  
- Run Classifier Example
  - Run python -m fca_dashboard.examples.classifier_example

## Procedure

### 1. Diagnosing Module Import Errors

1. Identify the correct module path format:
   - Use dot notation for module paths, not slashes
   - Correct: `python -m fca_dashboard.examples.classifier_example`
   - Incorrect: `python -m fca_dashboard/fca_dashboard.examples.classifier_example`
   - Incorrect: `python fca_dashboard/examples/classifier_example.py`

2. Verify the package is installed in development mode:

   ```bash
   pip install -e .
   ```

   This step is crucial for allowing the package to be imported using absolute imports. Without this step, you may encounter `ModuleNotFoundError` when trying to import modules from the package.

3. Use the Makefile's `install` target to ensure proper installation:

   ```bash
   make install
   ```

4. Check if you're using the correct Python interpreter:
   - Different terminals or IDEs might use different Python environments
   - Verify which Python is being used with:

     ```bash
     which python  # On Unix/Linux/macOS
     where python  # On Windows
     ```

   - Ensure you're using the Python from your virtual environment:

     ```
     # Should show a path like .venv/bin/python or .venv\Scripts\python.exe
     ```

5. Verify virtual environment activation:
   - On Windows:

     ```bash
     # Activate the environment
     .\.venv\Scripts\activate
     
     # Verify activation (should show (.venv) at the start of your prompt)
     # Also check Python path
     where python
     ```

   - On Unix/Linux/macOS:

     ```bash
     # Activate the environment
     source .venv/bin/activate
     
     # Verify activation
     which python
     ```

6. Check for conflicting installations:
   - Sometimes having the package installed both in development mode and via pip can cause conflicts
   - Uninstall any global installations of the package:

     ```bash
     pip uninstall fca-dashboard
     ```

   - Then reinstall in development mode:

     ```bash
     pip install -e .
     ```

### 2. Resolving Package Compatibility Issues

1. Identify package dependencies from requirements.txt:

   ```
   numpy>=1.20.0
   scikit-learn>=1.0.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   imbalanced-learn>=0.8.0
   ```

2. Reinstall problematic packages:

   ```bash
   pip uninstall -y <package-name>
   pip install <package-name>
   ```

3. For specific package issues:
   - NumPy: `pip uninstall -y numpy && pip install numpy`
   - SciPy: `pip uninstall -y scipy && pip install scipy`
   - Matplotlib: `pip uninstall -y matplotlib && pip install matplotlib`
   - Pillow (required by matplotlib): `pip uninstall -y pillow && pip install pillow`
   - Kiwisolver (dependency of matplotlib): `pip uninstall -y kiwisolver && pip install kiwisolver`

4. For circular import issues:
   - Reinstall the affected package and its dependencies
   - Check for version compatibility between packages

5. Common visualization dependency issues:
   - Matplotlib and its dependencies often cause issues due to C extensions
   - If you see errors related to C extensions (e.g., `_cext`, `_multiarray_umath`), try:

     ```bash
     # Reinstall the package with --force-reinstall to rebuild C extensions
     pip uninstall -y matplotlib
     pip install --force-reinstall matplotlib
     ```

   - For persistent issues, try installing pre-built binaries:

     ```bash
     pip install --only-binary=:all: matplotlib
     ```

6. Python version compatibility:
   - Some packages may have issues with specific Python versions
   - Check the package documentation for supported Python versions
   - Consider downgrading to a more stable Python version if needed

7. Operating system-specific issues:
   - Windows:
     - Visual C++ build tools might be required for some packages
     - Install from: <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
   - Linux:
     - Development libraries might be missing
     - Install with: `sudo apt-get install python3-dev` (Ubuntu/Debian)
   - macOS:
     - XCode command-line tools might be required
     - Install with: `xcode-select --install`

8. Verify package installation:
   - Check if packages are correctly installed:

     ```bash
     pip list | grep matplotlib
     pip show matplotlib
     ```

   - Try importing the package in a Python interpreter:

     ```python
     import matplotlib
     print(matplotlib.__version__)
     print(matplotlib.__file__)  # Shows where the package is installed
     ```

### 3. Verifying Data File Availability

1. Check the data file path in settings.yml:

   ```yaml
   classifier:
     data_paths:
       training_data: "fca_dashboard/classifier/ingest/eq_ids.csv"
   ```

2. Verify the file exists:

   ```bash
   # On Windows
   dir fca_dashboard\classifier\ingest\eq_ids.csv
   
   # On Unix/Linux/macOS
   ls -la fca_dashboard/classifier/ingest/eq_ids.csv
   ```

3. If the file is missing, check alternative locations or restore from repository:
   - Check if the file exists in a different location:

     ```bash
     # On Windows
     dir /s /b eq_ids.csv
     
     # On Unix/Linux/macOS
     find . -name eq_ids.csv
     ```

   - Restore from Git repository if available:

     ```bash
     git checkout -- fca_dashboard/classifier/ingest/eq_ids.csv
     ```

   - Check if the file is in a different format (e.g., .xlsx instead of .csv)

4. Verify the data file format and content:
   - Check if the CSV file is properly formatted:

     ```bash
     # View the first few lines
     head -n 5 fca_dashboard/classifier/ingest/eq_ids.csv
     
     # On Windows
     type fca_dashboard\classifier\ingest\eq_ids.csv | more
     ```

   - Verify the file has the expected columns:

     ```python
     import pandas as pd
     df = pd.read_csv("fca_dashboard/classifier/ingest/eq_ids.csv")
     print(df.columns)
     print(df.shape)  # Shows (rows, columns)
     ```

5. Check file permissions:
   - Ensure the file has read permissions:

     ```bash
     # On Unix/Linux/macOS
     chmod +r fca_dashboard/classifier/ingest/eq_ids.csv
     ```

6. Create a sample data file if needed:
   - If the original data file is unavailable, create a minimal sample file for testing
   - See the project documentation for the required format and columns

### 4. Creating a Simplified Version Without Visualizations

If visualization dependencies (matplotlib, seaborn) are causing issues, create a simplified version of the classifier example:

1. Create a simplified version of the classifier example that doesn't rely on matplotlib or seaborn:

   ```python
   # fca_dashboard/examples/classifier_example_simple.py
   """
   Simplified Example Usage of Equipment Classification Package
   
   This script demonstrates the core functionality of the equipment classification package
   without the visualization components.
   """
   
   import os
   import yaml
   from pathlib import Path
   
   # Import from the classifier package
   from fca_dashboard.classifier import (
       train_enhanced_model,
       predict_with_enhanced_model
   )
   
   # Import path utilities
   from fca_dashboard.utils.path_util import get_config_path, resolve_path
   
   
   def load_settings():
       """Load settings from the configuration file"""
       settings_path = get_config_path("settings.yml")
       
       try:
           with open(settings_path, 'r') as file:
               return yaml.safe_load(file)
       except FileNotFoundError:
           raise FileNotFoundError(f"Could not find settings file at: {settings_path}")
   
   
   def main():
       """Main function demonstrating the usage of the equipment classification package"""
       # Load settings
       settings = load_settings()
       classifier_settings = settings.get('classifier', {})
       
       # Get data path from settings
       data_path = classifier_settings.get('data_paths', {}).get('training_data')
       if not data_path:
           print("Warning: Training data path not found in settings, using default path")
           data_path = "fca_dashboard/classifier/ingest/eq_ids.csv"
       
       # Get output paths from settings
       example_settings = classifier_settings.get('examples', {})
       output_dir = example_settings.get('output_dir', 'fca_dashboard/examples/classifier/outputs')
       prediction_file = example_settings.get('prediction_file',
                                           os.path.join(output_dir, 'example_prediction.txt'))
       
       # Create output directory if it doesn't exist
       os.makedirs(output_dir, exist_ok=True)
       
       # Train enhanced model using the CSV file
       print(f"Training the model using data from: {data_path}")
       model, df = train_enhanced_model(data_path)
       
       # Example prediction with service life
       description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
       service_life = 20.0  # Example service life in years
       
       print("\nMaking a prediction for:")
       print(f"Description: {description}")
       print(f"Service Life: {service_life} years")
       
       prediction = predict_with_enhanced_model(model, description, service_life)
       
       print("\nEnhanced Prediction:")
       for key, value in prediction.items():
           print(f"{key}: {value}")
   
       # Save prediction results to file
       print(f"\nSaving prediction results to {prediction_file}")
       with open(prediction_file, 'w') as f:
           f.write("Enhanced Prediction Results\n")
           f.write("==========================\n\n")
           f.write("Input:\n")
           f.write(f"  Description: {description}\n")
           f.write(f"  Service Life: {service_life} years\n\n")
           f.write("Prediction:\n")
           for key, value in prediction.items():
               f.write(f"  {key}: {value}\n")
   
   
   if __name__ == "__main__":
       main()
   ```

2. Add a new Makefile target for running the simplified version:

   ```makefile
   # Run the simplified equipment classifier example (no visualizations)
   run-classifier-simple:
       # Run the simplified version without matplotlib/seaborn dependencies
       python -m fca_dashboard.examples.classifier_example_simple
   ```

3. Run the simplified version:

   ```bash
   make run-classifier-simple
   ```

### 5. Creating a Verification Script

Create a verification script to check if all necessary components are in place:

1. Create a verification script:

   ```python
   # fca_dashboard/utils/verify_classifier.py
   """
   Classifier Verification Script
   
   This script verifies that all necessary components are in place to run the classifier example.
   It checks for required packages, data files, and module imports.
   """
   
   import os
   import sys
   import importlib
   import pandas as pd
   from pathlib import Path
   
   
   def check_package_versions():
       """Check if all required packages are installed and compatible."""
       print("Checking package versions...")
       all_ok = True
       
       # Check numpy
       try:
           import numpy
           print(f"✓ numpy: {numpy.__version__}")
       except ImportError:
           print("✗ numpy: Not installed")
           all_ok = False
       
       # Check pandas
       try:
           import pandas
           print(f"✓ pandas: {pandas.__version__}")
       except ImportError:
           print("✗ pandas: Not installed")
           all_ok = False
       
       # Check scikit-learn
       try:
           import sklearn
           print(f"✓ scikit-learn: {sklearn.__version__}")
       except ImportError:
           print("✗ scikit-learn: Not installed")
           all_ok = False
       
       # Check matplotlib
       try:
           import matplotlib
           print(f"✓ matplotlib: {matplotlib.__version__}")
       except ImportError:
           print("✗ matplotlib: Not installed")
           all_ok = False
       
       # Check seaborn
       try:
           import seaborn
           print(f"✓ seaborn: {seaborn.__version__}")
       except ImportError:
           print("✗ seaborn: Not installed")
           all_ok = False
       
       # Check imbalanced-learn
       try:
           import imblearn
           print(f"✓ imbalanced-learn: {imblearn.__version__}")
       except ImportError:
           print("✗ imbalanced-learn: Not installed")
           all_ok = False
       
       return all_ok
   
   
   def check_data_file():
       """Check if the training data file exists."""
       # Try to load from settings
       try:
           import yaml
           from fca_dashboard.utils.path_util import get_config_path, resolve_path
           
           settings_path = get_config_path("settings.yml")
           with open(settings_path, 'r') as file:
               settings = yaml.safe_load(file)
               
           data_path = settings.get('classifier', {}).get('data_paths', {}).get('training_data')
           if not data_path:
               # Fallback to default path
               data_path = "fca_dashboard/classifier/ingest/eq_ids.csv"
           
           # Resolve the path to ensure it exists
           data_path = str(resolve_path(data_path))
       except Exception as e:
           print(f"Warning: Could not load settings: {e}")
           # Use absolute path as fallback
           data_path = str(Path(__file__).resolve().parent.parent / "classifier" / "ingest" / "eq_ids.csv")
       
       print(f"\nChecking data file: {data_path}")
       if os.path.exists(data_path):
           print(f"✓ Data file exists: {data_path}")
           try:
               df = pd.read_csv(data_path)
               print(f"✓ Data file can be read: {len(df)} rows, {len(df.columns)} columns")
               return True
           except Exception as e:
               print(f"✗ Error reading data file: {e}")
               return False
       else:
           print(f"✗ Data file not found: {data_path}")
           return False
   
   
   def check_module_imports():
       """Check if all required module imports work correctly."""
       print("\nChecking module imports...")
       all_ok = True
       
       modules_to_check = [
           ('fca_dashboard.classifier', 'train_enhanced_model'),
           ('fca_dashboard.classifier', 'predict_with_enhanced_model'),
           ('fca_dashboard.utils.path_util', 'get_config_path'),
           ('fca_dashboard.utils.path_util', 'resolve_path'),
       ]
       
       for module_name, attr_name in modules_to_check:
           try:
               module = importlib.import_module(module_name)
               attr = getattr(module, attr_name, None)
               if attr is not None:
                   print(f"✓ Successfully imported {module_name}.{attr_name}")
               else:
                   print(f"✗ Attribute {attr_name} not found in {module_name}")
                   all_ok = False
           except ImportError as e:
               print(f"✗ Error importing {module_name}: {e}")
               all_ok = False
           except Exception as e:
               print(f"✗ Unexpected error with {module_name}: {e}")
               all_ok = False
       
       return all_ok
   
   
   def main():
       """Run all verification checks."""
       print("=" * 60)
       print("CLASSIFIER EXAMPLE VERIFICATION")
       print("=" * 60)
       
       packages_ok = check_package_versions()
       data_ok = check_data_file()
       imports_ok = check_module_imports()
       
       print("\n" + "=" * 60)
       print("VERIFICATION SUMMARY")
       print("=" * 60)
       print(f"Packages: {'✓ OK' if packages_ok else '✗ Issues found'}")
       print(f"Data file: {'✓ OK' if data_ok else '✗ Issues found'}")
       print(f"Module imports: {'✓ OK' if imports_ok else '✗ Issues found'}")
       
       if packages_ok and data_ok and imports_ok:
           print("\nAll checks passed! You can run the classifier example with:")
           print("\n    python -m fca_dashboard.examples.classifier_example")
           return 0
       else:
           print("\nSome checks failed. Please fix the issues before running the classifier example.")
           return 1
   
   
   if __name__ == "__main__":
       sys.exit(main())
   ```

2. Run the verification script:

   ```bash
   python -m fca_dashboard.utils.verify_classifier
   ```

### 6. Setting Up a Clean Environment

If persistent issues occur, create a clean environment setup script:

1. Create a setup script:

   ```batch
   @echo off
   echo Setting up a clean environment for the classifier example...
   
   :: Create a new virtual environment
   echo Creating a new virtual environment...
   python -m venv .venv-classifier
   
   :: Activate the virtual environment
   echo Activating the virtual environment...
   call .venv-classifier\Scripts\activate
   
   :: Install dependencies
   echo Installing dependencies...
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   
   :: Verify installation
   echo Verifying installation...
   python -c "import numpy; import pandas; import sklearn; import matplotlib; import seaborn; import imblearn; print('All packages imported successfully!')"
   
   :: Run the classifier example
   echo Running the classifier example...
   python -m fca_dashboard.examples.classifier_example
   
   :: Deactivate the virtual environment
   call deactivate
   
   echo Done!
   ```

2. Run the setup script:

   ```bash
   .\setup_classifier_env.bat
   ```

### 7. Interpreting Model Output and Warnings

When running the classifier, you may encounter various warnings and outputs that need interpretation:

1. Understanding classification reports:
   - The classifier outputs detailed classification reports for each target variable
   - Key metrics to look for:
     - Precision: Ratio of true positives to all predicted positives
     - Recall: Ratio of true positives to all actual positives
     - F1-score: Harmonic mean of precision and recall
     - Support: Number of samples in each class
   - Low scores for specific classes may indicate:
     - Insufficient training data for that class
     - Class imbalance issues
     - Feature representation problems

2. Common warnings and their meanings:
   - `UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples`:
     - **Cause**: Some classes have no predicted samples in the test set
     - **Solution**: This is often normal for rare classes. Consider:
       - Using stratified sampling to ensure all classes appear in test set
       - Increasing test set size
       - Using a different evaluation metric for highly imbalanced classes

   - `ConvergenceWarning: Liblinear failed to converge, increase the number of iterations`:
     - **Cause**: The model optimization algorithm didn't reach convergence
     - **Solution**:
       - Increase max_iter parameter in the model
       - Normalize or standardize features
       - Reduce model complexity or use a different algorithm

   - `DataConversionWarning: A column-vector y was passed when a 1d array was expected`:
     - **Cause**: Format mismatch in target variable
     - **Solution**: Convert y to 1D array with `y = y.ravel()` or `y = np.ravel(y)`

3. Interpreting model performance:
   - Overall accuracy above 0.95 is excellent for this classifier
   - Pay attention to performance on minority classes
   - Check for classes with consistently low performance across metrics
   - Compare performance across different target variables (Equipment_Category, System_Type, etc.)

4. Analyzing misclassifications:
   - The classifier may output information about misclassified samples
   - Look for patterns in misclassifications:
     - Are certain classes frequently confused with each other?
     - Do misclassifications occur for samples with specific characteristics?
   - Use this information to improve:
     - Feature engineering
     - Training data quality
     - Model selection or hyperparameters

5. Visualizing model results:
   - The full classifier example generates visualizations:
     - Equipment category distribution
     - System type distribution
   - Use these visualizations to:
     - Identify class imbalance issues
     - Understand data distribution
     - Spot potential data quality issues

### 8. Optimizing Model Performance

If you need to improve the classifier's performance:

1. Feature engineering improvements:
   - Add more domain-specific features:
     - Extract more information from equipment descriptions
     - Include additional numerical features (e.g., dimensions, capacity)
     - Create interaction features between existing features
   - Improve text preprocessing:
     - Try different text vectorization methods (TF-IDF, word embeddings)
     - Experiment with n-gram ranges (unigrams, bigrams, trigrams)
     - Apply domain-specific text normalization

2. Model selection and tuning:
   - Try different classification algorithms:
     - Random Forest
     - Gradient Boosting
     - Support Vector Machines
     - Neural Networks
   - Perform hyperparameter tuning:
     - Use GridSearchCV or RandomizedSearchCV
     - Focus on key parameters for each algorithm
     - Consider cross-validation strategies

3. Handling class imbalance:
   - The current implementation uses RandomOverSampler
   - Alternative approaches:
     - SMOTE (Synthetic Minority Over-sampling Technique)
     - Class weights
     - Ensemble methods with balanced sampling
   - Evaluate impact on minority class performance

4. Ensemble methods:
   - Combine multiple models for better performance:
     - Voting classifiers
     - Stacking
     - Boosting
   - Especially effective for hierarchical classification tasks

5. Performance monitoring:
   - Track model performance over time
   - Implement monitoring for:
     - Data drift
     - Concept drift
     - Performance degradation
   - Establish retraining criteria and schedule

6. Optimizing for specific metrics:
   - Depending on the use case, prioritize:
     - Precision (minimize false positives)
     - Recall (minimize false negatives)
     - F1-score (balance precision and recall)
     - Accuracy (overall correctness)
   - Adjust classification thresholds accordingly

## Verification

1. Verify the simplified classifier example runs successfully:

   ```bash
   make run-classifier-simple
   ```

2. Verify the verification script runs successfully:

   ```bash
   python -m fca_dashboard.utils.verify_classifier
   ```

3. Check that the output files are created:
   - Prediction file: `fca_dashboard/examples/classifier/outputs/example_prediction.txt`

## Troubleshooting

### Module Import Errors

1. **ModuleNotFoundError: No module named 'fca_dashboard/fca_dashboard'**
   - **Cause**: Using slashes instead of dots in module path
   - **Solution**: Use `python -m fca_dashboard.examples.classifier_example`
   - **Example**: If you see this error when running `python -m fca_dashboard/examples/classifier_example.py`, change to `python -m fca_dashboard.examples.classifier_example`

2. **ModuleNotFoundError: No module named 'fca_dashboard'**
   - **Cause**: Package not installed in development mode
   - **Solution**: Run `pip install -e .` or `make install`
   - **Verification**: After installation, verify with `pip list | grep fca-dashboard`

3. **ImportError: attempted relative import with no known parent package**
   - **Cause**: Running a script directly instead of as a module
   - **Solution**: Use `python -m` syntax to run the script as a module
   - **Example**: Change `python fca_dashboard/examples/classifier_example.py` to `python -m fca_dashboard.examples.classifier_example`

4. **ImportError: cannot import name 'X' from 'Y'**
   - **Cause**: Missing or incorrect implementation of the imported item
   - **Solution**: Check that the imported item exists in the specified module and is spelled correctly
   - **Verification**: Inspect the module file to confirm the item exists

### Package Compatibility Issues

1. **ImportError: cannot import name '_imaging' from 'PIL'**
   - **Cause**: Pillow installation issue
   - **Solution**: Reinstall Pillow with `pip uninstall -y pillow && pip install pillow`
   - **Alternative**: Try `pip install --force-reinstall pillow`

2. **ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'**
   - **Cause**: NumPy installation issue
   - **Solution**: Reinstall NumPy with `pip uninstall -y numpy && pip install numpy`
   - **Alternative**: Try a specific version: `pip install numpy==1.23.5`

3. **ImportError: cannot import name '_c_internal_utils' from 'matplotlib'**
   - **Cause**: Matplotlib installation issue
   - **Solution**: Reinstall matplotlib with `pip uninstall -y matplotlib && pip install matplotlib`
   - **Verification**: Check if matplotlib works with `python -c "import matplotlib; print(matplotlib.__version__)"`

4. **ModuleNotFoundError: No module named 'kiwisolver._cext'**
   - **Cause**: Kiwisolver installation issue (dependency of matplotlib)
   - **Solution**: Reinstall kiwisolver with `pip uninstall -y kiwisolver && pip install kiwisolver`
   - **Details**: This is a common issue when C extensions aren't properly compiled. The reinstallation forces a rebuild of the C extensions.
   - **Alternative**: If reinstallation doesn't work, try `pip install --force-reinstall --no-binary kiwisolver kiwisolver`

5. **ValueError: numpy.ndarray size changed, may indicate binary incompatibility**
   - **Cause**: Version mismatch between NumPy and packages that depend on it
   - **Solution**: Reinstall all scientific packages in the correct order:

     ```bash
     pip uninstall -y numpy scipy pandas scikit-learn
     pip install numpy scipy pandas scikit-learn
     ```

### Data File Issues

1. **FileNotFoundError: Could not find settings file**
   - **Cause**: Settings file not found at expected location
   - **Solution**: Verify the path to settings.yml is correct
   - **Verification**: Check the actual path with `python -c "from fca_dashboard.utils.path_util import get_config_path; print(get_config_path('settings.yml'))"`

2. **FileNotFoundError: Could not find data file**
   - **Cause**: Training data file not found at expected location
   - **Solution**: Verify the path to eq_ids.csv is correct in settings.yml
   - **Verification**: Use the verification script to check data file availability

3. **UnicodeDecodeError when reading CSV file**
   - **Cause**: Encoding issues in the data file
   - **Solution**: Specify the correct encoding when reading the file:

     ```python
     df = pd.read_csv("path/to/file.csv", encoding='latin1')  # or try 'utf-8', 'cp1252', etc.
     ```

4. **pandas.errors.ParserError: Error tokenizing data**
   - **Cause**: CSV format issues (delimiters, quotes, etc.)
   - **Solution**: Inspect the file and specify the correct parameters:

     ```python
     df = pd.read_csv("path/to/file.csv", delimiter=',', quoting=csv.QUOTE_MINIMAL)
     ```

### Visualization Issues

1. **ImportError: matplotlib or seaborn related errors**
   - **Cause**: Issues with visualization dependencies
   - **Solution**: Use the simplified version without visualizations: `make run-classifier-simple`
   - **Alternative**: Fix the visualization dependencies as described in the "Package Compatibility Issues" section

2. **RuntimeError: Python is not installed as a framework (macOS)**
   - **Cause**: macOS-specific matplotlib issue
   - **Solution**: Create a file at `~/.matplotlib/matplotlibrc` with the content:

     ```
     backend: TkAgg
     ```

3. **UserWarning: Matplotlib is currently using agg, which is a non-GUI backend**
   - **Cause**: Running in an environment without display capabilities
   - **Solution**: Use `plt.savefig()` instead of `plt.show()` or set a different backend

### Model Training and Prediction Issues

1. **ValueError: Input contains NaN, infinity or a value too large**
   - **Cause**: Missing or invalid values in the input data
   - **Solution**: Clean the data before training:

     ```python
     import numpy as np
     df = df.replace([np.inf, -np.inf], np.nan).dropna()
     ```

2. **MemoryError during model training**
   - **Cause**: Dataset too large for available memory
   - **Solution**: Reduce dataset size or use incremental learning:

     ```python
     # Sample the dataset
     df = df.sample(frac=0.5, random_state=42)
     ```

3. **UndefinedMetricWarning: Precision/recall is ill-defined**
   - **Cause**: Some classes have no predicted samples in the test set
   - **Solution**: This is normal for rare classes. Use stratified sampling:

     ```python
     from sklearn.model_selection import StratifiedKFold
     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
     ```

### Virtual Environment Issues

1. **Virtual environment not activating properly**
   - **Cause**: Incorrect activation command or path
   - **Solution**:
     - Windows: `.\.venv\Scripts\activate`
     - Unix/Linux/macOS: `source .venv/bin/activate`
   - **Verification**: Check if the environment is active with `which python` or `where python`

2. **Packages installed in wrong environment**
   - **Cause**: Installing packages without activating the virtual environment
   - **Solution**: Ensure the virtual environment is activated before installing packages
   - **Verification**: Check the installation path with `pip show <package-name>`

### Environment-Specific Issues

#### Windows-Specific Issues

1. **Path length limitations**
   - **Cause**: Windows has a 260-character path length limit
   - **Solution**: Use shorter paths or enable long path support:

     ```
     # In PowerShell as Administrator
     Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
     ```

2. **Command prompt encoding issues**
   - **Cause**: Default encoding may not support all characters
   - **Solution**: Set UTF-8 encoding:

     ```
     chcp 65001
     ```

3. **Backslash vs. forward slash in paths**
   - **Cause**: Windows uses backslashes in paths, but Python often expects forward slashes
   - **Solution**: Use raw strings or double backslashes:

     ```python
     # Raw string
     path = r"C:\Repos\fca-dashboard4\fca_dashboard"
     
     # Double backslashes
     path = "C:\\Repos\\fca-dashboard4\\fca_dashboard"
     
     # Forward slashes (usually works in Python)
     path = "C:/Repos/fca-dashboard4/fca_dashboard"
     ```

4. **DLL load failed errors**
   - **Cause**: Missing Visual C++ Redistributable
   - **Solution**: Install the appropriate Visual C++ Redistributable from Microsoft

#### macOS-Specific Issues

1. **Matplotlib framework issues**
   - **Cause**: macOS requires matplotlib to be installed as a framework
   - **Solution**: Create a matplotlibrc file:

     ```
     # ~/.matplotlib/matplotlibrc
     backend: TkAgg
     ```

2. **XCode dependency issues**
   - **Cause**: Some packages require XCode command-line tools
   - **Solution**: Install with:

     ```bash
     xcode-select --install
     ```

3. **OpenSSL issues**
   - **Cause**: macOS may use its own SSL libraries
   - **Solution**: Install OpenSSL via Homebrew:

     ```bash
     brew install openssl
     export LDFLAGS="-L/usr/local/opt/openssl/lib"
     export CPPFLAGS="-I/usr/local/opt/openssl/include"
     ```

#### Linux-Specific Issues

1. **Missing system libraries**
   - **Cause**: Some Python packages require system libraries
   - **Solution**: Install required development packages:

     ```bash
     # Ubuntu/Debian
     sudo apt-get install python3-dev build-essential
     
     # CentOS/RHEL
     sudo yum install python3-devel gcc
     ```

2. **Permission issues**
   - **Cause**: Insufficient permissions to access files or directories
   - **Solution**: Adjust permissions:

     ```bash
     chmod -R 755 fca_dashboard
     ```

## References

- [ETL Pipeline v4 Implementation Guide](../../docs/guide/guide.md)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [Scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn documentation](https://seaborn.pydata.org/documentation.html)
- [Imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Python packaging guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
- [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- [Windows Long Paths](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)

## Revision History

| Version | Date       | Author   | Changes                                                                                                                                   |
| ------- | ---------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | 2025-03-04 | ETL Team | Initial version                                                                                                                           |
| 1.1     | 2025-03-04 | ETL Team | Expanded troubleshooting sections, added environment-specific issues, added detailed examples and verification steps, expanded references |
