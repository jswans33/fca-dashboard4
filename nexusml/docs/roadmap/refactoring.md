# Response to Code Review: NexusML Refactoring Plan

Thank you for the detailed code review! Your suggestions are excellent and will
significantly improve the maintainability and flexibility of the NexusML
codebase. I'll address each point and outline an implementation plan.

## Implementation Plan for Suggested Improvements

### 1. Clean Up Deprecated Models

I agree that consolidating the duplicate model files is a high priority. Here's
how we'll implement this:

```python
# nexusml/core/model.py

def build_enhanced_model(sampling_strategy="random_over", **kwargs):
    """
    Build an enhanced model with configurable sampling strategy

    Args:
        sampling_strategy (str): Sampling strategy to use ("random_over" or "smote")
        **kwargs: Additional parameters for the model
    """
    # Implementation that handles both strategies
    if sampling_strategy == "smote":
        from imblearn.over_sampling import SMOTE
        oversampler = SMOTE(random_state=42)
    else:  # default to random_over
        from imblearn.over_sampling import RandomOverSampler
        oversampler = RandomOverSampler(random_state=42)

    # Rest of model building logic
```

We'll then update `train_enhanced_model()` to use this parameterized function
and move the deprecated files to an `archive` folder with clear documentation.

### 2. Centralize Config and Remove Hardcoded Paths

We'll create a dedicated config module:

```python
# nexusml/config/__init__.py

import os
from pathlib import Path
import yaml

# Default paths
DEFAULT_PATHS = {
    "training_data": "ingest/data/eq_ids.csv",
    "output_dir": "outputs",
    "config_file": "config/settings.yml",
}

# Try to load from fca_dashboard if available (only once at import time)
try:
    from fca_dashboard.utils.path_util import get_config_path, resolve_path
    FCA_DASHBOARD_AVAILABLE = True
except ImportError:
    FCA_DASHBOARD_AVAILABLE = False

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def get_data_path(path_key="training_data"):
    """Get a data path from config or defaults."""
    # Implementation that handles both standalone and fca_dashboard contexts
```

Then we'll update all files to use this centralized config instead of hardcoded
paths.

### 3. Streamline SMOTE vs. Random OverSampler

We'll implement a unified class imbalance handling function:

```python
# nexusml/core/model.py

def handle_class_imbalance(X, y, method="random_over", **kwargs):
    """
    Handle class imbalance with configurable method

    Args:
        X: Features
        y: Target variables
        method (str): Method to use ("random_over" or "smote")
        **kwargs: Additional parameters for the oversampler

    Returns:
        Tuple: (Resampled features, resampled targets)
    """
    # Implementation that handles both methods
```

### 4. Externalize MasterFormat Mappings

We'll move the mappings to JSON files:

```
nexusml/
  config/
    mappings/
      masterformat_primary.json
      masterformat_equipment.json
```

And update the code to load these mappings:

```python
# nexusml/core/feature_engineering.py

import json
from nexusml.config import get_project_root

def load_masterformat_mappings():
    """Load MasterFormat mappings from JSON files."""
    root = get_project_root()

    with open(root / "config" / "mappings" / "masterformat_primary.json") as f:
        primary_mapping = json.load(f)

    with open(root / "config" / "mappings" / "masterformat_equipment.json") as f:
        equipment_specific_mapping = json.load(f)

    return primary_mapping, equipment_specific_mapping

def enhanced_masterformat_mapping(uniformat_class, system_type, equipment_category, equipment_subcategory=None):
    """Enhanced mapping with better handling of specialty equipment types."""
    primary_mapping, equipment_specific_mapping = load_masterformat_mappings()

    # Rest of the function remains the same
```

### 5. Consolidate Evaluation Functions

We'll ensure all evaluation code is in `nexusml/core/evaluation.py` and update
any references to use these centralized functions.

### 6. DRY up Examples Folder

We'll create a common utilities module for examples:

```python
# nexusml/examples/common.py

from nexusml.core.model import train_enhanced_model, predict_with_enhanced_model
from nexusml.config import get_data_path

def run_training_and_prediction(
    data_path=None,
    description="Heat Exchanger for Chilled Water system",
    service_life=20.0,
    output_dir=None
):
    """Run a standard training and prediction workflow."""
    # Use config for default paths
    if data_path is None:
        data_path = get_data_path("training_data")

    # Training
    model, df = train_enhanced_model(data_path)

    # Prediction
    prediction = predict_with_enhanced_model(model, description, service_life)

    # Output handling
    # ...

    return model, df, prediction
```

Then simplify the example scripts to use this common function.

### 7. Implement Unified Logging

We'll create a dedicated logging module:

```python
# nexusml/utils/logging.py

import logging
import sys
from pathlib import Path

# Try to use fca_dashboard logging if available
try:
    from fca_dashboard.utils.logging_config import configure_logging as fca_configure_logging
    FCA_LOGGING_AVAILABLE = True
except ImportError:
    FCA_LOGGING_AVAILABLE = False

def configure_logging(level="INFO", log_file=None, simple_format=False):
    """Configure application logging."""
    if FCA_LOGGING_AVAILABLE:
        return fca_configure_logging(level=level, log_file=log_file, simple_format=simple_format)

    # Fallback to standard logging
    # Implementation...

def get_logger(name="nexusml"):
    """Get a logger instance."""
    return logging.getLogger(name)
```

### 8. Unify Training Functions

We'll consolidate all training functions into a single, flexible implementation
in `nexusml/core/model.py`.

### 9. Fix Package Naming

We'll update the `pyproject.toml` to use "nexusml" as the project name:

```toml
[project]
name = "nexusml"
version = "0.1.0"
description = "Modern machine learning classification engine"
```

### 10. Apply Best Practices

We'll implement consistent:

- Type hints (following our updated mypy configuration)
- Docstrings (using Google style)
- Logging (replacing print statements)
- Error handling (with centralized utility functions)
