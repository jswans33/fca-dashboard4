# Feature Flags Documentation

## Overview

Feature flags are used in the training pipeline to maintain backward
compatibility while introducing new functionality. They allow users to toggle
between the new orchestrator-based implementation and the legacy implementation,
ensuring that existing scripts continue to work while new scripts can take
advantage of the improved architecture.

## Available Feature Flags

| Flag       | Default | Description                                       |
| ---------- | ------- | ------------------------------------------------- |
| `--legacy` | False   | Use legacy implementation instead of orchestrator |

## Implementation Details

### Command-Line Interface

The feature flags are implemented as command-line arguments in
`nexusml/core/cli/training_args.py`:

```python
# Feature flags
parser.add_argument(
    "--legacy",
    action="store_false",
    help="Use legacy implementation instead of orchestrator",
    dest="use_orchestrator",
)
```

The `--legacy` flag is implemented as a store_false action, which means that by
default, `use_orchestrator` is set to True. When the `--legacy` flag is
provided, `use_orchestrator` is set to False.

### TrainingArguments Class

The feature flags are stored in the `TrainingArguments` class:

```python
@dataclass
class TrainingArguments:
    # ...

    # Feature flags
    use_orchestrator: bool = True
```

### Entry Point Implementation

The feature flags are used in the entry point to determine which implementation
to use:

```python
def main():
    # ...

    if args.use_orchestrator:
        # Use the new orchestrator-based implementation
        model, metrics, viz_paths = train_with_orchestrator(args, logger)

        # Log metrics
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        # Log visualization paths if available
        if viz_paths:
            logger.info("Visualizations:")
            for key, path in viz_paths.items():
                logger.info(f"  {key}: {path}")
    else:
        # Use the legacy implementation
        logger.info("Using legacy pipeline implementation")
        classifier, df, metrics = train_model(
            data_path=args.data_path,
            feature_config_path=args.feature_config_path,
            sampling_strategy=args.sampling_strategy,
            test_size=args.test_size,
            random_state=args.random_state,
            optimize_params=args.optimize_hyperparameters,
            logger=logger,
        )

        # ...
```

## Usage Examples

### Using the New Implementation (Default)

```bash
python nexusml/train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv
```

### Using the Legacy Implementation

```bash
python nexusml/train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv --legacy
```

## Benefits of Feature Flags

### Backward Compatibility

Feature flags ensure that existing scripts continue to work with the new entry
point. This is important for maintaining compatibility with existing workflows
and avoiding disruption to users.

### Gradual Migration

Feature flags allow for a gradual migration from the legacy implementation to
the new implementation. Users can choose which implementation to use based on
their needs and gradually migrate to the new implementation as they become
comfortable with it.

### A/B Testing

Feature flags enable A/B testing between the legacy and new implementations.
This allows for comparing the performance, reliability, and other
characteristics of the two implementations.

### Risk Mitigation

Feature flags reduce the risk of introducing new functionality by allowing users
to fall back to the legacy implementation if issues are encountered with the new
implementation.

## Best Practices

### Clear Documentation

Document feature flags clearly, including their purpose, default values, and how
to use them. This helps users understand the available options and make informed
decisions.

### Sensible Defaults

Choose sensible defaults for feature flags. In this case, the default is to use
the new orchestrator-based implementation, which provides improved functionality
and follows best practices.

### Consistent Naming

Use consistent naming for feature flags. In this case, the flag is named
`--legacy` to indicate that it enables the legacy implementation, while the
internal variable is named `use_orchestrator` to indicate its purpose.

### Comprehensive Testing

Test both code paths thoroughly to ensure that both the legacy and new
implementations work correctly. This includes unit tests, integration tests, and
end-to-end tests.
