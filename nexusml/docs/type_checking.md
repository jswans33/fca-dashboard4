# Type Checking in NexusML

This document outlines the approach to type checking in the NexusML project.

## Configuration

NexusML uses [mypy](https://mypy.readthedocs.io/) for static type checking. The
configuration is defined in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
disallow_incomplete_defs = false
check_untyped_defs = true
ignore_missing_imports = true
```

## Type Checking Philosophy

We've adopted a balanced approach to type checking that aims to:

1. **Provide type safety** for critical components and public APIs
2. **Allow flexibility** for complex ML operations and internal implementation
   details
3. **Improve documentation** through meaningful type annotations
4. **Catch type-related bugs** early in the development process

## Guidelines for Type Annotations

### When to Use Type Annotations

- **Always annotate**:

  - Public API function parameters and return types
  - Class attributes
  - Complex data structures

- **Consider annotating**:

  - Internal helper functions
  - Local variables when the type is not obvious

- **Minimal annotations for**:
  - Very simple functions with obvious types
  - Complex ML operations where typing is cumbersome

### Type Annotation Best Practices

1. **Use specific types** when possible:

   ```python
   def process_data(data: pd.DataFrame) -> pd.DataFrame:
       # ...
   ```

2. **Use Union for multiple types**:

   ```python
   from typing import Union

   def handle_input(value: Union[str, int]) -> str:
       # ...
   ```

3. **Use Optional for nullable values**:

   ```python
   from typing import Optional

   def find_item(items: list, key: Optional[str] = None) -> Optional[dict]:
       # ...
   ```

4. **Document complex type annotations**:

   ```python
   # This operation returns a complex nested structure that's difficult to type precisely
   result = complex_operation()  # type: ignore
   ```

5. **Use TypedDict for dictionaries with known structure**:

   ```python
   from typing import TypedDict

   class ModelResult(TypedDict):
       accuracy: float
       f1_score: float
       predictions: list[str]
   ```

## Handling Third-Party Libraries

Many ML libraries have complex typing that can be difficult to represent
accurately. We use the following strategies:

1. **Use `ignore_missing_imports = true`** to handle libraries without type
   stubs
2. **Use specific type stubs** when available (e.g., pandas-stubs)
3. **Use `# type: ignore`** comments for specific lines when necessary, with
   explanatory comments

## Type Checking in CI/CD

Type checking is part of our CI/CD pipeline:

```bash
# Run mypy on the codebase
mypy nexusml
```

## Rationale for Current Configuration

- **disallow_untyped_defs = false**: Allows functions without type annotations,
  which is helpful for rapid prototyping and complex ML code
- **disallow_incomplete_defs = false**: Allows partial type annotations, which
  is useful when some parameters are difficult to type
- **check_untyped_defs = true**: Still checks the body of functions without
  annotations, providing some type safety
- **warn_return_any = true**: Encourages explicit return types rather than
  implicit Any
- **ignore_missing_imports = true**: Prevents errors from third-party libraries
  without type stubs

This configuration balances type safety with development flexibility, which is
particularly important for ML projects where complex data transformations are
common.
