# NexusML Scripts

This directory contains utility scripts for managing and maintaining the NexusML
project.

## Directory Structure Scripts

### 1. `migrate_outputs.py`

Consolidates multiple output directories into a single standardized output
directory structure.

**Purpose:**

- Moves files from `nexusml/outputs/` and `outputs/` to `nexusml/output/`
- Preserves the original directory structure within the target directory
- Creates appropriate subdirectories in the target location

**Usage:**

```bash
python -m nexusml.scripts.migrate_outputs
```

### 2. `migrate_examples.py`

Consolidates and organizes example files into a structured directory hierarchy.

**Purpose:**

- Moves examples from `examples/` to `nexusml/examples/`
- Categorizes examples by functionality (data_loading, pipeline, model_building,
  etc.)
- Creates README.md files for each category directory
- Preserves fca_dashboard examples in their original location

**Usage:**

```bash
python -m nexusml.scripts.migrate_examples
```

### 3. `remove_duplicates.py`

Identifies and removes duplicate files between different directories.

**Purpose:**

- Finds duplicate example files between `fca_dashboard/examples` and
  `nexusml/examples`
- Finds duplicate output files between `nexusml/output`, `nexusml/outputs`, and
  `outputs`
- Provides options for handling duplicates

**Options:**

- `--dry-run`: Only show what would be removed, don't actually delete files
- `--auto-remove`: Automatically remove all identified duplicates without
  prompting
- `--examples-only`: Only check for duplicate example files
- `--outputs-only`: Only check for duplicate output files

**Usage:**

```bash
# Interactive mode (default)
python -m nexusml.scripts.remove_duplicates

# Dry run - just show what would be removed
python -m nexusml.scripts.remove_duplicates --dry-run

# Auto-remove all duplicates
python -m nexusml.scripts.remove_duplicates --auto-remove

# Only check for duplicate example files
python -m nexusml.scripts.remove_duplicates --examples-only

# Only check for duplicate output files
python -m nexusml.scripts.remove_duplicates --outputs-only
```

### 4. `update_output_paths.py`

Updates references to old output directories in Python files.

**Purpose:**

- Finds Python files that reference old output paths (`outputs/` and
  `nexusml/outputs/`)
- Updates these references to use the standardized path (`nexusml/output/`)
- Only affects files within the nexusml directory and root Python files

**Options:**

- `--dry-run`: Only show what would be updated, don't actually modify files
- `--auto-update`: Automatically update all files without prompting

**Usage:**

```bash
# Interactive mode (default)
python -m nexusml.scripts.update_output_paths

# Dry run - just show what would be updated
python -m nexusml.scripts.update_output_paths --dry-run

# Auto-update all files
python -m nexusml.scripts.update_output_paths --auto-update
```

## Recommended Workflow

To standardize the directory structure, follow these steps:

1. **Migrate output files**:

   ```bash
   python -m nexusml.scripts.migrate_outputs
   ```

2. **Update path references in code**:

   ```bash
   python -m nexusml.scripts.update_output_paths
   ```

3. **Remove duplicate files**:

   ```bash
   python -m nexusml.scripts.remove_duplicates
   ```

4. **Migrate and organize examples**:
   ```bash
   python -m nexusml.scripts.migrate_examples
   ```

## Adding New Scripts

When adding new scripts to this directory:

1. Follow the naming convention: descriptive_name.py
2. Include a docstring at the top explaining the purpose and usage
3. Implement command-line argument parsing using argparse
4. Add logging with appropriate levels
5. Update this README.md with information about the new script
