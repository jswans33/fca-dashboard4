"""
Utility functions for NexusML.
"""

# Import utility functions to expose at the package level
from typing import List

from nexusml.utils.csv_utils import clean_omniclass_csv, read_csv_safe, verify_csv_file
from nexusml.utils.logging import configure_logging, get_logger
from nexusml.utils.path_utils import (
    ensure_nexusml_in_path,
    find_data_files,
    get_nexusml_root,
    get_project_root,
    resolve_path,
    setup_notebook_environment,
)

__all__: List[str] = [
    "configure_logging",
    "get_logger",
    "verify_csv_file",
    "read_csv_safe",
    "clean_omniclass_csv",
    "get_project_root",
    "get_nexusml_root",
    "ensure_nexusml_in_path",
    "resolve_path",
    "find_data_files",
    "setup_notebook_environment",
]
