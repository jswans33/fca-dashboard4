import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def get_root_dir() -> Path:
    """Return the project's root directory (assuming this module is within project)."""
    return Path(__file__).resolve().parents[2]


def get_config_path(filename: str = "settings.yml") -> Path:
    """Get absolute path to the config file, ensuring it exists."""
    config_path = get_root_dir() / "fca_dashboard" / "config" / filename
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
    return config_path


def get_logs_path(filename: str = "fca_dashboard.log") -> Path:
    """Get absolute path to the log file, ensuring the logs directory exists."""
    logs_dir = get_root_dir() / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    return logs_dir / filename


def resolve_path(path: Union[str, Path], base_dir: Union[Path, None] = None) -> Path:
    """
    Resolve a path relative to the base directory or project root.

    Args:
        path: The path to resolve.
        base_dir: Optional base directory; defaults to the project's root directory.

    Returns:
        Resolved Path object.
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    if path_obj.exists():
        return path_obj.resolve()

    if base_dir is None:
        base_dir = get_root_dir()

    candidate_paths = [
        base_dir / path_obj,
        base_dir / "fca_dashboard" / path_obj,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            logger.debug(f"Resolved path '{path}' to '{candidate.resolve()}'")
            return candidate.resolve()

    logger.debug(f"Could not resolve path: {path_obj}. Returning unresolved path.")
    return path_obj
