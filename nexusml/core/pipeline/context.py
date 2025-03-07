"""
Pipeline Context Module

This module provides the PipelineContext class, which is responsible for
managing state during pipeline execution, providing access to shared resources,
and collecting metrics.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd


class PipelineContext:
    """
    Context for pipeline execution.

    The PipelineContext class manages state during pipeline execution, provides
    access to shared resources, and collects metrics. It serves as a central
    repository for data and metadata that needs to be shared between pipeline
    components.

    Attributes:
        data: Dictionary containing data shared between pipeline components.
        metrics: Dictionary containing metrics collected during pipeline execution.
        logs: List of log messages generated during pipeline execution.
        start_time: Time when the pipeline execution started.
        end_time: Time when the pipeline execution ended.
        status: Current status of the pipeline execution.
        config: Configuration for the pipeline execution.
        logger: Logger instance for logging messages.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize a new PipelineContext.

        Args:
            config: Configuration for the pipeline execution.
            logger: Logger instance for logging messages.
        """
        self.data: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "initialized"
        self.config: Dict[str, Any] = config or {}
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self._component_execution_times: Dict[str, float] = {}
        self._current_component: Optional[str] = None
        self._component_start_time: Optional[float] = None
        self._accessed_keys: Set[str] = set()
        self._modified_keys: Set[str] = set()

    def start(self) -> None:
        """
        Start the pipeline execution.

        This method initializes the start time and sets the status to "running".
        """
        self.start_time = time.time()
        self.status = "running"
        self.logger.info("Pipeline execution started")

    def end(self, status: str = "completed") -> None:
        """
        End the pipeline execution.

        This method records the end time, calculates the total execution time,
        and sets the status to the provided value.

        Args:
            status: Final status of the pipeline execution.
        """
        self.end_time = time.time()
        self.status = status

        if self.start_time is not None:
            execution_time = self.end_time - self.start_time
            self.metrics["total_execution_time"] = execution_time
            self.logger.info(
                f"Pipeline execution {status} in {execution_time:.2f} seconds"
            )
        else:
            self.logger.warning(
                f"Pipeline execution {status} but start time was not recorded"
            )

    def start_component(self, component_name: str) -> None:
        """
        Start timing a component's execution.

        Args:
            component_name: Name of the component being executed.
        """
        self._current_component = component_name
        self._component_start_time = time.time()
        self.logger.info(f"Starting component: {component_name}")

    def end_component(self) -> None:
        """
        End timing a component's execution and record the execution time.
        """
        if (
            self._current_component is not None
            and self._component_start_time is not None
        ):
            execution_time = time.time() - self._component_start_time
            self._component_execution_times[self._current_component] = execution_time
            self.logger.info(
                f"Component {self._current_component} completed in {execution_time:.2f} seconds"
            )
            self._current_component = None
            self._component_start_time = None
        else:
            self.logger.warning(
                "Cannot end component timing: no component is currently being timed"
            )

    def get_component_execution_times(self) -> Dict[str, float]:
        """
        Get the execution times for all components.

        Returns:
            Dictionary mapping component names to execution times in seconds.
        """
        return self._component_execution_times.copy()

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the context data.

        Args:
            key: Key to store the value under.
            value: Value to store.
        """
        self.data[key] = value
        self._modified_keys.add(key)
        self.logger.debug(f"Set context data: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context data.

        Args:
            key: Key to retrieve the value for.
            default: Default value to return if the key is not found.

        Returns:
            Value associated with the key, or the default value if the key is not found.
        """
        value = self.data.get(key, default)
        self._accessed_keys.add(key)
        return value

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the context data.

        Args:
            key: Key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self.data

    def add_metric(self, key: str, value: Any) -> None:
        """
        Add a metric to the metrics collection.

        Args:
            key: Key to store the metric under.
            value: Metric value to store.
        """
        self.metrics[key] = value
        self.logger.debug(f"Added metric: {key}={value}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.

        Returns:
            Dictionary containing all metrics.
        """
        return self.metrics.copy()

    def log(self, level: str, message: str, **kwargs) -> None:
        """
        Log a message and store it in the logs collection.

        Args:
            level: Log level (e.g., "INFO", "WARNING", "ERROR").
            message: Log message.
            **kwargs: Additional data to include in the log entry.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }
        self.logs.append(log_entry)

        # Log to the logger as well
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get all logs.

        Returns:
            List of log entries.
        """
        return self.logs.copy()

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline execution.

        Returns:
            Dictionary containing execution summary information.
        """
        summary = {
            "status": self.status,
            "metrics": self.get_metrics(),
            "component_execution_times": self.get_component_execution_times(),
            "accessed_keys": list(self._accessed_keys),
            "modified_keys": list(self._modified_keys),
        }

        if self.start_time is not None:
            summary["start_time"] = datetime.fromtimestamp(self.start_time).isoformat()

        if self.end_time is not None:
            summary["end_time"] = datetime.fromtimestamp(self.end_time).isoformat()
            if self.start_time is not None:
                summary["total_execution_time"] = self.end_time - self.start_time

        return summary

    def save_data(self, key: str, data: pd.DataFrame, path: Union[str, Path]) -> None:
        """
        Save data to a file and store the path in the context.

        Args:
            key: Key to store the path under.
            data: DataFrame to save.
            path: Path to save the data to.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(path_obj, index=False)
        self.set(key, str(path_obj))
        self.logger.info(f"Saved data to {path_obj}")

    def load_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a file.

        Args:
            path: Path to load the data from.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path_obj}")

        if path_obj.suffix.lower() == ".csv":
            data = pd.read_csv(path_obj)
        elif path_obj.suffix.lower() in (".xls", ".xlsx"):
            data = pd.read_excel(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

        self.logger.info(f"Loaded data from {path_obj}")
        return data
