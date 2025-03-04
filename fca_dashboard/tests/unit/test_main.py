from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from fca_dashboard.main import main, parse_args, run_etl_pipeline
from fca_dashboard.utils.error_handler import ConfigurationError, DataExtractionError


@patch("sys.argv", ["main.py", "--config", "config/settings.yml"])
def test_main_runs_successfully() -> None:
    """Test that the main function runs successfully with default arguments."""
    exit_code = main()
    assert exit_code == 0


def test_parse_args_defaults() -> None:
    """Test that parse_args returns expected defaults."""
    with patch("sys.argv", ["main.py"]):
        args = parse_args()
        assert "settings.yml" in args.config
        assert args.log_level == "INFO"
        assert args.excel_file is None
        assert args.table_name is None


def test_parse_args_custom_values() -> None:
    """Test that parse_args handles custom arguments correctly."""
    with patch(
        "sys.argv",
        [
            "main.py",
            "--config",
            "custom_config.yml",
            "--log-level",
            "DEBUG",
            "--excel-file",
            "data.xlsx",
            "--table-name",
            "equipment",
        ],
    ):
        args = parse_args()
        assert args.config == "custom_config.yml"
        assert args.log_level == "DEBUG"
        assert args.excel_file == "data.xlsx"
        assert args.table_name == "equipment"


@patch("fca_dashboard.main.get_settings")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.resolve_path")
def test_main_with_excel_file_and_table(
    mock_resolve_path: MagicMock,
    mock_get_logger: MagicMock,
    mock_configure_logging: MagicMock,
    mock_get_settings: MagicMock,
) -> None:
    """Test main function with excel_file and table_name arguments."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    mock_settings.get.return_value = "sqlite:///test.db"
    mock_resolve_path.side_effect = lambda x: Path(f"/resolved/{x}")

    # Run with excel file and table name
    with patch(
        "sys.argv",
        ["main.py", "--config", "config/settings.yml", "--excel-file", "data.xlsx", "--table-name", "equipment"],
    ):
        exit_code = main()

    # Verify
    assert exit_code == 0
    assert mock_logger.info.call_count >= 5  # Multiple info logs
    # Check that excel file and table name were logged
    mock_resolve_path.assert_any_call("data.xlsx")
    # Check that the log message contains the excel file path (exact format may vary by OS)
    excel_log_found = False
    for call_args in mock_logger.info.call_args_list:
        if "Would process Excel file:" in call_args[0][0] and "data.xlsx" in call_args[0][0]:
            excel_log_found = True
            break
    assert excel_log_found, "Excel file log message not found"
    mock_logger.info.assert_any_call("Would process table: equipment")


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
@patch("fca_dashboard.main.ErrorHandler")
def test_main_file_not_found_error(
    mock_error_handler_class: MagicMock,
    mock_resolve_path: MagicMock,
    mock_configure_logging: MagicMock,
    mock_get_logger: MagicMock,
) -> None:
    """Test main function handling FileNotFoundError."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("nonexistent_file.yml")
    
    # Setup error handler mock
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = 1
    mock_error_handler_class.return_value = mock_error_handler

    # Simulate FileNotFoundError when trying to load settings
    with (
        patch("fca_dashboard.main.get_settings", side_effect=FileNotFoundError("File not found")),
        patch("sys.argv", ["main.py", "--config", "nonexistent_file.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 1
    mock_error_handler.handle_error.assert_called_once()
    # Verify the error passed to handle_error is a FileNotFoundError
    args, _ = mock_error_handler.handle_error.call_args
    assert isinstance(args[0], FileNotFoundError)


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
@patch("fca_dashboard.main.ErrorHandler")
def test_main_yaml_error(
    mock_error_handler_class: MagicMock,
    mock_resolve_path: MagicMock,
    mock_configure_logging: MagicMock,
    mock_get_logger: MagicMock,
) -> None:
    """Test main function handling YAMLError."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("invalid_yaml.yml")
    
    # Setup error handler mock
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = 2  # ConfigurationError code
    mock_error_handler_class.return_value = mock_error_handler

    # Simulate YAMLError when trying to load settings
    with (
        patch("fca_dashboard.main.get_settings", side_effect=yaml.YAMLError("Invalid YAML")),
        patch("sys.argv", ["main.py", "--config", "invalid_yaml.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 2  # ConfigurationError code
    mock_error_handler.handle_error.assert_called_once()
    # Verify the error passed to handle_error is a ConfigurationError
    args, _ = mock_error_handler.handle_error.call_args
    assert isinstance(args[0], ConfigurationError)
    assert "YAML configuration error" in str(args[0])


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
def test_main_unexpected_error(
    mock_resolve_path: MagicMock, mock_configure_logging: MagicMock, mock_get_logger: MagicMock
) -> None:
    """Test main function handling unexpected exceptions."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("config.yml")

    # Simulate unexpected exception
    with (
        patch("fca_dashboard.main.get_settings", side_effect=Exception("Unexpected error")),
        patch("sys.argv", ["main.py", "--config", "config.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 99  # Generic error code from ErrorHandler
    # The error is now handled by ErrorHandler, not directly in main


def test_run_etl_pipeline_success() -> None:
    """Test that run_etl_pipeline runs successfully with valid arguments."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.config = "config/settings.yml"
    mock_args.excel_file = None
    mock_args.table_name = None
    mock_log = MagicMock()

    with (
        patch("fca_dashboard.main.resolve_path", return_value=Path("config/settings.yml")),
        patch("fca_dashboard.main.get_settings", return_value={"databases.sqlite.url": "sqlite:///test.db"}),
    ):
        exit_code = run_etl_pipeline(mock_args, mock_log)

    # Verify
    assert exit_code == 0
    assert mock_log.info.call_count >= 4  # Multiple info logs


def test_run_etl_pipeline_configuration_error() -> None:
    """Test that run_etl_pipeline raises ConfigurationError for YAML errors."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.config = "invalid_config.yml"
    mock_log = MagicMock()

    with (
        patch("fca_dashboard.main.resolve_path", return_value=Path("invalid_config.yml")),
        patch("fca_dashboard.main.get_settings", side_effect=yaml.YAMLError("Invalid YAML")),
        pytest.raises(ConfigurationError) as exc_info,
    ):
        run_etl_pipeline(mock_args, mock_log)

    # Verify
    assert "YAML configuration error" in str(exc_info.value)


def test_run_etl_pipeline_excel_file_not_found() -> None:
    """Test that run_etl_pipeline raises DataExtractionError for missing Excel files."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.config = "config/settings.yml"
    mock_args.excel_file = "nonexistent.xlsx"
    mock_args.table_name = None
    mock_log = MagicMock()

    with (
        patch("fca_dashboard.main.resolve_path", side_effect=[
            Path("config/settings.yml"),  # First call for config file
            FileNotFoundError("Excel file not found"),  # Second call for Excel file
        ]),
        patch("fca_dashboard.main.get_settings", return_value={"databases.sqlite.url": "sqlite:///test.db"}),
        pytest.raises(DataExtractionError) as exc_info,
    ):
        run_etl_pipeline(mock_args, mock_log)

    # Verify
    assert "Excel file not found" in str(exc_info.value)


def test_main_with_error_handler() -> None:
    """Test that main uses ErrorHandler to handle exceptions."""
    # Setup mocks
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = 42

    with (
        patch("fca_dashboard.main.ErrorHandler", return_value=mock_error_handler),
        patch("fca_dashboard.main.run_etl_pipeline", side_effect=Exception("Test error")),
        patch("fca_dashboard.main.configure_logging"),
        patch("fca_dashboard.main.get_logger"),
        patch("sys.argv", ["main.py"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 42
    mock_error_handler.handle_error.assert_called_once()
