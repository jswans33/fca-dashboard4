from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from fca_dashboard.main import main, parse_args


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
def test_main_file_not_found_error(
    mock_resolve_path: MagicMock, mock_configure_logging: MagicMock, mock_get_logger: MagicMock
) -> None:
    """Test main function handling FileNotFoundError."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("nonexistent_file.yml")

    # Simulate FileNotFoundError when trying to load settings
    with (
        patch("fca_dashboard.main.get_settings", side_effect=FileNotFoundError("File not found")),
        patch("sys.argv", ["main.py", "--config", "nonexistent_file.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 1
    mock_logger.error.assert_called_once()
    assert "File not found" in mock_logger.error.call_args[0][0]


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
def test_main_yaml_error(
    mock_resolve_path: MagicMock, mock_configure_logging: MagicMock, mock_get_logger: MagicMock
) -> None:
    """Test main function handling YAMLError."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("invalid_yaml.yml")

    # Simulate YAMLError when trying to load settings
    with (
        patch("fca_dashboard.main.get_settings", side_effect=yaml.YAMLError("Invalid YAML")),
        patch("sys.argv", ["main.py", "--config", "invalid_yaml.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 1
    mock_logger.error.assert_called_once()
    assert "YAML configuration error" in mock_logger.error.call_args[0][0]


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
    assert exit_code == 1
    mock_logger.exception.assert_called_once()
    assert "Unexpected error" in mock_logger.exception.call_args[0][0]
