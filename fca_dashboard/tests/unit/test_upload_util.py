"""
Unit tests for the file upload utility module.

This module contains tests for the file upload utility functions in the
fca_dashboard.utils.upload_util module.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from fca_dashboard.utils.upload_util import FileUploadError, upload_file


def test_upload_file_success() -> None:
    """Test successful file upload."""
    # Create a temporary file to simulate an upload
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"dummy content")
        tmp_path = tmp.name

    try:
        # Destination could be a temporary directory
        with tempfile.TemporaryDirectory() as dest_dir:
            # Call the uploader function
            result = upload_file(tmp_path, dest_dir)
            assert result is True
            
            # Verify file exists in destination
            dest_path = os.path.join(dest_dir, os.path.basename(tmp_path))
            assert os.path.exists(dest_path)
            
            # Verify content was copied correctly
            with open(dest_path, "rb") as f:
                content = f.read()
                assert content == b"dummy content"
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_file_not_found() -> None:
    """Test upload with non-existent source file."""
    # Use a file path that doesn't exist
    non_existent_file = "nonexistent_file.txt"
    with tempfile.TemporaryDirectory() as dest_dir:
        with pytest.raises(FileNotFoundError):
            upload_file(non_existent_file, dest_dir)


def test_upload_file_invalid_destination() -> None:
    """Test upload with invalid destination directory."""
    # Create a dummy file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"dummy content")
        tmp_path = tmp.name
    
    try:
        # Provide an invalid destination (non-existent directory)
        invalid_dest = os.path.join(tempfile.gettempdir(), "nonexistent", "subdir")
        with pytest.raises(FileNotFoundError):
            upload_file(tmp_path, invalid_dest)
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@patch("fca_dashboard.utils.upload_util.shutil.copy")
def test_upload_file_copy_error(mock_copy) -> None:
    """Test handling of copy errors during upload."""
    # Mock shutil.copy to raise an exception
    mock_copy.side_effect = Exception("Copy failed")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"dummy content")
        tmp_path = tmp.name
    
    try:
        # Attempt to upload the file
        with tempfile.TemporaryDirectory() as dest_dir:
            with pytest.raises(FileUploadError):
                upload_file(tmp_path, dest_dir)
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_file_with_duplicate_handling() -> None:
    """Test upload with duplicate file handling."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"original content")
        tmp_path = tmp.name
        filename = os.path.basename(tmp_path)
    
    try:
        with tempfile.TemporaryDirectory() as dest_dir:
            # First upload
            upload_file(tmp_path, dest_dir)
            
            # Create a second file with different content but upload it with the same name
            with tempfile.NamedTemporaryFile(delete=False) as tmp2:
                tmp2.write(b"new content")
                tmp2_path = tmp2.name
            
            try:
                # Create a copy of the file in the destination directory to simulate a duplicate upload
                dest_file = os.path.join(dest_dir, filename)
                
                # Second upload with the same filename (directly use the temp file but specify the target filename)
                upload_file(tmp2_path, dest_dir, target_filename=filename)
                
                # Check that both files exist with different names
                files = os.listdir(dest_dir)
                assert len(files) == 2
                
                # Verify one file has the original name
                assert filename in files
                
                # Verify the other file has a suffix added
                renamed_files = [f for f in files if f != filename]
                assert len(renamed_files) == 1
                assert renamed_files[0].startswith(os.path.splitext(filename)[0])
                
                # Check content of both files
                with open(os.path.join(dest_dir, filename), "rb") as f:
                    assert f.read() == b"original content"
                
                with open(os.path.join(dest_dir, renamed_files[0]), "rb") as f:
                    assert f.read() == b"new content"
            finally:
                if os.path.exists(tmp2_path):
                    os.unlink(tmp2_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@patch("fca_dashboard.utils.upload_util.get_logger")
def test_upload_file_with_logging(mock_get_logger) -> None:
    """Test that upload operations are properly logged."""
    # Setup mock logger
    mock_logger = mock_get_logger.return_value
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"dummy content")
        tmp_path = tmp.name
    
    try:
        # Upload the file
        with tempfile.TemporaryDirectory() as dest_dir:
            upload_file(tmp_path, dest_dir)
            
            # Verify logging calls
            assert mock_logger.info.call_count >= 1
            # Check that the log message contains the filename
            assert any(os.path.basename(tmp_path) in str(args) for args, _ in mock_logger.info.call_args_list)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_file_concurrent_duplicates() -> None:
    """Test handling duplicate uploads in quick succession."""
    # Create a temporary source file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"concurrent content")
        tmp_path = tmp.name

    try:
        with tempfile.TemporaryDirectory() as dest_dir:
            target_name = os.path.basename(tmp_path)
            # First upload with a specific target filename
            upload_file(tmp_path, dest_dir, target_filename=target_name)
            # Immediately perform a second upload using the same target filename
            upload_file(tmp_path, dest_dir, target_filename=target_name)
            
            # List all files in the destination directory
            files = os.listdir(dest_dir)
            # Expect two files: the original name and one with a timestamp appended
            assert len(files) == 2
            assert target_name in files
            duplicate = [f for f in files if f != target_name]
            assert len(duplicate) == 1
            # Verify that the duplicate filename contains an underscore
            assert "_" in duplicate[0]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_file_permission_error(monkeypatch) -> None:
    """Test handling of permission errors during upload."""
    # Patch shutil.copy to simulate a PermissionError
    import shutil
    monkeypatch.setattr(shutil, "copy", lambda src, dst: (_ for _ in ()).throw(PermissionError("Permission denied")))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"test permission")
        tmp_path = tmp.name

    try:
        with tempfile.TemporaryDirectory() as dest_dir:
            with pytest.raises(FileUploadError) as excinfo:
                upload_file(tmp_path, dest_dir)
            assert "Permission denied" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_file_with_target_filename() -> None:
    """Test upload with a provided target filename."""
    # Create a temporary source file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"original")
        tmp_path = tmp.name

    try:
        with tempfile.TemporaryDirectory() as dest_dir:
            target_name = "custom_name.txt"
            # First upload using the target filename
            upload_file(tmp_path, dest_dir, target_filename=target_name)
            
            # Create a second temporary source file with different content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp2:
                tmp2.write(b"updated")
                tmp2_path = tmp2.name

            try:
                # Upload the second file with the same target filename
                upload_file(tmp2_path, dest_dir, target_filename=target_name)
                
                files = os.listdir(dest_dir)
                # Expect two files: one exactly as "custom_name.txt" and another with a timestamp appended
                assert len(files) == 2
                assert target_name in files
                duplicate_files = [f for f in files if f != target_name]
                assert duplicate_files, "Expected a duplicate file with a modified name."
                
                # Check content of both files
                with open(os.path.join(dest_dir, target_name), "rb") as f:
                    assert f.read() == b"original"
                
                with open(os.path.join(dest_dir, duplicate_files[0]), "rb") as f:
                    assert f.read() == b"updated"
            finally:
                if os.path.exists(tmp2_path):
                    os.unlink(tmp2_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)