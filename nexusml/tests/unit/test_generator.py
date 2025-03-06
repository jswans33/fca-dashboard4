"""
Unit tests for the generator module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nexusml.ingest.generator.omniclass import find_flat_sheet
from nexusml.ingest.generator.omniclass_description_generator import (
    AnthropicClient,
    BatchProcessor,
    OmniClassDescriptionGenerator,
)


class TestOmniClassGenerator:
    """Tests for the OmniClass generator module."""

    def test_find_flat_sheet(self):
        """Test the find_flat_sheet function."""
        # Test with a sheet name containing 'FLAT'
        sheet_names = ["Sheet1", "FLAT_VIEW", "Sheet3"]
        assert find_flat_sheet(sheet_names) == "FLAT_VIEW"

        # Test with no sheet name containing 'FLAT'
        sheet_names = ["Sheet1", "Sheet2", "Sheet3"]
        assert find_flat_sheet(sheet_names) is None

    @patch("nexusml.ingest.generator.omniclass_description_generator.AnthropicClient")
    def test_omniclass_description_generator(self, mock_client):
        """Test the OmniClassDescriptionGenerator class."""
        # Create a mock API client
        mock_client_instance = MagicMock()
        mock_client_instance.call.return_value = '[{"description": "Test description"}]'
        mock_client.return_value = mock_client_instance

        # Create test data
        data = pd.DataFrame({"OmniClass_Code": ["23-13 11 11"], "OmniClass_Title": ["Boilers"]})

        # Create generator
        generator = OmniClassDescriptionGenerator(api_client=mock_client_instance)

        # Test generate_prompt
        prompt = generator.generate_prompt(data)
        assert "Code: 23-13 11 11, Title: Boilers" in prompt

        # Test parse_response
        response = '[{"description": "Test description"}]'
        descriptions = generator.parse_response(response)
        assert descriptions == [{"description": "Test description"}]

        # Test generate
        descriptions = generator.generate(data)
        assert mock_client_instance.call.called
        assert len(descriptions) == 1

    @patch("nexusml.ingest.generator.omniclass_description_generator.OmniClassDescriptionGenerator")
    def test_batch_processor(self, mock_generator_class):
        """Test the BatchProcessor class."""
        # Create a mock generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["Test description"]
        mock_generator_class.return_value = mock_generator

        # Create test data
        data = pd.DataFrame({"OmniClass_Code": ["23-13 11 11"], "OmniClass_Title": ["Boilers"], "Description": [""]})

        # Create processor
        processor = BatchProcessor(generator=mock_generator, batch_size=1)

        # Test process
        result_df = processor.process(data)
        assert mock_generator.generate.called
        assert result_df["Description"][0] == "Test description"


if __name__ == "__main__":
    pytest.main()
