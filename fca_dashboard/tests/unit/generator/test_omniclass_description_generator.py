"""
Unit tests for the OmniClass description generator.

This module contains tests for the OmniClass description generator functionality,
including API client creation, prompt generation, response parsing, and batch processing.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add __init__.py to make the directory a package
with open(Path(__file__).parent / "__init__.py", "w") as f:
    f.write("# Generator tests package\n")

# Import the module to test
from fca_dashboard.generator.omniclass_description_generator import (
    AnthropicClient,
    ApiClientError,
    DescriptionGeneratorError,
    call_claude_api,
    create_client,
    generate_descriptions,
    generate_prompt,
    parse_response,
)


class TestOmniClassDescriptionGenerator(unittest.TestCase):
    """Test cases for the OmniClass description generator."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'OmniClass_Code': ['23-33 10 00', '23-33 20 00'],
            'OmniClass_Title': ['Air Distribution Systems', 'Heating Systems'],
            'Description': ['', '']
        })

    @patch('fca_dashboard.generator.omniclass_description_generator.anthropic.Anthropic')
    def test_create_client(self, mock_anthropic):
        """Test client creation with API key."""
        # Mock environment variable
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            # Call the function
            client = create_client()
            
            # Assert that Anthropic was called with the correct API key
            mock_anthropic.assert_called_once_with(api_key='test_key')
            
            # Assert that the client was returned
            self.assertEqual(client, mock_anthropic.return_value)

    @patch('fca_dashboard.generator.omniclass_description_generator.AnthropicClient')
    def test_generate_prompt(self, mock_anthropic_client):
        """Test prompt generation from batch data."""
        # Mock the AnthropicClient to avoid API key error
        mock_anthropic_client.return_value = MagicMock()
        
        # Generate prompt
        prompt = generate_prompt(self.sample_data)
        
        # Assert that the prompt contains the expected content
        self.assertIn('Air Distribution Systems', prompt)
        self.assertIn('Heating Systems', prompt)
        self.assertIn('23-33 10 00', prompt)
        self.assertIn('23-33 20 00', prompt)
        self.assertIn('JSON array', prompt)

    @patch('anthropic.Anthropic')
    def test_call_claude_api(self, mock_anthropic_class):
        """Test calling the Claude API with retry logic."""
        # Mock client and response
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '["Description 1", "Description 2"]'
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        # Mock the AnthropicClient to use our mocked client
        with patch('fca_dashboard.generator.omniclass_description_generator.AnthropicClient') as mock_anthropic_client:
            mock_api_client = MagicMock()
            mock_api_client.call.return_value = '["Description 1", "Description 2"]'
            mock_anthropic_client.return_value = mock_api_client
            
            # Call the function with a mock client that has an api_key attribute
            mock_client.api_key = 'test_key'
            result = call_claude_api(mock_client, "Test prompt")
            
            # Assert that the API client was called with the correct parameters
            mock_api_client.call.assert_called_once()
            
            # Assert that the result is correct
            self.assertEqual(result, '["Description 1", "Description 2"]')

    @patch('fca_dashboard.generator.omniclass_description_generator.AnthropicClient')
    def test_parse_response(self, mock_anthropic_client):
        """Test parsing the API response."""
        # Mock the AnthropicClient to avoid API key error
        mock_anthropic_client.return_value = MagicMock()
        
        # Test with valid JSON
        response = '{"some": "text"} ["Description 1", "Description 2"] more text'
        result = parse_response(response)
        self.assertEqual(result, ["Description 1", "Description 2"])
        
        # Test with invalid JSON
        response = 'No JSON here'
        result = parse_response(response)
        self.assertEqual(result, [])

    @patch('fca_dashboard.generator.omniclass_description_generator.OmniClassDescriptionGenerator')
    def test_generate_descriptions(self, mock_generator_class):
        """Test generating descriptions for OmniClass codes."""
        # Mock the generator instance
        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["Systems for distributing air in buildings", "Systems for heating buildings"]
        mock_generator_class.return_value = mock_generator
        
        # Create a temporary CSV file
        temp_input = 'temp_input.csv'
        temp_output = 'temp_output.csv'
        self.sample_data.to_csv(temp_input, index=False)
        
        try:
            # Call the function
            with patch('fca_dashboard.generator.omniclass_description_generator.AnthropicClient'):
                result_df = generate_descriptions(temp_input, temp_output)
            
            # Assert that the descriptions were added
            self.assertEqual(result_df.iloc[0]['Description'], "Systems for distributing air in buildings")
            self.assertEqual(result_df.iloc[1]['Description'], "Systems for heating buildings")
            
            # Assert that the output file was created
            self.assertTrue(os.path.exists(temp_output))
            
            # Load the output file and check its contents
            output_df = pd.read_csv(temp_output)
            self.assertEqual(output_df.iloc[0]['Description'], "Systems for distributing air in buildings")
            self.assertEqual(output_df.iloc[1]['Description'], "Systems for heating buildings")
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)

    @patch('fca_dashboard.generator.omniclass_description_generator.OmniClassDescriptionGenerator')
    def test_generate_descriptions_with_existing_descriptions(self, mock_generator_class):
        """Test that rows with existing descriptions are skipped."""
        # Create data with some existing descriptions
        data_with_descriptions = pd.DataFrame({
            'OmniClass_Code': ['23-33 10 00', '23-33 20 00', '23-33 30 00'],
            'OmniClass_Title': ['Air Distribution Systems', 'Heating Systems', 'Cooling Systems'],
            'Description': ['Existing description', '', '']
        })
        
        # Mock the generator instance
        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["Systems for heating buildings", "Systems for cooling buildings"]
        mock_generator_class.return_value = mock_generator
        
        # Create a temporary CSV file
        temp_input = 'temp_input_with_desc.csv'
        temp_output = 'temp_output_with_desc.csv'
        data_with_descriptions.to_csv(temp_input, index=False)
        
        try:
            # Call the function
            with patch('fca_dashboard.generator.omniclass_description_generator.AnthropicClient'):
                result_df = generate_descriptions(temp_input, temp_output)
            
            # Assert that the existing description was preserved
            self.assertEqual(result_df.iloc[0]['Description'], "Existing description")
            
            # Assert that new descriptions were added
            self.assertEqual(result_df.iloc[1]['Description'], "Systems for heating buildings")
            self.assertEqual(result_df.iloc[2]['Description'], "Systems for cooling buildings")
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)

    @patch('fca_dashboard.generator.omniclass_description_generator.BatchProcessor')
    def test_generate_descriptions_with_api_error(self, mock_batch_processor):
        """Test handling of API errors."""
        # Mock the batch processor to raise an error
        mock_processor = MagicMock()
        mock_processor.process.side_effect = DescriptionGeneratorError("Failed to generate descriptions: ANTHROPIC_API_KEY environment variable not set")
        mock_batch_processor.return_value = mock_processor
        
        # Create a temporary CSV file
        temp_input = 'temp_input_error.csv'
        temp_output = 'temp_output_error.csv'
        self.sample_data.to_csv(temp_input, index=False)
        
        try:
            # Call the function and expect an error
            with patch('fca_dashboard.generator.omniclass_description_generator.AnthropicClient'):
                with self.assertRaises(DescriptionGeneratorError):
                    generate_descriptions(temp_input, temp_output)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)


if __name__ == '__main__':
    unittest.main()