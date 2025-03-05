"""
PlantUML Utilities Test Script

This script tests the PlantUML utilities to ensure they are working correctly.
It checks that the diagrams can be loaded and rendered.

Usage:
    python test_puml.py
"""

import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the path so we can import the puml modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the puml modules
from puml import render_diagram, render_all_diagrams, DEFAULT_SOURCE_DIR

class TestPlantUML(unittest.TestCase):
    """Test case for PlantUML utilities."""
    
    def setUp(self):
        """Set up the test case."""
        # Get the directory containing this script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create a temporary output directory
        self.output_dir = os.path.join(self.script_dir, "test_output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after the test case."""
        # Remove the temporary output directory
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)
    
    def test_diagram_files_exist(self):
        """Test that the diagram files exist."""
        # Check that the diagram files exist
        diagram_files = [
            "classifier/classifier_model_diagram.puml",
            "classifier/classifier_data_model_diagram.puml",
            "classifier/classifier_pipeline_diagram.puml"
        ]
        
        for file in diagram_files:
            file_path = os.path.join(DEFAULT_SOURCE_DIR, file)
            self.assertTrue(os.path.exists(file_path), f"Diagram file not found: {file_path}")
    
    def test_render_diagram_png(self):
        """Test that a diagram can be rendered to PNG."""
        # Get the path to a diagram file
        diagram_file = os.path.join(DEFAULT_SOURCE_DIR, "classifier/classifier_model_diagram.puml")
        
        # Render the diagram with explicit output directory
        result = render_diagram(diagram_file, output_dir=self.output_dir, format="png")
        
        # Check that the rendering was successful
        self.assertTrue(result, "Diagram rendering failed")
        
        # Check that the output file was created (with preserved directory structure)
        output_file = os.path.join(self.output_dir, "classifier", "classifier_model_diagram.png")
        self.assertTrue(os.path.exists(output_file), f"Output file not found: {output_file}")
    
    def test_render_diagram_svg(self):
        """Test that a diagram can be rendered to SVG."""
        # Get the path to a diagram file
        diagram_file = os.path.join(DEFAULT_SOURCE_DIR, "classifier/classifier_model_diagram.puml")
        
        # Render the diagram with explicit output directory
        result = render_diagram(diagram_file, output_dir=self.output_dir, format="svg")
        
        # Check that the rendering was successful
        self.assertTrue(result, "Diagram rendering failed")
        
        # Check that the output file was created (with preserved directory structure)
        output_file = os.path.join(self.output_dir, "classifier", "classifier_model_diagram.svg")
        self.assertTrue(os.path.exists(output_file), f"Output file not found: {output_file}")

def main():
    """Run the tests."""
    unittest.main()

if __name__ == "__main__":
    main()