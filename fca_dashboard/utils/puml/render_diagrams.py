"""
PlantUML Diagram Renderer

This script renders PlantUML diagrams to PNG or SVG images using the plantuml Python package.
It can be used as an alternative to the HTML viewer if you prefer local rendering.

Requirements:
- Python 3.6+
- plantuml package (pip install plantuml)
- Internet connection (for the PlantUML server)

Usage:
python render_diagrams.py [--format=svg|png] [--source=<source_dir>]
"""

import os
import plantuml
import sys
import argparse
from pathlib import Path

# Default source directory for diagrams
DEFAULT_SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "docs", "diagrams")

def render_diagram(puml_file, output_dir=None, format="png"):
    """
    Render a PlantUML diagram to an image.
    
    Args:
        puml_file (str): Path to the PlantUML file
        output_dir (str): Directory to save the rendered image
        format (str): Output format (png or svg)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_SOURCE_DIR, "output")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Preserve directory structure relative to the source directory
    rel_path = os.path.relpath(os.path.dirname(puml_file), DEFAULT_SOURCE_DIR)
    if rel_path != '.':
        # Create subdirectory in output_dir to match the source structure
        output_subdir = os.path.join(output_dir, rel_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
    else:
        output_subdir = output_dir
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(puml_file))[0]
    output_file = os.path.join(output_subdir, f"{base_name}.{format}")
    
    # Create a PlantUML server instance with the appropriate URL for the format
    if format.lower() == "svg":
        plantuml_server = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/svg/')
    else:
        plantuml_server = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/img/')
    
    try:
        # Generate the diagram
        print(f"Rendering {puml_file} to {format.upper()}...")
        plantuml_server.processes_file(puml_file, outfile=output_file)
        print(f"Saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error rendering {puml_file}: {e}")
        return False

def render_all_diagrams(directory=DEFAULT_SOURCE_DIR, output_dir=None, format="png"):
    """
    Render all PlantUML diagrams in the specified directory.
    
    Args:
        directory (str): Directory containing PlantUML files
        output_dir (str): Directory to save rendered images (default: <directory>/diagram_images)
        format (str): Output format (png or svg)
    """
    # Find all .puml files in the directory and its subdirectories
    puml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.puml'):
                puml_files.append(os.path.join(root, file))
    
    if not puml_files:
        print(f"No .puml files found in {directory}")
        return
    
    # Create output directory if not specified
    if output_dir is None:
        # Default output directory is docs/diagrams/output
        output_dir = os.path.join(DEFAULT_SOURCE_DIR, "output")
    
    # Render each diagram
    success_count = 0
    for puml_path in puml_files:
        if render_diagram(puml_path, output_dir, format):
            success_count += 1
    
    # Print summary
    print(f"\nRendered {success_count} of {len(puml_files)} diagrams to {output_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render PlantUML diagrams to images")
    parser.add_argument("--format", choices=["png", "svg"], default="svg", help="Output format (png or svg)")
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="Source directory for PlantUML files")
    parser.add_argument("--output", default=None, help="Output directory for rendered images")
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Render all diagrams
    render_all_diagrams(args.source, args.output, args.format)
    
    print("\nDone!")

if __name__ == "__main__":
    main()