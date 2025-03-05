#!/usr/bin/env python
"""
PlantUML Command Line Interface

This script provides a command-line interface for working with PlantUML diagrams.

Usage:
    python cli.py render [--format=<format>] [--source=<source_dir>] [--output=<output_dir>] [--file=<file>]
    python cli.py view
    python cli.py help

Commands:
    render      Render PlantUML diagrams to images
    view        Open the HTML viewer in the default web browser
    help        Show this help message

Options:
    --format=<format>     Output format (svg or png, default: svg)
    --source=<source_dir> Source directory for PlantUML files (default: docs/diagrams)
    --output=<output_dir> Directory to save rendered images (default: docs/diagrams/output)
    --file=<file>         Specific .puml file to render (default: all .puml files)
"""

import os
import sys
import webbrowser
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import the render_diagrams module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the render_diagrams module
from puml.render_diagrams import render_diagram, render_all_diagrams, DEFAULT_SOURCE_DIR

def show_help():
    """Show the help message."""
    print(__doc__)

def render_command(args):
    """Handle the render command."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Render PlantUML diagrams to images")
    parser.add_argument("--format", choices=["png", "svg"], default="svg", help="Output format (png or svg)")
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="Source directory for PlantUML files")
    parser.add_argument("--output", default=os.path.join(DEFAULT_SOURCE_DIR, "output"), help="Output directory for rendered images")
    parser.add_argument("--file", help="Specific .puml file to render")
    
    # Parse the arguments
    args = parser.parse_args(args)
    
    # Render the diagrams
    if args.file:
        # Render a specific file
        file_path = os.path.join(args.source, args.file) if not os.path.isabs(args.file) else args.file
        if os.path.exists(file_path):
            render_diagram(file_path, args.output, args.format)
        else:
            print(f"Error: File not found: {file_path}")
            return 1
    else:
        # Render all diagrams
        render_all_diagrams(args.source, args.output, args.format)
    
    return 0

def view_command():
    """Handle the view command."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the path to the HTML viewer
    viewer_path = os.path.join(script_dir, "view_diagrams.html")
    
    # Check if the viewer exists
    if not os.path.exists(viewer_path):
        print(f"Error: HTML viewer not found: {viewer_path}")
        return 1
    
    # Open the viewer in the default web browser
    print(f"Opening HTML viewer: {viewer_path}")
    webbrowser.open(f"file://{os.path.abspath(viewer_path)}")
    
    return 0

def main():
    """Main function."""
    # Parse command-line arguments
    if len(sys.argv) < 2:
        show_help()
        return 1
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    # Handle commands
    if command == "render":
        return render_command(args)
    elif command == "view":
        return view_command()
    elif command == "help":
        show_help()
        return 0
    else:
        print(f"Error: Unknown command: {command}")
        show_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())