#!/usr/bin/env python
"""
Model Card Tool

A command-line tool for working with model cards.
"""

import argparse
import sys
from pathlib import Path

from nexusml.core.model_card.model_card import ModelCard
from nexusml.core.model_card.viewer import print_model_card_summary, export_model_card_html


def main():
    """Main entry point for the model card tool."""
    parser = argparse.ArgumentParser(description="Model Card Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View a model card")
    view_parser.add_argument("path", help="Path to the model card JSON file")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a model card to HTML")
    export_parser.add_argument("path", help="Path to the model card JSON file")
    export_parser.add_argument("--output", "-o", help="Path to save the HTML file")
    
    args = parser.parse_args()
    
    if args.command == "view":
        print_model_card_summary(args.path)
    elif args.command == "export":
        output_path = args.output or Path(args.path).with_suffix(".html")
        html_path = export_model_card_html(args.path, output_path)
        print(f"Exported model card to {html_path}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()