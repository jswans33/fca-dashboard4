# PlantUML Utilities

This directory contains utilities for working with PlantUML diagrams for the FCA Dashboard project.

## Overview

The PlantUML utilities provide tools for:

1. Rendering PlantUML diagrams to PNG or SVG images
2. Viewing diagrams in a web browser
3. Managing PlantUML diagrams through a command-line interface

## Diagram Location

**Important**: PlantUML files should be placed in the `docs/diagrams` directory. This is the default location where the utilities will look for diagrams to render.

For example:
```
docs/diagrams/classifier/classifier_model_diagram.puml
docs/diagrams/database/database_schema.puml
docs/diagrams/architecture/system_overview.puml
```

The utilities will recursively search through all subdirectories of `docs/diagrams` to find `.puml` files.

## Existing Diagrams

The following diagrams are currently available:

1. **classifier_model_diagram.puml** - Component-based ER diagram showing the overall architecture of the classifier model
2. **classifier_data_model_diagram.puml** - Traditional ER diagram focusing on the data entities and their relationships
3. **classifier_pipeline_diagram.puml** - Activity diagram showing the data processing pipeline

## Command-Line Interface

The CLI script provides a convenient way to work with the PlantUML diagrams:

### Using Make Commands

The simplest way to use the PlantUML utilities is through the Make commands. These commands can be run from either the project root directory or the `fca_dashboard/utils/puml` directory:

```bash
# From the project root directory
make render-diagrams
make view-diagrams

# OR from the fca_dashboard/utils/puml directory
cd fca_dashboard/utils/puml
make render-diagrams
make view-diagrams
```

Available Make commands:

```bash
# Render all diagrams to SVG (default)
make render-diagrams

# Render all diagrams to PNG
make render-diagrams-png

# Render a specific diagram
make render-diagram FILE=classifier/classifier_model_diagram.puml

# Open the HTML viewer in the default web browser
make view-diagrams

# Show help
make help
```

### Using the CLI Directly

You can also use the CLI script directly:

```bash
# Render all diagrams to SVG (default)
python -m fca_dashboard.utils.puml.cli render

# Render all diagrams to PNG
python -m fca_dashboard.utils.puml.cli render --format=png

# Render diagrams from a specific source directory
python -m fca_dashboard.utils.puml.cli render --source=/path/to/diagrams

# Save rendered images to a specific output directory
python -m fca_dashboard.utils.puml.cli render --output=/path/to/output

# Render a specific diagram
python -m fca_dashboard.utils.puml.cli render --file=classifier/classifier_model_diagram.puml

# Open the HTML viewer in the default web browser
python -m fca_dashboard.utils.puml.cli view

# Show help
python -m fca_dashboard.utils.puml.cli help
```

## HTML Viewer

The HTML viewer provides a web-based interface for viewing the rendered PlantUML diagrams.

Features:
- View all diagrams in one place
- Switch between PNG and SVG formats
- Access viewing instructions

To open the viewer:

```bash
# Using Make (recommended)
make -f fca_dashboard/utils/puml/Makefile view-diagrams
```

Or open `docs/diagrams/output/index.html` directly in a web browser.

**Note**: You must render the diagrams first using the `render-diagrams` command before viewing them.

## Python API

The Python API provides functions for rendering PlantUML diagrams programmatically:

```python
from fca_dashboard.utils.puml import render_diagram, render_all_diagrams

# Render a specific diagram to SVG
render_diagram('docs/diagrams/classifier/classifier_model_diagram.puml',
               output_dir='docs/diagrams/output',
               format='svg')

# Render all diagrams in a directory to PNG
render_all_diagrams('docs/diagrams',
                    output_dir='docs/diagrams/output',
                    format='png')

# Use the default output directory (docs/diagrams/output)
render_all_diagrams('docs/diagrams')
```

## Requirements

- For the HTML viewer: A web browser with JavaScript enabled
- For the Python API and CLI: Python 3.6+ and the `plantuml` package
- For local rendering: Java Runtime Environment (JRE) if using the PlantUML JAR directly

### Installation

Before using the PlantUML utilities, you need to install the required dependencies:

1. **Activate the virtual environment** (if using one):

   ```bash
   # On Windows Command Prompt/PowerShell
   .\.venv\Scripts\activate

   # On Git Bash/MINGW (Windows)
   source .venv/Scripts/activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

2. **Install the plantuml package**:

   ```bash
   # Install the plantuml package
   pip install plantuml

   # Or if you prefer
   python -m pip install plantuml
   ```

3. **Or add to requirements.txt** (if using the project's virtual environment):

   ```bash
   # Add plantuml to requirements.txt
   echo "plantuml" >> requirements.txt

   # Then install all dependencies
   pip install -r requirements.txt
   ```

**Note**: All commands that use the Python modules (including Make commands) must be run with the virtual environment activated if that's where the plantuml package is installed.

**For Git Bash/MINGW Users**: If you're using Git Bash on Windows, you may need to use forward slashes for paths and the `python` command directly:

```bash
# Activate virtual environment in Git Bash
source .venv/Scripts/activate

# Install plantuml
pip install plantuml

# Render diagrams
python -m fca_dashboard.utils.puml.cli render
```

## Adding New Diagrams

To add a new diagram:

1. Create a new `.puml` file in the `docs/diagrams` directory or a subdirectory
   ```
   # Example: Create a new diagram for database schema
   touch docs/diagrams/database/schema.puml
   ```

2. Edit the file with your PlantUML code
   ```plantuml
   @startuml "Database Schema"
   
   entity "User" as user {
     * id : int <<PK>>
     --
     * username : string
     * email : string
     * created_at : datetime
   }
   
   entity "Post" as post {
     * id : int <<PK>>
     * user_id : int <<FK>>
     --
     * title : string
     * content : text
     * created_at : datetime
   }
   
   user ||--o{ post
   
   @enduml
   ```

3. Render the diagram using Make
   ```bash
   # Render all diagrams
   make render-diagrams
   
   # Or render just the new diagram
   make render-diagram FILE=database/schema.puml
   ```

4. View the diagram in the HTML viewer
   ```bash
   make view-diagrams
   ```

The rendered images will be saved in the `docs/diagrams/output` directory, preserving the original directory structure.

## Viewing Options

There are several ways to view the PlantUML diagrams:

1. **Using the HTML Viewer**: Run `make -f fca_dashboard/utils/puml/Makefile view-diagrams` or open `docs/diagrams/output/index.html` directly in a web browser
2. **Using Make Commands**: Run `make -f fca_dashboard/utils/puml/Makefile render-diagrams` to generate SVG images in the `docs/diagrams/output` directory
3. **Using VS Code Extension**: Install the "PlantUML" extension by Jebbs and use Alt+D to preview .puml files directly in VS Code
4. **Using Online PlantUML Server**: Go to [http://www.plantuml.com/plantuml/uml/](http://www.plantuml.com/plantuml/uml/) and paste the diagram source

**Note**: For options 1 and 2, you can run the commands from the `fca_dashboard/utils/puml` directory without the `-f` flag:

```bash
cd fca_dashboard/utils/puml
make render-diagrams
make view-diagrams
```