# SOP-004: Using PlantUML Utilities

## Purpose

This procedure describes how to use the PlantUML utilities to create, render, and view diagrams for the FCA Dashboard project. PlantUML is a tool that allows you to create diagrams from text-based descriptions, making it easier to version control and maintain diagrams.

## Scope

This procedure covers:
- Creating PlantUML diagram files
- Rendering diagrams to SVG or PNG format
- Viewing rendered diagrams
- Adding new diagrams to the project
- Using the dynamic HTML viewer

This procedure does not cover:
- Installing PlantUML server
- Advanced PlantUML syntax and features

## Prerequisites

- Access to the FCA Dashboard project repository
- Python 3.6 or higher
- Virtual environment set up (optional but recommended)
- Basic understanding of PlantUML syntax

## Procedure

### 1. Install Required Dependencies

1. Activate the virtual environment (if using one):
   ```bash
   # For Windows Command Prompt/PowerShell
   .\.venv\Scripts\activate

   # For Git Bash/MINGW (Windows)
   source .venv/Scripts/activate

   # For macOS/Linux
   source .venv/bin/activate
   ```

2. Install the plantuml package:
   ```bash
   pip install plantuml
   ```

### 2. Create or Modify PlantUML Diagrams

1. Navigate to the `docs/diagrams` directory:
   ```bash
   cd docs/diagrams
   ```

2. Create a new directory for your diagrams if needed:
   ```bash
   mkdir -p your-category
   ```

3. Create or edit a PlantUML file (with `.puml` extension):
   ```bash
   # Example: Create a new diagram
   touch your-category/your-diagram.puml
   ```

4. Edit the file with your PlantUML code:
   ```plantuml
   @startuml "Your Diagram Title"
   
   ' Your PlantUML code here
   class Example {
     +attribute: Type
     +method(): ReturnType
   }
   
   @enduml
   ```

### 3. Render Diagrams

1. Return to the project root directory:
   ```bash
   cd /c/Repos/fca-dashboard4  # For Git Bash
   # OR
   cd c:\Repos\fca-dashboard4  # For Windows Command Prompt
   ```

2. Render all diagrams to SVG format (default):
   ```bash
   python -m fca_dashboard.utils.puml.cli render
   ```

3. Alternatively, render to PNG format:
   ```bash
   python -m fca_dashboard.utils.puml.cli render --format=png
   ```

4. To render a specific diagram:
   ```bash
   python -m fca_dashboard.utils.puml.cli render --file=your-category/your-diagram.puml
   ```

5. For best results, render both SVG and PNG formats:
   ```bash
   python -m fca_dashboard.utils.puml.cli render
   python -m fca_dashboard.utils.puml.cli render --format=png
   ```

### 4. View Diagrams with the Dynamic HTML Viewer

1. Open the HTML viewer in your default browser:
   ```bash
   # For Windows
   start docs/diagrams/output/index.html
   
   # For macOS
   open docs/diagrams/output/index.html
   
   # For Linux
   xdg-open docs/diagrams/output/index.html
   ```

2. The dynamic HTML viewer will:
   - Automatically detect all diagram folders
   - Create tabs for each folder
   - Display all diagrams in each folder
   - Allow switching between SVG and PNG formats

3. Using the viewer interface:
   - Click on folder tabs to switch between diagram categories
   - Use the PNG/SVG buttons to switch formats
   - Diagrams are displayed with formatted titles
   - New diagrams will appear automatically after rendering

### 5. Adding New Diagram Categories

1. To add a new diagram category:
   - Create a new directory under `docs/diagrams/`
   - Add your `.puml` files to this directory
   - Render the diagrams as described in section 3
   - The new category will automatically appear in the HTML viewer

2. No manual updates to the HTML viewer are required - it dynamically detects all available diagram folders and files.

### 6. Using Make Commands (Alternative)

1. From the project root directory, use Make commands:
   ```bash
   # Render all diagrams to SVG
   make -f fca_dashboard/utils/puml/Makefile render-diagrams
   
   # Render all diagrams to PNG
   make -f fca_dashboard/utils/puml/Makefile render-diagrams-png
   
   # View diagrams
   make -f fca_dashboard/utils/puml/Makefile view-diagrams
   ```

2. If you're in the `fca_dashboard/utils/puml` directory:
   ```bash
   # Render all diagrams to SVG
   make render-diagrams
   
   # Render all diagrams to PNG
   make render-diagrams-png
   
   # View diagrams
   make view-diagrams
   ```

## Verification

To verify that the procedure was completed successfully:

1. Check that the rendered diagrams exist in the `docs/diagrams/output` directory
2. Verify that the diagrams are displayed correctly in the HTML viewer
3. Confirm that both SVG and PNG formats can be viewed
4. Verify that all diagram categories appear as tabs in the viewer
5. Confirm that all diagrams within each category are displayed

## Troubleshooting

### Common Issues

1. **Module not found error: `plantuml`**
   - Solution: Install the plantuml package with `pip install plantuml`

2. **Path issues in Git Bash**
   - Solution: Use forward slashes for paths and ensure the virtual environment is activated with `source .venv/Scripts/activate`

3. **Diagrams not rendering**
   - Solution: Check that the PlantUML syntax is correct and that the file has a `.puml` extension

4. **HTML viewer not showing diagrams**
   - Solution: Ensure that diagrams have been rendered first with `python -m fca_dashboard.utils.puml.cli render`
   - Check browser console for any JavaScript errors
   - Verify that both SVG and PNG files exist in the output directory

5. **New diagrams not appearing in the viewer**
   - Solution: Refresh the browser page after rendering new diagrams
   - Check that the diagrams were rendered successfully
   - Verify the diagram files exist in the correct output subdirectory

6. **Make commands not working**
   - Solution: Ensure you're using the correct path to the Makefile with `-f fca_dashboard/utils/puml/Makefile`

## References

- [PlantUML Official Documentation](https://plantuml.com/en/guide)
- [PlantUML Syntax Guide](https://plantuml.com/en/guide)
- [FCA Dashboard PlantUML Utilities README](../../fca_dashboard/utils/puml/README.md)

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-03-04 | Roo | Initial version |
| 1.1 | 2025-03-05 | Roo | Updated for dynamic HTML viewer |