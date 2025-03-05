# SOP-004: Using PlantUML Utilities

## Purpose

This procedure describes how to use the PlantUML utilities to create, render, and view diagrams for the FCA Dashboard project. PlantUML is a tool that allows you to create diagrams from text-based descriptions, making it easier to version control and maintain diagrams.

## Scope

This procedure covers:
- Creating PlantUML diagram files
- Rendering diagrams to SVG or PNG format
- Viewing rendered diagrams
- Adding new diagrams to the project

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

### 4. View Diagrams

1. Open the HTML viewer in your default browser:
   ```bash
   # For Windows
   start docs/diagrams/output/index.html
   
   # For macOS
   open docs/diagrams/output/index.html
   
   # For Linux
   xdg-open docs/diagrams/output/index.html
   ```

2. Use the viewer interface to:
   - Switch between SVG and PNG formats
   - View different diagrams
   - Access instructions

### 5. Using Make Commands (Alternative)

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

5. **Make commands not working**
   - Solution: Ensure you're using the correct path to the Makefile with `-f fca_dashboard/utils/puml/Makefile`

## References

- [PlantUML Official Documentation](https://plantuml.com/en/guide)
- [PlantUML Syntax Guide](https://plantuml.com/en/guide)
- [FCA Dashboard PlantUML Utilities README](../../fca_dashboard/utils/puml/README.md)

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-03-04 | Roo | Initial version |