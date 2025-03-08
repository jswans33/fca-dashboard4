.PHONY: lint test format format-md lint-md install run clean init-db coverage test-unit test-integration nexusml-test nexusml-coverage nexusml-install nexusml-run-simple nexusml-run-advanced nexusml-example-basic nexusml-example-custom nexusml-example-config nexusml-example-di nexusml-examples nexusml-notebooks nexusml-notebooks-venv nexusml-verify-deps

lint:
	python -m black --check .
	python -m isort --check .
	python -m flake8 .
	python -m mypy .
	$(MAKE) lint-md

lint-md:
	-npx markdownlint-cli "**/*.md" --config .markdownlint.json

format-md:
	-npx markdownlint-cli "**/*.md" --config .markdownlint.json --fix

format:
	python -m black .
	python -m isort .
	$(MAKE) format-md

test:
	python -m pytest fca_dashboard/tests/

coverage:
	python -m pytest --cov=fca_dashboard --cov-report=html --cov-report=term fca_dashboard/tests/
	@echo "HTML coverage report generated in htmlcov/"
	python -c "import os, webbrowser; webbrowser.open('file://' + os.path.realpath('htmlcov/index.html'))"

test-unit:
	python -m pytest fca_dashboard/tests/unit/

test-integration:
	python -m pytest fca_dashboard/tests/integration/

# Run SQLite staging tests specifically
test-sqlite-staging:
	python -m pytest fca_dashboard/tests/unit/test_sqlite_staging_manager.py fca_dashboard/tests/integration/test_sqlite_staging_integration.py -v

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .
	
commit:
	git add .
	git commit -m "Update"
	git push

# Run the ETL pipeline with default settings
run:
	python -m fca_dashboard.main --config fca_dashboard/config/settings.yml

run-medtronic:
	python -m fca_dashboard.pipelines.pipeline_medtronics


# Run the SQLite staging example
run-sqlite-staging:
	python -m fca_dashboard.examples.sqlite_staging_example

# Run the Medtronics staging example
run-medtronics-staging:
	python -m fca_dashboard.examples.medtronics_staging_example

# Run the equipment classifier example
run-classifier-example:
	# Ensure we're using the virtual environment with seaborn installed
	.\.venv\Scripts\python -m fca_dashboard.examples.classifier_example

# Run the simplified equipment classifier example (no visualizations)
run-classifier-simple:
	# Run the simplified version without matplotlib/seaborn dependencies
	python -m fca_dashboard.examples.classifier_example_simple

# NexusML targets

# Test NexusML
nexusml-test:
	python -m pytest nexusml/tests/

# Run NexusML unit tests
nexusml-test-unit:
	python -m pytest nexusml/tests/unit/

# Run NexusML integration tests
nexusml-test-integration:
	python -m pytest nexusml/tests/integration/

# Generate coverage report for NexusML
nexusml-coverage:
	python -m pytest --cov=nexusml --cov-report=html --cov-report=term nexusml/tests/
	@echo "HTML coverage report generated in htmlcov/"
	python -c "import os, webbrowser; webbrowser.open('file://' + os.path.realpath('htmlcov/index.html'))"

# Install NexusML in the current environment
nexusml-install:
	python -m pip install --upgrade pip
	python -m pip install -e nexusml/

# Create a dedicated virtual environment for NexusML
nexusml-venv:
	python -m venv nexusml-venv
	@echo "Virtual environment created at nexusml-venv/"
	@echo "Activate with: source nexusml-venv/bin/activate (Linux/Mac) or nexusml-venv\\Scripts\\activate (Windows)"
	@echo "Then install with: make nexusml-install"

# Install uv package manager
install-uv:
	pip install uv

# Install NexusML using uv (recommended for monorepo)
nexusml-install-uv: install-uv
	uv pip install -e nexusml/

# Verify UV dependencies are up to date
nexusml-verify-deps: install-uv
	@echo "Verifying UV dependencies..."
	uv pip freeze > requirements-current.txt
	@echo "Comparing with requirements.txt..."
	python -c "import sys; sys.exit(0 if open('requirements.txt').read() == open('requirements-current.txt').read() else 1)" && \
	echo "Dependencies are up to date!" || \
	echo "Dependencies differ from requirements.txt. Please update. You can run 'uv pip install -r requirements.txt' to update."
	@del requirements-current.txt 2>nul || rm requirements-current.txt 2>/dev/null || echo "Cleanup completed"

# Run NexusML simple example
nexusml-run-simple:
	python -m nexusml.examples.simple_example

# Run NexusML advanced example
nexusml-run-advanced:
	python -m nexusml.examples.advanced_example

# Run NexusML documentation examples
nexusml-example-basic:
	@echo "Running NexusML basic usage example..."
	python nexusml/docs/examples/basic_usage.py

nexusml-example-custom:
	@echo "Running NexusML custom components example..."
	python nexusml/docs/examples/custom_components.py

nexusml-example-config:
	@echo "Running NexusML configuration example..."
	python nexusml/docs/examples/configuration.py

nexusml-example-di:
	@echo "Running NexusML dependency injection example..."
	python nexusml/docs/examples/dependency_injection.py

# Run all NexusML documentation examples
nexusml-examples: nexusml-example-basic nexusml-example-custom nexusml-example-config nexusml-example-di
	@echo "All NexusML documentation examples completed"

# Information about NexusML Jupyter Notebooks
nexusml-notebooks-info:
	mkdir -p nexusml/notebooks
	@echo ""
	@echo "=== NexusML Jupyter Notebooks ==="
	@echo ""
	@echo "The following notebook templates are available in the nexusml/notebooks directory:"
	@echo "- modular_template.ipynb: Original template for running experiments with NexusML"
	@echo "- enhanced_modular_template.ipynb: Enhanced template with improved error handling and features"
	@echo ""
	@echo "To use these notebooks in VSCode:"
	@echo "1. Open VSCode and navigate to the nexusml/notebooks directory"
	@echo "2. Click on the desired notebook file to open it"
	@echo "3. Select the Python kernel from your virtual environment (.venv)"
	@echo "4. Run the notebook cells using the VSCode Jupyter interface"
	@echo ""
	@echo "NOTE: The notebooks have been updated to automatically set the NEXUSML_CONFIG"
	@echo "environment variable to point to the correct configuration file."
	@echo ""
	@echo "To launch Jupyter Notebook server from the command line:"
	@echo "- Use 'make nexusml-notebook' for the standard launcher"
	@echo "- Use 'make nexusml-notebook-enhanced' for the enhanced launcher with better error handling"
	@echo ""
	@echo "=================================="

# Alternative target for launching Jupyter notebooks using the virtual environment
# This is now identical to nexusml-notebooks since VSCode can use the virtual environment
nexusml-notebooks-venv: nexusml-notebooks

# PlantUML Utilities

# Render all PlantUML diagrams to SVG (default)
render-diagrams:
	python -m fca_dashboard.utils.puml.cli render
	@echo "Diagrams saved to docs/diagrams/output"

# Render all PlantUML diagrams to PNG
render-diagrams-png:
	python -m fca_dashboard.utils.puml.cli render --format=png

# Render a specific PlantUML diagram
render-diagram:
	@echo "Usage: make render-diagram FILE=<file>"
	@if [ "$(FILE)" ]; then \
		python -m fca_dashboard.utils.puml.cli render --file=$(FILE); \
	fi

# Open the PlantUML HTML viewer
view-diagrams:
	# Open the HTML file directly instead of using the CLI
	start docs/diagrams/output/index.html 2>/dev/null || \
	open docs/diagrams/output/index.html 2>/dev/null || \
	xdg-open docs/diagrams/output/index.html 2>/dev/null || \
	echo "Could not open the HTML viewer automatically. Please open docs/diagrams/output/index.html manually."

# Clean up generated files
clean:
	python -c "import os, shutil; [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__' or d.endswith('.egg-info') or d.endswith('.egg') or d == '.pytest_cache' or d == 'htmlcov' or d == '.mypy_cache']"
	python -c "import os; [os.remove(os.path.join(root, f)) for root, _, files in os.walk('.') for f in files if f.endswith('.pyc') or f.endswith('.pyo') or f.endswith('.pyd') or f == '.coverage']"

# Create initial database schema
init-db:
	python -c "from fca_dashboard.core.models import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///etl.db'); Base.metadata.create_all(engine)"

#compile code: 

compile-core:
	cd fca_dashboard && npx repomix --ignore "tests/"

compile-tests:
	cd fca_dashboard && npx repomix --include "tests/"

# Launch Jupyter notebook for NexusML
nexusml-notebook:
	@echo ""
	@echo "=== Starting Jupyter Notebook for NexusML ==="
	@echo ""
	@echo "Once the server starts, copy and paste the URL with token into your browser."
	@echo "The URL will look like: http://localhost:8888/tree?token=<token>"
	@echo ""
	@echo "==============================================	@echo ""
	cd /c/Repos/fca-dashboard4/nexusml/notebooks && bash launch_jupyter.sh

# Launch enhanced Jupyter notebook for NexusML
nexusml-notebook-enhanced:
	@echo ""
	@echo "=== Starting Enhanced Jupyter Notebook for NexusML ==="
	@echo ""
	@echo "This uses the enhanced launcher with better error handling and dependency management."
	@echo "Once the server starts, copy and paste the URL with token into your browser."
	@echo "The URL will look like: http://localhost:8888/tree?token=<token>"
	@echo ""
	@echo "=============================================="
	@echo ""
	cd /c/Repos/fca-dashboard4/nexusml/notebooks && bash enhanced_launch_jupyter.sh

# Alternative target for launching Jupyter notebooks (same as nexusml-notebook)
nexusml-notebooks:
	@echo "Using the nexusml-notebook target instead..."
	$(MAKE) nexusml-notebook

# Alternative target for launching Jupyter notebooks using the virtual environment
nexusml-notebooks-venv:
	@echo "Using the nexusml-notebook target instead..."
	$(MAKE) nexusml-notebook

# Alternative target for launching enhanced Jupyter notebooks
nexusml-notebooks-enhanced:
	@echo "Using the nexusml-notebook-enhanced target instead..."
	$(MAKE) nexusml-notebook-enhanced
