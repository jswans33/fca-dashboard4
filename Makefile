.PHONY: lint test format format-md lint-md install run clean init-db coverage test-unit test-integration nexusml-test nexusml-coverage nexusml-install nexusml-run-simple nexusml-run-advanced

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

coverage-nexusml:
	python -m pytest --cov=nexusml/tests/ --cov-report=html --cov-report=term nexusml/tests/
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

# Run NexusML simple example
nexusml-run-simple:
	python -m nexusml.examples.simple_example

# Run NexusML advanced example
nexusml-run-advanced:
	python -m nexusml.examples.advanced_example

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
