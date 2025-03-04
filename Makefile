.PHONY: lint test format format-md lint-md install run clean init-db coverage test-unit test-integration

lint:
	black --check .
	isort --check .
	flake8 .
	mypy .
	$(MAKE) lint-md

lint-md:
	npx markdownlint-cli "**/*.md" --config .markdownlint.json

format-md:
	npx markdownlint-cli "**/*.md" --config .markdownlint.json --fix

format:
	black .
	isort .
	$(MAKE) format-md

test:
	pytest fca_dashboard/tests/

coverage:
	pytest --cov=fca_dashboard --cov-report=html --cov-report=term fca_dashboard/tests/
	@echo "HTML coverage report generated in htmlcov/"
	python -c "import os, webbrowser; webbrowser.open('file://' + os.path.realpath('htmlcov/index.html'))"

test-unit:
	pytest fca_dashboard/tests/unit/

test-integration:
	pytest fca_dashboard/tests/integration/

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

git-commit:
	git add .
	git commit -m "Update"
	git push

# Run the ETL pipeline with default settings
run:
	python fca_dashboard/main.py --config fca_dashboard/config/settings.yml

# Clean up generated files
clean:
	python -c "import os, shutil; [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__' or d.endswith('.egg-info') or d.endswith('.egg') or d == '.pytest_cache' or d == 'htmlcov' or d == '.mypy_cache']"
	python -c "import os; [os.remove(os.path.join(root, f)) for root, _, files in os.walk('.') for f in files if f.endswith('.pyc') or f.endswith('.pyo') or f.endswith('.pyd') or f == '.coverage']"

# Create initial database schema
init-db:
	python -c "from fca_dashboard.core.models import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///etl.db'); Base.metadata.create_all(engine)"
