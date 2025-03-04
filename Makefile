.PHONY: lint test format install

lint:
	black --check .
	isort --check .
	flake8 .
	mypy .

format:
	black .
	isort .

test:
	pytest tests/

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

	# Add these to your Makefile as needed
   
   # Run the ETL pipeline with default settings
   run:
    python fca_dashboard/main.py --config fca_dashboard/config/settings.yaml
   
   # Clean up generated files
   clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name "*.egg" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".coverage" -exec rm -rf {} +
    find . -type d -name "htmlcov" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
   
   # Create initial database schema
   init-db:
    python -c "from fca_dashboard.core.models import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///etl.db'); Base.metadata.create_all(engine)"