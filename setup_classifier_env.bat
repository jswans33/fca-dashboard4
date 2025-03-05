@echo off
echo Setting up a clean environment for the classifier example...

:: Create a new virtual environment
echo Creating a new virtual environment...
python -m venv .venv-classifier

:: Activate the virtual environment
echo Activating the virtual environment...
call .venv-classifier\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

:: Verify installation
echo Verifying installation...
python -c "import numpy; import pandas; import sklearn; import matplotlib; import seaborn; import imblearn; print('All packages imported successfully!')"

:: Run the classifier example
echo Running the classifier example...
python -m fca_dashboard.examples.classifier_example

:: Deactivate the virtual environment
call deactivate

echo Done!