"""
Setup script for the FCA Dashboard package.
"""

import re
from setuptools import find_packages, setup

# Get version from __init__.py
with open("fca_dashboard/__init__.py", encoding="utf-8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Split requirements into core and dev dependencies
core_requirements = []
dev_requirements = []

for req in requirements:
    if any(dev_keyword in req for dev_keyword in ["pytest", "black", "isort", "flake8", "mypy", "ruff", "types-", "coverage"]):
        dev_requirements.append(req)
    else:
        core_requirements.append(req)

setup(
    name="fca_dashboard",
    version=version,
    description="ETL Pipeline for FCA Dashboard",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ETL Team",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
