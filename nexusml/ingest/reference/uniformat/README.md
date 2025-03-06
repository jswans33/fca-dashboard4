# Uniformat Classification Data

This directory contains Uniformat classification data files used by the NexusML
classification system.

## File Format

Uniformat data should be provided as CSV files (`.csv`) with the following
columns:

| Column Name     | Description                    | Example                                      |
| --------------- | ------------------------------ | -------------------------------------------- |
| UniFormat Code  | The Uniformat code             | D3040                                        |
| UniFormat Title | The name of the classification | HVAC Distribution Systems                    |
| Description     | Detailed description           | Systems for distributing heating and cooling |

## Hierarchy

Uniformat codes follow a hierarchical structure:

- Level 1: Major group (e.g., D)
- Level 2: Group (e.g., D30)
- Level 3: Sub-group (e.g., D304)
- Level 4: Detail (e.g., D3040)

## Example Files

- uniformat_2010.csv
- uniformat_classifications.csv

## Usage

The system will automatically load and combine all CSV files in this directory.
Make sure all files follow the same column structure.
