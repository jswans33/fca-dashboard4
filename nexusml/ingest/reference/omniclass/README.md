# OmniClass Taxonomy Data

This directory contains OmniClass taxonomy data files used by the NexusML
classification system.

## File Format

OmniClass data should be provided as CSV files (`.csv`) with the following
columns:

| Column Name      | Description                    | Example                                           |
| ---------------- | ------------------------------ | ------------------------------------------------- |
| OmniClass Number | The OmniClass code             | 23-70-00                                          |
| OmniClass Title  | The name of the classification | Chillers                                          |
| Definition       | Detailed description           | Equipment that produces chilled water for cooling |

## Hierarchy

OmniClass codes follow a hierarchical structure with levels separated by
hyphens:

- Level 1: Major group (e.g., 23)
- Level 2: Medium group (e.g., 23-70)
- Level 3: Minor group (e.g., 23-70-00)

## Example Files

- omniclass.csv
- OmniClass_11_2013-02-26_2022.csv
- OmniClass_12_2012-10-30_2022.csv
- OmniClass_13_2012-05-16_2022.csv

## Usage

The system will automatically load and combine all CSV files in this directory.
Make sure all files follow the same column structure.
