# MasterFormat Classification Data

This directory contains MasterFormat classification data files used by the
NexusML classification system.

## File Format

MasterFormat data should be provided as CSV files (`.csv`) with the following
columns:

| Column Name        | Description                    | Example                                          |
| ------------------ | ------------------------------ | ------------------------------------------------ |
| MasterFormat Code  | The MasterFormat code          | 23 65 00                                         |
| MasterFormat Title | The name of the classification | Cooling Towers                                   |
| Description        | Detailed description           | Equipment that rejects heat from condenser water |

## Hierarchy

MasterFormat codes follow a hierarchical structure with levels separated by
spaces:

- Level 1: Division (e.g., 23)
- Level 2: Subdivision (e.g., 23 65)
- Level 3: Section (e.g., 23 65 00)

## Example Files

- masterformat_div_22_23.csv
- masterformat_2020.csv

## Usage

The system will automatically load and combine all CSV files in this directory.
Make sure all files follow the same column structure.
