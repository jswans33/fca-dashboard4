# ASHRAE Service Life Data

This directory contains ASHRAE (American Society of Heating, Refrigerating and
Air-Conditioning Engineers) service life data for equipment, used by the NexusML
classification system.

## File Format

ASHRAE service life data should be provided as CSV files (`.csv`) with the
following columns:

| Column Name    | Description                            | Example             |
| -------------- | -------------------------------------- | ------------------- |
| Equipment Type | Type of equipment                      | Centrifugal Chiller |
| Median Years   | Median service life in years           | 23                  |
| Min Years      | Minimum expected service life in years | 15                  |
| Max Years      | Maximum expected service life in years | 30                  |
| Source         | Source of the data                     | ASHRAE 2019         |

## Example Files

- ashrae_2019_service_life.csv
- ashrae_equipment_life.csv

## Usage

The system will automatically load and combine all CSV files in this directory.
Make sure all files follow the same column structure.

## Data Source

This data is typically extracted from ASHRAE publications such as:

- ASHRAE Handbook - HVAC Applications
- ASHRAE Equipment Life Expectancy chart
- ASHRAE Research Project 1237-TRP

ASHRAE provides median service life values for various types of HVAC equipment
based on statistical analysis of actual equipment performance data.
