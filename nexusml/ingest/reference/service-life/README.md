# Service Life Data

This directory contains service life data for equipment, used by the NexusML
classification system to predict the expected lifespan of different equipment
types.

## Subdirectories

- **ashrae/** - ASHRAE (American Society of Heating, Refrigerating and
  Air-Conditioning Engineers) service life data
- **energize-denver/** - Energize Denver service life data

## Usage

The system will automatically load and combine service life data from all
subdirectories. When multiple sources provide service life information for the
same equipment type, the system uses a priority order (ASHRAE first, then
Energize Denver).

Service life data is used to:

1. Predict the expected lifespan of equipment
2. Provide min/max ranges for service life
3. Enrich equipment data with service life information

This helps the classification system make more accurate predictions about
equipment replacement timelines.
