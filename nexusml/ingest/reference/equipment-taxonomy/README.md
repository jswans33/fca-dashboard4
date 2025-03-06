# Equipment Taxonomy

This directory contains equipment taxonomy data used by the NexusML
classification system. The taxonomy provides a hierarchical classification of
equipment types and subtypes.

## File Format

Equipment taxonomy data is provided as a CSV file (`equipment_taxonomy.csv`)
with the following structure:

| Column Name             | Description                           | Example                      |
| ----------------------- | ------------------------------------- | ---------------------------- |
| Asset Category          | Primary equipment category            | Air Handling Unit            |
| Equip Name ID           | Equipment name identifier             | AHU                          |
| Trade                   | Trade responsible for the equipment   | SM                           |
| Precon System           | Preconstruction system classification | Air Handling Units           |
| Operations System       | Operations system classification      | Air Handling Units           |
| Title                   | Equipment title                       | Air Handling Unit            |
| Drawing Abbreviation    | Abbreviation used in drawings         | AHU                          |
| Precon Tag              | Preconstruction tag                   | AHU                          |
| System Type ID          | System type identifier                | H                            |
| Sub System Type         | Subsystem type                        | Chilled Water, Heating Water |
| Sub System ID           | Subsystem identifier                  | CHW                          |
| Sub System Class        | Subsystem class                       | Custom                       |
| Class ID                | Class identifier                      | CST                          |
| Equipment Size          | Size of the equipment                 | 5000                         |
| Unit                    | Unit of measurement                   | CFM                          |
| Service Maintenance Hrs | Required maintenance hours            | 4                            |
| Service Life            | Expected service life in years        | 20                           |

## Example Files

- equipment_taxonomy.csv

## Usage

The system will automatically load the CSV file in this directory. The data is
used to:

1. Provide a standardized vocabulary for equipment types
2. Support hierarchical classification
3. Map between different naming conventions
4. Provide service life and maintenance information

## Sample Data

The equipment taxonomy includes categories such as:

- Air Handling Unit

  - Chilled Water, Heating Water (Custom/Packaged)
  - DX, Gas-Fired (Custom/Packaged)
  - DX, Heating Water (Custom/Packaged)
  - Other (Custom/Packaged)

- Chiller

  - Air Cooled (Package with Pump/Reciprocating/Screw/Scroll)
  - Water Cooled (Absorption/Centrifugal/Modular/Reciprocating/Screw/Scroll)

- Pump
  - Condensate (Duplex/Simplex)
  - Electronic Metering
  - End Suction
  - Inline
  - Sewage Ejector Pump (Duplex/Grinder Only/Simplex)
  - Split Case
  - Sump Pump
  - Vertical Turbine

## File Structure

The equipment taxonomy file organizes equipment in a hierarchical structure:

- Asset Category (e.g., Air Handling Unit)
  - System Type (e.g., Heating)
    - Sub System Type (e.g., Chilled Water, Heating Water)
      - Sub System Class (e.g., Custom, Packaged)
        - Equipment Size and Unit (e.g., 5000 CFM)

Each entry includes service maintenance hours and expected service life in
years.
