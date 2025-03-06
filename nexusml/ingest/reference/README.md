# Reference Data

This directory contains reference data files used by the NexusML classification
system. Each subdirectory contains a specific type of reference data, along with
a README file explaining the expected format.

## Directory Structure

```
reference/
├── omniclass/              # OmniClass taxonomy data
├── uniformat/              # Uniformat classification data
├── masterformat/           # MasterFormat classification data
├── mcaa-glossary/          # MCAA glossary and abbreviations
├── smacna-manufacturers/   # SMACNA manufacturer data
└── service-life/           # Service life data
    ├── ashrae/             # ASHRAE service life data
    └── energize-denver/    # Energize Denver service life data
```

## Configuration

Reference data paths and formats are configured in
`nexusml/config/reference_config.yml`. If you need to change the location of any
reference data, update this file.

## Validation

You can validate your reference data using the validation script:

```bash
cd nexusml
python test_reference_validation.py
```

This will check all reference data sources for issues and generate a validation
report.
