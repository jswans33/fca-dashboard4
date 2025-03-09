# Model Card: Equipment Classification Production Model

## Model Details

**Model Name**: Equipment Classification Production Model  
**Version**: 1.0.0  
**Date Created**: March 8, 2025  
**Type**: Multi-output classification model  
**Framework**: Scikit-learn  
**Architecture**: RandomForestClassifier with TF-IDF and numeric features  
**License**: Proprietary  
**Contact**: [Your Contact Information]

## Model Description

This model classifies mechanical equipment into standardized categories based on textual descriptions and metadata. It uses a combination of text features (processed with TF-IDF) and numeric features (like service life) to make predictions across multiple classification targets.

### Inputs

- **Text descriptions**: Equipment tags, manufacturer names, model numbers, etc.
- **Numeric features**: Service life

### Outputs

- **category_name**: Equipment category (e.g., "HVAC", "Plumbing")
- **uniformat_code**: Uniformat classification code (e.g., "D3050", "D2020")
- **mcaa_system_category**: MCAA system category (e.g., "Mechanical", "Plumbing Equipment")
- **Equipment_Type**: Hierarchical equipment type (e.g., "HVAC-Air Handling")
- **System_Subtype**: System subtype (e.g., "Mechanical-Cooling")
- **MasterFormat_Class**: MasterFormat classification (derived from other classifications)

## Intended Use

This model is designed for:

- Automatically classifying mechanical equipment in facility condition assessments
- Standardizing equipment categorization across different data sources
- Supporting data integration with building information systems
- Facilitating equipment lifecycle management

## Training Data

### Data Sources

The model is trained on a comprehensive dataset of mechanical equipment with the following characteristics:

- **Size**: [Number of examples] examples
- **Source**: [Data source information]
- **Time Period**: [Time period covered by the data]
- **Preprocessing**: Text cleaning, feature engineering, class balancing

### Data Format

The training data follows the production format with these key fields:

```
equipment_tag,manufacturer,model,category_name,omniclass_code,uniformat_code,
masterformat_code,mcaa_system_category,CategoryID,OmniClassID,UniFormatID,
MasterFormatID,MCAAID,LocationID,Precon_System,Drawing_Abbreviation,Precon_Tag,
System_Type_ID,Equip_Name_ID,Sub_System_ID,Sub_System_Class,Class_ID,Unit
```

#### Detailed Field Descriptions

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| equipment_tag | string | Equipment identifier or tag number | AHU-01 |
| manufacturer | string | Equipment manufacturer name | Trane |
| model | string | Equipment model number or identifier | CSAA012 |
| category_name | string | Primary equipment category | Air Handler |
| omniclass_code | string | OmniClass classification code | 23-33 13 13 |
| uniformat_code | string | Uniformat classification code | D3040 |
| masterformat_code | string | MasterFormat classification code | 23 74 13 |
| mcaa_system_category | string | MCAA system category | HVAC Equipment |
| CategoryID | integer | Category ID from reference database | 101 |
| OmniClassID | integer | OmniClass ID from reference database | 2333 |
| UniFormatID | integer | Uniformat ID from reference database | 3040 |
| MasterFormatID | integer | MasterFormat ID from reference database | 2374 |
| MCAAID | string | MCAA abbreviation (see MCAAID Format section) | H |
| LocationID | integer | Location ID from reference database | 1001 |

##### Additional Pricing Data System Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| Precon_System | string | Preconstruction system category | Air Handling Units |
| Drawing_Abbreviation | string | Standard drawing abbreviation | AHU |
| Precon_Tag | string | Preconstruction tag identifier | AHU |
| System_Type_ID | string | System type identifier (H, P, R) | H |
| Equip_Name_ID | string | Equipment name identifier | AHU |
| Sub_System_ID | string | Subsystem identifier | PKG |
| Sub_System_Class | string | Subsystem classification | Floor Mounted |
| Class_ID | string | Class identifier | FLR |
| Unit | string | Unit of measurement | CFM |

#### Sample Data

Here's an example of properly formatted training data:

```csv
equipment_tag,manufacturer,model,category_name,omniclass_code,uniformat_code,masterformat_code,mcaa_system_category,CategoryID,OmniClassID,UniFormatID,MasterFormatID,MCAAID,LocationID,Precon_System,Drawing_Abbreviation,Precon_Tag,System_Type_ID,Equip_Name_ID,Sub_System_ID,Sub_System_Class,Class_ID,Unit
AHU-01,Trane,CSAA012,Air Handler,23-33 13 13,D3040,23 74 13,HVAC Equipment,101,2333,3040,2374,H,1001,Air Handling Units,AHU,AHU,H,AHU,PKG,Floor Mounted,FLR,CFM
CH-01,Carrier,30XA,Chiller,23-33 13 19,D3040,23 64 23,HVAC Equipment,102,2334,3040,2364,H,1001,Chiller Plant,CH,CH,H,CH,AIR,Packaged,PKG,TONS
P-01,Grundfos,CR 32-2,Pump,23-33 13 25,D3020,23 21 23,Mechanical/Sheetmetal,103,2335,3020,2321,SM,1001,"Chilled, Condenser, Heating Water, Steam",PMP,PMP,H,PMP,HYD,Centrifugal,CNT,GPM
```

### MCAAID Format

The MCAAID field uses standard industry abbreviations that correspond to system categories:

| mcaa_system_category | MCAAID |
|----------------------|--------|
| HVAC Equipment       | H      |
| Plumbing Equipment   | P      |
| Mechanical/Sheetmetal| SM     |
| Process Cooling Water| R      |
| Hot Water Systems    | H      |
| Refrigeration        | R      |
| Electrical           | E      |
| Fire Protection      | F      |
| Controls             | C      |

### Standard Equipment Categories

The following standard equipment categories are taken from Tom's build op list and should be used for classification:

- Accessory
- Air Compressor
- Air Curtain
- Air Dryer
- Air Handler
- Air Receiver
- Air Rotator
- Air Scoop
- Air Separator
- Baseboard
- Boiler
- Bypass Filter
- Cabinet Unit
- Chiller
- Compressor
- Computer
- Condenser
- Connector
- Cooling Tower
- Coupon Rack
- DI/RO Equipment
- Direct Outdoor Air System
- Domestic
- Dual Duct
- Ductless Split
- Energy Recovery
- Evaporator
- Expansion
- Fan
- Fan Coil
- Fan Coil Unit
- Fan Power
- Fixture
- Furnace
- Glycol Feeder
- Grease Interceptor
- Grease Trap
- Heat Exchanger
- Heat Pump
- Heat Trace
- Humidifier
- Infrared
- Make-up Air
- Nitrogen
- Pot Feeder
- PRV
- Pump
- Radiant Panel
- Rear Door
- Return Air
- Roof Top Unit
- Sand/Oil Interceptor
- Tank
- Unit Heater
- Unit Ventilator
- Vacuum
- VAV Box
- Venturi
- Water Softener

These categories are from an established industry list and provide a standardized foundation for equipment categorization. When adding new equipment, try to match it to one of these categories whenever possible.

### Enumeration Reference Guide

The pricing data system fields use standardized enumerations from Tom's build op list. These enumerations are maintained in `files/training-data/enumeratins for training data.csv`, which serves as the authoritative reference for all valid values.

#### Enumeration Field Relationships

The enumeration fields form a hierarchical relationship:

1. **System Level**: Defines the broad system category
   - System_Type_ID (H, P, SM, etc.)
   - Precon_System (Air Handling Units, Chiller Plant, etc.)
   - Operations_System (Air Handling Units, Chilled Water, etc.)

2. **Equipment Level**: Defines the specific equipment type
   - Asset_Category (Air Handler, Chiller, Pump, etc.)
   - Equip_Name_ID (AHU, CH, PMP, etc.)
   - Drawing_Abbreviation (AHU, CH, PMP, etc.)
   - Precon_Tag (AHU, CH, PMP, etc.)

3. **Sub-System Level**: Defines equipment variants and characteristics
   - Sub_System_Type (Packaged, Air Cooled, Hydronic, etc.)
   - Sub_System_ID (PKG, AIR, HYD, etc.)
   - Sub_System_Class (Floor Mounted, Packaged, Centrifugal, etc.)
   - Class_ID (FLR, PKG, CNT, etc.)

4. **Measurement**: Defines how the equipment is measured
   - Unit (CFM, TONS, GPM, etc.)

#### How to Determine Correct Enumerations

When classifying new equipment, follow these steps:

1. **Identify the System Type**:
   - **System_Type_ID**: Use the primary system type
     - H: HVAC and heating systems
     - P: Plumbing systems
     - SM: Sheet Metal/Mechanical systems
     - R: Refrigeration systems
     - E: Electrical systems
     - F: Fire protection systems
     - C: Controls systems

2. **Determine the Precon and Operations Systems**:
   - **Precon_System**: Select from the first column in the enumerations file
     - Examples: "Air Handling Units", "Chiller Plant", "Heating Water Boiler Plant"
   - **Operations_System**: Select from the second column in the enumerations file
     - Examples: "Air Handling Units", "Chilled Water", "Heating Water"

3. **Identify the Equipment Type**:
   - **Asset_Category**: Select from the seventh column in the enumerations file
     - Examples: "Air Handler", "Chiller", "Pump"
   - **Equip_Name_ID**: Use the eighth column value
     - Examples: "AHU", "CH", "PMP"
   - **Drawing_Abbreviation**: Use the fourth column value
     - Examples: "AHU", "CH", "PMP"
   - **Precon_Tag**: Use the fifth column value
     - Examples: "AHU", "CH", "PMP"

4. **Determine the Sub-System Details**:
   - **Sub_System_Type**: Select from the ninth column
     - Examples: "Packaged", "Air Cooled", "Hydronic"
   - **Sub_System_ID**: Use the tenth column value
     - Examples: "PKG", "AIR", "HYD"
   - **Sub_System_Class**: Use the eleventh column value
     - Examples: "Floor Mounted", "Packaged", "Centrifugal"
   - **Class_ID**: Use the twelfth column value
     - Examples: "FLR", "PKG", "CNT"

5. **Determine the Measurement Unit**:
   - **Unit**: Use the thirteenth column value based on what the equipment measures
     - Examples: "CFM" (air flow), "TONS" (cooling capacity), "GPM" (flow rate)

#### Common Equipment Enumeration Examples

| Equipment Type | System_Type_ID | Precon_System | Asset_Category | Equip_Name_ID | Drawing_Abbreviation | Sub_System_ID | Sub_System_Class | Class_ID | Unit |
|---------------|---------------|---------------|---------------|--------------|---------------------|--------------|-----------------|----------|------|
| Air Handler | H | Air Handling Units | Air Handler | AHU | AHU | PKG | Floor Mounted | FLR | CFM |
| Chiller | H | Chiller Plant | Chiller | CH | CH | AIR | Packaged | PKG | TONS |
| Pump | SM | Chilled, Condenser, Heating Water, Steam | Pump | PMP | PMP | HYD | Centrifugal | CNT | GPM |
| Fan Coil Unit | H | Air Handling System Terminal Equipment | Fan Coil Unit | FCU | FCU | CLG | Direct Expansion | DX | CFM |
| Roof Top Unit | H | Air Handling Units | Roof Top Unit | RTU | RTU | PKG | DX, Gas-Fired | DXG | CFM |
| Boiler | H | Heating Water Boiler Plant | Boiler | BLR | HWB | HW | Packaged Fire Tube | PFT | MBH |

When in doubt, refer to similar equipment in the enumerations file and follow the same pattern. The enumerations file should be considered the source of truth for all valid values.

#### Complete List of Valid Values

For a complete list of valid values for each field, refer to the enumerations file at `files/training-data/enumeratins for training data.csv`. This file contains all possible values organized by column, with each column corresponding to one of the enumeration fields.

The file structure is as follows:
- Column 1: Precon System values
- Column 2: Operations System values
- Column 3: Title values (equipment names)
- Column 4: Drawing Abbreviation values
- Column 5: Precon Tag values
- Column 6: System Type ID values
- Column 7: Asset Category values
- Column 8: Equip Name ID values
- Column 9: Sub System Type values
- Column 10: Sub System ID values
- Column 11: Sub System Class values
- Column 12: Class ID values
- Column 13: Unit values

## Performance Metrics

### Overall Metrics

- **Accuracy**: [Overall accuracy]
- **F1 Score (macro)**: [F1 score]
- **Precision**: [Precision]
- **Recall**: [Recall]

### Per-Target Metrics

| Target | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| category_name | [Value] | [Value] | [Value] | [Value] |
| uniformat_code | [Value] | [Value] | [Value] | [Value] |
| mcaa_system_category | [Value] | [Value] | [Value] | [Value] |
| Equipment_Type | [Value] | [Value] | [Value] | [Value] |
| System_Subtype | [Value] | [Value] | [Value] | [Value] |

### "Other" Category Performance

Special attention has been paid to the performance on "Other" categories:

| Target | "Other" Accuracy | "Other" F1 | "Other" Precision | "Other" Recall |
|--------|------------------|------------|-------------------|----------------|
| category_name | [Value] | [Value] | [Value] | [Value] |
| uniformat_code | [Value] | [Value] | [Value] | [Value] |
| mcaa_system_category | [Value] | [Value] | [Value] | [Value] |
| Equipment_Type | [Value] | [Value] | [Value] | [Value] |
| System_Subtype | [Value] | [Value] | [Value] | [Value] |

## Limitations

- **Specialized Equipment**: The model may have limited accuracy for highly specialized or rare equipment types.
- **New Manufacturers/Models**: Performance may be reduced for manufacturers or models not represented in the training data.
- **Text Quality**: Classification accuracy depends on the quality and completeness of input text descriptions.
- **Domain Specificity**: The model is specifically trained for mechanical equipment in building systems and may not generalize to other domains.

## Ethical Considerations

- **Data Privacy**: The model does not contain or process personally identifiable information.
- **Bias**: The model has been trained on a diverse dataset to minimize bias in equipment classification.
- **Environmental Impact**: The model's computational requirements have been optimized to reduce environmental impact.

## Technical Specifications

### Model Architecture

- **Text Processing**: TF-IDF vectorization with n-gram range (1, 3)
- **Numeric Features**: Standardized numeric features (service_life)
- **Classifier**: MultiOutputClassifier with RandomForestClassifier
- **Hyperparameters**:
  - n_estimators: [Value]
  - max_depth: [Value]
  - min_samples_split: [Value]
  - min_samples_leaf: [Value]
  - class_weight: [Value]

### Feature Engineering

- **Text Combinations**: Combined text features from equipment_tag, manufacturer, model, etc.
- **Hierarchical Categories**: Created hierarchical categories like Equipment_Type and System_Subtype
- **Class Balancing**: Used RandomOverSampler for handling imbalanced classes

### Dependencies

- Python 3.8+
- scikit-learn 1.0+
- pandas 1.5+
- numpy 1.20+
- imbalanced-learn 0.8+

## Usage Guidelines

### Recommended Usage

```python
from nexusml.core.model import EquipmentClassifier

# Load the production model
classifier = EquipmentClassifier()
classifier.load_model("outputs/models/equipment_classifier_production.pkl")

# Make a prediction
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0
prediction = classifier.predict(description, service_life)

# Print prediction results
for key, value in prediction.items():
    if key != "attribute_template" and key != "master_db_mapping":
        print(f"{key}: {value}")
```

### Command-Line Usage

```bash
python -m nexusml.predict_v2 --model-path outputs/models/equipment_classifier_production.pkl --input-file path/to/data.csv --output-file predictions.csv --use-orchestrator
```
```bash
python -m nexusml.train_model_pipeline --data-path nexusml/data/training_data/production_training_data.csv
```

## Maintenance

### Monitoring

- Regular evaluation on new data to detect performance drift
- Tracking of misclassifications to identify areas for improvement

### Retraining

- Scheduled retraining every [timeframe] with updated data
- Incremental updates to include new equipment types and manufacturers

### Version History

| Version | Date | Changes | Performance |
|---------|------|---------|-------------|
| 1.0.0   | 2025-03-08 | Initial production model | [Metrics] |

## Contact Information

For questions, issues, or feedback regarding this model, please contact:

- **Name**: [Contact Name]
- **Email**: [Contact Email]
- **Department**: [Department]

## References

- NexusML Documentation
- MCAA Classification Standards
- [Other relevant references]