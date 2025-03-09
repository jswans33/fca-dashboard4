# Example: random_guessing.py

## Overview

The `random_guessing.py` script demonstrates how to use the NexusML equipment classifier model to make predictions on randomly generated or user-provided equipment descriptions. This example is particularly useful for testing the model's performance on a variety of inputs and for understanding how the model interprets different types of equipment descriptions.

Key features demonstrated:

1. **Model Loading**: Loading a pre-trained equipment classifier model
2. **Random Description Generation**: Creating realistic random equipment descriptions
3. **Custom Description Classification**: Classifying user-provided equipment descriptions
4. **Command-Line Interface**: Using argparse for a flexible command-line interface
5. **Batch Processing**: Processing multiple random samples in a single run

## Usage

```bash
python -m nexusml.examples.random_guessing [options]
```

### Options

- `--model-path MODEL_PATH`: Path to the trained model file (default: "outputs/models/equipment_classifier_20250306_161707.pkl")
- `--num-samples NUM_SAMPLES`: Number of random samples to generate (default: 5)
- `--custom CUSTOM`: Custom equipment description to classify

### Examples

#### Basic Usage

```bash
python -m nexusml.examples.random_guessing
```

This will load the default model and generate 5 random equipment descriptions for classification.

#### Custom Model Path

```bash
python -m nexusml.examples.random_guessing --model-path path/to/your/model.pkl
```

This will use a custom model file for classification.

#### More Random Samples

```bash
python -m nexusml.examples.random_guessing --num-samples 10
```

This will generate and classify 10 random equipment descriptions.

#### Custom Description

```bash
python -m nexusml.examples.random_guessing --custom "Trane XR80 Air Handler 2000 CFM variable-speed for office building"
```

This will classify the provided custom description in addition to the random samples.

## Code Walkthrough

### Random Description Generation

The script includes lists of sample equipment components for generating random descriptions:

```python
# Sample equipment components for generating random descriptions
MANUFACTURERS = [
    "Trane", "Carrier", "York", "Daikin", "Johnson Controls",
    # ... more manufacturers ...
]

EQUIPMENT_TYPES = [
    "Chiller", "Boiler", "Air Handler", "Rooftop Unit", "Heat Pump",
    # ... more equipment types ...
]

ATTRIBUTES = [
    "500 ton", "250 HP", "1000 kW", "480V", "2000 CFM",
    # ... more attributes ...
]

LOCATIONS = [
    "for mechanical room", "for rooftop installation", "for basement",
    # ... more locations ...
]
```

These lists are used to generate random equipment descriptions:

```python
def generate_random_description():
    """Generate a random equipment description."""
    manufacturer = random.choice(MANUFACTURERS)
    equipment_type = random.choice(EQUIPMENT_TYPES)
    attributes = random.sample(ATTRIBUTES, k=random.randint(1, 3))
    location = random.choice(LOCATIONS)

    model = f"{manufacturer[0]}{random.randint(100, 9999)}"

    description = (
        f"{manufacturer} {model} {equipment_type} {' '.join(attributes)} {location}"
    )
    return description
```

This function:
1. Selects a random manufacturer
2. Selects a random equipment type
3. Selects 1-3 random attributes
4. Selects a random location
5. Generates a random model number
6. Combines these components into a realistic equipment description

### Model Loading and Prediction

The script demonstrates how to load a pre-trained model and use it for prediction:

```python
# Load the model
print(f"Loading model from {args.model_path}")
classifier = EquipmentClassifier()
classifier.load_model(args.model_path)
print("Model loaded successfully\n")

# Process custom description if provided
if args.custom:
    print(f"Classifying custom description: {args.custom}")
    prediction = classifier.predict(args.custom)
    print_prediction(args.custom, prediction)
    print("\n" + "-" * 80 + "\n")

# Generate and process random descriptions
print(f"Generating {args.num_samples} random equipment descriptions:")
for i in range(args.num_samples):
    description = generate_random_description()
    prediction = classifier.predict(description)
    print(f"\nRandom Sample #{i+1}:")
    print_prediction(description, prediction)
    print("-" * 80)
```

This code:
1. Creates an EquipmentClassifier instance
2. Loads a pre-trained model from the specified path
3. Processes a custom description if provided
4. Generates and processes the specified number of random descriptions

### Command-Line Interface

The script uses argparse to create a flexible command-line interface:

```python
def main():
    """Main function to demonstrate random equipment guessing."""
    parser = argparse.ArgumentParser(
        description="Test the equipment classifier with random descriptions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/equipment_classifier_20250306_161707.pkl",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples to generate",
    )
    parser.add_argument(
        "--custom",
        type=str,
        default=None,
        help="Custom equipment description to classify",
    )
    args = parser.parse_args()
    
    # ... rest of the function ...
```

This code:
1. Creates an argument parser with a description
2. Adds arguments for model path, number of samples, and custom description
3. Parses the command-line arguments
4. Uses the parsed arguments to control the script's behavior

### Prediction Output

The script includes a function to print prediction results in a readable format:

```python
def print_prediction(description, prediction):
    """Print the prediction results in a readable format."""
    print(f"Description: {description}")
    print(f"Equipment Category: {prediction.get('category_name', 'Unknown')}")
    print(f"System Type: {prediction.get('mcaa_system_category', 'Unknown')}")
    print(f"Equipment Type: {prediction.get('Equipment_Type', 'Unknown')}")
    print(f"MasterFormat Class: {prediction.get('MasterFormat_Class', 'Unknown')}")
```

This function:
1. Prints the input description
2. Prints the predicted equipment category
3. Prints the predicted system type
4. Prints the predicted equipment type
5. Prints the predicted MasterFormat class

## Expected Output

```
Loading model from outputs/models/equipment_classifier_20250306_161707.pkl
Model loaded successfully

Classifying custom description: Trane XR80 Air Handler 2000 CFM variable-speed for office building
Description: Trane XR80 Air Handler 2000 CFM variable-speed for office building
Equipment Category: Air Handler
System Type: HVAC
Equipment Type: Air Handler-VAV
MasterFormat Class: 23 74 13

--------------------------------------------------------------------------------

Generating 5 random equipment descriptions:

Random Sample #1:
Description: Greenheck G5432 Exhaust Fan 3-phase high-efficiency for industrial facility
Equipment Category: Fan
System Type: HVAC
Equipment Type: Fan-Exhaust
MasterFormat Class: 23 34 23
--------------------------------------------------------------------------------

Random Sample #2:
Description: Lennox L789 Boiler 100 PSI water-cooled for hospital
Equipment Category: Boiler
System Type: HVAC
Equipment Type: Boiler-Hot Water
MasterFormat Class: 23 52 00
--------------------------------------------------------------------------------

Random Sample #3:
Description: Carrier C4567 Chiller 500 ton centrifugal for data center
Equipment Category: Chiller
System Type: HVAC
Equipment Type: Chiller-Centrifugal
MasterFormat Class: 23 64 16
--------------------------------------------------------------------------------

Random Sample #4:
Description: Grundfos G123 Pump vertical inline 50 GPM for mechanical room
Equipment Category: Pump
System Type: Plumbing
Equipment Type: Pump-Circulation
MasterFormat Class: 22 11 23
--------------------------------------------------------------------------------

Random Sample #5:
Description: Johnson Controls J8765 Heat Exchanger plate and frame shell and tube for basement
Equipment Category: Heat Exchanger
System Type: HVAC
Equipment Type: Heat Exchanger-Plate and Frame
MasterFormat Class: 23 57 00
--------------------------------------------------------------------------------
```

## Key Concepts Demonstrated

### 1. Pre-trained Model Usage

The example demonstrates how to use a pre-trained model for making predictions:

```python
classifier = EquipmentClassifier()
classifier.load_model(args.model_path)
prediction = classifier.predict(description)
```

This approach:
1. Creates an instance of the EquipmentClassifier class
2. Loads a pre-trained model from a file
3. Uses the model to make predictions on new descriptions

### 2. Random Data Generation

The example demonstrates how to generate realistic random data for testing:

```python
def generate_random_description():
    """Generate a random equipment description."""
    manufacturer = random.choice(MANUFACTURERS)
    equipment_type = random.choice(EQUIPMENT_TYPES)
    attributes = random.sample(ATTRIBUTES, k=random.randint(1, 3))
    location = random.choice(LOCATIONS)

    model = f"{manufacturer[0]}{random.randint(100, 9999)}"

    description = (
        f"{manufacturer} {model} {equipment_type} {' '.join(attributes)} {location}"
    )
    return description
```

This approach:
1. Uses predefined lists of components
2. Randomly selects components from each list
3. Combines the components into a realistic description
4. Provides a way to generate diverse test cases

### 3. Command-Line Interface

The example demonstrates how to create a flexible command-line interface:

```python
parser = argparse.ArgumentParser(
    description="Test the equipment classifier with random descriptions"
)
parser.add_argument(
    "--model-path",
    type=str,
    default="outputs/models/equipment_classifier_20250306_161707.pkl",
    help="Path to the trained model file",
)
# ... more arguments ...
args = parser.parse_args()
```

This approach:
1. Uses argparse to create a command-line interface
2. Provides default values for optional arguments
3. Includes help text for each argument
4. Makes the script more flexible and user-friendly

### 4. Batch Processing

The example demonstrates how to process multiple items in a batch:

```python
for i in range(args.num_samples):
    description = generate_random_description()
    prediction = classifier.predict(description)
    print(f"\nRandom Sample #{i+1}:")
    print_prediction(description, prediction)
    print("-" * 80)
```

This approach:
1. Generates multiple random descriptions
2. Processes each description and displays the results
3. Provides a way to test the model on a variety of inputs

## Dependencies

- **argparse**: Standard library module for command-line argument parsing
- **random**: Standard library module for random number generation
- **nexusml.core.model**: Core module containing the EquipmentClassifier class

## Notes and Warnings

- The script assumes that a pre-trained model file exists at the specified path
- The default model path may need to be updated to point to an existing model file
- The random descriptions are generated from predefined lists and may not cover all possible equipment types
- The script does not save the prediction results to a file
- The script does not include error handling for cases where the model file does not exist or is invalid
- The script does not include visualization components

## Extensions and Variations

### Saving Results to a File

To save the prediction results to a file:

```python
def main():
    # ... existing code ...
    
    # Open a file for writing results
    with open("random_guessing_results.txt", "w") as f:
        # Process custom description if provided
        if args.custom:
            f.write(f"Classifying custom description: {args.custom}\n")
            prediction = classifier.predict(args.custom)
            write_prediction(f, args.custom, prediction)
            f.write("\n" + "-" * 80 + "\n\n")
        
        # Generate and process random descriptions
        f.write(f"Generating {args.num_samples} random equipment descriptions:\n")
        for i in range(args.num_samples):
            description = generate_random_description()
            prediction = classifier.predict(description)
            f.write(f"\nRandom Sample #{i+1}:\n")
            write_prediction(f, description, prediction)
            f.write("-" * 80 + "\n")

def write_prediction(file, description, prediction):
    """Write the prediction results to a file."""
    file.write(f"Description: {description}\n")
    file.write(f"Equipment Category: {prediction.get('category_name', 'Unknown')}\n")
    file.write(f"System Type: {prediction.get('mcaa_system_category', 'Unknown')}\n")
    file.write(f"Equipment Type: {prediction.get('Equipment_Type', 'Unknown')}\n")
    file.write(f"MasterFormat Class: {prediction.get('MasterFormat_Class', 'Unknown')}\n")
```

### Adding Visualization

To add visualization of the prediction results:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    # ... existing code ...
    
    # Collect results for visualization
    results = []
    
    # Process custom description if provided
    if args.custom:
        prediction = classifier.predict(args.custom)
        results.append({
            "Description": args.custom,
            "Equipment Category": prediction.get('category_name', 'Unknown'),
            "System Type": prediction.get('mcaa_system_category', 'Unknown'),
            "Equipment Type": prediction.get('Equipment_Type', 'Unknown'),
            "Sample Type": "Custom"
        })
    
    # Generate and process random descriptions
    for i in range(args.num_samples):
        description = generate_random_description()
        prediction = classifier.predict(description)
        results.append({
            "Description": description,
            "Equipment Category": prediction.get('category_name', 'Unknown'),
            "System Type": prediction.get('mcaa_system_category', 'Unknown'),
            "Equipment Type": prediction.get('Equipment_Type', 'Unknown'),
            "Sample Type": "Random"
        })
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Visualize the distribution of equipment categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=results_df, x="Equipment Category")
    plt.title("Distribution of Equipment Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("equipment_category_distribution.png")
    
    # Visualize the distribution of system types
    plt.figure(figsize=(10, 6))
    sns.countplot(data=results_df, x="System Type")
    plt.title("Distribution of System Types")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("system_type_distribution.png")
```

### Adding Error Handling

To add error handling for cases where the model file does not exist or is invalid:

```python
def main():
    # ... existing code ...
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    try:
        classifier = EquipmentClassifier()
        classifier.load_model(args.model_path)
        print("Model loaded successfully\n")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # ... rest of the function ...
```

### Adding More Component Lists

To expand the random description generation with more component lists:

```python
# Additional component lists
SYSTEMS = [
    "HVAC",
    "Plumbing",
    "Electrical",
    "Fire Protection",
    "Building Automation",
    "Security",
    "Lighting",
    "Refrigeration",
    "Process",
    "Medical Gas",
]

MODELS = [
    "Standard",
    "Premium",
    "Elite",
    "Professional",
    "Commercial",
    "Industrial",
    "Residential",
    "High-Efficiency",
    "Energy Star",
    "Custom",
]

def generate_random_description():
    """Generate a random equipment description."""
    manufacturer = random.choice(MANUFACTURERS)
    equipment_type = random.choice(EQUIPMENT_TYPES)
    attributes = random.sample(ATTRIBUTES, k=random.randint(1, 3))
    location = random.choice(LOCATIONS)
    system = random.choice(SYSTEMS)
    model_type = random.choice(MODELS)

    model = f"{manufacturer[0]}{random.randint(100, 9999)}"

    description = (
        f"{manufacturer} {model} {model_type} {equipment_type} {' '.join(attributes)} "
        f"for {system} system {location}"
    )
    return description