#!/bin/bash
# Model Training Pipeline Script
# This script runs the model training pipeline with common options

# Default values
DATA_PATH=""
FEATURE_CONFIG=""
REFERENCE_CONFIG=""
OUTPUT_DIR="outputs/models"
MODEL_NAME="equipment_classifier"
TEST_SIZE=0.3
RANDOM_STATE=42
SAMPLING_STRATEGY="direct"
LOG_LEVEL="INFO"
OPTIMIZE=false
VISUALIZE=false

# Display help message
function show_help {
    echo "Usage: train_model.sh [options]"
    echo ""
    echo "Options:"
    echo "  -d, --data-path PATH       Path to the training data CSV file (required)"
    echo "  -f, --feature-config PATH  Path to the feature configuration YAML file"
    echo "  -r, --reference-config PATH Path to the reference configuration YAML file"
    echo "  -o, --output-dir DIR       Directory to save the trained model (default: outputs/models)"
    echo "  -n, --model-name NAME      Base name for the saved model (default: equipment_classifier)"
    echo "  -t, --test-size SIZE       Proportion of data to use for testing (default: 0.3)"
    echo "  -s, --random-state STATE   Random state for reproducibility (default: 42)"
    echo "  -g, --sampling-strategy STR Sampling strategy for class imbalance (default: direct)"
    echo "  -l, --log-level LEVEL      Logging level (default: INFO)"
    echo "  -p, --optimize             Perform hyperparameter optimization"
    echo "  -v, --visualize            Generate visualizations of model performance"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Example:"
    echo "  ./train_model.sh -d files/training-data/equipment_data.csv -p -v"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -f|--feature-config)
            FEATURE_CONFIG="$2"
            shift 2
            ;;
        -r|--reference-config)
            REFERENCE_CONFIG="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -t|--test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        -s|--random-state)
            RANDOM_STATE="$2"
            shift 2
            ;;
        -g|--sampling-strategy)
            SAMPLING_STRATEGY="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -p|--optimize)
            OPTIMIZE=true
            shift
            ;;
        -v|--visualize)
            VISUALIZE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if data path is provided
if [ -z "$DATA_PATH" ]; then
    echo "Error: Data path is required"
    show_help
    exit 1
fi

# Build the command
CMD="python -m nexusml.train_model_pipeline --data-path \"$DATA_PATH\""

if [ -n "$FEATURE_CONFIG" ]; then
    CMD="$CMD --feature-config \"$FEATURE_CONFIG\""
fi

if [ -n "$REFERENCE_CONFIG" ]; then
    CMD="$CMD --reference-config \"$REFERENCE_CONFIG\""
fi

CMD="$CMD --output-dir \"$OUTPUT_DIR\" --model-name \"$MODEL_NAME\""
CMD="$CMD --test-size $TEST_SIZE --random-state $RANDOM_STATE --sampling-strategy $SAMPLING_STRATEGY"
CMD="$CMD --log-level $LOG_LEVEL"

if [ "$OPTIMIZE" = true ]; then
    CMD="$CMD --optimize"
fi

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD