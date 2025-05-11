#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (two levels up)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Help message
show_help() {
    echo "Usage: $0 [dataset_path]"
    echo
    echo "If dataset_path is not provided, will use default path: \$PROJECT_ROOT/data/dataset.csv"
    exit 1
}

# Parse command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# Set dataset path
DATASET_PATH="$PROJECT_ROOT/data/dataset.csv"  # default path
if [ $# -ge 1 ]; then
    DATASET_PATH="$1"
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please provide a valid dataset path or generate the dataset first."
    echo "Usage: $0 [dataset_path]"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run Hybrid model training
echo "Starting Hybrid model training with dataset: $DATASET_PATH"
python "$PROJECT_ROOT/model_training/train_hybrid.py" --data_path "$DATASET_PATH"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Hybrid model training completed successfully"
else
    echo "Hybrid model training failed"
    exit 1
fi 