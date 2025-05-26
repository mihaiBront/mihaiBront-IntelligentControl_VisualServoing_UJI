#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

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
    if [ -f "$1" ]; then
        DATASET_PATH="$1"
    elif [ -f "$PROJECT_ROOT/$1" ]; then
        DATASET_PATH="$PROJECT_ROOT/$1"
    else
        echo "Error: Dataset not found at $1 or $PROJECT_ROOT/$1"
        show_help
    fi
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please provide a valid dataset path or generate the dataset first."
    show_help
fi

# Make all training scripts executable
chmod +x "$SCRIPT_DIR/model_training/"*.bash

# Initialize error counter
ERRORS=0

# Function to run a training script and track its success
run_training() {
    local script=$1
    echo "==============================================="
    echo "Running $script..."
    echo "==============================================="
    
    "$SCRIPT_DIR/model_training/$script" "$DATASET_PATH"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: $script failed!"
        ((ERRORS++))
    fi
    
    echo
}

# Run all training scripts found in the model_training directory
echo "Found training scripts:"
for script in "$SCRIPT_DIR/model_training/"*.bash; do
    echo "- $(basename "$script")"
done
echo

echo "Using dataset: $DATASET_PATH"
echo

for script in "$SCRIPT_DIR/model_training/"*.bash; do
    if [ -f "$script" ]; then
        run_training "$(basename "$script")"
    fi
done

# Report results
echo "==============================================="
echo "Training Complete!"
echo "==============================================="
if [ $ERRORS -eq 0 ]; then
    echo "All models trained successfully!"
    exit 0
else
    echo "WARNING: $ERRORS training script(s) failed!"
    exit 1
fi 