#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (two levels up)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Help message
show_help() {
    echo "Usage: $0 [dataset_path]"
    echo
    echo "If dataset_path is not provided, will use default path: \$PROJECT_ROOT/training_generation/data_0.csv"
    exit 1
}

# Parse command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# Set dataset path
DATASET_PATH="$PROJECT_ROOT/training_generation/data_0.csv"  # default path
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
    echo "Please provide a valid dataset path or generate the dataset first using:"
    echo "python generateDataset.py --num-sequences 20000 --output-dir training_generation"
    show_help
fi

# Ensure the web monitor is running
if ! pgrep -f "python web_view.py" > /dev/null; then
    echo "Starting web monitor..."
    python "$PROJECT_ROOT/web_view.py" &
    sleep 2  # Give it a moment to start
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run FNN training
echo "Starting FNN training with dataset: $DATASET_PATH"
echo "Monitor training progress at: http://localhost:5000"
python "$PROJECT_ROOT/model_training/train_fnn.py" --data_path "$DATASET_PATH"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "FNN training completed successfully"
    exit 0
else
    echo "FNN training failed"
    exit 1
fi 