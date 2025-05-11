#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
    
    "$SCRIPT_DIR/model_training/$script"
    
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