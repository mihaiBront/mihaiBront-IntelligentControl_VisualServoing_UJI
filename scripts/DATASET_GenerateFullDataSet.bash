#!/bin/bash

python generateDataset.py \
    --num-sequences 20000 \
    --output-dir training_generation \
    --max-iterations 500 \
    --savebunch-size 500
