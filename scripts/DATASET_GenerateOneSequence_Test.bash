#!/bin/bash

python generateDataset.py \
    --num-sequences 1 \
    --output-dir dataset_sample \
    --max-iterations 500 \
    --savebunch-size 50
