#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IBVS Dataset Generation for Neural Network Training

This script generates a dataset for training a neural network to replace the IBVS controller.
It runs the IBVS algorithm multiple times with randomly generated points and stores the necessary data.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
import os
import time
from tqdm import tqdm
import pandas as pd
import csv
import argparse
import json

# Import RVC3 libraries
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *

from lib.DatasetGeneration import IPVSDatasetGeneration as DsGen

# Set numpy print options for better readability
np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{0:8.4g}" if abs(x) < 1e-10 else f"{x:8.4g}"})

def main():
    """Main function to generate the dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate IBVS dataset for neural network training')
    parser.add_argument('--num-sequences', type=int, default=1000, help='Number of sequences to generate')
    parser.add_argument('--num-points', type=int, default=4, help='Number of points per sequence')
    parser.add_argument('--max-iterations', type=int, default=200, help='Maximum iterations per sequence')
    parser.add_argument('--lambda-value', type=float, default=0.1, help='Gain for the IBVS controller')
    parser.add_argument('--output-dir', type=str, default='dataset', help='Directory to save the dataset')
    parser.add_argument('--visualize', action='store_true', help='Visualize a sample sequence')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare data for neural network training')
    parser.add_argument('--load', action='store_true', help='Load existing dataset instead of generating a new one')
    args = parser.parse_args()
    
    # Create directory for saving data
    args.output_dir = "outputs/" + args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if args.load:
        # Load existing dataset
        dataset = DsGen.load_dataset_from_csv(args.output_dir)
    else:
        # Initialize dataset
        dataset = []
        
        # Generate sequences
        print(f"Generating {args.num_sequences} sequences...")
        for i in range(args.num_sequences):
            # Create camera with random pose
            camera = CentralCamera.Default(pose=DsGen.generate_random_camera_pose())
            
            # Generate random 3D points
            P = DsGen.generate_random_points(num_points=args.num_points)
            
            # Generate desired image points
            pd = DsGen.generate_desired_image_points(camera, num_points=args.num_points)
            
            # Run IBVS sequence
            data = DsGen.run_ibvs_sequence(camera, P, pd, max_iterations=args.max_iterations, lambda_value=args.lambda_value)
            
            # Add metadata
            data['sequence_id'] = i
            data['initial_camera_pose'] = camera.pose
            data['3d_points'] = P
            
            # Add to dataset
            dataset.append(data)
            
            # Save intermediate results every 100 sequences
            if (i + 1) % 100 == 0:
                DsGen.save_dataset_to_csv(dataset, args.output_dir)
                print(f"Saved {i+1} sequences to {args.output_dir}")
        
        # Save final dataset
        DsGen.save_dataset_to_csv(dataset, args.output_dir)
        print(f"Saved {args.num_sequences} sequences to {args.output_dir}")
    
    # Visualize a sample sequence if requested
    if args.visualize:
        print("Visualizing a sample sequence...")
        sequence_idx = np.random.randint(0, len(dataset))
        sequence = dataset[sequence_idx]
        DsGen.visualize_sequence(sequence, save_path=os.path.join(args.output_dir, "sample_sequence.png"))
        print(f"Visualization saved to {os.path.join(args.output_dir, 'sample_sequence.png')}")
    
    # Prepare data for neural network training if requested
    if args.prepare_data:
        print("Preparing data for neural network training...")
        X, y = DsGen.prepare_data_for_training(dataset)
        
        # Save prepared data
        np.savetxt(os.path.join(args.output_dir, "X.csv"), X, delimiter=',')
        np.savetxt(os.path.join(args.output_dir, "y.csv"), y, delimiter=',')
        
        print(f"Prepared {len(X)} training samples")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Data saved to {os.path.join(args.output_dir, 'X.csv')} and {os.path.join(args.output_dir, 'y.csv')}")

if __name__ == "__main__":
    main() 