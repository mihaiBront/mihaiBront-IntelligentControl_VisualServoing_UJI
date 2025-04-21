from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
import os
import time
import pandas as pd
import csv
import json

# Import RVC3 libraries
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *

class IPVSDatasetGeneration():
    def generate_random_points(num_points=4, min_distance=0.5, max_distance=3.0):
        """Generate random 3D points for the IBVS task.
        
        Args:
            num_points: Number of points to generate
            min_distance: Minimum distance from the origin
            max_distance: Maximum distance from the origin
            
        Returns:
            P: 3xN array of 3D points
        """
        # Generate random angles for spherical coordinates
        theta = np.random.uniform(0, 2*pi, num_points)  # azimuthal angle
        phi = np.random.uniform(0, pi, num_points)      # polar angle
        
        # Generate random distances
        r = np.random.uniform(min_distance, max_distance, num_points)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Stack into a 3xN array
        P = np.vstack((x, y, z))
        
        return P

    def generate_random_camera_pose(min_distance=1.0, max_distance=5.0):
        """Generate a random camera pose.
        
        Args:
            min_distance: Minimum distance from the origin
            max_distance: Maximum distance from the origin
            
        Returns:
            pose: SE3 object representing the camera pose
        """
        # Generate random position
        x = np.random.uniform(-max_distance, max_distance)
        y = np.random.uniform(-max_distance, max_distance)
        z = np.random.uniform(-max_distance, -min_distance)  # Camera is looking at the origin
        
        # Generate random rotation (Euler angles)
        roll = np.random.uniform(-pi/4, pi/4)
        pitch = np.random.uniform(-pi/4, pi/4)
        yaw = np.random.uniform(0, 2*pi)
        
        # Create SE3 object
        pose = SE3.Trans(x, y, z) * SE3.RPY(roll, pitch, yaw)
        
        return pose

    def generate_desired_image_points(camera, num_points=4, size=200):
        """Generate desired image points in a grid pattern.
        
        Args:
            camera: Camera object
            num_points: Number of points to generate
            size: Size of the grid
            
        Returns:
            pd: 2xN array of desired image points
        """
        # Calculate grid size based on number of points
        grid_size = int(np.ceil(np.sqrt(num_points)))
        
        # Generate grid points
        x = np.linspace(-size/2, size/2, grid_size)
        y = np.linspace(-size/2, size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Flatten and take only the required number of points
        pd = np.vstack((X.flatten()[:num_points], Y.flatten()[:num_points]))
        
        # Add camera principal point
        pd = pd + np.c_[camera.pp]
        
        return pd

    def run_ibvs_sequence(camera, P, pd, max_iterations=50, lambda_value=0.1, verbose=False):
        """Run an IBVS sequence and collect data.
        
        Args:
            camera: Camera object
            P: 3D points
            pd: Desired image points
            max_iterations: Maximum number of iterations
            lambda_value: Gain for the IBVS controller
            verbose: Whether to print progress
            
        Returns:
            data: Dictionary containing the collected data
        """
        # Create IBVS object
        ibvs = IBVS(camera, P=P, p_d=pd, lmbda=lambda_value)
        
        # Initialize data storage
        current_points = []
        desired_points = []
        camera_poses = []
        camera_velocities = []
        errors = []
        
        # Run IBVS for the specified number of iterations
        for i in range(max_iterations):
            # Store current data
            current_points.append(ibvs.P.copy())
            desired_points.append(pd.copy())  # Store the desired points at each timestep
            camera_poses.append(ibvs.camera.pose.copy())
            
            # Calculate error
            current_p = ibvs.camera.project_point(P, pose=ibvs.camera.pose)
            error = current_p - pd
            errors.append(error)
            
            # Run one iteration
            ibvs.step(i)  # Pass the iteration number as the time parameter
            
            # Store camera velocity - check if the attribute exists
            if hasattr(ibvs, 'v'):
                camera_velocities.append(ibvs.v.copy())
            else:
                # If 'v' attribute doesn't exist, try to get velocity from other attributes
                # This is a fallback and may need adjustment based on the actual IBVS implementation
                try:
                    # Try to get velocity from the error and Jacobian
                    J = ibvs.camera.visjac_p(ibvs.P, depth=1)
                    e = error.flatten()
                    v = lambda_value * np.linalg.pinv(J) @ e
                    camera_velocities.append(v)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not get camera velocity: {e}")
                    # Use a placeholder if we can't get the velocity
                    camera_velocities.append(np.zeros(6))
            
            # Check if converged
            error_norm = np.linalg.norm(error)
            if error_norm < 1e-3:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break
        
        # Compile data
        data = {
            'current_points': np.array(current_points),
            'desired_points': np.array(desired_points),  # Now this will have the same shape as current_points
            'camera_poses': camera_poses,
            'camera_velocities': np.array(camera_velocities),
            'errors': np.array(errors),
            'num_iterations': len(current_points),
            'final_error': np.linalg.norm(errors[-1]) if errors else 0
        }
        
        return data

    def prepare_data_for_training(dataset):
        """Prepare data for neural network training.
        
        Args:
            dataset: List of dictionaries containing IBVS sequence data
            
        Returns:
            X: Input features for neural network
            y: Target actions for neural network
        """
        X = []  # Input features
        y = []  # Target actions
        
        for sequence in dataset:
            # For each timestep in the sequence
            for t in range(len(sequence['current_points']) - 1):  # -1 because we don't have action for the last timestep
                # Input features: current points, desired points, and camera pose
                current_points = sequence['current_points'][t].flatten()
                desired_points = sequence['desired_points'][t].flatten()
                # Use the correct attribute for the SE3 object
                camera_pose = sequence['camera_poses'][t].A.flatten()
                
                # Combine features
                features = np.concatenate([current_points, desired_points, camera_pose])
                
                # Target: camera velocity (action taken by IBVS)
                action = sequence['camera_velocities'][t]
                
                # Add to dataset
                X.append(features)
                y.append(action)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X, y

    def visualize_sequence(sequence, save_path=None):
        """Visualize a sample sequence.
        
        Args:
            sequence: Dictionary containing IBVS sequence data
            save_path: Path to save the visualization (if None, display the plot)
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Image plane trajectory
        plt.subplot(2, 2, 1)
        for i in range(sequence['current_points'].shape[2]):  # For each point
            plt.plot(sequence['current_points'][:, 0, i], sequence['current_points'][:, 1, i], 'b-', label=f'Point {i+1}')
            
            # Handle different shapes of desired_points
            if len(sequence['desired_points'].shape) == 3:  # Shape is (T, 2, N)
                plt.plot(sequence['desired_points'][0, 0, i], sequence['desired_points'][0, 1, i], 'ro', label=f'Desired {i+1}')
            else:  # Shape is (2, N)
                plt.plot(sequence['desired_points'][0, i], sequence['desired_points'][1, i], 'ro', label=f'Desired {i+1}')
        
        plt.title('Image Plane Trajectory')
        plt.xlabel('u (pixels)')
        plt.ylabel('v (pixels)')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Camera velocity
        plt.subplot(2, 2, 2)
        t = np.arange(len(sequence['camera_velocities']))
        plt.plot(t, sequence['camera_velocities'][:, 0], 'r-', label='vx')
        plt.plot(t, sequence['camera_velocities'][:, 1], 'g-', label='vy')
        plt.plot(t, sequence['camera_velocities'][:, 2], 'b-', label='vz')
        plt.plot(t, sequence['camera_velocities'][:, 3], 'c-', label='ωx')
        plt.plot(t, sequence['camera_velocities'][:, 4], 'm-', label='ωy')
        plt.plot(t, sequence['camera_velocities'][:, 5], 'y-', label='ωz')
        plt.title('Camera Velocity')
        plt.xlabel('Iteration')
        plt.ylabel('Velocity')
        plt.grid(True)
        plt.legend()
        
        # Plot 3: Camera trajectory in 3D
        plt.subplot(2, 2, 3)
        camera_positions = np.array([pose.t for pose in sequence['camera_poses']])
        plt.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 'b-')
        plt.plot(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2], 'go', label='Start')
        plt.plot(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], 'ro', label='End')
        plt.title('Camera Trajectory')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.legend()
        
        # Plot 4: Error norm
        plt.subplot(2, 2, 4)
        errors = []
        for t in range(len(sequence['current_points'])):
            # Calculate error based on the shapes of current_points and desired_points
            if len(sequence['desired_points'].shape) == 3:  # Shape is (T, 2, N)
                error = sequence['current_points'][t, :2, :] - sequence['desired_points'][t, :, :]
            else:  # Shape is (2, N)
                error = sequence['current_points'][t, :2, :] - sequence['desired_points']
            
            error_norm = np.linalg.norm(error)
            errors.append(error_norm)
        
        plt.plot(errors, 'k-')
        plt.title('Error Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Error Norm (pixels)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def save_sequence_to_csv(sequence, output_dir, sequence_id):
        """Save a sequence to CSV files.
        
        Args:
            sequence: Dictionary containing IBVS sequence data
            output_dir: Directory to save the CSV files
            sequence_id: ID of the sequence
        """
        # Create directory for this sequence
        sequence_dir = os.path.join(output_dir, f"sequence_{sequence_id}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Save current points
        current_points_file = os.path.join(sequence_dir, "current_points.csv")
        np.savetxt(current_points_file, sequence['current_points'].reshape(sequence['current_points'].shape[0], -1), delimiter=',')
        
        # Save desired points
        desired_points_file = os.path.join(sequence_dir, "desired_points.csv")
        np.savetxt(desired_points_file, sequence['desired_points'].reshape(sequence['desired_points'].shape[0], -1), delimiter=',')
        
        # Save camera velocities
        camera_velocities_file = os.path.join(sequence_dir, "camera_velocities.csv")
        np.savetxt(camera_velocities_file, sequence['camera_velocities'], delimiter=',')
        
        # Save errors
        errors_file = os.path.join(sequence_dir, "errors.csv")
        np.savetxt(errors_file, sequence['errors'].reshape(sequence['errors'].shape[0], -1), delimiter=',')
        
        # Save camera poses (as matrices)
        camera_poses_file = os.path.join(sequence_dir, "camera_poses.csv")
        with open(camera_poses_file, 'w') as f:
            for i, pose in enumerate(sequence['camera_poses']):
                f.write(f"pose_{i}\n")
                # Use the correct attribute for the SE3 object
                np.savetxt(f, pose.A, delimiter=',')
                f.write("\n")
        
        # Save metadata
        metadata = {
            'sequence_id': sequence_id,
            'num_iterations': sequence['num_iterations'],
            'final_error': float(sequence['final_error']),
            'initial_camera_pose': {
                'translation': sequence['initial_camera_pose'].t.tolist(),
                'rotation': sequence['initial_camera_pose'].R.tolist()
            },
            '3d_points': sequence['3d_points'].tolist()
        }
        
        metadata_file = os.path.join(sequence_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_sequence_from_csv(output_dir, sequence_id):
        """Load a sequence from CSV files.
        
        Args:
            output_dir: Directory containing the CSV files
            sequence_id: ID of the sequence
            
        Returns:
            sequence: Dictionary containing IBVS sequence data
        """
        sequence_dir = os.path.join(output_dir, f"sequence_{sequence_id}")
        
        # Load current points
        current_points_file = os.path.join(sequence_dir, "current_points.csv")
        current_points = np.loadtxt(current_points_file, delimiter=',')
        
        # Load desired points
        desired_points_file = os.path.join(sequence_dir, "desired_points.csv")
        desired_points = np.loadtxt(desired_points_file, delimiter=',')
        
        # Load camera velocities
        camera_velocities_file = os.path.join(sequence_dir, "camera_velocities.csv")
        camera_velocities = np.loadtxt(camera_velocities_file, delimiter=',')
        
        # Load errors
        errors_file = os.path.join(sequence_dir, "errors.csv")
        errors = np.loadtxt(errors_file, delimiter=',')
        
        # Load camera poses
        camera_poses_file = os.path.join(sequence_dir, "camera_poses.csv")
        camera_poses = []
        with open(camera_poses_file, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith('pose_'):
                    matrix = np.zeros((4, 4))
                    for j in range(4):
                        matrix[j] = [float(x) for x in lines[i+j+1].strip().split(',')]
                    camera_poses.append(SE3(matrix))
                    i += 5
                else:
                    i += 1
        
        # Load metadata
        metadata_file = os.path.join(sequence_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Compile data
        sequence = {
            'current_points': current_points,
            'desired_points': desired_points,
            'camera_poses': camera_poses,
            'camera_velocities': camera_velocities,
            'errors': errors,
            'num_iterations': metadata['num_iterations'],
            'final_error': metadata['final_error'],
            'sequence_id': metadata['sequence_id'],
            'initial_camera_pose': SE3(metadata['initial_camera_pose']['translation'], metadata['initial_camera_pose']['rotation']),
            '3d_points': np.array(metadata['3d_points'])
        }
        
        return sequence

    def save_dataset_to_csv(dataset, output_dir):
        """Save the entire dataset to a single CSV file.
        
        Args:
            dataset: List of dictionaries containing IBVS sequence data
            output_dir: Directory to save the CSV file
        """
        # Create directory for the dataset
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a single CSV file for all sequences
        csv_file = os.path.join(output_dir, "ibvs_dataset.csv")
        
        # Define the header for the CSV file
        header = [
            'sequence_id', 'timestep', 
            'current_point1_x', 'current_point1_y', 
            'current_point2_x', 'current_point2_y',
            'current_point3_x', 'current_point3_y',
            'current_point4_x', 'current_point4_y',
            'desired_point1_x', 'desired_point1_y',
            'desired_point2_x', 'desired_point2_y',
            'desired_point3_x', 'desired_point3_y',
            'desired_point4_x', 'desired_point4_y',
            'camera_pose_11', 'camera_pose_12', 'camera_pose_13', 'camera_pose_14',
            'camera_pose_21', 'camera_pose_22', 'camera_pose_23', 'camera_pose_24',
            'camera_pose_31', 'camera_pose_32', 'camera_pose_33', 'camera_pose_34',
            'camera_pose_41', 'camera_pose_42', 'camera_pose_43', 'camera_pose_44',
            'camera_velocity_vx', 'camera_velocity_vy', 'camera_velocity_vz',
            'camera_velocity_wx', 'camera_velocity_wy', 'camera_velocity_wz',
            'error1_x', 'error1_y',
            'error2_x', 'error2_y',
            'error3_x', 'error3_y',
            'error4_x', 'error4_y',
            'initial_camera_pose_tx', 'initial_camera_pose_ty', 'initial_camera_pose_tz',
            'initial_camera_pose_r11', 'initial_camera_pose_r12', 'initial_camera_pose_r13',
            'initial_camera_pose_r21', 'initial_camera_pose_r22', 'initial_camera_pose_r23',
            'initial_camera_pose_r31', 'initial_camera_pose_r32', 'initial_camera_pose_r33',
            '3d_point1_x', '3d_point1_y', '3d_point1_z',
            '3d_point2_x', '3d_point2_y', '3d_point2_z',
            '3d_point3_x', '3d_point3_y', '3d_point3_z',
            '3d_point4_x', '3d_point4_y', '3d_point4_z',
            'final_error'
        ]
        
        # Write the CSV file
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for sequence_id, sequence in enumerate(dataset):
                try:
                    # Get the number of points in this sequence
                    num_points = sequence['current_points'].shape[2]  # Changed from shape[1] to shape[2]
                    
                    # For each timestep in the sequence
                    for t in range(len(sequence['current_points'])):
                        row = [sequence_id, t]
                        
                        # Add current points (pad with zeros if less than 4 points)
                        for i in range(4):
                            if i < num_points:
                                # Handle the case where current_points has 3 rows instead of 2
                                if sequence['current_points'].shape[1] == 3:
                                    # Use only the first two rows (x, y coordinates)
                                    row.extend([sequence['current_points'][t, 0, i], sequence['current_points'][t, 1, i]])
                                else:
                                    row.extend([sequence['current_points'][t, i, 0], sequence['current_points'][t, i, 1]])
                            else:
                                row.extend([0, 0])
                        
                        # Add desired points (pad with zeros if less than 4 points)
                        # The desired points are the same for all timesteps, so we use the first timestep
                        for i in range(4):
                            if i < num_points:
                                # Handle the case where desired_points has a different shape
                                if len(sequence['desired_points'].shape) == 3:  # Shape is (T, 2, N)
                                    row.extend([sequence['desired_points'][0, 0, i], sequence['desired_points'][0, 1, i]])
                                else:  # Shape is (2, N)
                                    row.extend([sequence['desired_points'][0, i], sequence['desired_points'][1, i]])
                            else:
                                row.extend([0, 0])
                        
                        # Add camera pose (4x4 matrix)
                        camera_pose = sequence['camera_poses'][t].A
                        for i in range(4):
                            for j in range(4):
                                row.append(camera_pose[i, j])
                        
                        # Add camera velocity (6x1 vector)
                        if t < len(sequence['camera_velocities']):
                            row.extend(sequence['camera_velocities'][t])
                        else:
                            row.extend([0, 0, 0, 0, 0, 0])
                        
                        # Add errors (pad with zeros if less than 4 points)
                        for i in range(4):
                            if i < num_points and t < len(sequence['errors']):
                                # Handle the case where errors has a different shape
                                if len(sequence['errors'].shape) == 3:  # Shape is (T, 2, N)
                                    row.extend([sequence['errors'][t, 0, i], sequence['errors'][t, 1, i]])
                                else:  # Shape is (2, N)
                                    row.extend([sequence['errors'][0, i], sequence['errors'][1, i]])
                            else:
                                row.extend([0, 0])
                        
                        # Add initial camera pose
                        initial_pose = sequence['initial_camera_pose']
                        row.extend(initial_pose.t)  # Translation
                        row.extend(initial_pose.R.flatten())  # Rotation matrix
                        
                        # Add 3D points (pad with zeros if less than 4 points)
                        for i in range(4):
                            if i < num_points:
                                row.extend(sequence['3d_points'][:, i])
                            else:
                                row.extend([0, 0, 0])
                        
                        # Add final error
                        row.append(sequence['final_error'])
                        
                        writer.writerow(row)
                except Exception as e:
                    print(f"Error processing sequence {sequence_id}: {e}")
                    print(f"Sequence data: {sequence.keys()}")
                    if 'desired_points' in sequence:
                        print(f"desired_points shape: {sequence['desired_points'].shape}")
                    if 'current_points' in sequence:
                        print(f"current_points shape: {sequence['current_points'].shape}")
                    # Skip this sequence and continue with the next one
                    continue
        
        # Save dataset metadata
        dataset_metadata = {
            'num_sequences': len(dataset),
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'IBVS dataset for neural network training',
            'csv_file': csv_file
        }
        
        dataset_metadata_file = os.path.join(output_dir, "dataset_metadata.json")
        with open(dataset_metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"Saved {len(dataset)} sequences to {csv_file}")

    def load_dataset_from_csv(output_dir):
        """Load the entire dataset from a single CSV file.
        
        Args:
            output_dir: Directory containing the CSV file
            
        Returns:
            dataset: List of dictionaries containing IBVS sequence data
        """
        # Load dataset metadata
        dataset_metadata_file = os.path.join(output_dir, "dataset_metadata.json")
        with open(dataset_metadata_file, 'r') as f:
            dataset_metadata = json.load(f)
        
        csv_file = dataset_metadata['csv_file']
        
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Get unique sequence IDs
        sequence_ids = data['sequence_id'].unique()
        
        # Initialize dataset
        dataset = []
        
        # Process each sequence
        for sequence_id in sequence_ids:
            # Get data for this sequence
            sequence_data = data[data['sequence_id'] == sequence_id]
            
            # Get the number of timesteps
            num_timesteps = len(sequence_data)
            
            # Initialize arrays for this sequence
            current_points = []
            desired_points = []
            camera_poses = []
            camera_velocities = []
            errors = []
            
            # Process each timestep
            for _, row in sequence_data.iterrows():
                # Extract current points
                current_point = np.zeros((2, 4))
                for i in range(4):
                    current_point[0, i] = row[f'current_point{i+1}_x']
                    current_point[1, i] = row[f'current_point{i+1}_y']
                current_points.append(current_point)
                
                # Extract desired points
                desired_point = np.zeros((2, 4))
                for i in range(4):
                    desired_point[0, i] = row[f'desired_point{i+1}_x']
                    desired_point[1, i] = row[f'desired_point{i+1}_y']
                desired_points.append(desired_point)
                
                # Extract camera pose
                camera_pose_matrix = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        camera_pose_matrix[i, j] = row[f'camera_pose_{i+1}{j+1}']
                camera_poses.append(SE3(camera_pose_matrix))
                
                # Extract camera velocity
                camera_velocity = np.array([
                    row['camera_velocity_vx'], row['camera_velocity_vy'], row['camera_velocity_vz'],
                    row['camera_velocity_wx'], row['camera_velocity_wy'], row['camera_velocity_wz']
                ])
                camera_velocities.append(camera_velocity)
                
                # Extract errors
                error = np.zeros((2, 4))
                for i in range(4):
                    error[0, i] = row[f'error{i+1}_x']
                    error[1, i] = row[f'error{i+1}_y']
                errors.append(error)
            
            # Extract initial camera pose
            initial_camera_pose_t = np.array([
                row['initial_camera_pose_tx'],
                row['initial_camera_pose_ty'],
                row['initial_camera_pose_tz']
            ])
            
            initial_camera_pose_R = np.array([
                [row['initial_camera_pose_r11'], row['initial_camera_pose_r12'], row['initial_camera_pose_r13']],
                [row['initial_camera_pose_r21'], row['initial_camera_pose_r22'], row['initial_camera_pose_r23']],
                [row['initial_camera_pose_r31'], row['initial_camera_pose_r32'], row['initial_camera_pose_r33']]
            ])
            
            initial_camera_pose = SE3(initial_camera_pose_t, initial_camera_pose_R)
            
            # Extract 3D points
            P = np.zeros((3, 4))
            for i in range(4):
                P[0, i] = row[f'3d_point{i+1}_x']
                P[1, i] = row[f'3d_point{i+1}_y']
                P[2, i] = row[f'3d_point{i+1}_z']
            
            # Compile sequence data
            sequence = {
                'current_points': np.array(current_points),
                'desired_points': np.array(desired_points),
                'camera_poses': camera_poses,
                'camera_velocities': np.array(camera_velocities),
                'errors': np.array(errors),
                'num_iterations': num_timesteps,
                'final_error': row['final_error'],
                'sequence_id': sequence_id,
                'initial_camera_pose': initial_camera_pose,
                '3d_points': P
            }
            
            dataset.append(sequence)
        
        print(f"Loaded {len(dataset)} sequences from {csv_file}")
        
        return dataset
