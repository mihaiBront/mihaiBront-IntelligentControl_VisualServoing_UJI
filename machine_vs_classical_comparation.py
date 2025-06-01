#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for WSL/headless environments

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import sys

# Add the lib directory to the path so we can import our custom classes
sys.path.append('lib')

# Standard RVC3 imports (same as chap15.ipynb)
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *

# Import our custom ML-based IBVS class
from lib.IBVSMachineLearning import IBVSMachineLearning

def ensure_resources_folder():
    """Create .resources folder if it doesn't exist"""
    if not os.path.exists('.resources'):
        os.makedirs('.resources')
        print("Created .resources folder for outputs")

def test_classical_ibvs():
    """Test the classical IBVS first for comparison"""
    print("=" * 60)
    print("Testing Classical IBVS (for comparison)")
    print("=" * 60)
    
    # Replicate EXACT training data conditions from IBVSHandler
    # Generate camera position like in training (random position behind target)
    camera = CentralCamera.Default(pose=SE3.Trans(1.5, 0.8, -1.5))  # Similar to training ranges
    
    # Generate world points exactly like training: mkgrid with 0 rotation, just translation
    # Training used: mkgrid(2, side=0.5, pose=SE3.Trans(x, y, z))
    P = mkgrid(2, side=0.5, pose=SE3.Trans(0.2, -0.3, 0.8))  # Random position like training
    
    # Use EXACT same desired points pattern as training
    # Training used: 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]
    pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]
    
    print(f"Camera pose (training-like): {camera.pose}")
    print(f"World points P (square, no rotation): {P.shape}")
    print(f"Desired points pd (training pattern): {pd.shape}")
    print(f"World points positions:")
    print(f"  P = {P}")
    print(f"Desired points pattern:")
    print(f"  pd = {pd}")
    
    # Create classical IBVS controller
    ibvs = IBVS(camera, P=P, p_d=pd, graphics=False)
    
    # Run simulation
    print("Running classical IBVS...")
    ibvs.run(50)
    
    print(f"Classical IBVS completed in {len(ibvs.history)} steps")
    print(f"Final error norm: {ibvs.history[-1].enorm:.6f}")
    
    # Clear any existing plots
    plt.close('all')
    
    # Create cleaner, custom plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error evolution
    ax1 = axes[0, 0]
    errors = [h.enorm for h in ibvs.history]
    ax1.plot(errors, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Error norm')
    ax1.set_title('Classical IBVS - Error Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Image plane trajectory (simplified)
    ax2 = axes[0, 1]
    # Plot desired points
    ax2.plot(pd[0, :], pd[1, :], 'r*', markersize=12, label='Desired', markeredgecolor='darkred')
    # Plot trajectory for each feature point
    colors = ['blue', 'green', 'orange', 'purple']
    for i in range(pd.shape[1]):
        p_traj = np.array([h.p[0, i] for h in ibvs.history])
        q_traj = np.array([h.p[1, i] for h in ibvs.history])
        ax2.plot(p_traj, q_traj, color=colors[i], linewidth=2, alpha=0.7, label=f'Point {i+1}')
        # Start and end markers
        ax2.plot(p_traj[0], q_traj[0], 'o', color=colors[i], markersize=8, markeredgecolor='black')
        ax2.plot(p_traj[-1], q_traj[-1], 's', color=colors[i], markersize=8, markeredgecolor='black')
    
    ax2.set_xlabel('u (pixels)')
    ax2.set_ylabel('v (pixels)')
    ax2.set_title('Classical IBVS - Image Plane Trajectory')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Velocity components
    ax3 = axes[1, 0]
    velocities = np.array([h.vel for h in ibvs.history])
    time_steps = range(len(ibvs.history))
    
    # Plot linear velocities
    ax3.plot(time_steps, velocities[:, 0], 'r-', linewidth=2, label='vx')
    ax3.plot(time_steps, velocities[:, 1], 'g-', linewidth=2, label='vy') 
    ax3.plot(time_steps, velocities[:, 2], 'b-', linewidth=2, label='vz')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Linear velocity (m/s)')
    ax3.set_title('Classical IBVS - Linear Velocities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Camera trajectory in 3D (top view)
    ax4 = axes[1, 1]
    poses = [h.pose for h in ibvs.history]
    x_traj = [pose.t[0] for pose in poses]
    y_traj = [pose.t[1] for pose in poses]
    
    ax4.plot(x_traj, y_traj, 'b-', linewidth=2, alpha=0.7, label='Camera path')
    ax4.plot(x_traj[0], y_traj[0], 'go', markersize=10, label='Start')
    ax4.plot(x_traj[-1], y_traj[-1], 'ro', markersize=10, label='End')
    
    # Plot world points projection
    ax4.plot(P[0, :], P[1, :], 'k*', markersize=12, label='Target points')
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Classical IBVS - Camera Trajectory (Top View)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig('.resources/classical_ibvs_results.png', dpi=150, bbox_inches='tight')
    print("Classical IBVS plots saved to .resources/classical_ibvs_results.png")
    
    return ibvs

def test_ml_ibvs(model_path="training_models"):
    """Test the ML-based IBVS"""
    print("=" * 60)
    print("Testing ML-based IBVS")
    print("=" * 60)
    
    # Check if model files exist
    required_files = [
        f"{model_path}/lstm_best.pth",
        f"{model_path}/lstm_feature_scaler.pkl",
        f"{model_path}/lstm_target_scaler.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Missing required model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you have trained the LSTM model and saved the required files.")
        return None
    
    # Use SAME conditions as classical IBVS (replicating training data conditions)
    camera = CentralCamera.Default(pose=SE3.Trans(1.5, 0.8, -1.5))  # Same as classical test
    P = mkgrid(2, side=0.5, pose=SE3.Trans(0.2, -0.3, 0.8))  # Same square pattern, no rotation
    pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]  # Same desired pattern
    
    print(f"Camera pose (training-like): {camera.pose}")
    print(f"World points P (square, no rotation): {P.shape}")
    print(f"Desired points pd (training pattern): {pd.shape}")
    
    try:
        # Create ML-based IBVS controller
        print(f"Loading ML model from: {model_path}")
        ml_ibvs = IBVSMachineLearning(camera, P=P, p_d=pd, model_path=model_path, graphics=False)
        print("ML model loaded successfully!")
        
        # Run simulation
        print("Running ML-based IBVS...")
        ml_ibvs.run(100)  # Allow more steps in case ML model needs it
        
        print(f"ML IBVS completed in {len(ml_ibvs.history)} steps")
        print(f"Final error norm: {ml_ibvs.history[-1].enorm:.6f}")
        
        # Clear any existing plots
        plt.close('all')
        
        # Create cleaner, custom plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Error evolution
        ax1 = axes[0, 0]
        errors = [h.enorm for h in ml_ibvs.history]
        ax1.plot(errors, 'r-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Error norm')
        ax1.set_title('ML IBVS - Error Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Image plane trajectory (simplified)
        ax2 = axes[0, 1]
        # Plot desired points
        ax2.plot(pd[0, :], pd[1, :], 'r*', markersize=12, label='Desired', markeredgecolor='darkred')
        # Plot trajectory for each feature point
        colors = ['blue', 'green', 'orange', 'purple']
        for i in range(pd.shape[1]):
            p_traj = np.array([h.p[0, i] for h in ml_ibvs.history])
            q_traj = np.array([h.p[1, i] for h in ml_ibvs.history])
            ax2.plot(p_traj, q_traj, color=colors[i], linewidth=2, alpha=0.7, label=f'Point {i+1}')
            # Start and end markers
            ax2.plot(p_traj[0], q_traj[0], 'o', color=colors[i], markersize=8, markeredgecolor='black')
            ax2.plot(p_traj[-1], q_traj[-1], 's', color=colors[i], markersize=8, markeredgecolor='black')
        
        ax2.set_xlabel('u (pixels)')
        ax2.set_ylabel('v (pixels)')
        ax2.set_title('ML IBVS - Image Plane Trajectory')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Plot 3: Velocity components
        ax3 = axes[1, 0]
        velocities = np.array([h.vel for h in ml_ibvs.history])
        time_steps = range(len(ml_ibvs.history))
        
        # Plot linear velocities
        ax3.plot(time_steps, velocities[:, 0], 'r-', linewidth=2, label='vx')
        ax3.plot(time_steps, velocities[:, 1], 'g-', linewidth=2, label='vy') 
        ax3.plot(time_steps, velocities[:, 2], 'b-', linewidth=2, label='vz')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Linear velocity (m/s)')
        ax3.set_title('ML IBVS - Linear Velocities')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Camera trajectory in 3D (top view)
        ax4 = axes[1, 1]
        poses = [h.pose for h in ml_ibvs.history]
        x_traj = [pose.t[0] for pose in poses]
        y_traj = [pose.t[1] for pose in poses]
        
        ax4.plot(x_traj, y_traj, 'r-', linewidth=2, alpha=0.7, label='Camera path')
        ax4.plot(x_traj[0], y_traj[0], 'go', markersize=10, label='Start')
        ax4.plot(x_traj[-1], y_traj[-1], 'ro', markersize=10, label='End')
        
        # Plot world points projection
        ax4.plot(P[0, :], P[1, :], 'k*', markersize=12, label='Target points')
        
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('ML IBVS - Camera Trajectory (Top View)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        plt.savefig('.resources/ml_ibvs_results.png', dpi=150, bbox_inches='tight')
        print("ML IBVS plots saved to .resources/ml_ibvs_results.png")
        
        return ml_ibvs
        
    except Exception as e:
        print(f"ERROR: Failed to run ML-based IBVS: {e}")
        return None

def compare_results(classical_ibvs, ml_ibvs):
    """Compare the results of classical and ML-based IBVS"""
    if ml_ibvs is None:
        print("Cannot compare results - ML IBVS failed")
        return
    
    print("=" * 60)
    print("Comparison of Classical vs ML-based IBVS")
    print("=" * 60)
    
    # Extract data from both controllers
    classical_steps = len(classical_ibvs.history)
    ml_steps = len(ml_ibvs.history)
    
    classical_final_error = classical_ibvs.history[-1].enorm
    ml_final_error = ml_ibvs.history[-1].enorm
    
    print(f"Steps to completion:")
    print(f"  Classical IBVS: {classical_steps}")
    print(f"  ML IBVS:        {ml_steps}")
    
    print(f"\nFinal error norm:")
    print(f"  Classical IBVS: {classical_final_error:.6f}")
    print(f"  ML IBVS:        {ml_final_error:.6f}")
    
    # Determine performance improvement
    if ml_final_error < classical_final_error:
        improvement = (classical_final_error - ml_final_error) / classical_final_error * 100
        print(f"  ML IBVS is {improvement:.1f}% better!")
    else:
        degradation = (ml_final_error - classical_final_error) / classical_final_error * 100
        print(f"  Classical IBVS is {degradation:.1f}% better")
    
    # Clear any existing plots
    plt.close('all')
    
    # Create a clean, focused comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Error evolution comparison
    ax1 = axes[0]
    classical_errors = [h.enorm for h in classical_ibvs.history]
    ml_errors = [h.enorm for h in ml_ibvs.history]
    
    ax1.plot(classical_errors, 'b-', label='Classical IBVS', linewidth=3, alpha=0.8)
    ax1.plot(ml_errors, 'r--', label='ML IBVS', linewidth=3, alpha=0.8)
    ax1.set_xlabel('Time step', fontsize=12)
    ax1.set_ylabel('Error norm', fontsize=12)
    ax1.set_title('Error Evolution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add final error annotations
    ax1.annotate(f'Final: {classical_final_error:.3f}', 
                xy=(classical_steps-1, classical_final_error), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                fontsize=10)
    ax1.annotate(f'Final: {ml_final_error:.3f}', 
                xy=(ml_steps-1, ml_final_error), 
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                fontsize=10)
    
    # Plot 2: Trajectory comparison in image plane (simplified)
    ax2 = axes[1]
    
    # Plot desired points (same for both)
    pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[classical_ibvs.camera.pp]
    ax2.plot(pd[0, :], pd[1, :], 'r*', markersize=15, label='Desired', 
             markeredgecolor='darkred', markeredgewidth=2)
    
    # Plot final positions for both methods
    classical_final_p = classical_ibvs.history[-1].p
    ml_final_p = ml_ibvs.history[-1].p
    
    ax2.plot(classical_final_p[0, :], classical_final_p[1, :], 'bo', 
             markersize=10, label='Classical final', markeredgecolor='darkblue')
    ax2.plot(ml_final_p[0, :], ml_final_p[1, :], 'rs', 
             markersize=10, label='ML final', markeredgecolor='darkred')
    
    # Connect desired to final positions with lines
    for i in range(pd.shape[1]):
        ax2.plot([pd[0, i], classical_final_p[0, i]], 
                [pd[1, i], classical_final_p[1, i]], 
                'b--', alpha=0.5, linewidth=1)
        ax2.plot([pd[0, i], ml_final_p[0, i]], 
                [pd[1, i], ml_final_p[1, i]], 
                'r--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('u (pixels)', fontsize=12)
    ax2.set_ylabel('v (pixels)', fontsize=12)
    ax2.set_title('Final Image Points Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('.resources/ibvs_comparison.png', dpi=150, bbox_inches='tight')
    print("Clean comparison plots saved to .resources/ibvs_comparison.png")
    
    # Create a single detailed error plot for publication
    plt.figure(figsize=(10, 6))
    plt.plot(classical_errors, 'b-', label='Classical IBVS', linewidth=3, alpha=0.8)
    plt.plot(ml_errors, 'r--', label='ML IBVS', linewidth=3, alpha=0.8)
    plt.xlabel('Time step', fontsize=14)
    plt.ylabel('Error norm', fontsize=14)
    plt.title('IBVS Performance Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add performance summary text box
    if ml_final_error < classical_final_error:
        improvement = (classical_final_error - ml_final_error) / classical_final_error * 100
        summary_text = f'ML IBVS achieves {improvement:.1f}% better accuracy\n' + \
                      f'Classical: {classical_final_error:.3f}\n' + \
                      f'ML: {ml_final_error:.3f}'
        box_color = 'lightgreen'
    else:
        degradation = (ml_final_error - classical_final_error) / classical_final_error * 100
        summary_text = f'Classical IBVS is {degradation:.1f}% better\n' + \
                      f'Classical: {classical_final_error:.3f}\n' + \
                      f'ML: {ml_final_error:.3f}'
        box_color = 'lightyellow'
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('.resources/error_comparison_detailed.png', dpi=150, bbox_inches='tight')
    print("Detailed error comparison saved to .resources/error_comparison_detailed.png")

def main():
    """Main comparison function"""
    print("Machine Learning vs Classical IBVS Comparison")
    print("Based on Chapter 15 examples from RVC3")
    print("=" * 60)
    
    # Ensure output folder exists
    ensure_resources_folder()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test classical IBVS first
    classical_ibvs = test_classical_ibvs()
    
    print("\n" + "=" * 60)
    print("Continuing with ML-based IBVS test...")
    
    # Test ML-based IBVS
    ml_ibvs = test_ml_ibvs()
    
    # Compare results if both succeeded
    if classical_ibvs and ml_ibvs:
        print("\n" + "=" * 60)
        print("Generating comparison plots...")
        compare_results(classical_ibvs, ml_ibvs)
    
    print("\n" + "=" * 60)
    print("Comparison completed!")
    print("All results saved in .resources/ folder:")
    print("  - classical_ibvs_results.png (4-panel detailed analysis)")
    if ml_ibvs:
        print("  - ml_ibvs_results.png (4-panel detailed analysis)")
        print("  - ibvs_comparison.png (clean side-by-side comparison)")
        print("  - error_comparison_detailed.png (publication-ready error plot)")
    print("\nThe plots are now much cleaner and easier to read!")
    print("Each individual plot focuses on specific aspects:")
    print("  • Error evolution with log scale")
    print("  • Image plane trajectories with clear markers") 
    print("  • Linear velocity components over time")
    print("  • Camera trajectory in top view")
    print("  • Clean comparison plots with performance metrics")

if __name__ == "__main__":
    main() 