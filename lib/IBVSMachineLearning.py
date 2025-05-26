#!/usr/bin/env python3

import numpy as np
from machinevisiontoolbox import IBVS
from spatialmath import SE3
import logging as log
import torch
import json

class IBVSMachineLearning(IBVS):
    """
    Image-Based Visual Servoing using Machine Learning for control law computation.
    This class inherits from IBVS and replaces the classical control law with a trained model.
    """
    
    def __init__(self, camera, P, p_d, model_path, model_type='fnn', graphics=True):
        """
        Initialize the ML-based IBVS controller
        
        Args:
            camera: Camera instance for visual servoing
            P: Current camera pose
            p_d: Desired image points
            model_path: Path to the trained model file
            model_type: Type of the machine learning model
            graphics: Whether to use graphics for visualization
        """
        super().__init__(camera, P, p_d, graphics)
        self.model_path = model_path
        self.model_type = model_type
        
        # Load model and scalers
        self.load_model()
            
    def load_model(self):
        """Load the trained model and its scalers"""
        # Import the appropriate trainer class
        if self.model_type == 'fnn':
            from model_training.train_fnn import FNNTrainer
            trainer = FNNTrainer()
        elif self.model_type == 'lstm':
            from model_training.train_lstm import LSTMTrainer
            trainer = LSTMTrainer()
        elif self.model_type == 'resnet':
            from model_training.train_resnet import ResNetTrainer
            trainer = ResNetTrainer()
        elif self.model_type == 'hybrid':
            from model_training.train_hybrid import HybridTrainer
            trainer = HybridTrainer()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create model architecture
        self.model = trainer.create_model()
        
        # Load model weights
        model_state = torch.load(f"{self.model_path}/{self.model_type}_best.pth")
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        # Load metadata to get scaler paths
        with open(f"{self.model_path}/{self.model_type}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load scalers
        trainer.training_metadata = metadata
        trainer.load_scalers()
        self.feature_scaler = trainer.feature_scaler
        self.target_scaler = trainer.target_scaler
            
    def prepare_input_features(self, current_points, desired_points, depth=None):
        """
        Prepare the input features for the ML model
        
        Args:
            current_points: Current image points (2xN array)
            desired_points: Desired image points (2xN array)
            depth: Optional depth values
            
        Returns:
            Processed input features as expected by the ML model
        """
        # Flatten the points in column-major order (F)
        current_features = current_points.flatten(order='F')
        desired_features = desired_points.flatten(order='F')
        
        # Combine features
        features = np.concatenate([current_features, desired_features])
        
        # Add depth if available
        if depth is not None:
            if np.isscalar(depth):
                depth_values = np.full(current_points.shape[1], depth)
            else:
                depth_values = np.array(depth)
            features = np.concatenate([features, depth_values])
        
        # Normalize features
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1))
        
        return features
    
    def predict_velocity(self, features):
        """
        Predict velocity using the ML model
        
        Args:
            features: Normalized input features
            
        Returns:
            Predicted velocity command
        """
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(features_tensor)
        
        # Denormalize prediction
        if self.target_scaler is not None:
            prediction = self.target_scaler.inverse_transform(prediction.numpy())
            
        return prediction.flatten()
    
    def update_velocity(self):
        """
        Update the camera velocity using the ML model
        """
        # Get current feature points
        current_points = self.camera.project_point(P=self.P, pose=self.camera.pose)
        
        # Prepare input features
        features = self.prepare_input_features(current_points, self.p_star)
        
        # Get velocity prediction
        velocity = self.predict_velocity(features)
        
        # Update camera velocity
        self.camera.set_velocity(velocity)
        
    def step(self, t):
        """
        Compute one timestep of ML-based IBVS simulation
        
        Args:
            t: Current timestep
            
        Returns:
            status: Simulation status (0: running, 1: completed, -1: error)
        """
        status = 0
        
        try:
            # Get current image feature points
            current_points = self.camera.project_point(self.P)
            
            # Prepare input features for the model
            features = self.prepare_input_features(
                current_points, 
                self.p_star,
                self.depth if hasattr(self, 'depth') else None
            )
            
            if self.model is None:
                raise ValueError("No model loaded. Call load_model() first.")
                
            # TODO: Replace this with actual model inference
            # Example for PyTorch:
            # with torch.no_grad():
            #     features_tensor = torch.FloatTensor(features)
            #     v = self.model(features_tensor).numpy()
            
            # HERE: Implement the actual model inference code
            #
            # Example for TensorFlow:
            # v = self.model.predict(features.reshape(1, -1))[0]
            
            # For now, using a placeholder
            v = np.zeros(6)  # Replace this with actual model prediction
            
            # Update the camera pose using the predicted velocity
            Td = SE3.Delta(v)
            self.camera.pose @= Td
            
            # Update history
            hist = self._history()
            hist.p = current_points
            hist.vel = v
            e = (current_points - self.p_star).flatten(order='F')
            hist.e = e
            hist.enorm = np.linalg.norm(e)
            hist.pose = self.camera.pose
            self.history.append(hist)
            
            # Check termination condition
            if hist.enorm < self.eterm:
                status = 1
                
        except Exception as e:
            log.error(f"Error in ML-IBVS step: {e}")
            status = -1
            
        return status 