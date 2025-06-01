#!/usr/bin/env python3

import numpy as np
from machinevisiontoolbox import IBVS
from spatialmath import SE3
import logging as log
import torch
import torch.nn as nn
import pickle
import json

class IBVSMachineLearning(IBVS):
    """
    Image-Based Visual Servoing using Machine Learning for control law computation.
    This class inherits from IBVS and replaces the classical control law with a trained model.
    """
    
    def __init__(self, camera, P, p_d, model_path, graphics=True):
        """
        Initialize the ML-based IBVS controller
        
        Args:
            camera: Camera instance for visual servoing
            P: Current camera pose
            p_d: Desired image points
            model_path: Path to the trained model file
            graphics: Whether to use graphics for visualization
        """
        # Store ML-specific parameters first
        self.model_path = model_path
        
        # Call parent constructor with exact same parameters as classical IBVS
        super().__init__(camera, P=P, p_d=p_d, graphics=graphics)
        
        # Load model and scalers after parent initialization
        self.load_model()
            
    def load_model(self):
        """Load the trained model and its scalers"""
        try:
            # Load metadata to get model configuration
            with open(f"{self.model_path}/lstm_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            model_config = metadata['model_config']
            print(f"Loading LSTM model with config: {model_config}")
            
            # Define LSTM model architecture with actual parameters
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.0):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                      batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
                    
                    # Fully connected head matching the saved model structure
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 32),  # 64 → 32
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),    # 0.2 dropout
                        nn.Linear(32, output_size)   # 32 → 3
                    )
                    
                def forward(self, x):
                    # Add sequence dimension if not present
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # Add sequence dimension
                    
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])  # Take last output
                    return out
            
            # Create model instance with actual parameters
            self.model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                output_size=model_config['output_size'],
                dropout_rate=model_config.get('dropout_rate', 0.0)
            )
            print("Model architecture created successfully")
            
            # Load model weights
            model_path = f"{self.model_path}/lstm_best.pth"
            print(f"Loading model weights from: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            print(f"State dict keys: {list(state_dict.keys())}")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model weights loaded successfully")
            
            # Load scalers
            print("Loading feature scaler...")
            with open(f"{self.model_path}/lstm_feature_scaler.pkl", 'rb') as f:
                self.feature_scaler = pickle.load(f)
                
            print("Loading target scaler...")
            with open(f"{self.model_path}/lstm_target_scaler.pkl", 'rb') as f:
                self.target_scaler = pickle.load(f)
                
            print("Scalers loaded successfully")
                
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise FileNotFoundError(f"Model files not found: {e}")
        except Exception as e:
            print(f"Exception during model loading: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error loading model: {e}")
        
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
            current_features = current_points.flatten(order='F')
            # Based on metadata, model expects only current features (8 values), not desired features
            features = current_features
            
            # Ensure features are float64 for scaler compatibility
            features = features.astype(np.float64)
            
            # Normalize features
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform(features.reshape(1, -1))
                # Scaler output is already 2D (1, n_features)
                features = features.flatten()
            
            # Convert to tensor with proper shape and type
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                prediction = self.model(features_tensor)
            
            # Denormalize prediction
            if self.target_scaler is not None:
                # Ensure prediction is numpy array and reshape for scaler
                prediction_np = prediction.cpu().numpy()
                v_linear = self.target_scaler.inverse_transform(prediction_np).flatten()
                
                # Model outputs only linear velocities (3 values), so pad with zeros for angular
                v = np.concatenate([v_linear, np.zeros(3)])  # [vx, vy, vz, 0, 0, 0]
            else:
                v_linear = prediction.cpu().numpy().flatten()
                v = np.concatenate([v_linear, np.zeros(3)])  # [vx, vy, vz, 0, 0, 0]
            
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