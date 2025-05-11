import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

from dataclasses import dataclass, field
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

import logging as log

@dataclass
class RowData():
    sequence_id: int = field(default=0)
    timestep: int = field(default=0)
    
    # Current feature points (flattened 2xN array)
    current_features: np.ndarray = field(default_factory=lambda: np.zeros(8))  # For 4 points (x,y)
    
    # Desired feature points (flattened 2xN array)
    desired_features: np.ndarray = field(default_factory=lambda: np.zeros(8))  # For 4 points (x,y)
    
    # Output velocity command
    velocity_command: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [vx, vy, vz, wx, wy, wz]
    
    # Error metrics
    feature_error: float = field(default=0.0)  # Norm of feature error
    
    def set_sequence_id(self, sequence_id):
        self.sequence_id = sequence_id
    
    def set_timestep(self, timestep):
        self.timestep = timestep
    
    def set_current_features(self, current_points):
        """Set current feature points from 2xN array"""
        self.current_features = current_points.flatten(order='F')
    
    def set_desired_features(self, desired_points):
        """Set desired feature points from 2xN array"""
        self.desired_features = desired_points.flatten(order='F')
    
    def set_velocity_command(self, velocity):
        """Set the 6-DOF velocity command"""
        self.velocity_command = np.array(velocity)
    
    def set_feature_error(self, error):
        """Set the feature error norm"""
        self.feature_error = error

    @classmethod
    def get_headers_row(cls):
        base_headers = ["sequence_id", "timestep"]
        
        # Add current feature headers
        current_features_headers = [f"current_feature_{i}" for i in range(8)]
        
        # Add desired feature headers
        desired_features_headers = [f"desired_feature_{i}" for i in range(8)]
        
        # Add velocity headers
        velocity_headers = ["vx", "vy", "vz", "wx", "wy", "wz"]
        
        # Add error header
        error_header = ["feature_error"]
        
        # Combine all headers
        all_headers = (base_headers + current_features_headers + 
                      desired_features_headers + velocity_headers + 
                      error_header)
        
        return ",".join(all_headers) + "\n"
    
    def get_csv_row(self):
        values = [
            self.sequence_id,
            self.timestep,
            *self.current_features,
            *self.desired_features,
            *self.velocity_command,
            self.feature_error
        ]
        return ",".join(map(str, values)) + "\n"
    
    @classmethod
    def fromIBVS(cls, ibvs: IBVS, iteration: int, step: int):
        obj = cls()
        obj.set_sequence_id(iteration)
        obj.set_timestep(step)
        
        try:
            # Get current feature points
            current_points = ibvs.camera.project_point(P=ibvs.P, pose=ibvs.camera.pose)
            obj.set_current_features(current_points)
            
            # Get desired feature points
            obj.set_desired_features(ibvs.p_star)
            
            # Get velocity command from history
            if len(ibvs.history) > 0:
                obj.set_velocity_command(ibvs.history[-1].vel)
            
            # Calculate and store feature error
            error = np.linalg.norm(current_points.flatten(order='F') - ibvs.p_star.flatten(order='F'))
            obj.set_feature_error(error)
            
        except Exception as e:
            log.error(f"Error creating RowData from IBVS: {e}")
            raise
        
        return obj
    
if __name__ == "__main__":
    rowData = RowData()
    print(RowData.get_headers_row())
    print(rowData.get_csv_row())
    
@dataclass
class DataStore():
    path: str = field(default="")
    lastStoredRowId: int = field(default=0)
    
    def start_file(self):
        log.info(f"Starting file {self.path}")
        self.path = os.path.join(self.path, f"data_{self.lastStoredRowId}.csv")
        with open(self.path, "w") as f:
            f.write(RowData.get_headers_row())
    
    def __post_init__(self):
        if self.path == "":
            log.error("Path is not set")
            raise ValueError("Path is not set")
        
        os.makedirs(self.path, exist_ok=True)
        log.info(f"Starting file {self.path}")
        self.start_file()
        log.info(f"File {self.path} initialized")
        
    def append_rows(self, rows: list[RowData]):
        log.info(f"Appending {len(rows)} rows to {self.path}")
        with open(self.path, "a") as f:
            for row in rows[self.lastStoredRowId:]:
                f.write(row.get_csv_row())
            self.lastStoredRowId = len(rows)
    