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

from lib.DataStoring import RowData

@dataclass
class IBVSHandler():
    central_camera: CentralCamera = field(default=CentralCamera.Default())
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    pd: np.ndarray = field(default_factory=lambda: np.array([]))
    min_distance: float = field(default=0.5)
    max_distance: float = field(default=1)
    max_iterations: int = field(default=300)
    lambda_value: float = field(default=0.1)
    verbose: bool = field(default=False)
        
    def generate_random_points(self, min_distance=0.3, max_distance=1) -> SE3:
        x = np.random.uniform(-max_distance, max_distance)
        y = np.random.uniform(-max_distance, max_distance)
        z = np.random.uniform(min_distance, max_distance)
        
        # Stack into a 3xN array
        self.P = mkgrid(2, side=0.5, pose=SE3.Trans(x, y, z))
        
    def generateCamera(self, min_distance=1.0, max_distance=2.0):
        x = np.random.uniform(-max_distance, max_distance)
        y = np.random.uniform(-max_distance, max_distance)
        z = np.random.uniform(-max_distance, -min_distance)
        
        
        self.central_camera = CentralCamera.Default(pose=SE3.Trans(x, y, z))
        
    def __post_init__(self):
        self.generateCamera()
        self.generate_random_points()
        self.pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[self.central_camera.pp]
        
    def randomizePositions(self):
        self.generateCamera()
        self.generate_random_points()
        
    def runIBVS(self, iteration: int):
        ibvs = IBVS(self.central_camera, P=self.P, p_d=self.pd, graphics=False)  # Disable graphics
        listData: list[RowData] = []
        
        for i in range(self.max_iterations):
            ibvs.step(i)
            listData.append(RowData.fromIBVS(ibvs, iteration=iteration, step=i))
            
            error_norm = np.linalg.norm(ibvs.history[-1].e)
            if error_norm < 1e-3:
                break
        return listData
            
            
        
    
        
 
        
        