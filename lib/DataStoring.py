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
    current_point1_x: float = field(default=0.0)
    current_point1_y: float = field(default=0.0)
    current_point2_x: float = field(default=0.0) 
    current_point2_y: float = field(default=0.0)
    current_point3_x: float = field(default=0.0)
    current_point3_y: float = field(default=0.0)
    current_point4_x: float = field(default=0.0)
    current_point4_y: float = field(default=0.0)
    desired_point1_x: float = field(default=0.0)
    desired_point1_y: float = field(default=0.0)
    desired_point2_x: float = field(default=0.0)
    desired_point2_y: float = field(default=0.0)
    desired_point3_x: float = field(default=0.0)
    desired_point3_y: float = field(default=0.0)
    desired_point4_x: float = field(default=0.0)
    desired_point4_y: float = field(default=0.0)
    robot_pose_11: float = field(default=0.0)
    robot_pose_12: float = field(default=0.0)
    robot_pose_13: float = field(default=0.0)
    robot_pose_14: float = field(default=0.0)
    robot_pose_21: float = field(default=0.0)
    robot_pose_22: float = field(default=0.0)
    robot_pose_23: float = field(default=0.0)
    robot_pose_24: float = field(default=0.0)
    robot_pose_31: float = field(default=0.0)
    robot_pose_32: float = field(default=0.0)
    robot_pose_33: float = field(default=0.0)
    robot_pose_34: float = field(default=0.0)
    robot_pose_41: float = field(default=0.0)
    robot_pose_42: float = field(default=0.0)
    robot_pose_43: float = field(default=0.0)
    robot_pose_44: float = field(default=0.0)
    robot_velocity_vx: float = field(default=0.0)
    robot_velocity_vy: float = field(default=0.0)
    robot_velocity_vz: float = field(default=0.0)
    robot_velocity_wx: float = field(default=0.0)
    robot_velocity_wy: float = field(default=0.0)
    robot_velocity_wz: float = field(default=0.0)
    final_error: float = field(default=0.0)
    
    def set_sequence_id(self, sequence_id):
        self.sequence_id = sequence_id
    
    def set_timestep(self, timestep):
        self.timestep = timestep
    
    def set_current_points(self, current_points):
        self.current_point1_x = current_points[0, 0]
        self.current_point1_y = current_points[1, 0]
        self.current_point2_x = current_points[0, 1]
        self.current_point2_y = current_points[1, 1]
        self.current_point3_x = current_points[0, 2]
        self.current_point3_y = current_points[1, 2]
        self.current_point4_x = current_points[0, 3]
        self.current_point4_y = current_points[1, 3]
    
    def set_desired_points(self, desired_points):
        self.desired_point1_x = desired_points[0, 0]
        self.desired_point1_y = desired_points[1, 0]
        self.desired_point2_x = desired_points[0, 1]
        self.desired_point2_y = desired_points[1, 1]
        self.desired_point3_x = desired_points[0, 2]
        self.desired_point3_y = desired_points[1, 2]
        self.desired_point4_x = desired_points[0, 3]
        self.desired_point4_y = desired_points[1, 3]
        
    def set_robot_pose(self, robot_pose):
        self.robot_pose_11 = robot_pose[0, 0]
        self.robot_pose_12 = robot_pose[0, 1]
        self.robot_pose_13 = robot_pose[0, 2]
        self.robot_pose_14 = robot_pose[0, 3]
        self.robot_pose_21 = robot_pose[1, 0]
        self.robot_pose_22 = robot_pose[1, 1]
        self.robot_pose_23 = robot_pose[1, 2]
        self.robot_pose_24 = robot_pose[1, 3]
        self.robot_pose_31 = robot_pose[2, 0]
        self.robot_pose_32 = robot_pose[2, 1]
        self.robot_pose_33 = robot_pose[2, 2]
        self.robot_pose_34 = robot_pose[2, 3]
        self.robot_pose_41 = robot_pose[3, 0]
        self.robot_pose_42 = robot_pose[3, 1]
        self.robot_pose_43 = robot_pose[3, 2]
        self.robot_pose_44 = robot_pose[3, 3]
        
    def set_robot_velocity(self, robot_velocity):
        self.robot_velocity_vx = robot_velocity[0]
        self.robot_velocity_vy = robot_velocity[1]
        self.robot_velocity_vz = robot_velocity[2]
        self.robot_velocity_wx = robot_velocity[3]
        self.robot_velocity_wy = robot_velocity[4]
        self.robot_velocity_wz = robot_velocity[5]
        
    def set_final_error(self, final_error):
        self.final_error = final_error

    @classmethod
    def get_headers_row(cls):
        temp = cls()
        params = temp.__dict__.keys()
        return ", ".join(params) + "\n"
    
    def get_csv_row(self):
        temp = self.__dict__.values()
        return ", ".join(map(str, temp)) + "\n"
    
    @classmethod
    def fromIBVS(cls, ibvs: IBVS, iteration: int, step: int):
        obj = cls()
        obj.set_sequence_id(iteration)
        obj.set_timestep(step)
        try:
            projected_points = ibvs.camera.project_point(P=ibvs.P, pose=ibvs.camera.pose)
        except Exception as e:
            log.error(f"Error projecting points: {e}")
        obj.set_current_points(projected_points)
        obj.set_desired_points(ibvs.p_star.copy())  # Store the desired points at each timestep
        obj.set_robot_pose(ibvs.history[-1].pose.A.copy())
        obj.set_robot_velocity(ibvs.history[-1].vel.copy())
        obj.set_final_error(np.linalg.norm(ibvs.history[-1].e.copy()))
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
        with open(self.path + ".csv", "w") as f:
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
        with open(self.path + ".csv", "a") as f:
            for row in rows[self.lastStoredRowId:]:
                f.write(row.get_csv_row())
            self.lastStoredRowId = len(rows)
    