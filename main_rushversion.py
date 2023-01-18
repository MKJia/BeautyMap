# Created: 2023-1-15 12:14
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import open3d as o3d
import numpy as np
import cupy as cp
from collections import defaultdict

from utils import load_view_point, save_view_point, bresenham, quat2mat
from utils.global_def import *

resolution = 1
range_m = 5


class process_pts:
    def __init__(self, file, range_m, resolution, T_MATRIX=np.eye(4)):
        self.MAP_FLAG = False
        ## 0. Read Point Cloud
        TIC()
        self.points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        
        # point the center points
        np.insert(self.points, 0, np.array([0,0,0,-1]),axis=0)

        TOC("Numpy read pt")

        self.range_m = range_m
        self.resolution = resolution
        self.dim_2d = (int)(range_m/resolution)
        self.twoD2ptindex = defaultdict(lambda  : defaultdict(list))
    
    def remove_ground(self):
        ## 1. REMOVE GROUND!!!
        self.rmg_pts = self.points
        # TODO
        TOC("Remove Ground pts")

    def all_process(self, center):

        self.remove_ground()
        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.rmg_pts[...,:2] - center,self.resolution) + (self.range_m/self.resolution)/2).astype(int)

        M_2d = np.zeros((self.dim_2d, self.dim_2d))

        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if ptidxy[0] < self.dim_2d and ptidxy[1]<self.dim_2d and ptidxy[1]>0 and ptidxy[0]>0:
                M_2d[ptidxy[0]][ptidxy[1]] = M_2d[ptidxy[0]][ptidxy[1]] + 1
                pt1xy = f"{ptidxy[0]}.{ptidxy[1]}"
                if pt1xy not in self.pts1idxy:
                    self.pts1idxy.append(pt1xy)
                self.twoD2ptindex[ptidxy[0]][ptidxy[1]].append(i)
        TOC("Stack to 2d array")

    def rayT_2d(self):
        ## 4. Ray Tracking in M_2d
        self.RayT_2d = np.zeros((self.dim_2d, self.dim_2d))
        for eid in self.pts1idxy:
            x1 = int(eid.split('.')[0])
            y1 = int(eid.split('.')[1])
            grid2one =bresenham(self.dim_2d//2,self.dim_2d//2, x1,y1)
            for sidxy in grid2one:
                self.RayT_2d[sidxy[0]][sidxy[1]] = 1
        TOC("Ray Casting")
        return self.RayT_2d

def calGMM(data):
    # TODO
    return None

def calKL(P1, P2):
    # TODO
    return None

## 5. Times Map and Query Matrix
import pandas as pd
df = pd.read_csv('data/TPB_poses_lidar2body.csv')
pose = df.values[100][2:]
xyz = np.array([pose[0],pose[1],pose[2]])
wxyz = np.array([pose[-1],pose[3],pose[5],pose[6]])
T_Q = np.eye(4)
T_Q[:3,:3] = quat2mat(wxyz)
T_Q[:4,-1] = xyz

Query_ = process_pts("data/bin/TPB_000100.bin", range_m, resolution, T_MATRIX=T_Q)
PrMap_ = process_pts("data/bin/TPB_global_map.bin", range_m, resolution)

Query_.all_process(Query_.points[:,0])
PrMap_.all_process(Query_.points[:,0])

Query2d = Query_.rayT_2d()
PrMap2d = PrMap_.rayT_2d()

KL_Matrix = M_2d = np.zeros((Query_.dim_2d, Query_.dim_2d))
## 6. The grid have one calculate the KL Diversity
Grids_Trigger_KL = list(zip(*np.where(Query2d.dot(PrMap2d) == 1)))
for item in Grids_Trigger_KL:
    Q_pts = Query2d.twoD2ptindex[item[0]][item[1]]
    M_pts = PrMap2d.twoD2ptindex[item[0]][item[1]]
    Q_Prob = calGMM(Q_pts) # Q_Prob: u1, u2, s1, s2 which is 2 gaussian
    M_Prob = calGMM(M_pts)
    KL_Matrix[item[0]][item[1]] = calKL(Q_Prob, M_Prob)
    
# TODO: view the Matrix with the pts also?
## 7. Difference Show to view the distribution with heatmap

## 8. remove the xy bins output the Global Map
# How to set the threshold??

print("All success")