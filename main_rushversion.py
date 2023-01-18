# Created: 2023-1-15 12:14
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import open3d as o3d
import numpy as np
import cupy as cp
from collections import defaultdict

from utils import load_view_point, save_view_point, bresenham
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
        
    def all_process(self):
        ## 1. REMOVE GROUND!!!
        rmg_pts = self.points
        TOC("Remove Ground pts")

        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(rmg_pts[...,:2],self.resolution) + (self.range_m/self.resolution)/2).astype(int)

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

    def rayT_2d(self, center):
        ## 4. Ray Tracking in M_2d
        self.RayT_2d = np.zeros((self.dim_2d, self.dim_2d))
        for eid in self.pts1idxy:
            x1 = int(eid.split('.')[0])
            y1 = int(eid.split('.')[1])
            grid2one =bresenham(center[0],center[1], x1,y1)
            for sidxy in grid2one:
                self.RayT_2d[sidxy[0]][sidxy[1]] = 1
        TOC("Ray Casting")

## 5. Times Map and Query Matrix
Query_ = process_pts("data/bin/frame_10_debug.bin", range_m, resolution)
PriorMap_ = process_pts("data/bin/global_map_debug.bin", range_m, resolution)

## 6. The grid have one calculate the KL Diversity
print("All success")