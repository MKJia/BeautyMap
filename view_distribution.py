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


TIC()
points = np.fromfile("data/bin/test.bin", dtype=np.float32).reshape(-1, 4)
# points = np.fromfile("data/bin/KTH- Setup 001_SPATIAL_SUBSAMPLED.bin", dtype=np.float32).reshape(-1, 4)
# the map is originally referenced in the GPS global frame, I offset everything by (-154100.0, -6581400.0, 0.0)
# offset_lc = [-154100.0, -6581400.0, 0.0]
# leica offset on xyz in csv
np.insert(points, 0, np.array(0,0,0,-1))
TOC("Numpy read pt")

## 1. REMOVE GROUND!!!

rmg_pts = points
TOC("Remove Ground pts")

## 2. Voxelize STACK TO 2D
idxy = (np.divide(rmg_pts[...,:2],resolution) + (range_m/resolution)/2).astype(int)

dim_2d = (int)(range_m/resolution)
M_2d = np.zeros((dim_2d, dim_2d))

twoD2ptindex = defaultdict(lambda  : defaultdict(list))
pts1idxy = []
for i, ptidxy in enumerate(idxy):
    if ptidxy[0] < dim_2d and ptidxy[1]<dim_2d and ptidxy[1]>0 and ptidxy[0]>0:
        M_2d[ptidxy[0]][ptidxy[1]] = M_2d[ptidxy[0]][ptidxy[1]] + 1
        pt1xy = f"{ptidxy[0]}.{ptidxy[1]}"
        if pt1xy not in pts1idxy:
            pts1idxy.append(pt1xy)
        twoD2ptindex[ptidxy[0]][ptidxy[1]].append(i)
        # print(f"id: {i}, IN, xy {ptidxy[0]},{ptidxy[1]}")
TOC("Stack to 2d array")

## 4. Ray Tracking in M_2d
RayT_2d = np.zeros((dim_2d, dim_2d))
for eid in pts1idxy:
    x1 = int(eid.split('.')[0])
    y1 = int(eid.split('.')[1])
    grid2one =bresenham(dim_2d//2,dim_2d//2, x1,y1)
    # print(grid2one)
    for sidxy in grid2one:
        RayT_2d[sidxy[0]][sidxy[1]] = 1
TOC("Ray Casting")
# cpoints = cp.fromfile("data/bin/KTH- Setup 001_SPATIAL_SUBSAMPLED.bin", dtype=cp.float32).reshape(-1, 4)
# est_cen = cp.mean(cpoints[...], axis=0)
# print(f"{est_cen}")
# TOC(chat="Cupy read pt")

# pts = o3d.geometry.PointCloud()
# pts.points = o3d.utility.Vector3dVector(new_pts[:,:3])
# pts.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(new_pts.shape[0], 3)))
# # pts.estimate_normals()
# TOC(chat="o3d")

# voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, voxel_size=1.0)
# voxels_all= voxels.get_voxels()
# print(f"voxels' origin Coorinate of the origin point: {voxels.origin}, len of voxels: {len(voxels_all)}")
# # o3d.visualization.draw_geometries([voxels])
view_thing = [pts, voxels]
save_view_point(view_thing, "data/lecai_viewpoint.json") # make sure you are quit as click the `q` button
load_view_point(view_thing, "data/lecai_viewpoint.json")

print("All success")