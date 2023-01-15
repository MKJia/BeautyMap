# Created: 2023-1-15 12:14
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import open3d as o3d
import numpy as np
import cupy as cp

from utils import load_view_point, save_view_point
from utils.global_def import *
resolution = 0.5
print("Testing IO for point cloud ...")
TIC()
points = np.fromfile("data/bin/test.bin", dtype=np.float32).reshape(-1, 4)
# points = np.fromfile("data/bin/KTH- Setup 001_SPATIAL_SUBSAMPLED.bin", dtype=np.float32).reshape(-1, 4)
est_cen = np.mean(points[...], axis=0)
# est_cen[2] = 0
# new_pts = points - est_cen
# est_cen = np.mean(new_pts[...], axis=0)
# est_cen[2] = 0
# print(f"{est_cen}")

TOC(chat="Numpy read pt")

idx = np.divide((points[...,:3] - est_cen[...,:3]),resolution).astype(int)
# TODO 保存成 xy 标准的matrix 255x255x5 (4: 2*2 gaussian, 1: pts id)， 这样方便直接对准
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
# view_thing = [pts, voxels]
# # save_view_point(view_thing, "data/lecai_viewpoint.json") # make sure you are quit as click the `q` button
# load_view_point(view_thing, "data/lecai_viewpoint.json")



print("All success")