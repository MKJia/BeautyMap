# Created: 2023-1-15 12:14
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import open3d as o3d
import numpy as np

# make sure relative package, check issue here: https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
# print(sys.path)

from utils import load_view_point, save_view_point
from utils.global_def import *


new_pts = np.fromfile(f"{BASE_DIR}/data/bin/TPB_global_map.bin", dtype=np.float32).reshape(-1, 4)
pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(new_pts[:,:3])
# pts.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(new_pts.shape[0], 3)))
# pts.estimate_normals()
TOC(chat="o3d")

voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, voxel_size=1.0)
voxels_all= voxels.get_voxels()
print(f"voxels' origin Coorinate of the origin point: {voxels.origin}, len of voxels: {len(voxels_all)}")
view_thing = [pts, voxels]
save_view_point(view_thing, f"{BASE_DIR}/data/o3d_view/TPB.json") # make sure you are quit as click the `q` button
# load_view_point(view_thing, f"{BASE_DIR}/data/o3d_view/lecai_viewpoint.json")

print("All codes run successfully, Close now..")