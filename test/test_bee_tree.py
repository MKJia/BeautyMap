# Created: 2023-2-2 11:20
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

# vis
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
import matplotlib.pyplot as plt

# self
from utils.global_def import *
from utils.pts_read import Points
from utils import quat2mat
from utils.bee_tree import BEETree


starttime = time.time()

RANGE = 10 # m, from cetner point to an square
RESOLUTION = 1.0 # m, resolution default 1m
H_RES = 0.2 # m, resolution default 1m
RANGE_16_RING = 8

TIC()

points_index2Remove = []

Mpts = BEETree()
Mpts.set_points_from_file("data/bin/TPB_global_map.bin")
Mpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
Mpts.set_global_center_from_file('data/TPB_poses_lidar2body.csv')
Mpts.non_negatification_all_map_points()
Mpts.calculate_matrix_order()
t1 = time.time()
Mpts.generate_binary_tree()
print(time.time() - t1)
# Mpts.set_pts_morton_id()
print("finished")

for id_ in range(90,93):

    Qpts = BEETree()
    Qpts.set_points_from_file(f"data/bin/TPB/{id_:06d}.bin")
    Qpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
    Qpts.transform_to_map_frame(Mpts.poses[id_], Mpts.coordinate_offset)

    Qpts.matrix_order = (int)(RANGE / RESOLUTION)
    Qpts.calculate_query_matrix_start_id()

    t1 = time.time()
    Qpts.generate_binary_tree()
    print(time.time() - t1)
    print("finished")

    # pre-process

    # RPG
    Qpts.RPGMat = Qpts.smoother()
    Qpts.RPGMask = Qpts.RPGMat > RESOLUTION**2 * 100

    # DEAR
    Qpts.RangeMask = Qpts.generate_range_mask(int(RANGE_16_RING/RESOLUTION))
    # Qpts.SightMask = TODO: generate sight mask
    # Qpts.DEARMask = Qpts.RangeMask & Qpts.SightMask

    binary_xor = Qpts.exclusive_with_other_binary_2d([i.binary_data] for i in Mpts.root_matrix)
    trigger = (~[i.binary_data] for i in Qpts.root_matrix) & binary_xor

    trigger = trigger & (trigger - 1) # for h_res = 0.5
    # trigger = trigger & (trigger - 1)
    trigger &= ~(Qpts.RPGMask - 1)
    trigger &= ~(Qpts.RangeMask - 1)
    # print(Qpts.binary_2d)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    # axs[0,0].imshow(np.log(Qpts.binary_2d), cmap='hot', interpolation='nearest')
    # axs[0,0].set_title('Query 2d')
    axs[0,1].imshow(np.log(binary_xor), cmap='hot', interpolation='nearest')
    axs[0,1].set_title('Prior Map bin 2d')
    axs[1,0].imshow(Qpts.RPGMask, cmap='hot', interpolation='nearest')
    axs[1,0].set_title('After RPG')
    axs[1,1].imshow(trigger, cmap='hot', interpolation='nearest')
    axs[1,1].set_title('After RPG Mask')
    plt.show()


    for (i,j) in list(zip(*np.where(trigger != 0))):
        for k in Mpts.twoD2ptindex[i][j]:
            if_delete = trigger[i][j] & (1<<Mpts.idz_c[k] if not(Mpts.idz_c[k]>62 or Mpts.idz_c[k]<0) else 0)
            if if_delete!=0:
                points_index2Remove = points_index2Remove + [k]

print(f"There are {len(points_index2Remove)} pts to remove")
TOC("All processes")

endtime = time.time()
print (endtime - starttime)

inlier_cloud = Mpts.select_by_index(points_index2Remove)
oulier_cloud = Mpts.select_by_index(points_index2Remove, invert=True)
Mpts.view_compare(inlier_cloud, oulier_cloud)

# fig, axs = plt.subplots(2, 2, figsize=(8,8))
# axs[0,0].imshow(Qpts.binary_2d, cmap='hot', interpolation='nearest')
# axs[0,0].set_title('Query 2d')
# axs[0,1].imshow(Mpts.binary_2d, cmap='hot', interpolation='nearest')
# axs[0,1].set_title('Prior Map bin 2d')
# axs[1,0].imshow(trigger, cmap='hot', interpolation='nearest')
# axs[1,0].set_title('After xor')
# plt.show()

print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")