# Created: 2023-2-2 11:20
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# vis
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
import matplotlib.pyplot as plt

# self
from utils.global_def import *
from utils.pts_read import Points

RANGE = 50 # m, from cetner point to an square
RESOLUTION = 2 # m, resolution default 1m
RANGE_LEICA = 80
offset = np.array([(-154100.0, -6581400.0, 0)])

TIC()
points_index2Remove = []
Mpts = Points("data/bin/KTH_1A2.bin", RANGE, RESOLUTION)

for id_ in range(1,4):
    Mpts.clear_result(id_)
    Qpts = Points(f"data/bin/KTH_00{id_}.bin", RANGE, RESOLUTION)

    df = pd.read_csv('data/kth_scan_poses.csv')
    center = np.array(df.values[id_-1][1:4], dtype=float) + offset
    print("point center is ", center)

    Qpts.centerlise_all_pts(center)
    Qpts.from_center_to_2d_binary()
    Mpts.centerlise_all_pts(center)
    Mpts.from_center_to_2d_binary()

    # pre-process

    #RPG
    Qpts.RPGMat = Qpts.smoother()
    Qpts.RPGMask = Qpts.RPGMat > RESOLUTION**2 * 100

    #DEAR
    Qpts.RangeMask = Qpts.generate_range_mask(int(RANGE_LEICA/RESOLUTION))


    binary_xor = Qpts.exclusive_with_other_binary_2d(Mpts.binary_2d)
    trigger = (~Qpts.binary_2d) & binary_xor & Mpts.binary_2d

    trigger &= ~(Qpts.RPGMask - 1)
    trigger &= ~(Qpts.RangeMask - 1)
    # trigger = trigger & (trigger - 1)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    axs[0,0].imshow(np.log(Qpts.binary_2d), cmap='hot', interpolation='nearest')
    axs[0,0].set_title('Query 2d')
    axs[0,1].imshow(np.log(binary_xor), cmap='hot', interpolation='nearest')
    axs[0,1].set_title('Prior Map bin 2d')
    axs[1,0].imshow(Qpts.RPGMask, cmap='hot', interpolation='nearest')
    axs[1,0].set_title('After RPG')
    axs[1,1].imshow(Qpts.RangeMask, cmap='hot', interpolation='nearest')
    axs[1,1].set_title('After RPG Mask')
    plt.show()

    for (i,j) in list(zip(*np.where(trigger != 0))):
        max_obj_length = trigger[i][j] & (trigger[i][j]<<5)
        if max_obj_length != 0:
            trigger[i][j] = trigger[i][j] & (~max_obj_length)& (~max_obj_length>>1)& (~max_obj_length>>2)& (~max_obj_length>>3)& (~max_obj_length>>4)& (~max_obj_length>>5)
        for k in Mpts.twoD2ptindex[i][j]:
            if_delete = trigger[i][j] & (1<<Mpts.idz[k] if not(Mpts.idz[k]>62 or Mpts.idz[k]<0) else 0)
            if if_delete!=0:
                points_index2Remove = points_index2Remove + [k]

print(f"There are {len(points_index2Remove)} pts to remove")
TOC("All processes")

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