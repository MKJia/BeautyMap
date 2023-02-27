# Created: 2023-2-2 11:20
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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

st = time.time()
points_index2Remove = []

# 0. read Data =====>
Mpts = Points("data/bin/KTH_1A2.bin", RANGE, RESOLUTION)
df = pd.read_csv('data/KTH_poses_sensor.csv')
all_center_pose = np.array(df.values[:,2:], dtype=float)
mean_center = np.mean(all_center_pose[:,:3], axis=0)

# 0. read Data =====>
print(f"According to the pose file, the mean center is: {np.round(mean_center,2)}")
print("Please make sure it's correct one.")

Mpts.GlobalMap_to_2d_binary(center=mean_center)

for id_ in range(1,3):
    # should be the same resolution with Map
    Qpts = Points(f"data/bin/KTH_00{id_}.bin", RANGE, Mpts.resolution)

    # select the small range
    center = all_center_pose[id_]

    [min_i_map, max_i_map], [min_j_map, max_j_map] = \
        Qpts.build_2d_binary_M_ref_select_roi(Mpts.range_m, Mpts.dim_2d,
        center=Mpts.global_center, pose_center=center)

    Mpts.SelectMap_based_on_Query_center(min_i_map, max_i_map, min_j_map, max_j_map)

    # pre-process

    # RPG
    Qpts.RPGMat = Qpts.smoother()
    Qpts.RPGMask = Qpts.RPGMat > RESOLUTION**2 * 100

    # DEAR
    Qpts.RangeMask = Qpts.generate_range_mask(int(RANGE_LEICA/RESOLUTION))

    binary_xor = Qpts.exclusive_with_other_binary_2d(Mpts.bqc_binary_2d)
    trigger = (~Qpts.binary_2d) & binary_xor

    trigger &= ~(Qpts.RPGMask - 1)
    trigger &= ~(Qpts.RangeMask - 1)
    # trigger = trigger & (trigger - 1)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    axs[0,0].imshow(np.log(Qpts.binary_2d), cmap='hot', interpolation='nearest')
    axs[0,0].set_title('Query 2d')
    axs[0,1].imshow(np.log(binary_xor), cmap='hot', interpolation='nearest')
    axs[0,1].set_title('Prior Map bin 2d')
    axs[1,0].imshow(np.log(Mpts.bqc_binary_2d), cmap='hot', interpolation='nearest')
    axs[1,0].set_title('Select Map')
    axs[1,1].imshow(np.log(trigger), cmap='hot', interpolation='nearest')
    axs[1,1].set_title('Trigger Map')
    plt.show()

    stt = time.time()
    for (i,j) in tqdm(list(zip(*np.where(trigger != 0))), desc=f"frame id {id_}: grids traverse"):
        max_obj_length = trigger[i][j] & (trigger[i][j]<<5)
        if max_obj_length != 0:
            trigger[i][j] = trigger[i][j] & (~max_obj_length)& (~max_obj_length>>1)& (~max_obj_length>>2)& (~max_obj_length>>3)& (~max_obj_length>>4)& (~max_obj_length>>5)
        for k in Mpts.twoD2ptindex[i+min_i_map][j+min_j_map]:
            if_delete = trigger[i][j] & (1<<Mpts.idz[k] if not(Mpts.idz[k]>62 or Mpts.idz[k]<0) else 0)
            if if_delete!=0:
                points_index2Remove = points_index2Remove + [k]
    print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("grid index 2 pts", ((time.time() - stt))*1000))
print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("All processes", ((time.time() - st))*1000))

print(f"There are {len(points_index2Remove)} pts to remove")
inlier_cloud = Mpts.select_by_index(points_index2Remove)
oulier_cloud = Mpts.select_by_index(points_index2Remove, invert=True)
Mpts.view_compare(inlier_cloud, oulier_cloud)

print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")