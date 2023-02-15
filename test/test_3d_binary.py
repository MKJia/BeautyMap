# Created: 2023-2-2 11:20
# Update:  2023-2-14 23:21
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
# sys
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

RANGE = 30 # m, from cetner point to an square
RESOLUTION = 5 # m, resolution default 1m
offset = np.array([-154100.0, -6581400.0, 0])

TIC()
points_index2Remove = []
Mpts = Points("data/bin/KTH_1A2.bin", RANGE, RESOLUTION)
# hard code here! -37 is because the pts heigh is really high
# normally we may hold default one is enough
Mpts.GlobalMap_to_2d_binary(center=np.array([0,0,37]))
for i in range(1,2):
    Qpts = Points(f"data/bin/KTH_00{i}.bin", RANGE, RESOLUTION)
    
    df = pd.read_csv('data/kth_scan_poses.csv')
    center = np.array(df.values[i-1][1:4], dtype=float) + offset
    print("point center is ", center)
    Qpts.centerlise_all_pts(center)
    Qpts.from_center_to_2d_binary()
    Mpts.SelectMap_based_on_Query_center(Qpts.dim_2d, center)
    # bqc: based on Query center
    binary_xor = Qpts.exclusive_with_other_binary_2d(Mpts.bqc_binary_2d)
    trigger = (~Qpts.binary_2d) & binary_xor
    for (i,j) in tqdm(list(zip(*np.where(trigger != 0))), desc="grids traverse"):
        # maybe BUG HERE since if padding we need based on padding also?
        # check detail
        i_in_Map = i + Mpts.center_xy_id[0] - Qpts.dim_2d//2
        j_in_Map = j + Mpts.center_xy_id[1] - Qpts.dim_2d//2
        for k in Mpts.twoD2ptindex[i_in_Map][j_in_Map]:
            if_delete = trigger[i][j] & (1<<Mpts.idz[k] if not(Mpts.idz[k]>62 or Mpts.idz[k]<0) else 0)
            if if_delete!=0:
                points_index2Remove = points_index2Remove + [k]

print(f"There are {len(points_index2Remove)} pts to remove")
TOC("All processes")

inlier_cloud = Mpts.select_by_index(points_index2Remove)
oulier_cloud = Mpts.select_by_index(points_index2Remove, invert=True)
Mpts.view_compare(inlier_cloud, oulier_cloud)

fig, axs = plt.subplots(2, 2, figsize=(8,8))
axs[0,0].imshow(np.log(Mpts.binary_2d), cmap='hot', interpolation='nearest')
axs[0,0].set_title('Global Map bin')
axs[1,1].imshow(np.log(Qpts.binary_2d), cmap='hot', interpolation='nearest')
axs[1,1].set_title('Query 2d')
axs[0,1].imshow(np.log(Mpts.bqc_binary_2d), cmap='hot', interpolation='nearest')
axs[0,1].set_title('RoI Map in Quer range')
axs[1,0].imshow(np.log(trigger), cmap='hot', interpolation='nearest')
axs[1,0].set_title('After xor')
plt.show()

print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")