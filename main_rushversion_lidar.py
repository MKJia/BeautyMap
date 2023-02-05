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
from utils import quat2mat

RANGE = 10 # m, from cetner point to an square
RESOLUTION = 0.5 # m, resolution default 1m

TIC()
points_index2Remove = []

Mpts = Points("data/bin/TPB_global_map.bin", RANGE, RESOLUTION)
df = pd.read_csv('data/TPB_poses_lidar2body.csv')

for id_ in range(90,93):
    Mpts.clear_result(id_)
    pose = df.values[id_][2:]
    wxyz = np.array([pose[6],pose[3],pose[4],pose[5]])
    T_Q = np.eye(4)
    T_Q[:3,:3] = quat2mat(wxyz)
    T_Q[:3,-1]= np.array([pose[0],pose[1],pose[2]])

    Qpts = Points(f"data/bin/TPB/{id_:06d}.bin", RANGE, RESOLUTION)
    Qpts.transform_from_TF_Matrix(T_Q)
    center = Qpts.points[0,:]
    print("point center is ", center)

    Qpts.centerlise_all_pts(center)
    Qpts.from_center_to_2d_binary()
    Mpts.centerlise_all_pts(center)
    Mpts.from_center_to_2d_binary()

    binary_xor = Qpts.exclusive_with_other_binary_2d(Mpts.binary_2d)
    trigger = (~Qpts.binary_2d) & binary_xor

    for (i,j) in list(zip(*np.where(trigger != 0))):
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