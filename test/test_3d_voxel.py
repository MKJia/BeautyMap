# Created: 2023-2-2 11:20
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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

RANGE = 50 # m, from cetner point to an square
RESOLUTION = 1 # m, resolution default 1m
TIC()
points_index2Remove = []
Qpts = Points("data/bin/KTH_001.bin", RANGE, RESOLUTION)
Mpts = Points("data/bin/KTH_1A2.bin", RANGE, RESOLUTION)
offset = np.array([(-154100.0, -6581400.0, 0.0)])
df = pd.read_csv('data/kth_scan_poses.csv')
center = np.array(df.values[1][1:4], dtype=float) + offset
print("point center is ", center)
Qpts.centerlise_all_pts(center)
Qpts.from_center_to_2d_grid()
Qpts.gmm_fit()
Mpts.centerlise_all_pts(center)
Mpts.from_center_to_2d_grid()

KL_Matrix = np.zeros((Qpts.dim_2d, Qpts.dim_2d))
Grids_Trigger = list(zip(*np.where((Qpts.bin_2d * Mpts.M_2d) == 1)))
for (i ,j) in Grids_Trigger:
    # KL_Matrix[i][j] = gmm_kl(Qpts.gmmFitMatrix[i][j], Mpts.gmmFitMatrix[i][j])
    proba = Qpts.gmmFitMatrix[i][j].predict_proba(Mpts.select_data_from_2DptIndex(i,j)[:,2].reshape(-1,1))
    Remove_ = (proba[:,0] < 0.68) * (proba[:,1]<0.68)
    indexOn2D = list(*np.where((Remove_ == True)))
    points_index2Remove = points_index2Remove + list(np.array(Mpts.twoD2ptindex[i][j])[indexOn2D])
        
inlier_cloud = Mpts.select_by_index(points_index2Remove)
oulier_cloud = Mpts.select_by_index(points_index2Remove, invert=True)
Mpts.view_compare(inlier_cloud, oulier_cloud)
TOC("All processes")
# # pts = Points("data/bin/TPB_000100.bin", RANGE, RESOLUTION)
# # pts.centerlise_all_pts()
# # pts.from_center_to_2d_grid()

fig, axs = plt.subplots(2, 2, figsize=(8,8))
axs[0,0].imshow(Qpts.M_2d, cmap='hot', interpolation='nearest')
axs[0,0].set_title('Query 2d')
axs[0,1].imshow(Mpts.M_2d, cmap='hot', interpolation='nearest')
axs[0,1].set_title('Prior Map bin 2d')
# axs[1,0].imshow(Query_.binT_2d, cmap='hot', interpolation='nearest')
# axs[1,0].set_title('Query bin map')
axs[1,1].imshow(np.clip(KL_Matrix, 0, 10), cmap='hot', interpolation='nearest')
axs[1,1].set_title('KL Matrix after normalization')
plt.show()
print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")