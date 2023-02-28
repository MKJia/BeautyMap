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
import operator

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
RANGE_16_RING = 8

st = time.time()
points_index2Remove = []

# 0. read Data =====>
Mpts = Points("data/bin/TPB_global_map.bin", RANGE, RESOLUTION)
df = pd.read_csv('data/TPB_poses_lidar2body.csv')
all_center_pose = np.array(df.values[:,2:], dtype=float)
mean_center = np.mean(all_center_pose[:,:3], axis=0)
# 0. read Data =====>

print(f"According to the pose file, the mean center is: {np.round(mean_center,2)}")
print("Please make sure it's correct one.")

Mpts.Generate_mat_tree(center=mean_center)

for id_ in range(96,101):

    pose = all_center_pose[id_]
    wxyz = np.array([pose[6],pose[3],pose[4],pose[5]])
    T_Q = np.eye(4)
    T_Q[:3,:3] = quat2mat(wxyz)
    T_Q[:3,-1]= np.array([pose[0],pose[1],pose[2]])

    Qpts = Points(f"data/bin/TPB/{id_:06d}.bin", RANGE, RESOLUTION)
    Mpts_ROI = Points(f"data/bin/TPB/{id_:06d}.bin", RANGE, RESOLUTION)
    Qpts_TMP = Points(f"data/bin/TPB/{id_:06d}.bin", RANGE, RESOLUTION)
    Qpts.transform_from_TF_Matrix(T_Q)
    center = Qpts.points[0,:]
    print("point center is ", center)

    [min_i_map, max_i_map], [min_j_map, max_j_map] = \
        Qpts.build_query_mat_tree_ref_select_roi(Mpts.range_m, Mpts.dim_2d,
        center=Mpts.global_center, pose_center=center)

    Mpts.SelectMap_based_on_Query_center(min_i_map, max_i_map, min_j_map, max_j_map)
    
    # pre-process

    # RPG
    Qpts.RPGMat = Qpts.smoother()
    Qpts.RPGMask = Qpts.RPGMat > RESOLUTION**2 * 100

    # DEAR
    Qpts.RangeMask = Qpts.generate_range_mask(int(RANGE_16_RING/RESOLUTION))
    # Qpts.SightMask = TODO: generate sight mask
    # Qpts.DEARMask = Qpts.RangeMask & Qpts.SightMask
    # LOGIC
    Mpts_ROI.threeD2ptindex = Mpts.threeD2ptindex
    Mpts_ROI.o3d_pts = Mpts.o3d_pts
    Mpts_ROI.LOGICMat = (0b11 * (Mpts.bqc_binary_2d & -Mpts.bqc_binary_2d)) 
    Mpts_ROI.LOGICPts = Mpts_ROI.select_lowest_pts(min_i_map, min_j_map, id_)
    print("pass")
    Qpts.LOGICMat = (0b111 * (Qpts.binary_2d & -Qpts.binary_2d)) >> 1
    Qpts.LOGICPts= Qpts.select_lowest_pts(min_i_map, min_j_map, id_)
    LOGIC_trigger = Mpts_ROI.calculate_ground_distribution(Qpts.LOGICPts, Qpts_TMP, min_i_map, max_i_map, min_j_map, max_j_map, Mpts.dim_2d, center)

    binary_xor = Qpts.exclusive_with_other_binary_2d(Mpts.bqc_binary_2d)
    trigger = (~Qpts.binary_2d) & binary_xor & Mpts.bqc_binary_2d

    # trigger = trigger & (trigger - 1) # for h_res = 0.5
    # trigger = trigger & (trigger - 1)
    trigger &= ~(Qpts.RPGMask - 1)
    trigger &= ~(Qpts.RangeMask - 1)
    trigger &= ~Mpts_ROI.LOGICMat 


    fig, axs = plt.subplots(3, 3, figsize=(8,8))
    axs[0,0].imshow(np.log(Qpts.binary_2d), cmap='hot', interpolation='nearest')
    axs[0,0].set_title('Query 2d')
    axs[0,1].imshow(np.log(Mpts.bqc_binary_2d), cmap='hot', interpolation='nearest')
    axs[0,1].set_title('Prior Map bin 2d')
    axs[1,0].imshow(Qpts.RPGMat, cmap='hot', interpolation='nearest')
    axs[1,0].set_title('After RPG')
    axs[1,1].imshow(Qpts.RPGMask, cmap='hot', interpolation='nearest')
    axs[1,1].set_title('After RPG Mask')
    axs[0,2].imshow(np.log(binary_xor), cmap='hot', interpolation='nearest')
    axs[0,2].set_title('binary_xor')
    axs[1,2].imshow(np.log(trigger), cmap='hot', interpolation='nearest')
    axs[1,2].set_title('trigger')
    axs[2,0].imshow(np.log(LOGIC_trigger), cmap='hot', interpolation='nearest')
    axs[2,2].set_title('LOGIC')
    axs[2,2].imshow(np.log(Mpts_ROI.LOGICMat), cmap='hot', interpolation='nearest')
    axs[2,2].set_title('Mpts_ROI.LOGICMat')
    axs[2,1].imshow(np.log(Qpts.LOGICMat), cmap='hot', interpolation='nearest')
    axs[2,1].set_title('Qpts.LOGICMat')
    plt.show()

    stt = time.time()
    for (i,j) in tqdm(list(zip(*np.where(trigger != 0))), desc=f"frame id {id_}: grids traverse"):
        max_obj_length = trigger[i][j] & (trigger[i][j]<<5)
        if max_obj_length != 0:
            trigger[i][j] = trigger[i][j] & (0b11111 * (~max_obj_length>>5))
        # for k in Mpts.twoD2ptindex[i+min_i_map][j+min_j_map]:
        #     if_delete = trigger[i][j] & (1<<Mpts.idz[k] if not(Mpts.idz[k]>62 or Mpts.idz[k]<0) else 0)
        #     if if_delete!=0:
        #         points_index2Remove = points_index2Remove + [k]
        all3d_indexs = Mpts.binTo3id(trigger[i][j])
        if (len(all3d_indexs)==0):
            continue
        elif(len(all3d_indexs)==1):
            points_index2Remove += Mpts.threeD2ptindex[i+min_i_map][j+min_j_map][all3d_indexs[0]]
        else:
            tupleOfTuples = operator.itemgetter(*all3d_indexs)(Mpts.threeD2ptindex[i+min_i_map][j+min_j_map])
            points_index2Remove += [element for tupl in tupleOfTuples for element in tupl]

    print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("grid index 2 pts", ((time.time() - stt))*1000))
print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("All processes", ((time.time() - st))*1000))

print(f"\nThere are {len(points_index2Remove)} pts to remove")
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