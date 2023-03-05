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
from collections import defaultdict

# vis
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
import matplotlib.pyplot as plt

# self
from utils.global_def import *
from utils.pts_read import Points
from utils import quat2mat

RANGE = 50 # m, from cetner point to an square
RESOLUTION = 2.0 # m, resolution default 1m
RANGE_LEICA = 80
DEBUG_PLOT = True # only run one frame and plot hot map

st = time.time()
points_index2Remove = []
points_index2Remove_in_ROI = []

# 0. read Data =====>
Mpts = Points("data/bin/KTH_1A2.bin", RANGE, RESOLUTION)
df = pd.read_csv('data/KTH_poses_sensor.csv')
all_center_pose = np.array(df.values[:,2:5], dtype=float)
mean_center = np.mean(all_center_pose[:,:3], axis=0)

# 0. read Data =====>
print(f"According to the pose file, the mean center is: {np.round(mean_center,2)}")
print("Please make sure it's correct one.")

Mpts.Generate_mat_tree(center=mean_center)

for id_ in range(1,3):
    # should be the same resolution with Map
    Qpts = Points(f"data/bin/KTH_00{id_}.bin", RANGE, Mpts.resolution)
    Mpts_ROI = Points(f"data/bin/KTH_1A2.bin", RANGE, Mpts.resolution)
    Qpts_TMP = Points(f"data/bin/KTH_00{id_}.bin", RANGE, Mpts.resolution)

    # select the small range
    center = all_center_pose[id_]

    [min_i_map, max_i_map], [min_j_map, max_j_map] = \
        Qpts.build_query_mat_tree_ref_select_roi(Mpts.dim_2d,
        center=Mpts.global_center, pose_center=center)

    Mpts.SelectMap_based_on_Query_center(min_i_map, max_i_map, min_j_map, max_j_map)

    # pre-process

    # RPG
    Qpts.RPGMat = Qpts.smoother()
    Qpts.RPGMask = Qpts.RPGMat > RESOLUTION**2 * 100

    # DEAR
    Qpts.RangeMask = Qpts.generate_range_mask(int(RANGE_LEICA/RESOLUTION))

    # 3. LOGIC [Lowest occupied-Grid Index Clear]  ======>
    Mpts_ROI.threeD2ptindex = Mpts.threeD2ptindex
    Mpts_ROI.o3d_pts = Mpts.o3d_pts

    # Explain: bin & -bin -> extract the lowest 1 [-bin = ~abs(bin) + 1 ]
    # TODO: not only 0b11 but based on resolution -> 2**n-1 感觉得由分辨率决定调用最下面 n 层点云
    Mpts_ROI.LOGICMat = (0b11 * (Mpts.bqc_binary_2d & -Mpts.bqc_binary_2d)) 
    Mpts_ROI.select_lowest_pts(min_i_map, min_j_map, id_, Mpts.global_center)

    # QUETION: Why is not 0b11?? but 0b111 >> 1
    Qpts.LOGICMat = (0b11 * (Qpts.binary_2d & -Qpts.binary_2d)) #>> 1
    Qpts.select_lowest_pts(min_i_map, min_j_map, id_, Mpts.global_center)

    Mpts_ROI.threeD2ptindex = defaultdict(lambda  : defaultdict(lambda : defaultdict(list)))

    LOGIC_trigger = \
    Mpts_ROI.calculate_ground_distribution(Qpts.LOGICPts, 
                                        Qpts_TMP, min_i_map, 
                                        max_i_map, min_j_map, 
                                        max_j_map, Mpts.dim_2d,
                                        pose_center=center)

    binary_xor = Qpts.exclusive_with_other_binary_2d(Mpts.bqc_binary_2d)
    trigger = (~Qpts.binary_2d) & binary_xor & Mpts.bqc_binary_2d


    trigger &= ~(Qpts.RPGMask - 1)
    trigger &= ~(Qpts.RangeMask - 1)
    trigger &= ~Mpts_ROI.LOGICMat
    # trigger = trigger & (trigger - 1)
    LOGIC_trigger &= ~(Qpts.RPGMask - 1)
    LOGIC_trigger &= ~(Qpts.RangeMask - 1)

    stt = time.time()
    for (i,j) in list(zip(*np.where(trigger != 0))):
        max_obj_length = trigger[i][j] & (trigger[i][j]<<5)
        if max_obj_length != 0:
            trigger[i][j] = trigger[i][j] & (0b11111 * (~max_obj_length>>5))
        all3d_indexs = Mpts.binTo3id(trigger[i][j])
        points_index2Remove += SELECT_Ptsindex_from_matrix(all3d_indexs, Mpts.threeD2ptindex, i,j,min_i_map, min_j_map)

    for (i,j) in list(zip(*np.where(LOGIC_trigger!= 0))):
        all3d_indexs = Mpts_ROI.binTo3id(LOGIC_trigger[i][j])
        points_index2Remove_in_ROI += SELECT_Ptsindex_from_matrix(all3d_indexs, Mpts_ROI.threeD2ptindex, i,j,min_i_map, min_j_map)
        
    if DEBUG_PLOT:
        np.seterr(divide = 'ignore') 
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
        axs[2,0].set_title('LOGIC trigger')
        axs[2,2].imshow(np.log(Mpts_ROI.LOGICMat), cmap='hot', interpolation='nearest')
        axs[2,2].set_title('Mpts_ROI.LOGICMat')
        axs[2,1].imshow(np.log(Qpts.LOGICMat), cmap='hot', interpolation='nearest')
        axs[2,1].set_title('Qpts.LOGICMat')
        plt.show()
        break
    
    print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("grid index 2 pts", ((time.time() - stt))*1000))
print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("All processes", ((time.time() - st))*1000))

print(f"There are {len(points_index2Remove)} pts to remove")
inlier_cloud = Mpts.select_by_index(points_index2Remove)
oulier_cloud = Mpts.select_by_index(points_index2Remove, invert=True)
Mpts.view_compare(inlier_cloud, oulier_cloud, others = Mpts_ROI.LOGIC_PCD.select_by_index(points_index2Remove_in_ROI),view_file="data/o3d_view/TPB.json")

print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")