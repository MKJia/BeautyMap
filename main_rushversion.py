# Created: 2023-5-1 16:29
# Copyright (C) 2022-now, KTH-RPL, HKUST-RamLab
# Author:
# Mingkai Jia
# Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)

from tqdm import tqdm
import sys, os
BASE_DIR = os.path.abspath(os.path.dirname( __file__ ))

# vis
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
import matplotlib.pyplot as plt

# self
from utils.global_def import *
from utils.bee_tree import BEETree
from utils.pcdpy3 import save_pcd

import itertools

starttime = time.time()
t_list = []

RANGE = 40 # m, from cetner point to an square
RESOLUTION = 1 # m, resolution default 1m
H_RES = 0.5 # m, resolution default 1m

DATA_FOLDER = f"/home/mjiaab/workspace/edo_ws/edomap_release/edomap/data/KITTI/01"
MAX_RUN_FILE_NUM = -1 # -1 for all files

print(f"We will process the data in folder: {bc.BOLD}{DATA_FOLDER}{bc.ENDC}")
points_index2Remove = []

# read raw map or gt cloud
raw_map_path = f"{DATA_FOLDER}/raw_map.pcd"
if not os.path.exists(raw_map_path):
    raw_map_path = f"{DATA_FOLDER}/gt_cloud.pcd"

Mpts = BEETree()
Mpts.set_points_from_file(raw_map_path)
Mpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
Mpts.non_negatification_all_map_points()
Mpts.calculate_matrix_order()
t1 = time.time()
Mpts.generate_map_binary_tree()
print("Generate binary tree in Map cost: ",time.time() - t1, " s")
Mpts.get_binary_matrix()
Mpts.get_minz_matrix()

print("finished M, cost: ", time.time() - t1, " s")

all_pcd_files = sorted(os.listdir(f"{DATA_FOLDER}/pcd"))

for file_cnt, pcd_file in tqdm(enumerate(all_pcd_files)):
    if file_cnt>MAX_RUN_FILE_NUM and MAX_RUN_FILE_NUM!=-1:
        break
    t1 = time.time()
    Qpts = BEETree()
    Qpts.set_points_from_file(f"{DATA_FOLDER}/pcd/{pcd_file}")
    Qpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
    k = Qpts.transform_on_points(Mpts.coordinate_offset)  * RESOLUTION / H_RES

    Qpts.matrix_order = (int)(RANGE / RESOLUTION)
    Qpts.calculate_query_matrix_start_id()

    Qpts.generate_query_binary_tree(Mpts.minz_matrix)
    Qpts.get_binary_matrix()

    # pre-process
    map_binary_matrix_roi = Qpts.calculate_map_roi(Mpts.binary_matrix)
    minz_matrix_roi = Qpts.calculate_map_roi(Mpts.minz_matrix)
    outlier_matrix_roi = Qpts.calculate_map_roi(Mpts.outlier_matrix)

    binary_xor = (~Qpts.binary_matrix) & map_binary_matrix_roi
    trigger = binary_xor

    # BGM
    Qpts.BlindGridMask = Qpts.generate_blind_grid_mask()
    # SRM
    Qpts.SightRangeMask = Qpts.generate_sight_range_mask(k, minz_matrix_roi, outlier_matrix_roi)
    trigger &= ~Qpts.SightRangeMask
    trigger &= ~(Qpts.BlindGridMask - 1)
    trigger &= ~(map_binary_matrix_roi & -map_binary_matrix_roi) # Remove the lowest of the trigger, which is further calculated in LOGIC
    # RV ray casting
    blocked_mask = Qpts.reverse_virtual_ray_casting(trigger, minz_matrix_roi)
    trigger &= ~blocked_mask

    ground_index_matrix = np.log2((trigger & -trigger) >> 1).astype(int)
    ground_trigger = Mpts.calculate_ground_mask(Qpts, ground_index_matrix)
    for (i,j) in list(zip(*np.where(trigger != 0))):
        z = Mpts.binTo3id(trigger[i][j])
        for idz in z:
            points_index2Remove += (Mpts.root_matrix[i+Qpts.start_id_x][j+Qpts.start_id_y].children[idz].pts_id).tolist()
        gz = Mpts.binTo3id(ground_trigger[i][j])
        for idgz in gz:
            points_index2Remove += (Mpts.root_matrix[i+Qpts.start_id_x][j+Qpts.start_id_y].children[ground_index_matrix[i][j]].children[idgz].pts_id).tolist()

    t_tmp = time.time() - t1
    print("finished Q, cost: ", t_tmp, " s")
    t_list.append(t_tmp)

mean_t = sum(t_list) / len(t_list)
np_t_list = np.array(t_list)
dev_t = np.std(np_t_list)
print("frame time: ", mean_t , "range: ", dev_t)

print(f"There are {len(points_index2Remove)} pts to remove")

visited = set()
dup_index2Remove = list({x for x in points_index2Remove if x in visited or (visited.add(x) or False)})
print(f"There are {len(dup_index2Remove)} pts to remove now")
print(f" running time: {time.time() - starttime}")
points_index2Retain =np.setdiff1d(np.arange(len(Mpts.original_points)), points_index2Remove)
outlier_cloud = Mpts.original_points[points_index2Retain]
# inlier_cloud = Mpts.o3d_original_points.select_by_index(points_index2Remove)
# oulier_cloud = Mpts.o3d_original_points.select_by_index(points_index2Remove, invert=True)

save_pcd(f"{DATA_FOLDER}/edomap_output.pcd", outlier_cloud) # static map

# save_pcd(f"{DATA_FOLDER}/edomap_output_r.pcd", np.array(inlier_cloud.points))

print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")