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
# pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dztimer
import dztimer # only py38 can use now, working to publish on pypi

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

# RANGE = 10 # m, from cetner point to an square
# RESOLUTION = 0.5 # m, resolution default 1m
# H_RES = 0.2 # m, resolution default 1m

# DATA_FOLDER = f"/home/mjiaab/workspace/edo_ws/edomap_release/edomap/data/cones_two_people"

RANGE = 40 # m, from cetner point to an square
RESOLUTION = 1 # m, resolution default 1m
H_RES = 0.5 # m, resolution default 1m

DATA_FOLDER = f"/home/mjiaab/workspace/edo_ws/edomap_release/edomap/data/KITTI/05"
MAX_RUN_FILE_NUM = -1 # -1 for all files

timer = dztimer.Timing()
timer.start("Total")
print(f"We will process the data in folder: {bc.BOLD}{DATA_FOLDER}{bc.ENDC}")
points_index2Remove = []

# read raw map or gt cloud
raw_map_path = f"{DATA_FOLDER}/raw_map.pcd"
if not os.path.exists(raw_map_path):
    raw_map_path = f"{DATA_FOLDER}/gt_cloud.pcd"

timer[0].start("Map Generation")
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
timer[0].stop()
print("finished M, cost: ", time.time() - t1, " s")

global_trigger = np.zeros([Mpts.matrix_order, Mpts.matrix_order], dtype=int)
global_ground_trigger = np.zeros([Mpts.matrix_order, Mpts.matrix_order], dtype=int)

all_pcd_files = sorted(os.listdir(f"{DATA_FOLDER}/pcd"))

for file_cnt, pcd_file in tqdm(enumerate(all_pcd_files)):
    if file_cnt>MAX_RUN_FILE_NUM and MAX_RUN_FILE_NUM!=-1:
        break
    timer[4].start("One Scan Cost")
    timer[1].start("Query BeeTree")
    Qpts = BEETree()
    Qpts.set_points_from_file(f"{DATA_FOLDER}/pcd/{pcd_file}")
    Qpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
    k = Qpts.transform_on_points(Mpts.coordinate_offset)  * RESOLUTION / H_RES

    Qpts.matrix_order = (int)(RANGE / RESOLUTION)
    Qpts.calculate_query_matrix_start_id()
    

    # pre-process
    map_binary_matrix_roi = Qpts.calculate_map_roi(Mpts.binary_matrix)
    minz_matrix_roi = Qpts.calculate_map_roi(Mpts.minz_matrix)
    outlier_matrix_roi = Qpts.calculate_map_roi(Mpts.outlier_matrix)


    Qpts.generate_query_binary_tree(minz_matrix_roi)
    Qpts.get_binary_matrix()



    binary_xor = (~Qpts.binary_matrix) & map_binary_matrix_roi
    trigger = binary_xor
    timer[1].stop()

    timer[2].start("Static Restoration")
    # SRM
    Qpts.BlindGridMask = Qpts.generate_blind_grid_mask()
    Qpts.SightRangeMask = Qpts.generate_sight_range_mask(k, minz_matrix_roi, outlier_matrix_roi)
    trigger &= ~Qpts.SightRangeMask
    trigger &= ~(Qpts.BlindGridMask - 1)
    trigger &= ~(map_binary_matrix_roi & -map_binary_matrix_roi) # Remove the lowest of the trigger, which is further calculated in LOGIC
    # RV ray casting
    blocked_mask = Qpts.reverse_virtual_ray_casting(trigger, minz_matrix_roi)
    trigger &= ~blocked_mask
    timer[2].stop()

    timer[3].start("Fine Ground Seg")
    ground_index_matrix = np.log2((trigger & -trigger) >> 1).astype(int)
    ground_trigger = Mpts.calculate_ground_mask(Qpts, ground_index_matrix)

    timer[3].stop()

    global_trigger[Qpts.start_id_x:Qpts.start_id_x+Qpts.matrix_order, Qpts.start_id_y:Qpts.start_id_y+Qpts.matrix_order] = trigger
    global_ground_trigger[Qpts.start_id_x:Qpts.start_id_x+Qpts.matrix_order, Qpts.start_id_y:Qpts.start_id_y+Qpts.matrix_order] = ground_trigger

    timer[4].stop()

global_ground_index_matrix = np.log2((global_trigger & -global_trigger) >> 1).astype(int)
for (i,j) in list(zip(*np.where(global_trigger != 0))):
    z = Mpts.binTo3id(global_trigger[i][j])
    for idz in z:
        points_index2Remove += (Mpts.root_matrix[i][j].children[idz].pts_id).tolist()
    gz = Mpts.binTo3id(global_ground_trigger[i][j])
    for idgz in gz:
        points_index2Remove += (Mpts.root_matrix[i][j].children[global_ground_index_matrix[i][j]].children[idgz].pts_id).tolist()
# points_index2Remove = list(set(points_index2Remove))



print(f"There are {len(points_index2Remove)} pts to remove")

print(f" running time: {time.time() - starttime}")
points_index2Retain =np.setdiff1d(np.arange(len(Mpts.original_points)), points_index2Remove)
outlier_cloud = Mpts.original_points[points_index2Retain]

save_pcd(f"{DATA_FOLDER}/beautymap_output.pcd", outlier_cloud) # static map

timer.stop()
timer.print(title="BeautyMap",random_colors=True, bold=True)
print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")