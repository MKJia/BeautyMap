# Created: 2023-5-1 16:29
# Copyright (C) 2022-now, KTH-RPL, HKUST-RamLab
# Author:
# Mingkai Jia  (https://mkjia.github.io/)
# Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)

from tqdm import tqdm
import os
import dztimer

# self
from utils.global_def import *
from lib.bee_tree import BEETree
from utils.pcdpy3 import save_pcd

starttime = time.time()
t_list = []

'''
For semi-indoor dataset
'''
# RANGE = 10 # m, from center point to an square
# RESOLUTION = 0.5 # m, resolution default 1m
# H_RES = 0.2 # m, resolution default 1m
# DATA_FOLDER = f"/path/to/your/data/cones_two_people"

'''
For KITTI dataset
'''
RANGE = 40 # m, from center point to an square
RESOLUTION = 1 # m, resolution default 1m
H_RES = 0.5 # m, resolution default 1m
# DATA_FOLDER = f"/path/to/your/data/KITTI/xx"
DATA_FOLDER = f"/home/mjiaab/workspace/edo_ws/edomap_release/edomap/data/KITTI/01"

MAX_RUN_FILE_NUM = -1 # -1 for all files

timer = dztimer.Timing()
timer.start("Total")
print(f"We will process the data in folder: {bc.BOLD}{DATA_FOLDER}{bc.ENDC}")


# read raw map or gt cloud
raw_map_path = f"{DATA_FOLDER}/raw_map.pcd"
if not os.path.exists(raw_map_path):
    raw_map_path = f"{DATA_FOLDER}/gt_cloud.pcd"

# map generation
timer[0].start("Map Generation")
Mpts = BEETree()
Mpts.set_points_from_file(raw_map_path)
Mpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
Mpts.non_negatification_all_map_points()
Mpts.calculate_matrix_order()
Mpts.generate_map_binary_tree()
Mpts.get_binary_matrix()
Mpts.get_minz_matrix()
timer[0].stop()

# initialize global triggers
global_trigger = np.zeros([Mpts.matrix_order, Mpts.matrix_order], dtype=int)
global_ground_trigger = np.zeros([Mpts.max_height, Mpts.matrix_order, Mpts.matrix_order], dtype=int)

# process
all_pcd_files = sorted(os.listdir(f"{DATA_FOLDER}/pcd"))
for file_cnt, pcd_file in tqdm(enumerate(all_pcd_files), total=len(all_pcd_files), ncols=100, desc="Processing"):
    if file_cnt>MAX_RUN_FILE_NUM and MAX_RUN_FILE_NUM!=-1:
        break
    timer[5].start("One Scan Cost")

    # query init
    timer[1].start("Query Generation")
    Qpts = BEETree()
    Qpts.matrix_order = (int)(RANGE / RESOLUTION)
    Qpts.set_points_from_file(f"{DATA_FOLDER}/pcd/{pcd_file}")
    Qpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
    k = Qpts.transform_on_points(Mpts.coordinate_offset)  * RESOLUTION / H_RES
    Qpts.calculate_query_matrix_start_id()

    # get roi matrix
    map_binary_matrix_roi = Qpts.calculate_map_roi(Mpts.binary_matrix)
    minz_matrix_roi = Qpts.calculate_map_roi(Mpts.minz_matrix)
    outlier_matrix_roi = Qpts.calculate_map_roi(Mpts.outlier_matrix)

    # query generation
    Qpts.generate_query_binary_tree(minz_matrix_roi)
    Qpts.get_binary_matrix()
    timer[1].stop()

    # matrix comparison
    timer[2].start("Matrix Comparison")
    binary_xor = (~Qpts.binary_matrix) & map_binary_matrix_roi
    trigger = binary_xor
    timer[2].stop()

    # static restoration
    timer[3].start("Static Restoration")
    # out-of-sight protection
    Qpts.BlindGridMask = Qpts.generate_blind_grid_mask()
    Qpts.SightRangeMask = Qpts.generate_sight_range_mask(k, minz_matrix_roi, outlier_matrix_roi)
    trigger &= ~Qpts.SightRangeMask
    trigger &= ~(Qpts.BlindGridMask - 1)
    trigger &= ~(map_binary_matrix_roi & -map_binary_matrix_roi) # Remove the lowest of the trigger, which is further calculated in LOGIC
    # reverse virtual ray casting
    blocked_mask = Qpts.reverse_virtual_ray_casting(trigger, minz_matrix_roi)
    trigger &= ~blocked_mask
    timer[3].stop()

    # fine ground segmentation
    timer[4].start("Fine Ground Seg")
    ground_index_matrix = np.log2((trigger & -trigger) >> 1).astype(int)
    ground_trigger = Mpts.calculate_ground_mask(Qpts, ground_index_matrix)
    global_trigger[Qpts.start_id_x:Qpts.start_id_x+Qpts.matrix_order, Qpts.start_id_y:Qpts.start_id_y+Qpts.matrix_order] |= trigger
    global_ground_trigger[:, Qpts.start_id_x:Qpts.start_id_x+Qpts.matrix_order, Qpts.start_id_y:Qpts.start_id_y+Qpts.matrix_order] |= ground_trigger
    timer[4].stop()
    timer[5].stop()

# extract all dynamic points from trigger matrix
points_index2Remove = []
for (i,j) in list(zip(*np.where(global_trigger != 0))):
    z = Mpts.binTo3id(global_trigger[i][j])
    for idz in z:
        points_index2Remove += (Mpts.root_matrix[i][j].children[idz].pts_id).tolist()
    for cid in range(len(global_ground_trigger)):
        gz = Mpts.binTo3id(global_ground_trigger[cid][i][j])
        for idgz in gz:
            points_index2Remove += (Mpts.root_matrix[i][j].children[cid].children[idgz].pts_id).tolist()    # points_index2Remove = list(set(points_index2Remove))
print(f"There are {len(points_index2Remove)} pts to remove")
print(f" running time: {time.time() - starttime}")

# save static map
points_index2Retain =np.setdiff1d(np.arange(len(Mpts.original_points)), points_index2Remove)
outlier_cloud = Mpts.original_points[points_index2Retain]
save_pcd(f"{DATA_FOLDER}/beautymap_output.pcd", outlier_cloud) # static map

# show timer
timer.stop()
timer.print(title="BeautyMap",random_colors=True, bold=True)
print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")