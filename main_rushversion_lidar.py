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

starttime = time.time()

RANGE = 10 # m, from cetner point to an square
RESOLUTION = 0.5 # m, resolution default 1m
H_RES = 0.5 # m, resolution default 1m
RANGE_16_RING = 8
GROUND_THICK = 0.5
DATA_FOLDER = f"{BASE_DIR}/data/three_people_behind"
# DATA_FOLDER = f"{BASE_DIR}/data/KITTI/00"
MAX_RUN_FILE_NUM = 10 # -1 for all files

print(f"We will process the data in folder: {bc.BOLD}{DATA_FOLDER}{bc.ENDC}")
points_index2Remove = []
points_ground2Protect = []

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
Mpts.generate_binary_tree(Mpts.minz_matrix)
print("Generate binary tree in Map cost: ",time.time() - t1, " s")
Mpts.get_binary_matrix()
Mpts.get_minz_matrix()
ijh_index = np.log2((Mpts.binary_matrix & -Mpts.binary_matrix)).astype(int) #* (2**int(GROUND_THICK/H_RES+2)-1)
Mpts.get_ground_hierachical_binary_matrix(ijh_index) # for i,j in matrix, we extract the ground hierachical in the h_th height
points_ground2Protect = Mpts.get_ground_points_id(ijh_index)
ground_mask = Mpts.calculate_ground_distribution_mask()

print("finished M, cost: ", time.time() - t1, " s")
# view_pts = Mpts.view_tree(ijh_index, 1)
# o3d.visualization.draw_geometries([view_pts])

all_pcd_files = sorted(os.listdir(f"{DATA_FOLDER}/pcd"))
for file_cnt, pcd_file in tqdm(enumerate(all_pcd_files)):
    if file_cnt>MAX_RUN_FILE_NUM and MAX_RUN_FILE_NUM!=-1:
        break
    
    Qpts = BEETree()
    Qpts.set_points_from_file(f"{DATA_FOLDER}/pcd/{pcd_file}")
    Qpts.set_unit_params(RESOLUTION, RESOLUTION, H_RES)
    Qpts.transform_on_points(Mpts.coordinate_offset)

    Qpts.matrix_order = (int)(RANGE / RESOLUTION)
    Qpts.calculate_query_matrix_start_id()

    t1 = time.time()
    Qpts.generate_binary_tree(Mpts.minz_matrix)
    print(time.time() - t1)
    Qpts.get_binary_matrix()
    print("finished Q, cost: ", time.time() - t1, " ms")

    # pre-process

    # RPG
    Qpts.RPGMat = Qpts.smoother()
    Qpts.RPGMask = Qpts.RPGMat > (RESOLUTION * RANGE)**2

    # DEAR
    Qpts.RangeMask = Qpts.generate_range_mask(int(RANGE_16_RING/RESOLUTION))
    # Qpts.SightMask = TODO: generate sight mask
    # Qpts.DEARMask = Qpts.RangeMask & Qpts.SightMask

    map_binary_matrix_roi = Qpts.calculate_map_roi(Mpts.binary_matrix)
    binary_xor = (~Qpts.binary_matrix) & map_binary_matrix_roi
    trigger = binary_xor #(~Qpts.binary_matrix) & binary_xor

    trigger &= ~(Qpts.RPGMask - 1)
    trigger &= ~(Qpts.RangeMask - 1)
    trigger &= ~(map_binary_matrix_roi & -map_binary_matrix_roi)
    # print(Qpts.binary_2d)


    ground_index_matrix = np.log2((trigger & -trigger) >> 1).astype(int) #* (2**int(GROUND_THICK/H_RES+2)-1)
    ground_trigger = Mpts.calculate_ground_mask(Qpts, ground_index_matrix)
    # map_ground_binary_matrix_roi = Qpts.calculate_map_roi(Mpts.ground_binary_matrix)
    # map_ground_mask_roi = Qpts.calculate_map_roi(ground_mask)
    # Mpts.get_ground_hierachical_binary_matrix(ground_index_matrix)

    # for i in range(len(map_ground_binary_matrix_roi)):
    #     print("================================================================")
    #     for j in range(len(map_ground_binary_matrix_roi[0])):
    #         print(bin(map_ground_binary_matrix_roi[i][j]).zfill(32))
    # fig, axs = plt.subplots(2, 2, figsize=(8,8))
    # axs[0,0].imshow(np.log2(Qpts.binary_matrix), cmap='hot', interpolation='nearest')
    # axs[0,0].set_title('Query 2d')
    # axs[0,1].imshow(np.log2(map_binary_matrix_roi), cmap='hot', interpolation='nearest')
    # axs[0,1].set_title('Prior Map bin 2d')
    # axs[1,0].imshow(np.log2(map_ground_binary_matrix_roi), cmap='hot', interpolation='nearest')
    # axs[1,0].set_title('After RPG')
    # axs[1,1].imshow(np.log2(trigger), cmap='hot', interpolation='nearest')
    # axs[1,1].set_title('After RPG Mask')
    # plt.show()

    t = time.time()
    for (i,j) in list(zip(*np.where(trigger != 0))):
        z = Mpts.binTo3id(trigger[i][j])
        for idz in z:
            points_index2Remove += (Mpts.root_matrix[i+Qpts.start_id_x][j+Qpts.start_id_y].children[idz].pts_id).tolist()
        gz = Mpts.binTo3id(ground_trigger[i][j])
        for idgz in gz:
            points_index2Remove += (Mpts.root_matrix[i+Qpts.start_id_x][j+Qpts.start_id_y].children[ground_index_matrix[i][j]].children[idgz].pts_id).tolist()
    print(time.time() - t)

print(f"There are {len(points_index2Remove)} pts to remove")

visited = set()
dup_index2Remove = list({x for x in points_index2Remove if x in visited or (visited.add(x) or False)})
print(f"There are {len(dup_index2Remove)} pts to remove now")
TOC("All processes")

endtime = time.time()
print (endtime - starttime)

inlier_cloud = Mpts.o3d_original_points.select_by_index(points_index2Remove)
oulier_cloud = Mpts.o3d_original_points.select_by_index(points_index2Remove, invert=True)
save_pcd(f"{DATA_FOLDER}/edomap_output.pcd", np.array(oulier_cloud.points))
print(f"Saved {len(oulier_cloud.points)} data points to edomap_output.pcd.")
# Mpts.view_compare(inlier_cloud, oulier_cloud)

# fig, axs = plt.subplots(2, 2, figsize=(8,8))
# axs[0,0].imshow(Qpts.binary_2d, cmap='hot', interpolation='nearest')
# axs[0,0].set_title('Query 2d')
# axs[0,1].imshow(Mpts.binary_2d, cmap='hot', interpolation='nearest')
# axs[0,1].set_title('Prior Map bin 2d')
# axs[1,0].imshow(trigger, cmap='hot', interpolation='nearest')
# axs[1,0].set_title('After xor')
# plt.show()

print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")