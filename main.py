'''
# Created: 2023-5-1 16:29
# Copyright (C) 2023-now, KTH-RPL, HKUST-RamLab
# Author:
# Mingkai Jia  (https://mkjia.github.io/)
# Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
'''

# math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)

from tqdm import tqdm
import os, fire
import dztimer

# self
from lib.bee_tree import BEETree
from utils.pcdpy3 import save_pcd

def main(
    data_dir: str = "data/00",
    dis_range: float = 40,
    xy_resolution: float = 1.0,
    h_res: float = 0.5,
    run_file_num: int = -1, # -1 for all files
):

    timer = dztimer.Timing()
    timer.start("Total")
    print(f"We will process the data in folder: \033[1m {data_dir} \033[0m")

    # read raw map or gt cloud
    raw_map_path = f"{data_dir}/raw_map.pcd"
    if not os.path.exists(raw_map_path):
        # it's only for reading raw map no label will be used.
        raw_map_path = f"{data_dir}/gt_cloud.pcd"

    if not os.path.exists(raw_map_path):
        print("No raw map found, Please check your data_dir.")
        return

    # map generation
    timer[0].start("Map Generation")
    Mpts = BEETree()
    Mpts.set_points_from_file(raw_map_path)
    Mpts.set_unit_params(xy_resolution, xy_resolution, h_res)
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
    all_pcd_files = sorted(os.listdir(f"{data_dir}/pcd"))
    for file_cnt, pcd_file in tqdm(enumerate(all_pcd_files), total=len(all_pcd_files), ncols=100, desc="Processing Frames"):
        if file_cnt>run_file_num and run_file_num!=-1:
            break
        timer[5].start("One Scan Cost")

        # query init
        timer[1].start("Query Generation")
        Qpts = BEETree()
        Qpts.matrix_order = (int)(dis_range / xy_resolution)
        Qpts.set_points_from_file(f"{data_dir}/pcd/{pcd_file}")
        Qpts.set_unit_params(xy_resolution, xy_resolution, h_res)
        k = Qpts.transform_on_points(Mpts.coordinate_offset)  * xy_resolution / h_res
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
        timer[2].stop()

        # static restoration
        timer[3].start("Static Restoration")
        # out-of-sight protection
        binary_xor &= ~Qpts.generate_static_restoration_mask(k, minz_matrix_roi, outlier_matrix_roi)
        binary_xor &= ~(map_binary_matrix_roi & -map_binary_matrix_roi) # Remove the lowest of the binary_xor, which is further calculated in Fine Ground Seg
        # reverse virtual ray casting
        binary_xor &= ~Qpts.reverse_virtual_ray_casting(binary_xor, minz_matrix_roi)
        timer[3].stop()

        # fine ground segmentation
        timer[4].start("Fine Ground Seg")
        ground_index_matrix = np.log2((binary_xor & -binary_xor) >> 1).astype(int)
        ground_trigger = Mpts.calculate_ground_mask(Qpts, ground_index_matrix)
        global_trigger[Qpts.start_id_x:Qpts.start_id_x+Qpts.matrix_order, Qpts.start_id_y:Qpts.start_id_y+Qpts.matrix_order] |= binary_xor
        global_ground_trigger[:, Qpts.start_id_x:Qpts.start_id_x+Qpts.matrix_order, Qpts.start_id_y:Qpts.start_id_y+Qpts.matrix_order] |= ground_trigger
        timer[4].stop()
        timer[5].stop()

    # extract all dynamic points from binary_xor matrix
    points_index2Remove = []
    print("shape:", global_ground_trigger.shape)
    print("len print:", len(global_ground_trigger))
    for (i,j) in list(zip(*np.where(global_trigger != 0))):
        z = Mpts.binTo3id(global_trigger[i][j])
        for idz in z:
            points_index2Remove += (Mpts.root_matrix[i][j].children[idz].pts_id).tolist()
        for cid in range(len(global_ground_trigger)):
            gz = Mpts.binTo3id(global_ground_trigger[cid][i][j])
            for idgz in gz:
                points_index2Remove += (Mpts.root_matrix[i][j].children[cid].children[idgz].pts_id).tolist()
    print(f"There are {len(points_index2Remove)} pts to remove")

    # save static map
    points_index2Retain =np.setdiff1d(np.arange(len(Mpts.original_points)), points_index2Remove)
    save_pcd(f"{data_dir}/beautymap_output.pcd", Mpts.original_points[points_index2Retain]) # static map

    # show timer
    timer.stop()
    timer.print(title="BeautyMap",random_colors=True, bold=True)
    print(f"{os.path.basename( __file__ )}: All codes run successfully, Close now..")


if __name__ == '__main__':
    fire.Fire(main)