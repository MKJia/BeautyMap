# Updated: 2023-3-06 13:20
# Copyright (C) 2022-now, KTH-RPL, HKUST-RamLab
# Author:
# Mingkai Jia  (https://mkjia.github.io/)
# Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import itertools
import numpy as np

import copy
from utils.pcdpy3 import load_pcd
from tqdm import tqdm

MAX_HEIGHT = 32
MIN_Z_RES = 0.1 # minimum hierarchical height resolution
GPNR = 0.85 # Ground Points Number Ratio
RANGE_OF_SIGHT = 15 # m Points over this range become unreliable

class BEETree: # Binary-Encoded Eliminate Tree
    def __init__(self):
        # points
        self.original_points = None
        self.non_negtive_points = None

        # unit parameters
        self.unit_x = None
        self.unit_y = None
        self.unit_z = None

        # calculate parameters
        self.center = None
        self.non_negtive_center = None
        self.coordinate_offset = None
        self.start_xy = None
        self.matrix_order = None
        self.start_id_x = 0
        self.start_id_y = 0
        self.viewpoint_z = 0
        self.max_height = MAX_HEIGHT

        # map structure
        self.root_matrix = None
        self.pts_num_in_unit = None
        self.binary_matrix = None
        self.minz_matrix = None
        self.outlier_matrix = None

    def set_points_from_file(self, filename):
        ## 0. Read Point Cloud
        PointCloudData = load_pcd(filename) # x, y, z, qw, qx, qy, qz
        self.sensor_origin_pose = np.array(list(PointCloudData.get_metadata()['viewpoint']))
        self.original_points = PointCloudData.np_data
        self.original_points[:,:3] += self.sensor_origin_pose[:3] # hotfix0717: tranform map points by viewpoint params
        self.center = np.mean(self.original_points[:,:3],axis=0)
        self.start_xy = np.array([.0,.0]) # Only for map, once

    def set_unit_params(self, unit_x, unit_y, unit_z):
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_z = unit_z

    def non_negatification_all_map_points(self):
        """Makes all points x,y,z are non-negtive ones to store in matrix & binary tree, calculate new center and save offset for Querys

        Find the minimum x,y,z and process all points as an numpy array, the two points variables have the same index ONLY FOR MAP
        """
        min_xyz = np.array([min(self.original_points[...,0]),
                             min(self.original_points[...,1]),
                             min(self.original_points[...,2])])
        self.coordinate_offset = min_xyz
        print(f"[LOG] Min xyz: {min_xyz[0]}, {min_xyz[1]}, {min_xyz[2]} ")
        self.non_negtive_points = self.original_points[:,:3] - min_xyz
        self.non_negtive_center = self.center - min_xyz

    def calculate_matrix_order(self):
        max_x = max(self.non_negtive_points[...,0])
        max_y = max(self.non_negtive_points[...,1])
        min_x = min(self.non_negtive_points[...,0])
        min_y = min(self.non_negtive_points[...,1])
        print(f"[LOG] Max/Min value on x: {max_x}/{min_x}, y: {max_y}/{min_y}")
        self.matrix_order = max((max_x - min_x)/ self.unit_x, (max_y - min_y) / self.unit_y).astype(int) + 1
        print(f"[LOG] Matrix order: {self.matrix_order}")
        self.minz_matrix = np.zeros([self.matrix_order, self.matrix_order], dtype=float) + float("inf") # Only for map, once

    def generate_map_binary_tree(self):
        """Generates a binary tree

        Matrix -> BEETree(ROOT) -> BEENode...
        """
        points_in_map_frame = self.non_negtive_points - [self.start_xy[0], self.start_xy[1], 0]
        idxyz = (np.divide(points_in_map_frame,[self.unit_x, self.unit_y, self.unit_z])).astype(int)[:,:3]

        ori_id = np.lexsort([idxyz[:,2], idxyz[:,1], idxyz[:,0]])
        newidxyz = idxyz[ori_id]

        self.root_matrix = np.empty([self.matrix_order, self.matrix_order], dtype=object)
        self.pts_num_in_unit = np.zeros([self.matrix_order, self.matrix_order], dtype=int)
        self.outlier_matrix = np.zeros([self.matrix_order, self.matrix_order], dtype=int)

        id_begin = np.array([],dtype=int)
        id_end = np.array([],dtype=int)

        x_diff = newidxyz[1:, 0] != newidxyz[:-1, 0]
        y_diff = newidxyz[1:, 1] != newidxyz[:-1, 1]
        id_begin = (np.flatnonzero(x_diff | y_diff) + 1 ).tolist()
        id_begin.insert(0, 0)

        id_end = copy.deepcopy(id_begin)
        id_end.remove(0)
        id_end.append(len(newidxyz))

        for iid in tqdm(range(len(id_begin)), desc='Generating Map Binary tree', ncols=100):
            ib = id_begin[iid]
            ie = id_end[iid]
            idx = newidxyz[ib][0]
            idy = newidxyz[ib][1]
            if idx < 0 or idy < 0 or idx >= self.matrix_order or idy >= self.matrix_order:
                continue
            pts_id = ori_id[ib:ie]
            pts = points_in_map_frame[pts_id]
            self.root_matrix[idx][idy] = BEENode()
            min_z = min(pts[...,2])
            neighbour_array = self.calculate_median(idx, idy)

            # coarse ground extraction
            if len(neighbour_array) != 0:
                min_z_median = np.median(neighbour_array)
                min_z_MAD = np.median(np.abs(neighbour_array - min_z_median))
                if min_z < min_z_median - 3 * min_z_MAD:
                    min_z = min_z_median - 3 * min_z_MAD
                    self.outlier_matrix[idx][idy] = 1
                elif min_z > min_z_median + 3 * min_z_MAD:
                    min_z = min_z_median + 3 * min_z_MAD

            self.root_matrix[idx][idy].min_z = min_z
            idz = np.divide(pts[...,2] - self.root_matrix[idx][idy].min_z, self.unit_z).astype(int)
            self.root_matrix[idx][idy].register_points(pts, idz, self.unit_z, pts_id)
            self.pts_num_in_unit[idx][idy] = ie - ib

    def calculate_median(self, idx, idy):
        min_z_list = []
        near_len = 5
        effective_grids_num = int(near_len ** 2 * 0.60)
        for x, y in itertools.product(range(idx-near_len, idx), range(idy-near_len, idy)):
            if x < 0 or y < 0:
                continue
            if self.root_matrix[x][y] == None:
                continue
            min_z_list.append(self.root_matrix[x][y].min_z)
        if len(min_z_list) < effective_grids_num:
            min_z_list = []
        return np.asarray(min_z_list)

    def generate_query_binary_tree(self, minz_matrix_ori):
        """Generates a simple query binary tree
        """
        points_in_map_frame = self.non_negtive_points - [self.start_xy[0], self.start_xy[1], 0]
        points_in_range_id = (points_in_map_frame[:, 0] >= 0) & (points_in_map_frame[:, 1] >= 0) & (points_in_map_frame[:, 0] < self.matrix_order*self.unit_x) & (points_in_map_frame[:, 1] < self.matrix_order*self.unit_y)
        points_in_range = points_in_map_frame[points_in_range_id]
        if len(points_in_range) == 0:
            return False
        idxy = (np.divide(points_in_range,[self.unit_x, self.unit_y, self.unit_z])).astype(int)[:,:2]
        ori_id = np.lexsort([idxy[:,1], idxy[:,0]])
        newidxy = idxy[ori_id]

        self.root_matrix = np.empty([self.matrix_order, self.matrix_order], dtype=object)

        id_begin = np.array([],dtype=int)
        id_end = np.array([],dtype=int)
        x_diff = newidxy[1:, 0] != newidxy[:-1, 0]
        y_diff = newidxy[1:, 1] != newidxy[:-1, 1]
        id_begin = (np.flatnonzero(x_diff | y_diff) + 1 ).tolist()
        id_begin.insert(0, 0)

        id_end = copy.deepcopy(id_begin)
        id_end.remove(0)
        id_end.append(len(newidxy))

        for iid in range(len(id_begin)):
            ib = id_begin[iid]
            ie = id_end[iid]
            idx = newidxy[ib][0]
            idy = newidxy[ib][1]
            pts_id = ori_id[ib:ie]
            pts = points_in_range[pts_id]
            self.root_matrix[idx][idy] = BEENode()
            self.root_matrix[idx][idy].min_z = minz_matrix_ori[idx][idy]
            idz = np.divide(pts[...,2] - self.root_matrix[idx][idy].min_z, self.unit_z).astype(int)

            idz_sorted = np.unique(idz)
            overheight_index_search = np.searchsorted(idz_sorted, MAX_HEIGHT - 2, side='right')
            filtered_array_search = idz_sorted[np.searchsorted(idz_sorted, 0, side='left'):overheight_index_search]
            exists_overheight = overheight_index_search < len(idz_sorted) - 1 and idz_sorted[overheight_index_search+1] > MAX_HEIGHT - 2

            self.root_matrix[idx][idy].binary_data = np.sum(1 << filtered_array_search)
            if (exists_overheight):
                self.root_matrix[idx][idy].binary_data |= 1 << MAX_HEIGHT - 2

        return True

    def get_binary_matrix(self):
        self.binary_matrix = np.zeros([self.matrix_order, self.matrix_order], dtype=int)
        for i in range(self.matrix_order):
            for j in range(self.matrix_order):
                if self.root_matrix[i][j] is None:
                    self.binary_matrix[i][j] = 0
                else:
                    self.binary_matrix[i][j] = self.root_matrix[i][j].binary_data

    def get_minz_matrix(self):
        self.minz_matrix = np.zeros([self.matrix_order, self.matrix_order], dtype=float)
        for i in range(self.matrix_order):
            for j in range(self.matrix_order):
                if self.root_matrix[i][j] is not None:
                    self.minz_matrix[i][j] = self.root_matrix[i][j].min_z

    def transform_on_points(self, coordinate_offset):
        # hotfix0717: tranform map points by viewpoint params
        # NOTE: In our standard data pre-processing pipe, query frames need to be transormed into map coordinate!
        # You can also note the follow line:225 to ALLOW viewpoint transformation for query frames
        self.original_points[:,:3] -= self.sensor_origin_pose[:3]
        # non-negtificate query points
        self.non_negtive_points = np.asarray(self.original_points[:,:3] - coordinate_offset)
        self.non_negtive_center = self.sensor_origin_pose[:3] - coordinate_offset
        tmp1 = self.non_negtive_points - self.non_negtive_center
        tmp2 = np.sqrt(tmp1[:,0] ** 2 + tmp1[:,1] ** 2)
        return max(tmp1[:,2]/tmp2)

    def generate_static_restoration_mask(self, k, minz_matrix, outlier_matrix):
        # range mask (optional)
        blind_range = int(2.0 / self.unit_x) + 1
        range_mask = np.ones((self.matrix_order, self.matrix_order), dtype=int)
        for i, j in itertools.product(range(-blind_range, blind_range), range(-blind_range, blind_range)):
            range_mask[int(self.matrix_order/2)+i][int(self.matrix_order/2)+j] = 0
        # visibility mask
        r = self.matrix_order
        sight_mask = np.zeros((r, r), dtype=int)
        no_ground = self.binary_matrix & (-2) # -0xffff fffe
        ground = self.binary_matrix & 1
        highest_bit = (np.log2(no_ground+1)).astype(int)
        self.viewpoint_z = (self.non_negtive_center[2] - minz_matrix[int(self.matrix_order/2)][int(self.matrix_order/2)]) / self.unit_z + 0.5
        for i, j in itertools.product(range(0, 0+r), range(0, 0+r)):
            highest_sight_bit = min(MAX_HEIGHT-1, int(self.viewpoint_z + np.sqrt((r/2-i)**2+(r/2-j)**2) * k + 0.5))
            if ground[i][j] == 0 and (i - r/2)**2 + (j - r/2)**2 >= (RANGE_OF_SIGHT / self.unit_y)**2:
                sight_mask[i][j] = 2 ** MAX_HEIGHT - 1
            elif highest_bit[i][j] > 0 and outlier_matrix[i][j] == 0:
                sight_mask[i][j] = 2 ** MAX_HEIGHT - 2 ** highest_bit[i][j]
            else:
                sight_mask[i][j] = 2 ** MAX_HEIGHT - 2 ** highest_sight_bit
        return sight_mask | (range_mask - 1)

    def generate_blind_grid_mask(self):
        r = int(2.0 / self.unit_x) + 1 # for 2m+ range blind
        # r = 0 # (optional) disable range blind
        BlindGridMask = np.ones((self.matrix_order, self.matrix_order), dtype=int)
        for i, j in itertools.product(range(-r, r), range(-r, r)):
            BlindGridMask[int(self.matrix_order/2)+i][int(self.matrix_order/2)+j] = 0
        return BlindGridMask

    def calculate_map_roi(self, map_binary_matrix):
        map_binary_matrix_roi = map_binary_matrix[self.start_id_x:self.start_id_x+self.matrix_order][:, self.start_id_y:self.start_id_y+self.matrix_order]
        return map_binary_matrix_roi

    def calculate_query_matrix_start_id(self):
        start_point_x = (int)(self.non_negtive_center[0]) - self.matrix_order / 2.0 * self.unit_x
        start_point_y = (int)(self.non_negtive_center[1]) - self.matrix_order / 2.0 * self.unit_y
        self.start_xy = np.array([start_point_x, start_point_y])
        self.start_id_x = (int)(self.start_xy[0] / self.unit_x)
        self.start_id_y = (int)(self.start_xy[1] / self.unit_y)

    def calculate_ground_mask(self, Qpts, ground_index_matrix):
        ground_mask = np.zeros([MAX_HEIGHT, Qpts.matrix_order, Qpts.matrix_order], dtype=int)
        for ii in range(Qpts.matrix_order):
            i = ii + Qpts.start_id_x
            for jj in range(Qpts.matrix_order):
                j = jj + Qpts.start_id_y
                cid = ground_index_matrix[ii][jj]
                if cid >= 0 and self.root_matrix[i][j] is not None and self.root_matrix[i][j].children[cid] is not None:# if has ground 2-nd children
                    all_pts_num = self.root_matrix[i][j].children[cid].pts_num
                    pts_num = 0
                    next_flag = False
                    for k in range(MAX_HEIGHT):
                        if self.root_matrix[i][j].children[cid].children[k] is not None:
                            pts_num += self.root_matrix[i][j].children[cid].children[k].pts_num
                            if pts_num *1.0 / all_pts_num >= GPNR:
                                ground_mask[cid][ii][jj] |= ((1 & next_flag) << k) # 1 for triggered and 0 for protected
                                next_flag = True
        return ground_mask

    def reverse_virtual_ray_casting(self, trigger, minz_matrix):
        blocked_mask = np.zeros((self.matrix_order, self.matrix_order), dtype=int)
        origin_x = origin_y = int(self.matrix_order / 2)
        for i, j in itertools.product(range(self.matrix_order), range(self.matrix_order)):
            if trigger[i][j] != 0:
                direction_x = i - origin_x
                direction_y = j - origin_y
                if direction_x == 0 and direction_y != 0:
                    step_x = 0
                    step_y = direction_y / abs(direction_y)
                elif direction_y == 0 and direction_x != 0:
                    step_y = 0
                    step_x = direction_x / abs(direction_x)
                elif direction_x != 0 and direction_y != 0:
                    sign_x = direction_x / abs(direction_x)
                    sign_y = direction_y / abs(direction_y)
                    if abs(direction_x) > abs(direction_y):
                        step_y = sign_y * abs(direction_y / direction_x)
                        step_x = sign_x
                    else:
                        step_x = sign_x * abs(direction_x / direction_y)
                        step_y = sign_y
                else:
                    continue

                z_list = self.binTo3id(trigger[i][j])
                current_x = origin_x + step_x
                current_y = origin_y + step_y
                ds = np.sqrt(step_x**2+step_y**2)
                s = np.sqrt(direction_x**2+direction_y**2)
                step_z = ds

                while 0 < current_x < self.matrix_order and 0 < current_y < self.matrix_order:
                    if int(abs(current_x - origin_x)) + int(abs(current_y - origin_y)) > 3:
                        query_z = self.binary_matrix[int(current_x)][int(current_y)]
                        if query_z != 0:
                            current_z = [int(minz_matrix[int(current_x)][int(current_y)]-minz_matrix[origin_x][origin_y]+((z - self.viewpoint_z) * step_z / s)+ self.viewpoint_z+0.5) for z in z_list]
                            current_occupied_list = self.binTo3id(query_z)
                            blocked_index = [index for index, num in enumerate(current_z) if num in current_occupied_list] # index is same as z_list
                            for id in blocked_index:
                                blocked_mask[i][j] |= (1 << z_list[id])
                        if abs(current_x - i) <= 1 and abs(current_y - j) <= 1:
                            break
                    step_z += ds
                    current_x += step_x
                    current_y += step_y
        return blocked_mask

    @staticmethod
    def binTo3id(t):
        tmp = []
        cnt = 0
        while t:
            if t & 1 == 1:
                tmp.append(cnt)
            t = t >> 1
            cnt += 1
        return tmp

class BEENode:
    def __init__(self):
        self.binary_data = 0
        self.children = None
        self.pts_id = None
        self.pts_num = None
        self.min_z = float("inf")

    def register_points(self, pts, idz, unit_z, pts_id):
        """Register all points in BEENodes

        BEETree(ROOT) -> N*N BEENodes ->Children

        Args:
            pts: ndarray list of points selected by index
            idz: ndarray list of idz selected by index
        """
        if unit_z > MIN_Z_RES:
            hierarchical_unit_z = max(MIN_Z_RES, unit_z / (MAX_HEIGHT-1))
            self.children = np.empty(MAX_HEIGHT, dtype=object)
        elif unit_z == MIN_Z_RES:
            hierarchical_unit_z = 0.0
            self.children = np.empty(MAX_HEIGHT, dtype=object)
        else:
            self.children = None
            self.pts_id = pts_id
            self.pts_num = len(pts)
            return 0
        in_ground_flag = False
        for i in range(MAX_HEIGHT):
            overheight_id = np.where(idz>=i)
            in_node_id = np.where(idz==i)
            upper_node_id = np.where(idz==i+1)
            if i == MAX_HEIGHT - 1 and overheight_id[0].size != 0:
                ii = MAX_HEIGHT - 2 # to protect MAX_HEIGHT - 1 if using MAX_HEIGHT == 64 for int64
                index = overheight_id
            elif in_node_id[0].size == 0:
                self.children[i] = None
                if(in_ground_flag and upper_node_id[0].size == 0):
                    break
                else:
                    continue
            else:
                ii = i
                index = in_node_id
                in_ground_flag = True
            self.children[ii] = BEENode()
            self.children[ii].min_z = min(pts[index][...,2])
            new_idz = (np.divide(pts[in_node_id][...,2] - i * unit_z - self.min_z, hierarchical_unit_z)).astype(int)
            self.children[ii].register_points(pts[index], new_idz, hierarchical_unit_z, pts_id[index])
            self.binary_data |= 1<<ii
        self.pts_id = pts_id
        self.pts_num = len(pts)

    def get_num(self):
        l = []
        for i in range(MAX_HEIGHT):
            if self.children[i] is not None:
                l.append(self.children[i].pts_num)
            else:
                l.append(0)
        return l