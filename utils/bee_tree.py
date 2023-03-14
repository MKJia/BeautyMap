# Updated: 2023-3-06 13:20

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import itertools
import numpy as np
import pandas as pd

import open3d as o3d
import matplotlib.pyplot as plt
from utils import quat2mat
import time
from collections import defaultdict
import copy
from . import load_view_point

SIZE_OF_INT64 = 64
SIZE_OF_INT32 = 32
MIN_Z_RES = 0.02

class BEETree: # Binary-Encoded Eliminate Tree (Any B Number in my mind?)
    def __init__(self): 
        # points 
        self.original_points = None
        self.non_negtive_points = None
        self.o3d_original_points = None

        # unit parameters
        self.unit_x = None
        self.unit_y = None
        self.unit_z = None
        
        # calculate parameters
        self.center = None
        self.non_negtive_center = None
        self.coordinate_offset = None
        self.poses = None
        self.start_xy = None
        self.matrix_order = None

        # tree structure
        self.root_matrix = None
        self.pts_num_in_unit = None
        self.binary_matrix = None

        #RPG
        self.RPGMat = None
        self.RPGMask = None
        self.RangeMask = None


    def set_points_from_file(self, filename):
        ## 0. Read Point Cloud
        self.original_points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        self.o3d_original_points = o3d.geometry.PointCloud()
        self.o3d_original_points.points = o3d.utility.Vector3dVector(self.original_points[:,:3]) 
        print(f"number of points: {self.original_points.shape}")

    def set_unit_params(self, unit_x, unit_y, unit_z):
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_z = unit_z

    def set_global_center_from_file(self, filename):
        self.poses = np.array(pd.read_csv(filename).values[:,2:], dtype=np.float32)
        self.center = np.mean(self.poses[:,:3], axis=0)
        self.start_xy = np.array([.0,.0]) # Only for map, once
        
    def non_negatification_all_map_points(self):
        """Makes all points x,y,z are non-negtive ones to store in matrix & binary tree, calculate new center and save offset for Querys

        Find the minimum x,y,z and process all points as an numpy array, the two points variables have the same index ONLY FOR MAP
        """
        min_xyz = np.array([min(self.original_points[...,0]), 
                             min(self.original_points[...,1]), 
                             min(self.original_points[...,2])])
        self.coordinate_offset = min_xyz
        self.non_negtive_points = self.original_points[:,:3] - min_xyz
        self.non_negtive_center = self.center - min_xyz 

    def calculate_matrix_order(self): # def calculate_matrix_column_row(self):
        max_x = max(self.non_negtive_points[...,0])
        max_y = max(self.non_negtive_points[...,1])
        min_x = min(self.non_negtive_points[...,0])
        min_y = min(self.non_negtive_points[...,1])
        print(max_x, min_x, max_y, min_y)
        self.matrix_order = max((max_x - min_x)/ self.unit_x, (max_y - min_y) / self.unit_y).astype(int) + 1  # old version self.dim_2d
        print(self.matrix_order)

        ## Maybe the matrix do not need to be a square matrix? @Kin
        # self.matrix_columns = (max_x / self.unit_x).astype(int) + 1
        # self.matrix_rows = (max_y / self.unit_y).astype(int) + 1

    def generate_binary_tree(self):
        """Generates a binary tree

        Matrix -> BEETree(ROOT) -> BEENode...
        230307 TODO: remove np.where and change to another method for binary tree generation
            np.where needs to search all the list, though sometimes the aimed value is just the first one, which cost much time.
        """
        points_in_map_frame = self.non_negtive_points - [self.start_xy[0], self.start_xy[1], 0]
        idxyz = (np.divide(points_in_map_frame,[self.unit_x, self.unit_y, self.unit_z])).astype(int)[:,:3]

        ori_id = np.lexsort([idxyz[:,2], idxyz[:,1], idxyz[:,0]])
        newidxyz = idxyz[ori_id]
        # if(len(newidxyz) < 1000000):
        #     f = open("./log.txt", 'w+')
        #     print((newidxyz), file=f)
        #     f.close()
        # idx = idxyz[:,0]
        # idy = idxyz[:,1]

        self.root_matrix = np.empty([self.matrix_order, self.matrix_order], dtype=object)
        self.pts_num_in_unit = np.zeros([self.matrix_order, self.matrix_order], dtype=int)
        # for i in range(self.matrix_order):
        #     for j in range(self.matrix_order):
        #         ptid_in_unit_ij = (idx == i) & (idy == j)
        #         pts_z = idxyz[:,2][ptid_in_unit_ij]
        #         if len(pts_z) == 0:
        #             continue
        #         self.root_matrix[i][j] = BEENode()
        #         self.root_matrix[i][j].register_points(points_in_map_frame[ptid_in_unit_ij], pts_z, self.unit_z, ptid_in_unit_ij)
        #         self.pts_num_in_unit[i][j] = len(pts_z)
        id_begin = np.array([],dtype=int)
        id_end = np.array([],dtype=int)
        t = time.time()
        id_begin = [i for i, v in enumerate(newidxyz) if i == 0 or (v[0] != newidxyz[i-1][0] or v[1] != newidxyz[i-1][1])]
        print(time.time()-t)
        id_end = copy.deepcopy(id_begin)
        id_end.remove(0)
        id_end.append(len(newidxyz))
        for iid in range(len(id_begin)):
            ib = id_begin[iid]
            ie = id_end[iid]
            idx = newidxyz[ib][0]
            idy = newidxyz[ib][1]
            if idx < 0 or idy < 0 or idx >= self.matrix_order or idy >= self.matrix_order:
                continue
            idz = newidxyz[ib:ie][:,2]
            pts_id = ori_id[ib:ie]
            pts = points_in_map_frame[pts_id]
            self.root_matrix[idx][idy] = BEENode()
            self.root_matrix[idx][idy].register_points(pts, idz, self.unit_z, pts_id)
            self.pts_num_in_unit[idx][idy] = ie - ib

    def get_binary_matrix(self):
        self.binary_matrix = np.zeros([self.matrix_order, self.matrix_order], dtype=int)
        for i in range(self.matrix_order):
            for j in range(self.matrix_order):
                if self.root_matrix[i][j] is None:
                    self.binary_matrix[i][j] = 0
                else:
                    self.binary_matrix[i][j] = self.root_matrix[i][j].binary_data

    def transform_to_map_frame(self, pose, coordinate_offset):
        wxyz = np.array([pose[6],pose[3],pose[4],pose[5]])
        T_Q = np.eye(4)
        T_Q[:3,:3] = quat2mat(wxyz)
        T_Q[:3,-1] = np.array([pose[0],pose[1],pose[2]])


        self.original_points = np.insert(self.original_points, 0, np.array([0,0,0,0]), axis = 0)
        self.o3d_original_points.points = o3d.utility.Vector3dVector(self.original_points[:,:3])
        self.o3d_original_points.transform(T_Q)
        self.non_negtive_points = np.asarray(self.o3d_original_points.points - coordinate_offset)
        self.non_negtive_center = self.non_negtive_points[0,:]

    def smoother(self):
        m,n = len(self.pts_num_in_unit), len(self.pts_num_in_unit[0])
        res = [[0] * n for _ in range(m)]
        for i, j in itertools.product(range(m), range(n)):
            count = 0
            for x, y in itertools.product(range(i-1, i+2), range(j-1, j+2)):
                if 0 <= x < m and 0 <= y < n:
                    res[i][j] += self.pts_num_in_unit[x][y]
                    count += 1
            res[i][j] //= count
        return np.asarray(res,dtype='int')
    
    def generate_range_mask(self, r):
        range_mask = np.zeros((self.matrix_order, self.matrix_order), dtype=int)
        i = j = (self.matrix_order - r)//2
        for x, y in itertools.product(range(i, i+r), range(j, j+r)):
            range_mask[x][y] = 1
        return range_mask
    
    def exclusive_with_other_binary_2d(self, map_binary_matrix):
        # compute the exclusive or
        start_id_x = (int)(self.start_xy[0] / self.unit_x)
        start_id_y = (int)(self.start_xy[1] / self.unit_y)
        map_binary_matrix_roi = map_binary_matrix[start_id_x:start_id_x+self.matrix_order][:, start_id_y:start_id_y+self.matrix_order]
        print(start_id_x, start_id_y)
        return map_binary_matrix_roi ^ self.binary_matrix
  
    def calculate_query_matrix_start_id(self):
        # compute the exclusive or
        start_point_x = self.non_negtive_center[0] - self.matrix_order / 2.0 * self.unit_x
        start_point_y = self.non_negtive_center[1] - self.matrix_order / 2.0 * self.unit_y
        self.start_xy = np.array([start_point_x, start_point_y])

    def view_compare(self, inlier, outlier, others=None, view_file = None):
        view_things = [outlier]
        if others is not None:
            others.paint_uniform_color([0.0, 0.0, 0.0])
            view_things.append(others)
        inlier.paint_uniform_color([1.0, 0, 0])
        view_things.append(inlier)
        load_view_point(view_things, filename=view_file)
class BEENode:
    def __init__(self):
        self.binary_data = 0 # int64
        self.children = np.empty(SIZE_OF_INT64, dtype=object) # ndarray(BEENode)[63]
        self.pts_id = None
        self.pts_num = None

    def register_points(self, pts, idz, unit_z, pts_id):
        """Register all points in BEENodes

        BEETree(ROOT) -> N*N BEENodes(BRANCH) ->Children BEENodes(LEAF)

        Args:
            pts: ndarray list of points selected by index(ptid_in_unit_ij),
            one function can register all points in one unit
            idz: ndarray list of idz selected by index(ptid_in_unit_ij),
            each single value needs to be 0 <= idz[i] <= 62 for int64,
            fortunately we have non-negatificated points so it obviously >= 0,
            and 1 << idz[i] will become 0 for those bigger than 63.
            So we only need to deal with 63 (Or it will ERROR :ufunc 'bitwise_or' 
            not supported for the input types, and the inputs could not be safely 
            coerced to any supported types according to the casting rule ''safe'')
        """
        if unit_z > MIN_Z_RES: 
            hierarchical_unit_z = max(MIN_Z_RES, unit_z / (SIZE_OF_INT64-1))
        elif unit_z == MIN_Z_RES:
            hierarchical_unit_z = 0.0001
        else:
            self.children = None
            self.pts_id = pts_id
            return 0
        for i in range(SIZE_OF_INT64):
            overheight_id = np.where(idz>=i)
            in_node_id = np.where(idz==i)
            if (i == SIZE_OF_INT64 - 1 or i == SIZE_OF_INT64) and overheight_id[0].size != 0:
                self.children[i] = BEENode()
                new_idz = (np.divide(pts[overheight_id][...,2] - i * unit_z, hierarchical_unit_z)).astype(int)
                self.children[i].register_points(pts[overheight_id], new_idz, hierarchical_unit_z, pts_id[overheight_id])
                i = SIZE_OF_INT64 - 2 # to protect SIZE_OF_INT - 1
            elif in_node_id[0].size == 0:
                self.children[i] = None
                continue
            else:
                self.children[i] = BEENode()
                new_idz = (np.divide(pts[in_node_id][...,2] - i * unit_z, hierarchical_unit_z)).astype(int)
                self.children[i].register_points(pts[in_node_id], new_idz, hierarchical_unit_z, pts_id[in_node_id])
            self.binary_data |= 1<<i
        self.pts_id = pts_id
