# Updated: 2023-3-06 13:20

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
import pandas as pd

import open3d as o3d
import matplotlib.pyplot as plt

SIZE_OF_INT64 = 64
MIN_Z_RES = 0.02

class BEETree: # Binary-Encoded Eliminate Tree (Any B Number in my mind?)
    def __init__(self, unit_x, unit_y, unit_z): 
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

        # tree structure
        self.root_matrix = None


    def set_points_from_file(self, filename):
        ## 0. Read Point Cloud
        self.original_points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

    def set_unit_params(self, unit_x, unit_y, unit_z):
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.unit_z = unit_z

    def set_center_from_file(self, filename):
        poses = np.array(pd.read_csv(filename).values[:,2:5], dtype=np.float32)
        self.center = np.mean(poses, axis=0)
        
    def non_negatification_all_map_points(self):
        """Makes all points x,y,z are non-negtive ones to store in matrix & binary tree, calculate new center and save offset for Querys

        Find the minimum x,y,z and process all points as an numpy array
        """
        min_xyz = np.array([min(self.original_points[...,0]), 
                             min(self.original_points[...,1]), 
                             min(self.original_points[...,2])])
        self.coordinate_offset = min_xyz
        self.non_negtive_points = self.original_points - min_xyz
        self.non_negtive_center = self.center - min_xyz 
        
    def calculate_matrix_order(self): # def calculate_matrix_column_row(self):
        max_x = max(self.non_negtive_points[...,0])
        max_y = max(self.non_negtive_points[...,1])
        self.matrix_order = max(max_x / self.unit_x, max_y / self.unit_y).astype(int) + 1  # old version self.dim_2d

        ## Maybe the matrix do not need to be a square matrix? @Kin
        # self.matrix_columns = (max_x / self.unit_x).astype(int) + 1
        # self.matrix_rows = (max_y / self.unit_y).astype(int) + 1

    def generate_map_binary_tree(self):
        idx, idy, idz = (np.divide(self.non_negtive_points,[self.unit_x, self.unit_y, self.unit_z])).astype(int)
        self.root_matrix = np.empty([self.matrix_order, self.matrix_order], dtype=object)

        for i in range(self.matrix_order):
            for j in range(self.matrix_order):
                self.root_matrix[i][j] = BEENode()
                ptid_in_unit_ij = np.where(idx==i and idy == j)
                self.root_matrix[i][j].register_points(self.non_negtive_points[ptid_in_unit_ij], idz[ptid_in_unit_ij], self.unit_z)


class BEENode:
    def __init__(self):
        self.binary_data = 0 # int64
        self.children = np.empty(SIZE_OF_INT64, dtype=object) # ndarray(BEENode)[63]
        self.points = None

    def register_points(self, pts, idz, unit_z):
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
        hierarchical_unit_z = max(MIN_Z_RES, unit_z / (SIZE_OF_INT64-1))
        if unit_z == hierarchical_unit_z: 
                self.children = None
                self.points = pts
                return 0
        for i in range(SIZE_OF_INT64):
            overheight_id = np.where(idz>=i)
            in_node_id = np.where(idz==i)
            if i==SIZE_OF_INT64-1 and overheight_id[0].size != 0:
                self.children[i] = BEENode()
                new_idz = (np.divide(pts[overheight_id][...,2] - i * unit_z, hierarchical_unit_z)).astype(int)
                self.children[i].register_points(pts[overheight_id], new_idz)
                i = SIZE_OF_INT64 # to protect 63
            elif in_node_id[0].size == 0:
                self.children[i] = None
                continue
            else:
                self.children[i] = BEENode()
                self.children[i].register_points(pts[in_node_id], idz[in_node_id])
            
            self.binary_data |= 1<<i
        self.points = pts