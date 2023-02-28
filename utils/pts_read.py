# Updated: 2023-2-14 21:36
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# math
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.mixture import GaussianMixture
RANDOM_SEED = 2023
OPERATOR_BIT_UB = 62 # 64-bit with sign bit*1, valid range: 0-62
LOWEST_GRID_NUM = 2
LOGIC_BIT_UB = OPERATOR_BIT_UB//(LOWEST_GRID_NUM)

import open3d as o3d
import matplotlib.pyplot as plt
from . import load_view_point

import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from utils.global_def import *

from collections import defaultdict
import time
from tqdm import tqdm
import operator
class Points:
    def __init__(self, file, range_m, resolution, h_res=0.2):
        ## 0. Read Point Cloud
        # st = time.time()
        self.points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        # print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("Numpy read pt", ((time.time() - st))*1000))

        # parameters
        self.range_m = range_m * 2 # since range is half of dis
        self.resolution = resolution
        self.dim_2d = (int)(self.range_m/resolution)
        self.h_res = h_res
        
        # 为了使得 传感器相对位置下的 z 能被计入计算 HARD CODE HERE
        self.idz_offset = 2

        # results
        self.twoD2ptindex = defaultdict(lambda  : defaultdict(list))
        self.threeD2ptindex = defaultdict(lambda  : defaultdict(lambda : defaultdict(list)))
        self.gmmFitMatrix = defaultdict(lambda  : defaultdict(list))
        self.bin_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        self.LOGICMat = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        self.LOGIC_idx = []

        # open3d
        self.o3d_pts = o3d.geometry.PointCloud()
        self.o3d_pts.points = o3d.utility.Vector3dVector(self.points[:,:3])
        self.points = np.asarray(self.o3d_pts.points)
        self.LOGICPts = np.asarray(self.o3d_pts.points)
        self.LOGIC_PCD = o3d.geometry.PointCloud()

    def transform_from_TF_Matrix(self, T_MATRIX=np.eye(4)):
        self.points = np.insert(self.points, 0, np.array([0,0,0]),axis=0)
        self.o3d_pts.points = o3d.utility.Vector3dVector(self.points[:,:3])
        self.o3d_pts.transform(T_MATRIX)
        self.points = np.asarray(self.o3d_pts.points)

    def centerline_all_pts(self, center=np.array([0,0,0])):
        self.points = self.points[:,:3] - center
        # print(f"number of points: {self.points.shape}")

    def GlobalMap_to_2d_binary(self, center=np.array([0,0,0])):
        '''
        NOTE: Here is only for global map to 2d binary once. NO need in the loop function anymore.
        '''
        st = time.time()
        self.global_center = center
        self.centerline_all_pts(self.global_center)
        ## 1. save max min for range, refresh the range based on max and min
        self.max_xyz = np.array([max(abs(self.points[...,0])), \
                                 max(abs(self.points[...,1])), \
                                 max(abs(self.points[...,2]))])

        self.range_m = max(self.max_xyz[0], self.max_xyz[1]) * 2
        self.dim_2d = (int)(self.range_m/self.resolution)

        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)

        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points[...,:2],self.resolution)).astype(int) + self.dim_2d//2
        self.idz = (np.divide(self.points[...,2], self.h_res)).astype(int) + self.idz_offset

        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            idx = ptidxy[0]
            idy = ptidxy[1]
            if idx < self.dim_2d and idy<self.dim_2d and idx>=0 and idy>=0:
                self.M_2d[idx][idy] = 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[idx][idy].append(i)
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if not(self.idz[i]>62 or self.idz[i]<0):
                    self.binary_2d[idx][idy] = self.binary_2d[idx][idy] | (1<<(self.idz[i]).astype(int))
        print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("Global Map Stack to 2D", ((time.time() - st))*1000))

    def Generate_mat_tree(self, center=np.array([0,0,0])):
        '''
        NOTE: Here is only for global map to 2d binary once. 
              NO need in the loop function anymore.
            This function will change the variance in the class like: dim_2d, range_m etc we may set before.
        '''
        st = time.time()
        self.global_center = center
        self.centerline_all_pts(self.global_center)
        ## 1. save max min for range, refresh the range based on max and min
        self.max_xyz = np.array([max(abs(self.points[...,0])), \
                                 max(abs(self.points[...,1])), \
                                 max(abs(self.points[...,2]))])

        self.range_m = max(self.max_xyz[0], self.max_xyz[1]) * 2
        self.dim_2d = (int)(self.range_m/self.resolution)

        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)

        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points[...,:2],self.resolution)).astype(int) + self.dim_2d//2
        idz = (np.divide(self.points[...,2], self.h_res)).astype(int) + self.idz_offset

        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            idx = ptidxy[0]
            idy = ptidxy[1]
            if idx < self.dim_2d and idy<self.dim_2d and idx>=0 and idy>=0:
                self.M_2d[idx][idy] = 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[idx][idy].append(i)
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if idz[i]<=62 and idz[i]>=0:
                    self.binary_2d[idx][idy] = self.binary_2d[idx][idy] | (1<<(idz[i]).astype(int))
                    self.threeD2ptindex[idx][idy][idz[i]].append(i)
        print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("Global Map Stack to 3D", ((time.time() - st))*1000))

    def SelectMap_based_on_Query_center(self, min_i_map, max_i_map, min_j_map, max_j_map):
        '''
        input: q_dim means query frame dimension, it should be different value
        output: 2d dict but save the points' index in the 2d grid.
        '''
        # st = time.time()
        self.bqc_binary_2d = self.binary_2d[min_i_map:max_i_map, min_j_map:max_j_map]
        # print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("Select QRoI", ((time.time() - st))*1000))

    def build_query_mat_tree_ref_select_roi(self, map_dim, center = np.array([0,0,0]), pose_center = np.array([0,0,0])):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        st = time.time()

        # Centerlize the pts
        self.centerline_all_pts(center)

        # Voxlise to grid index
        idxy = (np.divide(self.points[...,:2],self.resolution)).astype(int) + map_dim//2
        idz = (np.divide(self.points[...,2], self.h_res)).astype(int) + self.idz_offset
        
        M_2d = np.zeros((map_dim, map_dim))
        N_2d = np.zeros((map_dim, map_dim))
        binary_2d = np.zeros((map_dim, map_dim), dtype=int)

        pc_id = np.divide((pose_center - center)[...,:2], self.resolution).astype(int) + map_dim//2
        min_i_map = max(pc_id[0]-self.dim_2d//2, 0)
        max_i_map = min(pc_id[0]+self.dim_2d//2, map_dim - 1)
        min_j_map = max(pc_id[1]-self.dim_2d//2, 0)
        max_j_map = min(pc_id[1]+self.dim_2d//2, map_dim - 1)

        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if ptidxy[0] <= max_i_map and ptidxy[1] <= max_j_map \
           and ptidxy[0] >= min_i_map and ptidxy[1] >= min_j_map:
                M_2d[ptidxy[0]][ptidxy[1]] = 1
                N_2d[ptidxy[0]][ptidxy[1]] += 1
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if idz[i]<=62 and idz[i]>=0:
                    binary_2d[ptidxy[0]][ptidxy[1]] = binary_2d[ptidxy[0]][ptidxy[1]] | (1<<(idz[i]).astype(int))
                    self.threeD2ptindex[ptidxy[0]][ptidxy[1]][idz[i]].append(i)
        print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %("Stack to 2D", ((time.time() - st))*1000))
        
        self.M_2d = M_2d[min_i_map:max_i_map, min_j_map:max_j_map]
        self.N_2d = N_2d[min_i_map:max_i_map, min_j_map:max_j_map]
        self.binary_2d = binary_2d[min_i_map:max_i_map, min_j_map:max_j_map]

        return [min_i_map, max_i_map], [min_j_map, max_j_map]

    def exclusive_with_other_binary_2d(self, compare_b2):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        # compute the exclusive or
        return self.binary_2d ^ compare_b2
        
        
    def gmm_fit(self):
        Grids_Trigger = list(zip(*np.where(self.M_2d == 1)))
        TIC()
        for (i ,j) in Grids_Trigger:
            points_selected = self.select_data_from_2DptIndex(i,j)
            if points_selected.shape[0] > 2:
                self.gmmFitMatrix[i][j] = GaussianMixture(n_components=2, random_state=RANDOM_SEED).fit(points_selected[:,2].reshape(-1,1))
                self.bin_2d[i][j] = 1
        TOC("GMM Fits")
        return
    
    
    def gmm_fit(self):
        Grids_Trigger = list(zip(*np.where(self.M_2d == 1)))
        TIC()
        for (i ,j) in Grids_Trigger:
            points_selected = self.select_data_from_2DptIndex(i,j)
            if points_selected.shape[0] > 2:
                self.gmmFitMatrix[i][j] = GaussianMixture(n_components=2, random_state=RANDOM_SEED).fit(points_selected[:,2].reshape(-1,1))
                self.bin_2d[i][j] = 1
        TOC("GMM Fits")
        return
    
    def select_by_index(self, index_list, invert=False):
        return self.o3d_pts.select_by_index(index_list, invert=invert)

    def smoother(self):
        m,n = len(self.N_2d), len(self.N_2d[0])
        res = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                count = 0
                for x in range(i-1, i+2):
                    for y in range(j-1, j+2):
                        if 0 <= x < m and 0 <= y < n:
                            res[i][j] += self.N_2d[x][y]
                            count += 1
                res[i][j] //= count
        return np.asarray(res,dtype='int')

    def generate_range_mask(self, r):
        self.range_mask = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        i = j = (self.dim_2d - r)//2
        for x in range(i, i+r):
            for y in range(j, j+r):
                self.range_mask[x][y] = 1
        return self.range_mask

    def set_pts_morton_id(self, center):
            self.idx = np.asarray(self.points[:,0] / self.resolution,dtype='int')
            self.idy = np.asarray(self.points[:,1] / self.resolution,dtype='int')
            self.idz = np.asarray(self.points[:,2] / self.h_res,dtype='int')

            cidx, cidy, cidz = (center / [self.resolution, self.resolution, self.h_res]).astype(int)
            half_of_dim = self.dim_2d//2
            tmp  = np.where((self.idx < cidx + half_of_dim) * (self.idx > cidx - half_of_dim) * (self.idy < cidy + half_of_dim) * (self.idy > cidy - half_of_dim), 1, 0)
            self.idx = self.idx[np.where(tmp != 0)]
            self.idy = self.idy[np.where(tmp != 0)]
            self.idz = self.idz[np.where(tmp != 0)]
            self.points = self.points[np.where(tmp != 0)]
            self.idxyz = np.c_[self.idx, self.idy, self.idz]

            out = self.idx * 0

            for xx in [self.idz,self.idy,self.idx]:
                xx = (xx | (xx << 16)) & 0x030000FF
                xx = (xx | (xx <<  8)) & 0x0300F00F
                xx = (xx | (xx <<  4)) & 0x030C30C3
                xx = (xx | (xx <<  2)) & 0x09249249
                out |= xx
                out << 1
            # t = time.time()
            # tt = time.time()
            # print(1000*(tt-t))
            self.points_with_id = np.c_[self.points, out, self.idxyz]
            self.points_with_id = self.points_with_id[np.lexsort([self.idy, self.idx])]
            _, unique_id = np.unique(self.idxyz, return_index=True, axis=0)
            self.unique_pts_with_id = self.points_with_id[unique_id]

            print(self.points_with_id)

    def get_xyz_from_morton_id(self, input):
        x = input &        0x09249249
        y = (input >> 1) & 0x09249249
        z = (input >> 2) & 0x09249249

        x = ((x >> 2) | x) & 0x030C30C3
        x = ((x >> 4) | x) & 0x0300F00F
        x = ((x >> 8) | x) & 0x030000FF
        x = ((x >>16) | x) & 0x000003FF

        y = ((y >> 2) | y) & 0x030C30C3
        y = ((y >> 4) | y) & 0x0300F00F
        y = ((y >> 8) | y) & 0x030000FF
        y = ((y >>16) | y) & 0x000003FF

        z = ((z >> 2) | z) & 0x030C30C3
        z = ((z >> 4) | z) & 0x0300F00F
        z = ((z >> 8) | z) & 0x030000FF
        z = ((z >>16) | z) & 0x000003FF

        return x,y,z

    def from_morton_to_2d_binary(self, center):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''

        self.M_2d = np.zeros((self.dim_2d, self.dim_2d),dtype='int')
        self.N_2d = np.zeros((self.dim_2d, self.dim_2d),dtype='int')
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d),dtype='int')
        
        cidx, cidy, cidz = (center / [self.resolution, self.resolution, self.h_res]).astype(int)
        self.unique_pts_with_id[...,4] = self.unique_pts_with_id[...,4] - cidx + self.dim_2d//2
        self.unique_pts_with_id[...,5] = self.unique_pts_with_id[...,5] - cidx + self.dim_2d//2
        self.unique_pts_with_id[...,6] = self.unique_pts_with_id[...,6] - cidx + self.dim_2d//2
        

        for i in range(len(self.unique_pts_with_id)):
            tmpx = self.unique_pts_with_id[i][4].astype(int)
            tmpy = self.unique_pts_with_id[i][5].astype(int)
            tmpz = self.unique_pts_with_id[i][6].astype(int)

            if tmpx < self.dim_2d and tmpy<self.dim_2d and tmpy>=0 and tmpx>=0:
                self.M_2d[tmpx][tmpy] = 1
                # self.N_2d[self.unique_pts_with_id[i]][tmpy] += 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[tmpx][tmpy].append(i)
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if not(tmpz>62 or tmpz<0):
                    self.binary_2d[tmpx][tmpy] = self.binary_2d[tmpx][tmpy] | (1<<(tmpz).astype(int))

        # for i in range(self.dim_2d):
        #     for j in range(self.dim_2d):
        #         if(np.isin(i,self.idx_c) and np.isin(j,self.idy_c)):
        #             self.M_2d[i][j] = 1
        #             in_grid_pts_id = np.where(self.idx_c==i,1,0) * np.where(self.idy_c==j,1,0)
        #             self.N_2d[i][j] = np.sum(in_grid_pts_id)
        #             for k in range(63):
        #                 if(np.isin(k,self.idz_c)):
        #                     self.binary_2d[i][j] = self.binary_2d[i][j] | (1<<(self.idz_c[i]).astype(int))
        TOC("Stack to binary 2D") 

    def select_lowest_pts(self, min_i_map, min_j_map, id, center):
        lowest_pts_idx = []
        for (i,j) in tqdm(list(zip(*np.where(self.LOGICMat != 0))), desc=f"frame id {id}: grids traverse", ncols=80): # lowest 2 voxel has points
            all3d_indexs = self.binTo3id(self.LOGICMat[i][j])
            lowest_pts_idx += \
                SELECT_Ptsindex_from_matrix(all3d_indexs, self.threeD2ptindex, 
                                            i, j, min_i_map, min_j_map)
        lowest_pts = self.select_by_index(lowest_pts_idx)
        self.LOGIC_PCD = lowest_pts
        self.LOGIC_idx = lowest_pts_idx
        self.LOGICPts = lowest_pts.points - center

    def calculate_ground_distribution(self, Q_LOGICpts, T, min_i_map, max_i_map, min_j_map, max_j_map, map_dim, pose_center = np.array([0,0,0])):
        ## 1. save max min for range, refresh the range based on max and min
        min_z = ((np.log2(self.LOGICMat+1)-0.5).astype(int) - 2 - self.idz_offset) * self.h_res


        self.resolution = self.resolution / 2 # smaller the resolution
        self.dim_2d = (int)(self.range_m/(self.resolution*2)) # keep dim_2d not large
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)

        binary_2d = np.zeros((map_dim, map_dim), dtype=int)
        Q_LOGICpts_T = np.asarray(Q_LOGICpts)


        ## 2. Voxelize Query STACK TO 2D TMP
        idxy = (np.divide(Q_LOGICpts_T[...,:2],self.resolution)).astype(int) + map_dim//2
        idz = (np.divide(Q_LOGICpts_T[...,2], max(0.05, self.h_res / LOGIC_BIT_UB))).astype(int)
        adaptive_idz = (np.divide(Q_LOGICpts_T[...,2], self.h_res)).astype(int) # just initalize
        for i, ptidxy in enumerate(idxy):
            idx, idy = ptidxy[0], ptidxy[1]
            if idx <= max_i_map and idy<= max_j_map and idx>= min_i_map and idy>= min_j_map:
                adaptive_idz[i] = (idz[i] - np.divide(min_z[idx - min_i_map - 1][idy - min_j_map - 1], max(0.05, self.h_res / LOGIC_BIT_UB))).astype(int)
                if adaptive_idz[i]<=OPERATOR_BIT_UB and adaptive_idz[i]>=0:
                    T.threeD2ptindex[idx][idy][adaptive_idz[i]].append(i)
                    binary_2d[idx][idy] = binary_2d[idx][idy] | (1<<adaptive_idz[i])
        T.binary_2d = binary_2d[min_i_map:max_i_map, min_j_map:max_j_map]

        # BUG are between this one ===============>
        ## 3. Voxelize Map ROI STACK TO 2D self
        idxy = (np.divide(np.asarray(self.LOGICPts)[...,:2],self.resolution)).astype(int) + self.dim_2d//2
        idz = (np.divide(np.asarray(self.LOGICPts)[...,2], max(0.05, self.h_res / LOGIC_BIT_UB))).astype(int)
        adaptive_idz = (np.divide(np.asarray(self.LOGICPts)[...,2], self.h_res)).astype(int) # just initalize

        for i, ptidxy in enumerate(idxy):
            idx, idy = ptidxy[0], ptidxy[1]
            if idx < self.dim_2d and idy<self.dim_2d and idx>=0 and idy>=0:
                adaptive_idz[i] = (idz[i] - np.divide(min_z[idx][idy], max(0.05, self.h_res / LOGIC_BIT_UB))).astype(int)
                if adaptive_idz[i]<=OPERATOR_BIT_UB and adaptive_idz[i]>=0:
                    self.threeD2ptindex[idx][idy][adaptive_idz[i]].append(i)
                    self.binary_2d[idx][idy] = self.binary_2d[idx][idy] | (1<<adaptive_idz[i])
        # BUG are between this one ===============<
        return (T.binary_2d ^ self.binary_2d) & self.binary_2d # in fact this is the same as ~T.binary_2d & self.binary_2d

    @staticmethod
    def view_compare(inlier, outlier, others=None, view_file = None):
        view_things = [outlier]
        if others is not None:
            others.paint_uniform_color([0.0, 0.0, 0.0])
            view_things.append(others)
        inlier.paint_uniform_color([1.0, 0, 0])
        view_things.append(inlier)
        load_view_point(view_things, filename=view_file)
    @staticmethod
    def view(pts):
        o3d.visualization.draw_geometries([pts])

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