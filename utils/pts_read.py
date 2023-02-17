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

import open3d as o3d
import matplotlib.pyplot as plt
from . import load_view_point

import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from utils.global_def import *

from collections import defaultdict
class Points:
    def __init__(self, file, range_m, resolution, h_res=0.5):
        ## 0. Read Point Cloud
        TIC()
        self.points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        TOC("Numpy read pt")

        # parameters
        self.range_m = range_m * 2 # since range is half of dis
        self.resolution = resolution
        self.dim_2d = (int)(self.range_m/resolution)
        self.h_res = h_res

        # results
        self.twoD2ptindex = defaultdict(lambda  : defaultdict(list))
        self.gmmFitMatrix = defaultdict(lambda  : defaultdict(list))
        self.bin_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)

        # open3d
        self.o3d_pts = o3d.geometry.PointCloud()
        self.o3d_pts.points = o3d.utility.Vector3dVector(self.points[:,:3])
        self.points = np.asarray(self.o3d_pts.points)

    def clear_result(self, id=0):
        print(f"Clear result =============> id: {id} ===============>")
        # results
        self.twoD2ptindex = defaultdict(lambda  : defaultdict(list))
        self.gmmFitMatrix = defaultdict(lambda  : defaultdict(list))
        self.bin_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)
        self.points = np.asarray(self.o3d_pts.points)

    def transform_from_TF_Matrix(self, T_MATRIX=np.eye(4)):
        self.points = np.insert(self.points, 0, np.array([0,0,0]),axis=0)
        self.o3d_pts.points = o3d.utility.Vector3dVector(self.points[:,:3])
        self.o3d_pts.transform(T_MATRIX)
        self.points = np.asarray(self.o3d_pts.points)

    def centerlise_all_pts(self, center=np.array([0,0,0])):
        self.points = self.points[:,:3] - center
        print(f"number of points: {self.points.shape}")

    def GlobalMap_to_2d_binary(self, center=np.array([0,0,0])):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        self.points = self.points[:,:3] - center
        ## 1. save max min for range, refresh the range based on max and min
        self.max_xyz = np.array([max(abs(self.points[...,0])), max(abs(self.points[...,1])), max(abs(self.points[...,2]))])
        # self.min_xyz = np.array([min(abs(self.points[...,0])), min(abs(self.points[...,1])), min(self.points[...,2])])
        self.range_m = max(self.max_xyz[0], self.max_xyz[1]) * 2
        self.dim_2d = (int)(self.range_m/self.resolution)

        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        self.binary_2d = np.zeros((self.dim_2d, self.dim_2d), dtype=int)

        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points,self.resolution) + (self.range_m/self.resolution)/2).astype(int)
        # HARD CODE HERE!! + 5
        self.idz = (np.divide(self.points[...,2], self.h_res)).astype(int) + 5

        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if self.idx[i] < self.dim_2d and self.idy[i]<self.dim_2d and self.idy[i]>=0 and self.idx[i]>=0:
                self.M_2d[self.idx[i]][self.idy[i]] = 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[self.idx[i]][self.idy[i]].append(i)
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if not(self.idz[i]>62 or self.idz[i]<0):
                    self.binary_2d[self.idx[i]][self.idy[i]] = self.binary_2d[self.idx[i]][self.idy[i]] | (1<<(self.idz[i]).astype(int))
        TOC("Global Map Stack to 2D")

    def SelectMap_based_on_Query_center(self, q_dim, center=np.array([0,0,0])):
        '''
        input: q_dim means query frame dimension, it should be different value
        output: 2d dict but save the points' index in the 2d grid.
        '''
        self.center_xy_id =(np.divide(center,self.resolution) + (self.range_m/self.resolution)/2).astype(int)[:2]
        # bqc: based on Query center, maybe BUG HERE need padding if exceed the max range
        self.bqc_binary_2d = np.zeros((q_dim, q_dim), dtype=int)
        self.bqc_binary_2d = self.binary_2d[
        (self.center_xy_id[0]-q_dim//2):(self.center_xy_id[0]+q_dim//2),
        (self.center_xy_id[1]-q_dim//2):(self.center_xy_id[1]+q_dim//2)
        ]
        TOC("Select QRoI")

    def from_center_to_2d_grid(self, center=np.array([0,0,0])):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points - center,self.resolution) + (self.range_m/self.resolution)/2).astype(int)

        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        
        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if self.idx[i] < self.dim_2d and self.idy[i]<self.dim_2d and self.idy[i]>=0 and self.idx[i]>=0:
                self.M_2d[self.idx[i]][self.idy[i]] = 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[self.idx[i]][self.idy[i]].append(i)
        TOC("Stack to 2D grid")

    def from_center_to_2d_binary(self, center=np.array([0,0,0])):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points - center,self.resolution) + (self.range_m/self.resolution)/2).astype(int)
        # HARD CODE HERE!! + 5
        self.idz = (np.divide(self.points[...,2] - center[2], self.h_res)).astype(int) + 5
        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        self.N_2d = np.zeros((self.dim_2d, self.dim_2d))
        
        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if self.idx[i] < self.dim_2d and self.idy[i]<self.dim_2d and self.idy[i]>=0 and self.idx[i]>=0:
                self.M_2d[self.idx[i]][self.idy[i]] = 1
                self.N_2d[self.idx[i]][self.idy[i]] += 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[self.idx[i]][self.idy[i]].append(i)
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if not(self.idz[i]>62 or self.idz[i]<0):
                    self.binary_2d[self.idx[i]][self.idy[i]] = self.binary_2d[self.idx[i]][self.idy[i]] | (1<<(self.idz[i]).astype(int))
        TOC("Stack to binary 2D")

    def select_data_from_2DptIndex(self, i, j):
        return self.points[self.twoD2ptindex[i][j]]

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