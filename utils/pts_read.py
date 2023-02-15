# math
import numpy as np
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
        self.range_m = range_m
        self.resolution = resolution
        self.dim_2d = (int)(range_m/resolution)
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
        self.o3d_pts.transform(T_MATRIX)
        self.points = np.asarray(self.o3d_pts.points)

    def centerlise_all_pts(self, center=np.array([0,0,0])):
        self.points = self.points[:,:3] - center
        print(f"number of points: {self.points.shape}")

    def from_center_to_2d_grid(self, center=np.array([0,0,0])):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points - center,self.resolution) + (self.range_m/self.resolution)/2).astype(int)

        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        
        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if ptidxy[0] < self.dim_2d and ptidxy[1]<self.dim_2d and ptidxy[1]>=0 and ptidxy[0]>=0:
                self.M_2d[ptidxy[0]][ptidxy[1]] = 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[ptidxy[0]][ptidxy[1]].append(i)
        TOC("Stack to 2D")

    def from_center_to_2d_binary(self, center=np.array([0,0,0])):
        '''
        output: 2d dict but save the points' index in the 2d grid.
        '''
        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.points - center,self.resolution) + (self.range_m/self.resolution)/2).astype(int)
        self.idz = (np.divide(self.points[...,2] - center[2], self.h_res)).astype(int) + 5
        self.M_2d = np.zeros((self.dim_2d, self.dim_2d))
        self.N_2d = np.zeros((self.dim_2d, self.dim_2d))
        
        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if ptidxy[0] < self.dim_2d and ptidxy[1]<self.dim_2d and ptidxy[1]>=0 and ptidxy[0]>=0:
                self.M_2d[ptidxy[0]][ptidxy[1]] = 1
                self.N_2d[ptidxy[0]][ptidxy[1]] += 1
                # 500.167847 ms -> 643.996239 ms
                self.twoD2ptindex[ptidxy[0]][ptidxy[1]].append(i)
                # Give the bin based on the z axis, e.g. idz = 10 1<<10 to point out there is occupied
                if not(self.idz[i]>62 or self.idz[i]<0):
                    self.binary_2d[ptidxy[0]][ptidxy[1]] = self.binary_2d[ptidxy[0]][ptidxy[1]] | (1<<(self.idz[i]).astype(int))
        TOC("Stack to 2D")

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