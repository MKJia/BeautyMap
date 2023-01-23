# Created: 2023-1-15 12:14
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import open3d as o3d
import numpy as np

from collections import defaultdict
from sklearn.mixture import GaussianMixture

from utils import load_view_point, save_view_point, bresenham, quat2mat
from utils.global_def import *
from scipy.special import rel_entr

resolution = 1 # default 16-LiDAR, 1 meter
range_m = 10 # default 16-LiDAR, 20 meter
RANDOM_SEED = 10
X_AXIS = 0 # ground plane x
Y_AXIS = 1 # ground plane y
Z_AXIS = 2
class o3d_point:
    def __init__(self,points):
        self.pts = o3d.geometry.PointCloud()
        self.pts.points = o3d.utility.Vector3dVector(points[:,:3])
    
    def transform(self, T_MATRIX):
        self.pts.transform(T_MATRIX)
    def removeGround(self):
        # 1. Remove the ground points (TODO better ground remove!!! HERE TODO)
        plane_model, inliers = self.pts.segment_plane(distance_threshold=0.1,
                                                ransac_n=16,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = self.pts.select_by_index(inliers)
        self.rmg_pts = self.pts.select_by_index(inliers, invert=True)
        # save_view_point([inlier_cloud.paint_uniform_color([1.0, 0, 0]), self.rmg_pts], "data/TPB.json")
        load_view_point([inlier_cloud.paint_uniform_color([1.0, 0, 0]), self.rmg_pts], "data/TPB.json")
        
        TOC("REMOVE GROUND")
    def view_compare(self, inlier, outlier):
        inlier.paint_uniform_color([1.0, 0, 0])
        load_view_point([inlier, outlier], "data/TPB.json")
    def view(self,pts):
        o3d.visualization.draw_geometries([pts])
        
class process_pts:
    def __init__(self, file, range_m, resolution, T_MATRIX=np.eye(4), sh=0.8):
        self.MAP_FLAG = False
        ## 0. Read Point Cloud
        TIC()
        self.points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        # point the center points
        np.insert(self.points, 0, np.array([0,0,0,-1]),axis=0)
        TOC("Numpy read pt")

        self.o3d_pts = o3d_point(self.points)
        self.o3d_pts.transform(T_MATRIX)
        self.o3d_pts.removeGround()
        self.rmg_pts = np.asarray(self.o3d_pts.rmg_pts.points)
        self.range_m = range_m
        self.resolution = resolution
        self.dim_2d = (int)(range_m/resolution)
        self.twoD2ptindex = defaultdict(lambda  : defaultdict(list))
        self.binT_2d = np.zeros((self.dim_2d, self.dim_2d))
        TOC("Initial steps")

    def all_process(self, center):

        ## 2. Voxelize STACK TO 2D
        idxy = (np.divide(self.rmg_pts[...,[X_AXIS,Y_AXIS]] - center[[X_AXIS,Y_AXIS]],self.resolution) + (self.range_m/self.resolution)/2).astype(int)

        M_2d = np.zeros((self.dim_2d, self.dim_2d))
        
        self.pts1idxy = []
        for i, ptidxy in enumerate(idxy):
            if ptidxy[0] < self.dim_2d and ptidxy[1]<self.dim_2d and ptidxy[1]>0 and ptidxy[0]>0:
                M_2d[ptidxy[0]][ptidxy[1]] = M_2d[ptidxy[0]][ptidxy[1]] + 1
                pt1xy = f"{ptidxy[0]}.{ptidxy[1]}"
                if pt1xy not in self.pts1idxy:
                    self.pts1idxy.append(pt1xy)
                    self.binT_2d[ptidxy[0]][ptidxy[1]] = 1
                self.twoD2ptindex[ptidxy[0]][ptidxy[1]].append(i)
        TOC("Stack to 2D")

    def rayT_2d(self):
        ## 4. Ray Tracking in M_2d
        self.RayT_2d = np.zeros((self.dim_2d, self.dim_2d))
        for eid in self.pts1idxy:
            x1 = int(eid.split('.')[0])
            y1 = int(eid.split('.')[1])
            grid2one =bresenham(self.dim_2d//2,self.dim_2d//2, x1,y1)
            for sidxy in grid2one:
                self.RayT_2d[sidxy[0]][sidxy[1]] = 1
        TOC("Ray Casting")
        return self.RayT_2d

def calGMM(data):
    
    if len(data)<=1:
        gm= GaussianMixture(n_components=2, random_state=RANDOM_SEED).fit((-1*np.ones((100,1))).reshape(-1,1))
    else:
        # data = np.repeat(data, 2).reshape(-1,1)
        gm = GaussianMixture(n_components=2, random_state=RANDOM_SEED).fit(data)
    return gm

# reference: https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms?noredirect=1&lq=1
# TODO sample again not smart way? is there any continues way to do that?
def gmm_kl(gmm_p, gmm_q, n_samples=10**3):
    X = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X[0])
    log_q_X = gmm_q.score_samples(X[0])
    return log_p_X.mean() - log_q_X.mean()

## 5. Times Map and Query Matrix
import pandas as pd
df = pd.read_csv('data/TPB_poses_lidar2body.csv')
pose = df.values[100][2:]
wxyz = np.array([pose[6],pose[3],pose[4],pose[5]])
T_Q = np.eye(4)
T_Q[:3,:3] = quat2mat(wxyz)
T_Q[:3,-1]= np.array([pose[0],pose[1],pose[2]])

Query_ = process_pts("data/bin/TPB_000100.bin", range_m, resolution, T_MATRIX=T_Q)
PrMap_ = process_pts("data/bin/TPB_global_map.bin", range_m, resolution)

# debug: make sure the aglignment is corrected
# Query_.o3d_pts.view_compare(Query_.o3d_pts.pts, PrMap_.o3d_pts.pts)

Query_.all_process(Query_.points[0,:])
PrMap_.all_process(Query_.points[0,:])

Query2d = Query_.rayT_2d()
KL_Matrix = np.zeros((Query_.dim_2d, Query_.dim_2d))
## 6. The grid have one calculate the KL Diversity
Grids_Trigger_KL = list(zip(*np.where(Query2d.dot(PrMap_.binT_2d) == 1)))
for item in Grids_Trigger_KL:
    Q_pts_id = Query_.twoD2ptindex[item[0]][item[1]]
    M_pts_id = PrMap_.twoD2ptindex[item[0]][item[1]]
    Q_Prob = calGMM(Query_.points[Q_pts_id][:,Z_AXIS].reshape(-1,1)) # Q_Prob: u1, u2, s1, s2 which is 2 gaussian
    M_Prob = calGMM(PrMap_.points[M_pts_id][:,Z_AXIS].reshape(-1,1))
    KL_Matrix[item[0]][item[1]] = gmm_kl(Q_Prob, M_Prob)

if len(Grids_Trigger_KL)!=0:
    # TODO: view the Matrix with the pts also?
    ## 7. Difference Show to view the distribution with heatmap
    import matplotlib.pyplot as plt
    plt.imshow(KL_Matrix, cmap='hot', interpolation='nearest')
    plt.show()

    ## 8. remove the xy bins output the Global Map
    # How to set the threshold??
    Grids_Trigger_KL2pt = list(zip(*np.where(KL_Matrix > 5)))
    points_index2Remove = []
    for item in Grids_Trigger_KL2pt:
        Remove_ptid = PrMap_.twoD2ptindex[item[0]][item[1]]
        points_index2Remove = points_index2Remove + Remove_ptid

    inlier_cloud = PrMap_.o3d_pts.rmg_pts.select_by_index(points_index2Remove)
    oulier_cloud = PrMap_.o3d_pts.rmg_pts.select_by_index(points_index2Remove, invert=True)
    PrMap_.o3d_pts.view_compare(inlier_cloud, oulier_cloud)

print("All success")