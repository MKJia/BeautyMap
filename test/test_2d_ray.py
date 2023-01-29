import numpy as np
import matplotlib.pyplot as plt

import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from utils import RayOutside
from utils.global_def import *

T_map = np.array([
[0,0,0,0,1, 1,1,0,0,0,0,0,0,0,1, 1,1,0,0,0],
[0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0, 0,0,1,0,0],
[0,0,0,1,0, 0,0,0,1,0,0,0,0,1,0, 0,0,0,1,0],
[0,1,0,0,0, 0,0,0,0,1,0,1,0,0,0, 0,0,0,0,1],
[1,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,1],
[1,0,0,0,0, 0,0,0,0,0,1,0,0,0,0, 0,0,0,1,0],
[0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,1,0,0],
[0,0,0,1,0, 0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0],
[0,0,0,1,0, 0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0],
[0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
[0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0, 1,1,0,0,0],
[0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,1,0,0],
[0,0,0,1,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,1,0],
[0,1,0,0,0, 0,0,0,0,0,0,1,0,0,0, 0,0,0,0,1],
[1,0,0,0,0, 0,0,0,0,0,1,0,0,0,0, 0,0,0,0,1],
[1,0,0,0,0, 0,0,0,1,0,1,0,0,0,0, 1,0,0,1,0],
[0,1,0,0,0, 0,0,0,0,0,0,1,0,0,0, 0,0,1,0,0],
[0,0,0,1,0, 0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0],
[0,0,0,1,1, 1,0,0,0,0,0,0,0,1,0, 1,0,0,0,0],
[0,0,0,0,1, 0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0],
],dtype =np.float32)
# T_map = np.array([
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 1,1,1,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
# ],dtype =np.float32)
def rayT_2d(TMap, dim_2d=20):
    ## 4. Ray Tracking in M_2d
    RayT_2d = np.ones((dim_2d, dim_2d))
    for x1,y1 in list(zip(*np.where(TMap == 1))):
        grid2one =RayOutside(dim_2d//2,dim_2d//2, x1,y1)
        for sidxy in grid2one:
            if sidxy[0]>=dim_2d or sidxy[1]>=dim_2d:
                print(f"exceed the max dim: end: {x1}, {y1}, through: {sidxy[0]}, {sidxy[1]}")
                continue
            RayT_2d[sidxy[0]][sidxy[1]] = 0
    return RayT_2d
# T_map[11][10] = 1.0
Query2d = rayT_2d(T_map)
T_map[10][10] = 0.5
Query2d[10][10] = 0.5

fig, axs = plt.subplots(1, 2, figsize=(8,8))
axs[0].imshow(T_map)
axs[0].set_title('origin map')
axs[1].imshow(Query2d)
axs[1].set_title('After Ray outside')
plt.show()