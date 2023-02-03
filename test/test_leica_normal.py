# Created: 2023-2-2 11:20
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import matplotlib.pyplot as plt

import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from utils import RayOutside, quat2mat
from utils.global_def import *
import pandas as pd

# df = pd.read_csv('data/TPB_poses_lidar2body.csv')
# pose = df.values[100][2:]
# wxyz = np.array([pose[6],pose[3],pose[4],pose[5]])
# T_Q = np.eye(4)
# T_Q[:3,:3] = quat2mat(wxyz)
# T_Q[:3,-1]= np.array([pose[0],pose[1],pose[2]])

Query_ = process_pts("data/bin/TPB_000100.bin", range_m, resolution, T_MATRIX=T_Q)
PrMap_ = process_pts("data/bin/TPB_global_map.bin", range_m, resolution)
