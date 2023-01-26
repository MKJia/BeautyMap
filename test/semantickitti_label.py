# Created: 2023-1-25 22:57
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import open3d as o3d
import numpy as np
import yaml
# make sure relative package, check issue here: https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
# print(sys.path)

from utils.global_def import *
from utils.o3d_view import ViewControl
points = np.fromfile(f"{BASE_DIR}/data/test/000000.bin", dtype=np.float32).reshape(-1, 4)[:,:3]
labels = np.fromfile(f"{BASE_DIR}/data/test/000000.label", dtype=np.uint32) & 0xFFFF # since we only care about semantic (lower 16 for semantic, higher 16 for instance)
CFG = yaml.safe_load(open(f"{BASE_DIR}/test/semantic_label/semantic-label.yaml", 'r'))

pc = o3d.geometry.PointCloud()
colors = []
for i in range(len(labels)):
    # please check semantic-kitti-mos.yaml for more segmentic label
    # 40:road , 72: terrain, 60: lane-marking, 48: sidewalk, 49: other-ground
    # 52:other-structure, 70 是vegatation 也是有一些土壤点的
    # 40,72,60,48,49,52,44
    # colors.append(list(np.array(CFG["color_map"][labels[i]])/255))
    # for ground segmentation:
    if labels[i] in [40,72,60,48,49,44]:
        colors.append([0,1,0])
    else:
        colors.append([0,0,0])

pc.points=o3d.utility.Vector3dVector(np.asarray(points))
pc.colors=o3d.utility.Vector3dVector(colors)
# pc.paint_uniform_color(colors)

# 1. define
viz = o3d.visualization.VisualizerWithVertexSelection()
# 2. create
viz.create_window()
# 3. add geometry
viz.add_geometry(pc)
# 4. get control !!! must step by step
ctr = viz.get_view_control()

o3d_vctrl = ViewControl(ctr, f'{BASE_DIR}/data/o3d_view/kitti_view.json')

viz.run()
viz.destroy_window()
print("\033[92mAll codes run successfully, Close now..\033[0m See you!")