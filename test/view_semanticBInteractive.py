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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from utils.global_def import *
from utils.o3d_view import ViewControl
# ========> CONFIG HERE
DataFolder = f"{BASE_DIR}/data/test/teaser" # make sure the subfolder is labels and points
# each points need have `.label` file to match
MaxFramePlay = 100 # if you want to play all of them, set -1

CFG = yaml.safe_load(open(f"{BASE_DIR}/test/semantic_label/semantic-label.yaml", 'r'))
# <======== CONFIG HERE

if __name__ == "__main__":
    pts_folder = f"{DataFolder}/points"
    label_folder = f"{DataFolder}/labels"
    pts_files = sorted(os.listdir(pts_folder))

    # view
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pc = o3d.geometry.PointCloud()
    viz = o3d.visualization.VisualizerWithKeyCallback()
    # 2. create
    viz.create_window(window_name="view semantic kitti, press v for next frame", width=960,height=540)
    # 3. add geometry
    viz.add_geometry(pc)
    # 4. get control !!! must step by step
    ctr = viz.get_view_control()
    view_setting_file = f'{BASE_DIR}/data/o3d_view/kitti_view.json'
    o3d_vctrl = ViewControl(ctr, view_setting_file)

    N = len(pts_files)

    class thread:
        def __init__(self):
            self.id = 0
        def next_frame(self,viz):
            pts_file = pts_files[self.id]
            label_file = pts_file.split('.')[0]+'.label'
            points = np.fromfile(f"{pts_folder}/{pts_file}", dtype=np.float32).reshape(-1, 4)[:,:3]
            labels = np.fromfile(f"{label_folder}/{label_file}", dtype=np.uint32) & 0xFFFF # since we only care about semantic (lower 16 for semantic, higher 16 for instance)
            colors = []
            for i in range(len(labels)):
                colors.append(list(np.array(CFG["color_map"][labels[i]])/255))

            pc.points=o3d.utility.Vector3dVector(points)
            pc.colors=o3d.utility.Vector3dVector(colors)
            viz.add_geometry(pc)
            # need set view again, check this issue: https://github.com/isl-org/Open3D/issues/2219
            o3d_vctrl.read_viewTfile(view_setting_file)
            viz.update_geometry(pc)
            # viz.update_renderer()
            print(f"id: {self.id}, next frame..")
            self.id = self.id+1

    myThread = thread()
    myThread.next_frame(viz)
    # 9 is `Tab` keyboard
    viz.register_key_callback(86, myThread.next_frame)
    viz.run()
    viz.destroy_window()
    print("All codes run successfully, Close now..")