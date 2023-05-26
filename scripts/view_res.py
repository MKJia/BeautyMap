# Created: 2023-05-11 22:35
# @Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# @Author: Kin ZHANG  (https://kin-zhang.github.io/)

# If you find this repo helpful, please cite the respective publication in DUFOMap.
# This script is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import pyvista as pv
def view_pcd2img(pcd_np_data, show_false_diff=False, show_pred_dynamic=False):
    label_colors = {0: [145, 191, 219], # blue static TN
                    1: [239, 133, 84],   # orange FN
                    2: [255, 255, 191], # yellow TP
                    3: [240, 59, 32]   # red FP
                    }
    plotter = pv.Plotter()
    plotter.enable_eye_dome_lighting()
    plotter.enable_trackball_style()
    plotter.set_background([236, 235, 235])
    # if show_dynamic:
    cloud = pv.PolyData(pcd_np_data[:,:3])
    labels = pcd_np_data[:,3]
    show_label = [0,1,2,3]
    
    if show_pred_dynamic and not show_false_diff:
        labels[labels==3] = 0
        labels[labels==1] = 2
        show_label = [0,2]
    elif show_pred_dynamic and show_false_diff: # show output static
        labels[labels==3] = 0
        show_label = [0,1,2]
    elif not show_pred_dynamic and show_false_diff:
        show_label = [0,3]
    elif not show_pred_dynamic and not show_false_diff: # show output static with single color (blue)
        labels[labels==3] = 0
        show_label = [0]

    cloud["labels"] = labels.flatten()

    for label, color in label_colors.items():
        if label not in show_label:
            continue
        draw_pt = cloud.extract_points(cloud["labels"] == label)
        if draw_pt.n_points>0:
            plotter.add_mesh(draw_pt, point_size=5, color=color)
    plotter.show()

import sys, os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from utils.pcdpy3 import load_pcd
from utils import check_file_exists
Result_Folder = "/home/kin/workspace/DUFOMap/data" 
data_name = "00"
gt_pcd_path = f"{Result_Folder}/{data_name}/gt_cloud.pcd"
check_file_exists(gt_pcd_path)
gt_pc_ = load_pcd(gt_pcd_path)

et_pcd_path = f"{Result_Folder}/{data_name}/eval/edomap_output_exportGT.pcd"
check_file_exists(et_pcd_path)
et_pc_ = load_pcd(et_pcd_path)
et_pc_.np_data[(et_pc_.np_data[:,3] == 0) * (gt_pc_.np_data[:,3] == 0), 3]=0 # TN
et_pc_.np_data[(et_pc_.np_data[:,3] == 1) * (gt_pc_.np_data[:,3] == 0), 3]=1 # FN
et_pc_.np_data[(et_pc_.np_data[:,3] == 1) * (gt_pc_.np_data[:,3] == 1), 3]=2 # TP
et_pc_.np_data[(et_pc_.np_data[:,3] == 0) * (gt_pc_.np_data[:,3] == 1), 3]=3 # FP
view_pcd2img(et_pc_.np_data, show_pred_dynamic=False, show_false_diff=True)