import numpy as np
import pyvista as pv

import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)

from utils.pcdpy3 import load_pcd, save_pcd
from utils import check_file_exists


if __name__ == "__main__":
    if len(sys.argv) > 1:
        off_screen = True
    else:
        off_screen = False
    DATA_FOLDER = "/home/kin/data/Dynamic_Papers_assets/BeautyMap/05"
    METHOD_NAME = "dynablox"
    gt_pcd = load_pcd(f"{DATA_FOLDER}/gt_cloud.pcd")
    eval_pcd = load_pcd(f"{DATA_FOLDER}/eval/{METHOD_NAME}_output_exportGT.pcd")
    eval_pcd.np_data[(eval_pcd.np_data[:,3] == 0) * (gt_pcd.np_data[:,3] == 1), 3]=2 # FN
    cloud = pv.PolyData(eval_pcd.np_data[:,:3])
    labels = eval_pcd.np_data[:,3]

    # Create a PyVista point cloud
    cloud["labels"] = labels.flatten()

    # Color based on this label color
    label_colors = {0: [145, 191, 219], 2: [239, 133, 84], 1: [255, 255, 191]}  # RGB values of #91BFDB and #FFFFBF #ef8554

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background([236, 235, 235])
    plotter.enable_eye_dome_lighting()
    plotter.enable_trackball_style()

    camera_viewpoint_file = os.path.join(os.path.dirname( __file__ ), "camera_position_full.txt")
    if os.path.exists(camera_viewpoint_file):
        loaded_camera_position = []
        with open(camera_viewpoint_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                loaded_camera_position.append(eval(line.strip()))
            
        loaded_camera_position = tuple(loaded_camera_position)
        plotter.camera_position = loaded_camera_position

    # Set the color mapping for the labels
    for label, color in label_colors.items():
        if label==1:
            continue
        plotter.add_mesh(cloud.extract_points(cloud["labels"] == label), point_size=5, color=color)

    def save_callback():
        print("Camera position updated.")
        camera_position = plotter.camera_position
        # Store the camera position to a file for later use
        with open(camera_viewpoint_file, "w") as file:
            for position in camera_position:
                file.write(str(position) + "\n")
        return
    plotter.add_key_event("p", save_callback)
    plotter.show()
