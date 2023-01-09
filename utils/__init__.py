# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/visualization/load_save_viewpoint.py

import numpy as np
import open3d as o3d


def save_view_point(viewThings: list, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for sth in viewThings:
        vis.add_geometry(sth)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(viewThings: list, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    for sth in viewThings:
        vis.add_geometry(sth)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()
