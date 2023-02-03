import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from global_def import *
from . import load_view_point

class Points:
    def __init__(self, file):
        self.MAP_FLAG = False
        ## 0. Read Point Cloud
        TIC()
        self.points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        TOC("Numpy read pt")

        self.o3d_pts = o3d.geometry.PointCloud()
        self.o3d_pts.points = o3d.utility.Vector3dVector(self.points[:,:3])
        TOC("Initial steps")

    def view_compare(self, inlier, outlier, others=None):
        view_things = [outlier]
        if others is not None:
            others.paint_uniform_color([0.0, 0.0, 0.0])
            view_things.append(others)
        inlier.paint_uniform_color([1.0, 0, 0])
        view_things.append(inlier)
        load_view_point(view_things, "data/o3d_view/TPB.json")
        
    def view(self,pts):
        o3d.visualization.draw_geometries([pts])