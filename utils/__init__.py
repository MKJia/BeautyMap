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

# https://github.com/openai/mujoco-worldgen/blob/master/mujoco_worldgen/util/rotation.py
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def RayOutside(x0, y0, x1, y1):
    dim = int(x0*2)
    rayLists = []
    deltaY = y1-y0
    deltaX = x1-x0

    if deltaX==0 or deltaY==0:
        print("line")
    else:
        xi = x1
        yi = y1
        slope = deltaY/deltaX
        # 第一 三象限
        if slope>0:
            minusO = -1 if deltaX<0 else 1 # 第三象限  x y 都需要相减
            if slope>1: # y 增长速度更快
                while 0<yi<dim:
                    rayLists.append([int(xi),int(yi)])
                    yi = yi + 1*minusO
                    xi = x0 + (yi-y0)/slope
                    # yi = slope*(xi-x0) + y0
            else:
                while 0<xi<dim:
                    rayLists.append([int(xi),int(yi)])
                    xi = xi + 1*minusO
                    yi = slope*(xi-x0) + y0
        # 第二 四象限
        else:
            minusY = 1 if deltaX<0 else -1 # y+1
            minusX = 1 if deltaY<0 else -1 # x-1
            if abs(slope)>1: # y 增长速度更快
                while 0<yi<dim:
                    rayLists.append([int(xi),int(yi)])
                    yi = yi + 1*minusY
                    xi = x0 + (yi-y0)/slope
            else:
                while 0<xi<dim:
                    rayLists.append([int(xi),int(yi)])
                    xi = xi + 1*minusX
                    yi = slope*(xi-x0) + y0
    return rayLists