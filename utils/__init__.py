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

# https://github.com/daQuincy/Bresenham-Algorithm/blob/master/bresenham.py
def bresenham(x0,y0,x1,y1):
    """
    Bresenham's Line Generation Algorithm
    https://www.youtube.com/watch?v=76gp2IAazV4
    """
    line_pixel = []
    line_pixel.append((x0,y0))

    # step 2 calculate difference
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    if dx==0:
        for i in range(dy):
            line_pixel.append([x0,y0+i])
        return line_pixel
    m = dy/dx
    
    # step 3 perform test to check if pk < 0
    flag = True
    
    step = 1
    if x0>x1 or y0>y1:
        step = -1

    mm = False   
    if m < 1:
        x0, x1 ,y0 ,y1 = y0, y1, x0, x1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        mm = True
        
    p0 = 2*dx - dy
    x = x0
    y = y0
    
    for i in range(abs(y1-y0)):
        if flag:
            x_previous = x0
            p_previous = p0
            p = p0
            flag = False
        else:
            x_previous = x
            p_previous = p
            
        if p >= 0:
            x = x + step

        p = p_previous + 2*dx -2*dy*(abs(x-x_previous))
        y = y + 1
        
        if mm:
            line_pixel.append([y,x])
        else:
            line_pixel.append([x,y])
            
    line_pixel = np.array(line_pixel).astype(int)
    
    return line_pixel