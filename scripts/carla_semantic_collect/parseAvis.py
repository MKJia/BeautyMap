import numpy as np
import carla
import copy
import cv2

# modify from world on rail code
def lidar_to_bev(lidar, min_x=-24,max_x=24,min_y=-16,max_y=16, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1,:]

# modify from manual control
def parse_image_cb(type_name, image):
    if type_name.split('_')[-1]=="depth":
        image.convert(carla.ColorConverter.LogarithmicDepth)
    # log.info(type_name)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

# reference code: https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/open3d_lidar.py
def parse_lidar_cb(lidar_data):
    # points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    data = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T
    # points = np.array([data['x'], data['y'], data['z']]).T
    points = np.append(points,np.zeros((points.shape[0],1)),axis=1)
    # Colorize the pointcloud based on the CityScapes color palette
    instances = np.array(data['ObjIdx']).astype(np.uint32)
    labels = ((instances<<16) + np.array(data['ObjTag']).astype(np.int16)).astype(np.int32)
    
    # print(points.shape)
    # points = copy.deepcopy(points)
    # points = np.reshape(points, (int(points.shape[0] / 3), 3))
    return points, labels

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))

# modify from world on rail code
def visualize_data(rgb, lidar, imu_yaw=None, gnss=None, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):

    canvas = np.array(rgb[...,::-1])

    if lidar is not None:
        lidar[:,1] = -lidar[:,1]
        lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz,cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([canvas, cv2.resize(lidar_viz.astype(np.uint8), (canvas.shape[0], canvas.shape[0]))], axis=1)

    # cv2.putText(canvas, f'yaw angle: {imu_yaw:.3f}', (4, 10), *text_args)
    # cv2.putText(canvas, f'log: {gnss[0]:.3f} alt: {gnss[1]:.3f} brake: {gnss[2]:.3f}', (4, 20), *text_args)

    return canvas
    
# Following code from: https://github.com/matthew-brett/transforms3d/blob/76213cc352/transforms3d/euler.py
import math
import numpy as np
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0
import numpy as np
def euler2quat(ai, aj, ak, axes='sxyz'):
    """Return `quaternion` from Euler angles and axis sequence `axes`
    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format
    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q