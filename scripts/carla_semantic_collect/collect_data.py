
#!/usr/bin/env python
# Created: 2022-03-25 23:10
# Updated: 2023-1-13 19:23
# This Version: 2023-1-26 19:38
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# Part of these codes are from some reference please check comments

import glog as log
import open3d as o3d
import carla
import random, math
import numpy as np
from queue import Queue, Empty

# make sure relative package, check issue here: https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)

from utils.global_def import *
from parseAvis import *

random.seed(0)

# args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--host', metavar='H',    default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p',           default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port',              default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--ego-spawn', type=list, default=None, help='[x,y] in world coordinate')
parser.add_argument('--top-view',             default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map',                  default='Town04', help='Town Map')
parser.add_argument('--sync',                 default=True, help='Synchronous mode execution')
parser.add_argument('--sensor-h',             default=2.4, help='Sensor Height')
# 给绝对路径 记得改位置哦！
parser.add_argument('--save-path',            default='/home/kin/bags/hus_data/', help='Synchronous mode execution')
args = parser.parse_args()

# 图片大小可自行修改
IM_WIDTH = 1024
IM_HEIGHT = 768
update_hz = 10
actor_list, sensor_list = [], []

SENSOR_TYPES = ['rgb','lidar','depth', 'labels']

CV_VIEW_WINDOW = 2 # 1 small, 2 medium, 3 large
def main(args):
    poses2save = ""
    # We start creating the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    
    # world = client.get_world()
    world = client.load_world('Town01')
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    
    try:
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        # 手动规定
        # transform_vehicle = carla.Transform(carla.Location(0, 10, 0), carla.Rotation(0, 0, 0))
        # 自动选择
        transform_vehicle = carla.Transform(carla.Location(92.1, 37.1, 0.1), carla.Rotation(0,-90, -3.05))#random.choice(world.get_map().get_spawn_points())
        
        ego_vehicle_bp = random.choice(blueprint_library.filter("model3"))
        ego_vehicle_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform_vehicle)
        
        path2follow = [carla_map.get_waypoint(carla.Location(92.1, 37.1, 0.1)).transform.location,
                       carla_map.get_waypoint(carla.Location(134.6, 1.5, 0.1)).transform.location,
                       carla_map.get_waypoint(carla.Location(153.6, 38.5, 0.1)).transform.location,
                       carla_map.get_waypoint(carla.Location(136.6, 55.3, 0.1)).transform.location,
                       carla_map.get_waypoint(carla.Location(115.2, 55.4, 0.1)).transform.location,
                       carla_map.get_waypoint(carla.Location(92.1, 37.1, 0.1)).transform.location,
                       ]

        # 设置traffic manager
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        # 是否忽略红绿灯
        tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1-10%)=27km/h
        tm.global_percentage_speed_difference(-10.0)
        ego_vehicle.set_autopilot(True, tm.get_port())
        tm.set_path(ego_vehicle, path2follow)
        actor_list.append(ego_vehicle)

        #-------------------------- 进入传感器部分 --------------------------#
        sensor_queue = Queue()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_depth_bp = blueprint_library.find('sensor.camera.depth')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
        # lidar_bp.set_attribute('noise_stddev', '0.2')
        imu_bp = blueprint_library.find('sensor.other.imu')
        gnss_bp = blueprint_library.find('sensor.other.gnss')

        # 可以设置一些参数 set the attribute of camera
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "66.5")
        cam_depth_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_depth_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_depth_bp.set_attribute("fov", "66.5")
        # cam_bp.set_attribute('sensor_tick', '0.1')

        cam01 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h),carla.Rotation(yaw=0)), attach_to=ego_vehicle)
        cam01.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_front"))
        sensor_list.append(cam01)

        cam02 = world.spawn_actor(cam_depth_bp, carla.Transform(carla.Location(z=args.sensor_h),carla.Rotation(yaw=0)), attach_to=ego_vehicle)
        cam02.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_depth"))
        sensor_list.append(cam02)

        # set similar to kitti: https://www.cvlibs.net/datasets/kitti/setup.php
        # velodyne features: https://www.goetting-agv.com/components/hdl-64E
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '3000000') # around 2,200,000
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', str(int(1/settings.fixed_delta_seconds)*2)) #
        lidar_bp.set_attribute('upper_fov', str(2))
        lidar_bp.set_attribute('lower_fov', str(-24.9))
        lidar01 = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=args.sensor_h)), attach_to=ego_vehicle)
        lidar01.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))
        sensor_list.append(lidar01)
        #-------------------------- 传感器设置完毕 --------------------------#


        start_frame=0
        no_vel = True
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            v=ego_vehicle.get_velocity()
            if math.sqrt(v.x**2 + v.y**2 + v.z**2)<1 and start_frame==0:
                ego_vehicle.set_autopilot(True, tm.get_port())
                tm.set_path(ego_vehicle, path2follow)
            else:
                no_vel = False
            try:
                rgbs = []
                now_pose = None
                for i in range (0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        rgbs.append(parse_image_cb(s_name, s_data))
                    elif sensor_type == 'lidar':
                        lidar, labels = parse_lidar_cb(s_data)
                        s_lidar = s_data

                        # reference: https://github.com/carla-simulator/ros-bridge/blob/master/carla_common/src/carla_common/transforms.py
                        tf = ego_vehicle.get_transform()

                        # Considers the conversion from left-handed system (unreal) to right-handed system (ROS)
                        roll = math.radians(tf.rotation.roll)
                        pitch=-math.radians(tf.rotation.pitch)
                        yaw = -math.radians(tf.rotation.yaw)

                        quat = euler2quat(roll, pitch, yaw)
                        now_pose = "{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}\n".format(tf.location.x, -tf.location.y, tf.location.z, quat[0],quat[1],quat[2],quat[3])
                
                if no_vel or len(rgbs)<2:
                    continue
                # 仅用来可视化 可注释
                img_rgb = rgbs[0]
                img_dep = rgbs[1]
                rgb=np.concatenate(rgbs, axis=1)[...,:3]
                imS = cv2.resize(visualize_data(rgb, lidar), (int(704*CV_VIEW_WINDOW), int(192*CV_VIEW_WINDOW)))                # Resize image
                cv2.imshow('vizs', imS)
                cv2.waitKey(100)
                if now_pose is not None and (w_frame-start_frame) > update_hz:
                    # 检查是否有各自传感器的文件夹
                    mkdir_folder(args.save_path, SENSOR_TYPES)
                    frame_index_string = "{:06d}".format(w_frame)
                    filename = args.save_path +'rgb/'+str(frame_index_string)+'.png'
                    cv2.imwrite(filename, np.array(img_rgb[...,::-1]))
                    filename = args.save_path +'depth/'+str(frame_index_string)+'.png'
                    cv2.imwrite(filename, np.array(img_dep[...,::-1]))
                    filename = args.save_path +'lidar/'+str(frame_index_string)+'.bin'
                    lidar.astype('float32').tofile(filename)
                    filename = args.save_path +'labels/'+str(frame_index_string)+'.label'
                    labels.astype('int32').tofile(filename)
                    poses2save = poses2save + now_pose
                    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(lidar[...,:3])
                    # o3d.io.write_point_cloud(filename, pcd)
                    # np.save(filename, lidar)
                    start_frame = w_frame
                # else:
                #     print(f"{bc.WARNING} No camera data now....{bc.ENDC}")
            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        with open(f'{args.save_path}/pose.txt', 'w') as file:
            file.write(poses2save)
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        for sensor in sensor_list:
            sensor.destroy()
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')