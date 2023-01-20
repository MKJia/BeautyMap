EDOMap
---

Title: Eliminate Dynamic Obstacle points  in the Global map (ERASOR++

Authors: Qingwen ZHANG, Mingkai JIA, Ruoyu GENG

Not only limited on the scenarios, point clouds are enough for remove! TESTED SENSOR: Velodyne-16, Velodyne-64, MEMS, Leica BLK360

To improve the ERASOR in speed and remove theory in math. Check [Kin's notion page](https://www.notion.so/kinzhang/EDOMap-Eliminate-Dynamic-Obstacle-points-in-the-Global-map-ERASOR-6732884af87d430e9405c1e5e5c6ad73) for more detail improvement thinking.



**<u>Test computer and System:</u>**

Desktop setting: i9-12900KF, GPU 3090, CUDA 11.3, cuDNN 8.2.1

System setting: Ubuntu 20.04, ROS noetic (**<u>Python 3.8</u>**)






## Others

Some hints about point cloud data set:
```bash
print(f"{points.shape}")
# [1394189,4](lecai) [25380, 4](16-velodyne) 
# 4: (x, y, z, intensity)
```