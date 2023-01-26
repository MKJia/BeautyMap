# CARLA Semantic LiDAR Dataset

此部分仅在以下系统进行测试：

Desktop setting: i9-12900KF, GPU 3090, CUDA 11.3, cuDNN 8.2.1

System setting: Ubuntu 20.04, ROS noetic (**<u>Python 3.8</u>**)

CARLA Version: 0.9.14 (**<u>Python 3.8</u>**)



主要就是用CARLA收集一下 语义雷达信息 保持和semantickitti一致的格式形式

首先介绍一下SemanticKITTI数据集：

```bash
└── your_folder
    ├── labels
        ├── 000000.label (int32)
        ├── ...
    └── velodyne
        ├── 000000.bin (float16)
        ├── ...
    └── image (optional, only to view)
        ├── 000000.png
        ├── ...
    ├── poses.txt
    ├── calib.txt
```

labels 文件夹里的label呢 是int32位，前16位是instance level，后16位是segmentic level

```python
labels = np.fromfile("label/000000.label", dtype=np.uint32) & 0xFF  
```

比如这里就是取了后16位 读到了label 然后转成了int 这样就能和config里的直接对应好了



## Collect CARLA

那么CARLA部分遵循与上面相同的方案 和 数据格式

```bash
└── your_folder
    ├── labels
        ├── 000000.label (int32)
        ├── ...
    └── lidar
        ├── 000000.bin (float16)
        ├── ...
    └── rgbs (optional, only to view)
        ├── 000000.png
        ├── ...
    ├── poses.txt
```



2023/1/26 21:16 完成初稿

运行后，设定好文件夹 即可生成数据：

```bash
python3 scripts/carla_semantic_collect/collect_data.py

# 如果要有其他NPC的话 麻烦记得也运行一下 CARLA/PythonAPI下有个example文件夹内的spawn npc可行
# 如果没有的话 见quickly-carla: https://github.com/Kin-Zhang/quickly-carla/blob/master/spawn_npc.py
python3 quickly-carla/spawn_npc.py -n 150 -w 80
```

收集数据的代码原理 与 张聪明的此篇中文博客文章 一毛一样：https://www.cnblogs.com/kin-zhang/p/16057173.html

实时截图：

![image](https://user-images.githubusercontent.com/35365764/214942630-fb1e0d26-1bc3-4d33-8dd2-63f7d76910b2.png)

生成数据查看如下：【查看代码在本repo的test 文件夹下

```bash
python3 test/view_semanticBInteractive.py 
```

然后按 键盘 `V`就是下一帧

![image](https://user-images.githubusercontent.com/35365764/214946087-f9d39f88-de2b-44af-ac2b-62ad664e7a36.png)