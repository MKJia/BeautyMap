EDOMap
---

To imporve the ERASOR in speed and remove theory

Check the notion page for more detail improvment thinking.



**<u>Test computer and System:</u>**

Desktop setting: i9-12900KF, GPU 3090, CUDA 11.3, cuDNN 8.2.1

System setting: Ubuntu 20.04, ROS noetic (Python 3.8)

## Install
CuDNN official install manual: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html



[TODO make detail] Here is the Example on my computer:

Download `.deb` from https://developer.nvidia.com/rdp/cudnn-archive

```bash
sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo apt-get install libcudnn8 libcudnn8-dev libcudnn8-samples
```

Tested if you are successfully installed:
```bash
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
```
If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:
```bash
Test passed!
```

For python to run, one more step:
```bash
pip install cupy-cuda11x

# based on my env
pip install cupy-cuda113
```

## Others

Some hints about point cloud data set:
```bash
print(f"{points.shape}")
# [1394189,4](lecai) [25380, 4](16-velodyne) 
# 4: (x, y, z, intensity)
```