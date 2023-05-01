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
