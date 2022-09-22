# Environment

## Install

1.Basically follow https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation
```angular2html
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
chmod 777 Miniconda3-py38_4.11.0-Linux-x86_64.sh
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
install gcc
```
conda install -c psi4 gcc-5 # gcc-5.2 for mmcv-full

conda install -c omgarcia gcc-6 # if the above command dose not work
conda install gxx_linux-64
```

mmcv-full
```
pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

mmdet
```
pip install mmdet==2.19.0
```

mmseg
```
pip install mmsegmentation==0.20.0
```

mmdet3d
```
git clone https://gitee.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.18.1
python setup.py install
```

timm
```
pip install timm==0.5.4
```

waymo-open-dataset
```
pip install waymo-open-dataset-tf-2-6-0==1.4.1 # for waymo open dataset
pip install lingvo==0.10.0
```


```
# git config credential.helper store
git clone https://gitee.com/lizhiqi-gitee/bevformer.git
```
