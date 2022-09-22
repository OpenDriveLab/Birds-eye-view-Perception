# Data Preparation

## Datasets

1. Download the Waymo mini dataset from xxx and unzip it. 

## Pretrained Model

1. Download pretrained model from xxx 

## Folder Structure
```
|── docs
├── code
|   ├── projects
│   |   ├── configs
│   |   ├── mmdet3d_plugin
|   ├── mmdetection3d
|   ├── nuscenes-devkit
|   ├── tools
|   ├── ckpts
│   |   ├── fcos3d.pth
|   ├── data
│   |   ├── waymo_mini
│   |   │   ├── training
│   |   │   ├── waymo_calibs.pkl
│   │   |   ├── waymo_mini_infos_train.pkl
│   │   |   ├── waymo_infos_val.pkl
│   │   |   ├── gt.bin
|   |   |   ├── filter_waymo.txt
```


<!-- # 这个命令会最后生成一个bin的文件，代码会输出这个bin的路径，会在test/$CONFIG_NAME 下面
./tools/dist_test_video.sh ./projects/configs/waymo/waymo_imp.py ./path/to/ckpt.pth 8

# 然后用下面的命令获得最终结果

python ./projects/mmdet3d_plugin/datasets/eval_waymo.py  -b path/to/bin

# note:两个代码串在一起，一直跑不起来
``` -->