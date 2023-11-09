# Data Preparation

## Datasets

### Full Waymo v1.3 dataset
Download the data from the [website](https://waymo.com/open/) of Waymo Open Dataset. Then prepare Waymo data in Kitti format by running
```shell
cd code
python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --extra-tag waymo
```

### Waymo mini dataset
We also provide a mini Waymo dataset in Kitti format which includes 1/5 Waymo v1.3 dataset.
You can download the Waymo mini dataset from [Google Drive](https://drive.google.com/drive/folders/1gOz4XcyaKQFKCnD4TAY1vREt82_QsHPM?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1ho1_orT_iNanJpZYla5fNg?pwd=w7zz#list/path=%2F), and extract the data following the [folder structure](#structure). As for the extraction, you can use the commands below.
```
cat training.tar.gz* | tar -xzv
```

## Pretrained Model

Download pretrained model from [here](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth).

## Folder Structure
The folder structure should like this.
```
|── docs
|── bev_toolbox
├── waymo_playground/
|   ├── projects
│   |   ├── configs
│   |   ├── mmdet3d_plugin
|   ├── mmdetection3d
|   ├── nuscenes-devkit
|   ├── toolbox
|   ├── tools
|   ├── ckpts
│   |   ├── r101_dcn_fcos3d_pretrain.pth
|   ├── data
│   |   ├── waymo_mini
│   |   │   ├── training
│   |   │   ├── waymo_calibs.pkl
│   │   |   ├── waymo_mini_infos_train.pkl
│   │   |   ├── waymo_mini_infos_val.pkl
│   │   |   ├── gt.bin
│   │   |   ├── gt.bin.pkl
|   |   filter_waymo.txt
...
```
