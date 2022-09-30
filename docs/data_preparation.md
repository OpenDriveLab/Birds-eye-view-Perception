# Data Preparation

## Datasets

### Full Waymo v1.3 dataset
Download the data from xxx. Then prepare Waymo data in Kitti format by running
```shell
cd code
python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --extra-tag waymo
```

### Waymo mini dataset
We also provide a mini Waymo dataset in Kitti format which includes 1/5 Waymo v1.3 dataset.
You can download the Waymo mini dataset from xxx and extract the data following the [folder structure](#structure). 

## Pretrained Model

Download pretrained model from xxx.

## <div id='structure'>Folder Structure</div>
The folder structure should like this.
```
|── docs
├── code
|   ├── projects
│   |   ├── configs
│   |   ├── mmdet3d_plugin
|   ├── mmdetection3d
|   ├── nuscenes-devkit
|   ├── toolbox
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
│   │   |   ├── gt.bin.pkl
|   |   filter_waymo.txt
...
```