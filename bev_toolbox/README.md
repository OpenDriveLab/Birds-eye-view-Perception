# Table of contents
- [<div id='guideline'>BEV Toolbox</div>](#div-idguidelinebev-toolboxdiv)
  - [<div id='guideline'>Get Started</div>](#div-idguidelineget-starteddiv)
    - [<div id='example'>Installation</div>](#div-idexampleinstallationdiv)
    - [<div id='example'>A simple example</div>](#div-idexamplea-simple-examplediv)
    - [Use BEV toolbox with `mmdet3d`](#use-bev-toolbox-with-mmdet3d)
    - [Use BEV-toolbox with `detectron2`](#use-bev-toolbox-with-detectron2)
  - [Playground on Waymo](#playground-on-waymo)
    - [Setup](#setup)
    - [Config with Performance](#config-with-performance)
- [<div id='todo'>Ongoing Features</div>](#div-idtodoongoing-featuresdiv)

## <div id='guideline'>BEV Toolbox</div>

### <div id='guideline'>Get Started</div>

#### <div id='example'>Installation</div>

```shell
pip install numpy opencv-python
pip install bev-toolbox
```

#### <div id='example'>A simple example</div>

We provide an example with a sample from Waymo dataset to introduce the usage of this toolbox.

```python
import cv2
import numpy as np
from bev_toolbox.data_aug import RandomScaleImageMultiViewImage

# Declare an augmentation pipeline
transform = RandomScaleImageMultiViewImage(scales=[0.9, 1.0, 1.1])

# multiple-view images
imgs = [cv2.imread(f'example/cam{i}_img.jpg') for i in range(5)]
# intrinsic parameters of cameras
cam_intr = [np.load(f'example/cam{i}_intrinsic.npy') for i in range(5)]
# extrinsic parameters of cameras
cam_extr = [np.load(f'example/cam{i}_extrinsic.npy') for i in range(5)]
# transformations from lidar to image
lidar2img = [np.load(f'example/cam{i}_lidar2img.npy') for i in range(5)]

# Augment an image
imgs_new, cam_intr_new, lidar2img_new = transform(imgs, cam_intr, cam_extr, lidar2img)
```

For more details like the coordinate systems or visualization, please refer to [example.md](example/example.md)

#### Use BEV toolbox with `mmdet3d`

We provide wrappers of this BEV toolbox for mmdet3d and detectron2. 

1. Add the following code to [train_video.py](waymo_playground/tools/train_video.py#L93) or [test_video.py](waymo_playground/tools/test_video.py#L110).
```
from bev_toolbox.init_toolbox import init_toolbox_mmdet3d
init_toolbox_mmdet3d()
```
2. Use functions in the toolbox just like mmdet3d. For example, you can just add ```RandomScaleImageMultiViewImage``` to the configure file.
```python
train_pipeline = [
    ...
    dict(type='RandomScaleImageMultiViewImage', scales=[0.9, 1.0, 1.1]),
    ...
]
```

#### Use BEV-toolbox with `detectron2`

We plan to make this toolbox compatible with detectron2 in the future.

### Playground on Waymo

We provide a suitable playground on the Waymo dataset, including hands-on tutorial and small-scale dataset (1/5 WOD in kitti format) to validate idea.

#### Setup
Please refer to [waymo_setup.md](docs/waymo_setup.md) about how to run experiments on Waymo.

#### Config with Performance

We provide the improvement of each trick compared with the baseline on the Waymo validation set. All the models are trained with 1/5 training data of Waymo v1.3 which is represented as Waymo mini here. It's worthy noting that the results were run on data with *png* format. We are revalidating these results on the data with *jpg* format. So, the actual performance may be different.

✓: DONE, ☐: TODO.

| Backbone  | Head  | Train data | Trick and corresponding config           | LET-mAPL | LET-mAPH | L1/mAPH (Car) | Status |
| :-------: | :---: | :--------: | :--------------------------------------- | :------: | :------: | :-----------: | :----: |
| ResNet101 | DETR  | Waymo mini | Baseline                                 |   34.9   |   46.3   |     25.5      |   ✓    |
| ResNet101 | DETR  | Waymo mini | Multi-scale resize, Flip                 |   35.6   |   46.9   |     26.8      |   ✓    |
| ResNet101 | DETR  | Waymo mini | Conv offset in TSA                       |   35.9   |   48.1   |     25.6      |   ☐    |
| ResNet101 | DETR  | Waymo mini | Deformable view encoder                  |   36.1   |   48.1   |     25.9      |   ☐    |
| ResNet101 | DETR  | Waymo mini | Corner pooling                           |   35.6   |   46.9   |     26.0      |   ☐    |
| ResNet101 | DETR  | Waymo mini | 2x BEV scale]                            |    -     |    -     |     25.5      |   ☐    |
| ResNet101 | DETR  | Waymo mini | Sync BN                                  |    -     |    -     |     25.5      |   ☐    |
| ResNet101 | DETR  | Waymo mini | EMA                                      |    -     |    -     |     25.6      |   ☐    |
| ResNet101 | DETR  | Waymo mini | 2d auxiliary loss                        |   35.3   |   47.4   |     24.6      |   ☐    |
| ResNet101 | DETR  | Waymo mini | 2d auxiliary loss, Learnable loss weight |   36.2   |   48.1   |     25.4      |   ☐    |
| ResNet101 | DETR  | Waymo mini | Smooth L1 loss                           |    -     |    -     |     26.2      |   ☐    |
| ResNet101 | DETR  | Waymo mini | Label smoothing                          |   36.0   |   46.7   |       -       |   ☐    |


## <div id='todo'>Ongoing Features</div>

**Literature Survey**
- [ ] Add new papers.

**BEV toolbox**
- [x] Data augmentation methods for BEV perception
  - [x] Random horizontal flip
  - [x] Random scale
  - [ ] Grid mask
  - [ ] New data augmentation
- [ ] Integrate more tricks
  - [ ] Post-process
    - [ ] Test-time Augmentation
    - [ ] Weighted Box Fusion
    - [ ] Two-stage Ensemble
  - [ ] BEV Encoder
  - [ ] Loss Family
- [ ] Add Visualization in BEV
- [ ] Improve the current implementations.
- [ ] Add documentation to introduce the APIs of the toolbox
