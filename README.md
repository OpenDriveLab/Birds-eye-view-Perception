# BEVPerception-Survey-Recipe

Awesome BEV perception papers and toolbox for achieving SOTA results. [🤝Fundamental Vision](https://github.com/fundamentalvision)

## Table of contents
- [BEVPerception-Survey-Recipe](#bevperception-survey-recipe)
  - [<div id='intro'>Introduction</div>](#div-idintrointroductiondiv)
    - [Major Features](#major-features)
  - [<div id='update'>What's New</div>](#div-idupdatewhats-newdiv)
  - [<div id='overview'>Literature Survey</div>](#div-idoverviewliterature-surveydiv)
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
  - [<div id='license'>License</div>](#div-idlicenselicensediv)
  - [<div id='cite'>Citation</div>](#div-idcitecitationdiv)



## <div id='intro'>Introduction</div>

This repo is associated with the survey paper "[Delving into the Devils of Bird’s-eye-view Perception: A Review, Evaluation and Recipe](https://arxiv.org/abs/2209.05324)", which provides an up-to-date literature survey for BEVPercption and an open source BEV toolbox based on PyTorch. In the literature survey, it includes different modalities (camera, lidar and fusion) and tasks (detection and segmentation). As for the toolbox, it provides useful recipe for BEV camera-based 3D object detection, including solid data augmentation strategies, efficient BEV encoder design, loss function family, useful test-time augmentation, ensemble policy, and so on. We hope this repo can not only be a good starting point for new beginners but also help current researchers in the BEV perception community.

### Major Features

* **Up-to-date Literature Survey for BEV Perception** <br> We summarized important methods in recent years about BEV perception including different modalities and tasks.
* **Convenient BEVPerception Toolbox** <br> We integrate bag of tricks in the BEV toolbox that help us achieve 1st in the camera-based detection track of the Waymo Open Challenge 2022, which can be grouped as four types -- data augmentation, design of BEV encoder, loss family and post-process policy. This toolbox can be used indedependly or as a plug-in for `mmdet3d` and `detectron2`. 
<div align="center">
  <b>Bag of Tricks</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="middle">
      <td>
        <b>Multiple View Data Augmentation</b>
      </td>
      <td>
        <b> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  BEV encoder  &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp </b>
      </td>
      <td>
        <b> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  Loss family  &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp </b>
      </td>
      <td>
        <b>Post-Process</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="code/projects/configs/bevformer/data_aug">Random Flip</a></li>
          <li><a href="code/projects/configs/bevformer/data_aug">Random Multi-scale Resize</a></li>
          <li>Grid Mask</li>
        </ul>
      </td>
      <td>
        <!-- <ul> -->
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  TBA  &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
            <!-- <li><a href="tba">TBA</a></li> -->
      <!-- </ul> -->
      </td>
      <td>
        <!-- <ul> -->
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  TBA  &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          <!-- <li><a href="tba">TBA</a></li> -->
        <!-- </ul> -->
      </td>
      <td>
        <ul>
          <li>Test-time Augmentation</li>
          <li>Weighted Box Fusion</li>
          <li>Two-stage Ensemble</li>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>


* **Support Waymo Open Dataset (WOD) for camera-only detection** <br> We provide a suitable playground for new-beginners in this area, including hands-on tutorial and small-scale dataset (1/5 WOD in kitti format) to validate idea.


## <div id='update'>What's New</div>

v0 was released in 09/30/2022.
* Integrate some practical data augmentation methods for BEV camera-based 3D detection in the toolbox.
* Offer a pipeline to process the Waymo dataset (camera-based 3D detection).
* Release a baseline (with config) for Waymo dataset and also 1/5 Waymo dataset in Kitti format.

Please refer to [changelog.md](docs/changelog.md) for details and release history.

## <div id='overview'>Literature Survey</div>

![](figs/general_overview.jpg)
The general picture of BEV perception at a glance, where consists of three sub-parts based on the input modality. BEV perception is a general task built on top of a series of fundamental tasks. For better completeness of the whole perception algorithms in autonomous driving, we list other topics as well. More detail can be found in the [survey paper](https://arxiv.org/abs/2209.05324).

We have summarized important datasets and methods in recent years about BEV perception in academia and also different roadmaps used in industry. 
* [Academic Summary of BEV Perception](docs/paper_list/academia.md)
  * [Datasets for BEV Perception](docs/paper_list/dataset.md)
  * [BEV Camera](docs/paper_list/bev_camera.md)
  * [BEV Lidar](docs/paper_list/bev_lidar.md)
  * [BEV Fusion](docs/paper_list/bev_fusion.md)
* [Industrial Roadmap of BEV Perception](docs/paper_list/industry.md)
  
We have also summarized some conventional methods for different tasks.
* [Conventional Methods Camera 3D Object Detection](docs/paper_list/camera_detection.md)
* [Conventional Methods LiDAR Detection](docs/paper_list/lidar_detection.md)
* [Conventional Methods LiDAR Segmentation](docs/paper_list/lidar_segmentation.md)
* [Conventional Methods Sensor Fusion](docs/paper_list/sensor_fusion.md)

## <div id='guideline'>BEV Toolbox</div>

### <div id='guideline'>Get Started</div>

#### <div id='example'>Installation</div>

```shell
pip install numpy opencv-python
pip install bev-toolbox
```

#### <div id='example'>A simple example</div>

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

#### Use BEV toolbox with `mmdet3d`

We provide wrappers of this BEV toolbox for mmdet3d and detectron2. 

1. Add the following code to [train_video.py](experiments/tools/train_video.py#L93) or [test_video.py](experiments/tools/test_video.py#L110).
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
| ResNet101 | DETR  | Waymo mini | Baseline                                 |   34.6   |   46.1   |     25.5      |   ✓    |
| ResNet101 | DETR  | Waymo mini | Multi-scale resize, Flip                 |    -     |    -     |     26.8      |   ☐    |
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


## <div id='license'>License</div>
This project is released under the [Apache 2.0 license](LICENSE).

## <div id='cite'>Citation</div>

If you find this project useful in your research, please consider cite:

```BibTeX
@article{li2022bevsurvey,
  title={Delving into the Devils of Bird's-eye-view Perception: A Review, Evaluation and Recipe},
  author={Li, Hongyang and Sima, Chonghao and Dai, Jifeng and Wang, Wenhai and Lu, Lewei and Wang, Huijie and Xie, Enze and Li, Zhiqi and Deng, Hanming and Tian, Hao and Zhu, Xizhou and Chen, Li and Gao, Yulu and Geng, Xiangwei and Zeng, Jia and Li, Yang and Yang, Jiazhi and Jia, Xiaosong and Yu, Bohan and Qiao, Yu and Lin, Dahua and Liu, Si and Yan, Junchi and Shi, Jianping and Luo, Ping},
  journal={arXiv preprint arXiv:2209.05324},
  year={2022}
}
```
```BibTeX
@misc{bevtoolbox2022,
  title={{BEVPerceptionx-Survey-Recipe} toolbox for general BEV perception},
  author={BEV-Toolbox Contributors},
  howpublished={\url{https://github.com/OpenPerceptionX/BEVPerception-Survey-Recipe}},
  year={2022}
}
```

## Acknowledgement

Many thanks to these excellent open source projects and also the stargazers and forkers:
- [detr3d](https://github.com/WangYueFt/detr3d) 
- [BEVFormer](https://github.com/zhiqi-li/BEVFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)


### &#8627; Stargazers
[![Stargazers repo roster for @OpenPerceptionX/BEVPerception-Survey-Recipe](https://reporoster.com/stars/OpenPerceptionX/BEVPerception-Survey-Recipe)](https://github.com/OpenPerceptionX/BEVPerception-Survey-Recipe/stargazers)

### &#8627; Forkers
[![Forkers repo roster for @OpenPerceptionX/BEVPerception-Survey-Recipe](https://reporoster.com/forks/OpenPerceptionX/BEVPerception-Survey-Recipe)](https://github.com/OpenPerceptionX/BEVPerception-Survey-Recipe/network/members)
