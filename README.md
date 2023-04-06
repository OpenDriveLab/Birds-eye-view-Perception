# Bird's-eye-view Perception: A Survey and Collection

Awesome BEV perception papers and toolbox for achieving SOTA results by OpenDriveLab.

## Table of contents
- [<div id='intro'>Introduction</div>](#div-idintrointroductiondiv)
  - [Major Features](#major-features)
- [<div id='update'>What's New</div>](#div-idupdatewhats-newdiv)
- [<div id='guideline'>Playground on nuScenes</div>](#playground-on-nuscenes)
- [<div id='guideline'>BEV Toolbox</div>](#div-idguidelinebev-toolboxdiv)
- [<div id='todo'>Ongoing Features</div>](#div-idtodoongoing-featuresdiv)
- [<div id='license'>License & Citation & Acknowledgement</div>](#div-idlicenselicensediv)


## <div id='intro'>Introduction</div>

This repo is associated with the survey paper "[Delving into the Devils of Bird’s-eye-view Perception: A Review, Evaluation and Recipe](https://arxiv.org/abs/2209.05324)", which provides an up-to-date literature survey for BEVPercption and an open source BEV toolbox based on PyTorch. In the literature survey, it includes different modalities (camera, lidar and fusion) and tasks (detection and segmentation). As for the toolbox, it provides useful recipe for BEV camera-based 3D object detection, including solid data augmentation strategies, efficient BEV encoder design, loss function family, useful test-time augmentation, ensemble policy, and so on. We hope this repo can not only be a good starting point for new beginners but also help current researchers in the BEV perception community.


`If you find some work popular enough to be cited below, shoot us email or simply open a PR!`

Currently, the BEV perception community is very active and growing fast. There are also some good repos of BEV Perception, _e.g_.

* [BEVFormer](https://github.com/fundamentalvision/BEVFormer) <img src="https://img.shields.io/github/stars/fundamentalvision/BEVFormer?style=social"/>
                <!-- <a class="github-button" href="https://github.com/fundamentalvision/BEVFormer" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. A cutting-edge baseline for camera-based detection via spatiotemporal transformers.
* [BEVDet](https://github.com/HuangJunJie2017/BEVDet) <img src="https://img.shields.io/github/stars/HuangJunJie2017/BEVDet?style=social"/>
                <!-- <a class="github-button" href="https://github.com/HuangJunJie2017/BEVDet" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. Official codes for the camera-based detection methods - BEVDet series, including BEVDet, BEVDet4D and BEVPoolv2.
* [PETR](https://github.com/megvii-research/PETR) <img src="https://img.shields.io/github/stars/megvii-research/PETR?style=social"/>
                <!-- <a class="github-button" href="https://github.com/megvii-research/PETR" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. Implicit BEV representation for camera-based detection and Segmentation, including PETR and PETRv2.
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) <img src="https://img.shields.io/github/stars/Megvii-BaseDetection/BEVDepth?style=social"/>
                <!-- <a class="github-button" href="https://github.com/Megvii-BaseDetection/BEVDepth" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. Official codes for the BEVDepth and BEVStereo, which use LiDAR or temporal stereo to enhance depth estimation.
* [Lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot) <img src="https://img.shields.io/github/stars/nv-tlabs/lift-splat-shoot?style=social"/>
                <!-- <a class="github-button" href="https://github.com/nv-tlabs/lift-splat-shoot" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. Implicitly Unprojecting camera image features to 3D for the segmentation task. 
* [BEVFusion (MIT)](https://github.com/mit-han-lab/bevfusion) <img src="https://img.shields.io/github/stars/mit-han-lab/bevfusion?style=social"/>
                <!-- <a class="github-button" href="https://github.com/mit-han-lab/bevfusion" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. Unifies camera and LiDAR features in the shared bird's-eye view (BEV) representation space for the detection and map segmentation tasks.
* [BEVFusion (ADLab)](https://github.com/ADLab-AutoDrive/BEVFusion) <img src="https://img.shields.io/github/stars/ADLab-AutoDrive/BEVFusion?style=social"/>
                <!-- <a class="github-button" href="https://github.com/ADLab-AutoDrive/BEVFusion" data-icon="octicon-star"
                    data-show-count="true" aria-label="Star ntkme/github-buttons on GitHub">Star</a> -->. A simple and robust LiDAR-Camera fusion framework for the detection task. 

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
        <b>   BEV encoder   </b>
      </td>
      <td>
        <b>   Loss & Heads family   </b>
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
              TBA  
            <!-- <li><a href="tba">TBA</a></li> -->
      <!-- </ul> -->
      </td>
      <td>
        <!-- <ul> -->
            TBA  
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
[2023/04/06]: A new paper [Geometric-aware Pretraining for Vision-centric 3D Object Detection]() is comming soon.

[2022/10/13]: v0.1 was released.
* Integrate some practical data augmentation methods for BEV camera-based 3D detection in the toolbox.
* Offer a pipeline to process the Waymo dataset (camera-based 3D detection).
* Release a baseline (with config) for Waymo dataset and also 1/5 Waymo dataset in Kitti format.

Please refer to [changelog.md](docs/changelog.md) for details and release history.

<!-- ## <div id='overview'>Literature Survey</div>

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
* [Conventional Methods Sensor Fusion](docs/paper_list/sensor_fusion.md) -->

## Playground on nuScenes
The nuScenes playground provides new advancements for BEV camera-based 3D object detection, such as plug-and-play distillation methods that enhance the performance of camera-based detectors and pre-training distillation methods that effectively utilize geometry information from the LiDAR BEV feature.

**GAPretrain**
* [Geometric-aware Pretraining for Vision-centric 3D Object Detection]().
(paper coming soon)

## <div id='guideline'>BEV Toolbox</div>
The toolbox provides useful recipe for BEV camera-based 3D object detection, including solid data augmentation strategies, efficient BEV encoder design, loss function family, useful test-time augmentation, ensemble policy, and so on. Please refer to [README.md](bev_toolbox/README.md) for more details.

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

**nuScences playGround**
- [ ] Add the code of the paper GAPretrain.



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
