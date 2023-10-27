# Bird's-eye-view (BEV) Perception: A Survey and Collection

Awesome BEV perception papers and toolbox for achieving state-of-the-arts performance.

## Table of contents
- [Bird's-eye-view Perception: A Survey and Collection](#birds-eye-view-perception-a-survey-and-collection)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Major Features](#major-features)
  - [What's New](#whats-new)
  - [BEV Algorithm Family](#bev-algorithm-family)
  - [BEV Toolbox](#bev-toolbox)
  - [Literature Survey](#literature-survey)
  - [License and Citation](#license-and-citation)


## <div id='intro'>Introduction</div>

This repo is associated with the survey paper "[Delving into the Devils of Bird’s-eye-view Perception: A Review, Evaluation and Recipe](https://arxiv.org/abs/2209.05324)", which provides an up-to-date literature survey for BEV percption and an open source BEV toolbox based on PyTorch. We also introduce BEV algorithm family, incluidng follow-up work on BEV percepton such as [VCD](https://arxiv.org/abs/2310.15670), [GAPretrain](https://arxiv.org/abs/2304.03105) and [FocalDistiller](https://openaccess.thecvf.com/content/CVPR2023/html/Zeng_Distilling_Focal_Knowledge_From_Imperfect_Expert_for_3D_Object_Detection_CVPR_2023_paper.html). We hope this repo can not only be a good starting point for new beginners but also help current researchers in the BEV perception community.
<!-- In the literature survey, it includes different modalities (camera, lidar and fusion) and tasks (detection and segmentation). As for the toolbox, it provides useful recipe for BEV camera-based 3D object detection, including solid data augmentation strategies, efficient BEV encoder design, loss function family, useful test-time augmentation, ensemble policy, and so on. -->

`If you find some work popular enough to be cited below, shoot us email or simply open a PR!`

<!-- Currently, the BEV perception community is very active and growing fast. There are also some good repos of BEV Perception, _e.g_. -->

<!-- * [BEVFormer](https://github.com/fundamentalvision/BEVFormer) <img src="https://img.shields.io/github/stars/fundamentalvision/BEVFormer?style=social"/>. A cutting-edge baseline for camera-based detection via spatiotemporal transformers.
* [BEVDet](https://github.com/HuangJunJie2017/BEVDet) <img src="https://img.shields.io/github/stars/HuangJunJie2017/BEVDet?style=social"/>. Official codes for the camera-based detection methods - BEVDet series, including BEVDet, BEVDet4D and BEVPoolv2.
* [PETR](https://github.com/megvii-research/PETR) <img src="https://img.shields.io/github/stars/megvii-research/PETR?style=social"/>. Implicit BEV representation for camera-based detection and Segmentation, including PETR and PETRv2.
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) <img src="https://img.shields.io/github/stars/Megvii-BaseDetection/BEVDepth?style=social"/>. Official codes for the BEVDepth and BEVStereo, which use LiDAR or temporal stereo to enhance depth estimation.
* [Lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot) <img src="https://img.shields.io/github/stars/nv-tlabs/lift-splat-shoot?style=social"/>. Implicitly Unprojecting camera image features to 3D for the segmentation task. 
* [BEVFusion (MIT)](https://github.com/mit-han-lab/bevfusion) <img src="https://img.shields.io/github/stars/mit-han-lab/bevfusion?style=social"/>. Unifies camera and LiDAR features in the shared bird's-eye view (BEV) representation space for the detection and map segmentation tasks.
* [BEVFusion (ADLab)](https://github.com/ADLab-AutoDrive/BEVFusion) <img src="https://img.shields.io/github/stars/ADLab-AutoDrive/BEVFusion?style=social"/>. A simple and robust LiDAR-Camera fusion framework for the detection task.  -->

### Major Features

* **SOTA BEV Algorithm Family** <br> We include important follow-up works of BEVFormer/BEVDet/BEVDepth in different aspects, ranging from plug-and-play tricks to knowledge distillation([GAPretrain](https://arxiv.org/abs/2304.03105), [FocalDistiller]((https://openaccess.thecvf.com/content/CVPR2023/html/Zeng_Distilling_Focal_Knowledge_From_Imperfect_Expert_for_3D_Object_Detection_CVPR_2023_paper.html))). More detail of each paper can be found in each README.md file under [here](./nuScenes_playground/).
* **Convenient BEVPerception Toolbox** <br> We integrate bag of tricks in the BEV toolbox that help us achieve 1st in the camera-based detection track of the Waymo Open Challenge 2022, which can be used indedependly or as a plug-in for `mmdet3d` and `detectron2`. Moreover, we provide a suitable playground for new-beginners in this area, including hands-on tutorial and small-scale dataset (1/5 WOD in kitti format) to validate idea. More detail can be found [here](./bev_toolbox/README.md).
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
        <b>   Heads family   </b>
      </td>
      <td>
        <b>   Loss   </b>
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
          <li>BEVFormer encoder</li>
          <li>Voxel-SPVCNN encoder</li>
            <!-- <li><a href="tba">TBA</a></li> -->
      <!-- </ul> -->
      </td>
      <td>
        <!-- <ul> -->
          <li>Deformable DETR head</li>
          <li>FreeAnchor head</li>
          <li>Centerpoint head</li>
            <!-- <li><a href="tba">TBA</a></li> -->
      <!-- </ul> -->
      </td>
      <td>
        <!-- <ul> -->
          Camera-only detection:
          <li>auxiliary 2D detection loss</li>
          <li>auxiliary depth supervision</li>
          LiDAR segmentation:
          <li>Geo loss</li>
          <li>Lovasz loss</li>
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

* **Up-to-date Literature Survey for BEV Perception** <br> We summarize important methods in recent years about BEV perception including different modalities (camera, LIDAR, Fusion) and tasks (Detection, Segmentation, Occupancy). More detail of the survey paper list can be found [here](./docs/paper_list/).

## <div id='update'>What's New</div>
[2023/10/26]: A new paper [VCD](https://arxiv.org/abs/2310.15670) are comming soon with official implementation.

[2023/09/06]: We have a new version of the [survey](https://arxiv.org/abs/2209.05324). Check it out!

[2023/04/06]: Two new paper [GAPretrain](https://arxiv.org/abs/2304.03105) and [FocalDistiller](https://openaccess.thecvf.com/content/CVPR2023/html/Zeng_Distilling_Focal_Knowledge_From_Imperfect_Expert_for_3D_Object_Detection_CVPR_2023_paper.html) are comming soon with official implementation.

[2022/10/13]: v0.1 was released.
* Integrate some practical data augmentation methods for BEV camera-based 3D detection in the toolbox.
* Offer a pipeline to process the Waymo dataset (camera-based 3D detection).
* Release a baseline (with config) for Waymo dataset and also 1/5 Waymo dataset in Kitti format.

Please refer to [changelog.md](docs/changelog.md) for details and release history of the toolbox code.

## <div id='algo_family'>BEV Algorithm Family</div>
The BEV algorithm family includes follow-up works of BEVFormer in different aspects, ranging from plug-and-play tricks to pre-training distillation. All paper summary is under [nuscenes_playground](nuScenes_playground) alongwith official implementation, check it out!
* **VCD** <br> [Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection](https://arxiv.org/abs/2310.15670). More detail can be found in [nuScenes_playground/VCD.md](./nuScenes_playground/VCD.md).
* **GAPretrain** <br> [Geometric-aware Pretraining for Vision-centric 3D Object Detection](https://arxiv.org/abs/2304.03105). More detail can be found in [nuScenes_playground/GAPretrain.md](./nuScenes_playground/GAPretrain.md).
* **FocalDistiller** <br> [Distilling Focal Knowledge from Imperfect Expert for 3D object Detection (https://openaccess.thecvf.com/content/CVPR2023/html/Zeng_Distilling_Focal_Knowledge_From_Imperfect_Expert_for_3D_Object_Detection_CVPR_2023_paper.html). More detail can be found in [nuScenes_playground/FocalDistiller.md](./nuScenes_playground/FocalDistiller.md).
## <div id='guideline'>BEV Toolbox</div>
The BEV toolbox provides useful recipe for BEV camera-based 3D object detection, including solid data augmentation strategies, efficient BEV encoder design, loss function family, useful test-time augmentation, ensemble policy, and so on. Please refer to [bev_toolbox/README.md](bev_toolbox/README.md) for more details.

<!-- The BEV algorithm family includes follow-up works of BEVFormer in different aspects, ranging from plug-and-play tricks to pre-training distillation. All paper summary is in [nuscenes_playground/README.md](nuScenes_playground/README.md) alongwith official implementation, check it out! -->

<!-- 
**GAPretrain**
* Geometric-aware Pretraining for Vision-centric 3D Object Detection.
(paper coming soon) 
The nuScenes playground provides new advancements for BEV camera-based 3D object detection, such as plug-and-play distillation methods that enhance the performance of camera-based detectors and pre-training distillation methods that effectively utilize geometry information from the LiDAR BEV feature.
-->
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

## <div id='license & citation'>License and Citation</div>
This project is released under the [Apache 2.0 license](LICENSE).

If you find this project useful in your research, please consider cite:

```BibTeX
@article{li2022bevsurvey,
  title={Delving into the Devils of Bird's-eye-view Perception: A Review, Evaluation and Recipe},
  author={Li, Hongyang and Sima, Chonghao and Dai, Jifeng and Wang, Wenhai and Lu, Lewei and Wang, Huijie and Zeng, Jia and Li, Zhiqi and Yang, Jiazhi and Deng, Hanming and Tian, Hao and Xie, Enze and Xie, Jiangwei and Chen, Li and Li, Tianyu and Li, Yang and Gao, Yulu and Jia, Xiaosong and Liu, Si and Shi, Jianping and Lin, Dahua and Qiao, Yu},
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
