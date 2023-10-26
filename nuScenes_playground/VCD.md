## VCD
> This repository is the official implementation of the NeurIPS 2023 paper ["Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection"](https://arxiv.org/abs/2310.15670). 
>
> Authors: Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li

### Abstract
Current research is primarily dedicated to advancing the accuracy of camera-only 3D object detectors (apprentice) through the knowledge transferred from LiDAR- or multi-modal-based counterparts (expert). However, the presence of the domain gap between LiDAR and camera features, coupled with the inherent incompatibility in temporal fusion, significantly hinders the effectiveness of distillation-based enhancements for apprentices. Motivated by the success of uni-modal distillation, an apprentice-friendly expert model would predominantly rely on camera features, while still achieving comparable performance to multi-modal models. To this end, we introduce VCD, a framework to improve the camera-only apprentice model, including an apprentice-friendly multi-modal expert and temporal-fusion-friendly distillation supervision. The multi-modal expert VCD-E adopts an identical structure as that of the camera-only apprentice in order to alleviate the feature disparity, and leverages LiDAR input as a depth prior to reconstruct the 3D scene, achieving the performance on par with other heterogeneous multi-modal experts. Additionally, a fine-grained trajectory-based distillation module is introduced with the purpose of individually rectifying the motion misalignment for each object in the scene. With those improvements, our camera-only apprentice VCD-A sets new state-of-the-art on nuScenes with a score of 63.1% NDS.

### Main results

Models and results under main metrics are provided below.

| Methods     | Backbone        | Image Size     | Frames    | mAP    | NDS  | mATE  | mASE  | mAOE  | mAVE | mAAE |
|--------------------------------------|-----------------|------------------------|-----------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|------------------|
| BEVDet | ResNet-50       | 256 $\times$ 704       | 1         | 0.298               | 0.379               | 0.725               | 0.279               | 0.589               | 0.860            | 0.245            |
| PETR  | ResNet-50       | 384 $\times$ 1056      | 1         | 0.313               | 0.381               | 0.768               | 0.278               | 0.564               | 0.923            | 0.225            |
| BEVDet4D | ResNet-50       | 256 $\times$ 704       | 2         | 0.322               | 0.457               | 0.703               | 0.278               | 0.495               | 0.354            | 0.206            |
| BEVDepth | ResNet-50       | 256 $\times$ 704       | 2         | 0.351               | 0.475               | 0.639               | 0.267               | 0.479               | 0.428            | 0.198            |
| BEVStereo  | ResNet-50       | 256 $\times$ 704       | 2         | 0.372               | 0.500               | 0.598               | 0.270               | 0.438               | 0.367            | 0.190            |
| STS| ResNet-50       | 256 $\times$ 704       | 2         | 0.377               | 0.489               | 0.601               | 0.275               | 0.450               | 0.446            | 0.212            |
| VideoBEV  | ResNet-50       | 256 $\times$ 704       | 8         | 0.422               | 0.535               | 0.564               | 0.276               | 0.440               | 0.286            | 0.198            |
| SOLOFusion | ResNet-50       | 256 $\times$ 704       | 16+1      | 0.427               | 0.534               | 0.567               | 0.274               | 0.411               | 0.252   | 0.188   |
| StreamPETR | ResNet-50       | 256 $\times$ 704       | 8         | 0.432               | 0.540               | 0.581               | 0.272               | 0.413               | 0.295            | 0.195            |
| Baseline | ResNet-50       | 256 $\times$ 704       | 8+1       | 0.401               | 0.515               | 0.595               | 0.279               | 0.489               | 0.291            | 0.198            |
| VCD-A   | ResNet-50 | 256 $\times$ 704 | 8+1 | 0.426    | 0.540          | 0.547          | 0.271          | 0.433          | 0.268       | 0.207       |
| Baseline     | ResNet-50       | 256 $\times$ 704       | 8+1       | 0.418               | 0.542               | 0.522               | 0.267               | 0.428               | 0.262            | 0.188  |
| VCD-A  | ResNet-50 | 256 $\times$ 704 | 8+1  | 0.446 | 0.566 | 0.497 | 0.260 | 0.350 | 0.257       | 0.203       |





## License

All assets and code are under the [Apache 2.0 license](https://github.com/increase24/FocalDistiller/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

<!-- ```bibtex
@inproceedings{zeng2023distilling,
  title={Distilling Focal Knowledge from Imperfect Expert for 3D Object Detection},
  author={Zeng, Jia and Chen, Li and Deng, Hanming and Lu, Lewei and Yan, Junchi and Qiao, Yu and Li, Hongyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={992--1001},
  year={2023}
}
``` -->
## Acknowledgement

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)