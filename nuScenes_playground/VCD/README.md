## VCD
> This repository is the official implementation of the NeurIPS 2023 paper ["Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection"](https://arxiv.org/abs/2310.15670). 
>
> Authors: Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li

### Abstract
Current research is primarily dedicated to advancing the accuracy of camera-only 3D object detectors (apprentice) through the knowledge transferred from LiDAR- or multi-modal-based counterparts (expert). However, the presence of the domain gap between LiDAR and camera features, coupled with the inherent incompatibility in temporal fusion, significantly hinders the effectiveness of distillation-based enhancements for apprentices. Motivated by the success of uni-modal distillation, an apprentice-friendly expert model would predominantly rely on camera features, while still achieving comparable performance to multi-modal models. To this end, we introduce VCD, a framework to improve the camera-only apprentice model, including an apprentice-friendly multi-modal expert and temporal-fusion-friendly distillation supervision. The multi-modal expert VCD-E adopts an identical structure as that of the camera-only apprentice in order to alleviate the feature disparity, and leverages LiDAR input as a depth prior to reconstruct the 3D scene, achieving the performance on par with other heterogeneous multi-modal experts. Additionally, a fine-grained trajectory-based distillation module is introduced with the purpose of individually rectifying the motion misalignment for each object in the scene. With those improvements, our camera-only apprentice VCD-A sets new state-of-the-art on nuScenes with a score of 63.1% NDS.

### Main results

Models and results under main metrics are provided below.

| Methods     | Backbone        | Image Size     | Frames    | mAP    | NDS  |
|--------------------------------------|-----------------|------------------------|-----------|---------------------|---------------------
| Baseline | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.401               | 0.515               | 
| VCD-A    | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.426               | 0.540               | 
| Baseline | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.418               | 0.542               | 
| VCD-A    | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.446               | 0.566               |



## License

All assets and code are under the [Apache 2.0 license](https://github.com/increase24/FocalDistiller/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@article{huang2023leveraging,
  title={Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection},
  author={Huang, Linyan and Li, Zhiqi and Sima, Chonghao and Wang, Wenhai and Wang, Jingdong and Qiao, Yu and Li, Hongyang},
  journal={arXiv preprint arXiv:2310.15670},
  year={2023}
}
```
## Acknowledgement

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
