## GAPretrain
[Geometric-aware Pretraining for Vision-centric 3D Object Detection.](https://arxiv.org/abs/2304.03105)
(code coming soon)

Multi-camera 3D object detection for autonomous driving is a challenging problem that has garnered notable attention from both academia and industry. An obstacle encountered in vision-based techniques involves the precise extraction of geometry-conscious features from RGB images. Recent approaches have utilized geometric-aware image backbones pretrained on depth-relevant tasks to acquire spatial information. However, these approaches overlook the critical aspect of view transformation, resulting in inadequate performance due to the misalignment of spatial knowledge between the image backbone and view transformation. To address this issue, we propose a novel geometric-aware pretraining framework called GAPretrain. Our approach incorporates spatial and structural cues to camera networks by employing the geometric-rich modality as guidance during the pretraining phase. The transference of modal-specific attributes across different modalities is non-trivial, but we bridge this gap by using a unified bird's-eye-view (BEV) representation and structural hints derived from LiDAR point clouds to facilitate the pretraining process. GAPretrain serves as a plug-and-play solution that can be flexibly applied to multiple state-of-the-art detectors. Our experiments demonstrate the effectiveness and generalization ability of the proposed method. We achieve 46.2 mAP and 55.5 NDS on the nuScenes val set using the BEVFormer method, with a gain of 2.7 and 2.1 points, respectively.




## License

All assets and code are under the [Apache 2.0 license](https://github.com/increase24/FocalDistiller/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@article{huang2023geometricaware,
  title={Geometric-aware Pretraining for Vision-centric 3D Object Detection},
  author={Linyan Huang and Huijie Wang and Jia Zeng and Shengchuan Zhang and Liujuan Cao and Rongrong Ji and Junchi Yan and Hongyang Li},
  journal={arXiv preprint arXiv:2304.03105},
  year={2023}
}
```
## Acknowledgement

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [DETR3D](https://github.com/WangYueFt/detr3d)