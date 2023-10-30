<div align="center">
<h3>[NeurIPS2023] Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection</h3>
</div>

## Introduction

This repository is an official implementation of VCD.


## Results on NuScenes Val Set.

$^*$ depicts that the size of BEV feature is 256 $\times$ 256.
| Methods     | Backbone        | Image Size     | Frames    | mAP    | NDS  |
|--------------------------------------|-----------------|------------------------|-----------|---------------------|---------------------
| Baseline | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.401               | 0.515               | 
| VCD-A    | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.426               | 0.540               | 
| Baseline $^*$ | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.418               | 0.542               | 
| VCD-A $^*$   | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.446               | 0.566               |




## Currently Supported Features

- [x] VCD online code
- [ ] Checkpoints
- [ ] VCD offline code




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