<div align="center">
<h2>[NeurIPS 2023] Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection</h2>
</div>

## Introduction

This repository is an official implementation of VCD.


## Offline Results on NuScenes Val Set.

$^*$ depicts that the size of BEV feature is 256 $\times$ 256.
| Methods     | Backbone        | Image Size     | Frames    | mAP    | NDS  |
|--------------------------------------|-----------------|------------------------|-----------|---------------------|---------------------
| Baseline | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.401               | 0.515               | 
| VCD-A    | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.426               | 0.540               | 
| Baseline $^*$ | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.418               | 0.542               | 
| VCD-A $^*$   | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.446               | 0.566               |




## Currently Supported Features

- [x] VCD online code
- [ ] VCD online checkpoints
- [ ] VCD offline code
- [ ] VCD offline checkpoints




## License

All assets and code are under the [Apache 2.0 license](../../LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

 ```bibtex
@article{huang2023leveraging,
  title={Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection},
  author={Huang, Linyan and Li, Zhiqi and Sima, Chonghao and Wang, Wenhai and Wang, Jingdong and Qiao, Yu and Li, Hongyang},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
``` 
## Acknowledgement

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
