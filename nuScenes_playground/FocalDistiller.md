## FocalDistiller
> This repository is the official implementation of the CVPR 2023 paper ["Distilling Focal Knowledge from Imperfect Expert for 3D object Detection"]() (code and paper coming soon). 
>
> Authors: Jia Zeng, [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en&authuser=1), Hanming Deng, Lewei Lu, Junchi Yan, Yu Qiao, [Hongyang Li](https://lihongyang.info/)

### Abstract
Multi-camera 3D object detection blossoms in recent years and most of state-of-the-art methods are built up on the bird's-eye-view (BEV) representations. Albeit remarkable performance, these works suffer from low efficiency. Typically, knowledge distillation can be used for model compression. However, due to unclear 3D geometry reasoning, expert features usually contain some noisy and confusing areas. In this work, we investigate on how to distill the knowledge from an imperfect expert. We propose FD3D, a Focal Distiller for 3D object detection. Specifically, a set of queries are leveraged to locate the instance-level areas for masked feature generation, to intensify feature representation ability in these areas. Moreover, these queries search out the representative fine-grained positions for refined distillation. We verify the effectiveness of our method by applying it to two popular detection models, BEVFormer and DETR3D. The results demonstrate that our method achieves improvements of 4.07 and 3.17 points respectively in terms of NDS metric on nuScenes benchmark. 

### Main results

Models and results under main metrics are provided below.

| Method | Back-bone | Image Res. | BEV Res. | NDS | mAP | GFLOPS | FPS | config | ckpt |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: | :---: | :---: |
| BEVFormer-Base* (E) | R101 | 900x1600 | 200x200 | 47.37  | 36.44 | 1845.36 | 2.0 | TBA | TBA |
| BEVFormer-Tiny (A)  | R50  | 450x800  | 100x100 | 39.02 | 26.87 | 381.95  | 7.3 | TBA | TBA |
| + FD3D (ours)       | R50  | 450x800  | 100x100 | 43.09 (↑4.07) | 31.00 (↑4.13) | 381.95  | 7.3  | TBA | TBA |
| BEVFormer-Base (E) | R101-DCN | 900x1600 | 200x200 | 51.74  | 41.64 | 1323.41 | 1.8 | TBA | TBA |
| BEVFormer-Small (A)  | R101-DCN  | 450x800  | 100x100 | 46.26 | 34.56 | 416.46  | 5.9 | TBA | TBA |
| + FD3D (ours)   | R101-DCN  | 450x800  | 100x100 | 48.73 (↑2.47) | 37.64 (↑3.08) | 416.46  | 5.9  | TBA | TBA |
| DETR3D-R101 (E) | R101-DCN | 900x1600 | - | 42.5  | 34.6 | 1016.83 | 2.5 | TBA | TBA |
| DETR3D-R50 (A)  | R50  | 900x1600  | - | 35.78 | 28.85 | 876.94  | 4.0 | TBA | TBA |
| + FD3D (ours)   | R50  | 900x1600  | - | 38.95 (↑3.17) | 31.33 (↑2.48) | 876.94  | 4.0  | TBA | TBA |




## License

All assets and code are under the [Apache 2.0 license](https://github.com/increase24/FocalDistiller/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{Zeng2023Distilling,
 title={Distilling focal knowledge from imperfect
expert for 3D object detection}, 
 author={Jia Zeng and Li Chen and Hanming Deng and Lewei Lu and Junchi Yan and Yu Qiao and Hongyang Li},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 year={2023},
}
```
## Acknowledgement

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [DETR3D](https://github.com/WangYueFt/detr3d)