<div align="center">
<h2>[NeurIPS 2023] Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection</h2>
</div>

## Introduction

This repository is an official implementation of VCD.


## Offline Results on NuScenes Val Set.

$^*$ depicts that the size of BEV feature is 256 $\times$ 256.
| Methods     | Backbone        | Image Size     | Frames    | mAP    | NDS  |
|--------|-------|------|--------|--------|-----|
| VCD-E | ConvNext-B    | 768 $\times$ 1408       | 8+1       | 0.677               | 0.711               | 
| Baseline | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.401               | 0.515               | 
| VCD-A    | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.426               | 0.540               | 
| Baseline $^*$ | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.418               | 0.542               | 
| VCD-A $^*$   | ResNet-50    | 256 $\times$ 704       | 8+1       | 0.446               | 0.566               |


## Online Results on NuScenes Val Set.

$^*$ depicts that the size of BEV feature is 256 $\times$ 256.
| Methods     | Backbone        | Image Size     | Frames    | mAP    | NDS  | config | ckpt |
|--------|----|----------|----|---------|-----|---|---|
| VCD-E | ConvNext-B   | 256 $\times$ 704       | 8+1     | 0.538      | 0.606     | [config](nuScenes_playground/VCD/configs/vcd/vcd-convnext-base.py) | [weight](https://drive.google.com/file/d/1oqpqmQYC6MNdKYhZxTPiOjPiyb4Q2UtJ/view?usp=sharing) |
| Baseline | ResNet-50    | 256 $\times$ 704    | 8+1     | 0.389      | 0.493     | [config](nuScenes_playground/VCD/configs/vcd/vcd-r50.py) | [weight](https://drive.google.com/file/d/1QQCzlPSUfRZ3lz6JYevQ1boEnfQEzKUW/view?usp=sharing) |
| VCD-A    | ResNet-50    | 256 $\times$ 704    | 8+1     | 0.410      | 0.518     |  [config](nuScenes_playground/VCD/configs/vcd/MotionDistiller_convnext-r50-4motion.py) | [weight](https://drive.google.com/file/d/1zi8HtIv7PJrh2VrhIVldSdSMP-W63Mwb/view?usp=sharing) |


## Installation

```bash
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```
* We recommend you to follow the guide of [BEVDet2.0](https://github.com/HuangJunJie2017/BEVDet/tree/dev2.0) for environment configuration and dataset preparation.


## Training

```bash
# train a expert
bash tools/dist_train.sh configs/vcd/vcd-convnext-base.py 8

# train a baseline
bash tools/dist_train.sh configs/vcd/vcd-r50.py 8
```


## Inference
```bash
# inference expert or baseline
bash tools/dist_test.sh configs/vcd/vcd-convnext-base.py ckpts/vcd-convnext-base.pth 8

# inference apprentice models
bash tools/dist_test_stu.sh configs/vcd/MotionDistiller_convnext-r50-4motion.py  ckpts/MotionDistiller_convnext-r50-4motion.pth 8
```


## Distillation
```bash
bash tools/dist_distill.sh configs/vcd/MotionDistiller_convnext-r50-4motion.py 8
```

## Currently Supported Features

- [x] VCD online code
- [x] VCD online checkpoints
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

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDet4D](https://github.com/HuangJunJie2017/BEVDet)
