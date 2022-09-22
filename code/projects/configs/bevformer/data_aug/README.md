# Multiple View Data Augmentation

## Introduction

We apply two kinds of data augmentation which are static augmentation involving color variation alone and spatial transformation moving pixels around.
For augmentations involving spatial transformation, apart from ground truth transformed accordingly, calibration in camera parameter is also necessary. Common augmentations adopted in recent work are color jitter, flip, multi-scale resize, rotation, crop and grid mask. 
Since BEVFormer uses sequence input, we ensure that the transformations are consistent for each frame of the sequence.

### Multi-scale Resize

[config](waymo_mini_r101_ms_flip.py#173) | [code](../../../mmdet3d_plugin/datasets/pipelines/transform_3d.py#L237)

The input image is scaled by a factor between 0.5 and 1.2.
Coresponding Config: dict(type='RandomScaleImageMultiViewImage', scales=[0.9, 1.0, 1.1]) in train_pipeline.

### Flip

[config](waymo_mini_r101_ms_flip.py#172) | [code](../../../mmdet3d_plugin/datasets/pipelines/transform_3d.py#L319)

The input is fliped by a ratio of 0.5. We simply flip images, image orders, ground truth and camera parameters accordingly to maintain coherence in overlappedvarea between images, which resembles to flip the whole 3D space symmetrically.

### Grid Mask

The maximum 30% of total area is randomly masked with square masks.


## Results and Models

### Waymo validation

| Backbone  | Head  |              Trick and corresponding config              | LET-mAPL | LET-mAPH | L1/mAPH (Car) | Download |
| :-------: | :---: | :------------------------------------------------------: | :------: | :------: | :-----------: | :------: |
| ResNet101 | DETR  |       [Baseline](./../waymo_mini_r101_baseline.py)       |   34.6   |   46.1   |     25.5      | [model]  |
| ResNet101 | DETR  | [Multi-scale resize, Flip](./waymo_mini_r101_ms_flip.py) |    -     |    -     |     26.8      | [model]  |