######################################################################
# Copyright (c) 2022 OpenPerceptionX. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################

######################################################################
# This file includes the wrappers of the data augmentation methods for
# mmdet3d. For more details, please refer to data_aug/transfroms.py and
# data_aug/functional.py
######################################################################

from typing import Dict
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from ..data_aug.transforms import RandomScaleImageMultiViewImage
from ..data_aug.transforms import RandomHorizontalFlipMultiViewImage


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(RandomScaleImageMultiViewImage):
    """Resize the multiple-view images with the same scale selected randomly.  .
    Wrapper for mmdet3d
    Args:
        scales (tuple of float): ratio for resizing the images. Every time, select one ratio randomly.
    """

    def __call__(self, results: Dict, seed=None) -> Dict:
        """Call function to randomly resize multiple-view images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            results (dict): Updated result dict.
        """
        imgs = results['img']
        cam_intrinsics = results['cam_intrinsic']
        lidar2cam = results['lidar2cam']
        lidar2img = results['lidar2img']
        imgs_new, cam_intrinsics_new, lidar2img_new = self.forward(imgs,
                                                                   cam_intrinsics,
                                                                   lidar2cam,
                                                                   lidar2img,
                                                                   seed=seed)
        results['img'] = imgs_new
        # results['cam_intrinsic'] = cam_intrinsics_new
        results['lidar2img'] = lidar2img_new
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


@PIPELINES.register_module()
class RandomHorizontalFlipMultiViewImage(RandomHorizontalFlipMultiViewImage):
    """Horizontally flip the multiple-view images with bounding boxes, camera parameters and can bus randomly.  .
    Wrapper for mmdet3d. Support coordinate systems like Waymo (https://waymo.com/open/data/perception/) or 
    Nuscenes (https://www.nuscenes.org/public/images/data.png).
    Args:
        flip_ratio (float 0~1): probability of the images being flipped. Default value is 0.5.
        dataset (string): Specify 'waymo' coordinate system or 'nuscenes' coordinate system.
    """

    def __call__(self, results: Dict, seed=None) -> Dict:
        """Call function to randomly horizontally flip the multiple-view images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            results (dict): Updated result dict.
        """

        if len(results['bbox3d_fields']) == 0:  # test mode
            results['bbox3d_fields'].append('empty_box3d')
            results['empty_box3d'] = results['box_type_3d'](np.array([], dtype=np.float32))
        assert len(results['bbox3d_fields']) == 1

        imgs = results['img']
        bboxes_3d = results['gt_bboxes_3d'].tensor.numpy()
        cam_intrinsics = results['cam_intrinsic']
        cam_extrinsics = results['lidar2cam']
        lidar2imgs = results['lidar2img']
        canbus = results['can_bus']
        flip_flag, imgs_flip, bboxes_3d_flip, cam_intrinsics_flip, cam_extrinsics_flip, lidar2imgs_flip, canbus_flip = self.forward(
            imgs, bboxes_3d, cam_intrinsics, cam_extrinsics, lidar2imgs, canbus, seed=seed)
        if flip_flag:
            results['flip'] = True
            results['img'] = imgs_flip
            bboxes_3d_flip = torch.Tensor(bboxes_3d_flip).to(results['gt_bboxes_3d'].tensor.device)
            results['gt_bboxes_3d'].tensor = bboxes_3d_flip
            results['lidar2cam'] = cam_extrinsics_flip
            results['lidar2img'] = lidar2imgs_flip
            results['can_bus'] = canbus_flip

        return results
