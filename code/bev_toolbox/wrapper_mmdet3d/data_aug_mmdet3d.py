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

from typing import Dict
from mmdet.datasets.builder import PIPELINES
from ..data_aug.transforms import RandomScaleImageMultiViewImage_naive


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(RandomScaleImageMultiViewImage_naive):
    """Resize the multiple-view images with the same scale selected randomly.  .
    Wrapper for mmdet3d
    Args:
        scales (tuple of float): ratio for resizing the images. Every time, select one ratio randonly.
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
        imgs_new, cam_intrinsics_new, lidar2img_new = self.forward(imgs, cam_intrinsics, lidar2cam, lidar2img, seed=seed)
        results['img'] = imgs_new
        results['cam_intrinsic'] = cam_intrinsics_new
        results['lidar2img'] = lidar2img_new
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results
