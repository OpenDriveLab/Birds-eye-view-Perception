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

from typing import List, Tuple
import numpy as np
from .functional import scale_image_multiple_view


class RandomScaleImageMultiViewImage_naive(object):
    """Resize the multiple-view images with the same scale selected randomly.  .
    Args:
        scales (tuple of float): ratio for resizing the images. Every time, select one ratio 
        randomly.
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales
        self.seed = 0

    def forward(self,
                imgs: List[np.ndarray],
                cam_intrinsics: List[np.ndarray],
                cam_extrinsics: List[np.ndarray],
                lidar2img: List[np.ndarray],
                seed=None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Args:
            img (list of numpy.array): Multiple-view images to be resized.
            cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras.
            cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras that transform from lidar to cameras.
            lidar2img (list of numpy.array): Transformations from lidar to images.
            seed (int): Seed for generating random number. 
        Returns:
            imgs_new (list of numpy.array): Updated multiple-view images
            cam_intrinsics_new (list of numpy.array): Updated intrinsic parameters of different cameras.
            lidar2img_new (list of numpy.array): Updated Transformations from lidar to images.
        """
        if seed is not None:
            np.random.seed(int(seed))
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]
        imgs_new, cam_intrinsic_new, lidar2img_new = scale_image_multiple_view(imgs, cam_intrinsics, cam_extrinsics,
                                                                               lidar2img, rand_scale)

        return imgs_new, cam_intrinsic_new, lidar2img_new

    def __call__(self,
                 imgs: List[np.ndarray],
                 cam_intrinsics: List[np.ndarray],
                 cam_extrinsics: List[np.ndarray],
                 lidar2img: List[np.ndarray],
                 seed=None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        return self.forward(imgs, cam_intrinsics, cam_extrinsics, lidar2img, seed)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str
