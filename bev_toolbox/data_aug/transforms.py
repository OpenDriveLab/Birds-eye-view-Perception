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
# This file integrates different data augmentation methods. For the detailed
# implementation, please refer to functional.py
######################################################################

from typing import List, Tuple
import numpy as np
from .functional import scale_image_multiple_view
from .functional import horizontal_flip_image_multiview, horizontal_flip_bbox, horizontal_flip_cam_params, horizontal_flip_canbus


class RandomScaleImageMultiViewImage(object):
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
            imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                    img shape: [H, W, 3].
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras. Transformations from camera 
                    to image. len(cam_intrinsics) is the number of camera. For each camera, shape is 4 * 4.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras. Transformations from
                lidar to cameras. len(cam_extrinsics) is the number of camera. For each camera, shape is 4 * 4.
        lidar2img (list of numpy.array): Transformations from lidar to images. len(lidar2img) is the number
                of camera. For each camera, shape is 4 * 4.
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


class RandomHorizontalFlipMultiViewImage(object):
    """Horizontally flip the multiple-view images with bounding boxes, camera parameters and can bus randomly.  .
    Support coordinate systems like Waymo (https://waymo.com/open/data/perception/) or Nuscenes (https://www.nuscenes.org/public/images/data.png).
    Args:
        flip_ratio (float 0~1): probability of the images being flipped. Default value is 0.5.
        dataset (string): Specify 'waymo' coordinate system or 'nuscenes' coordinate system.
    """

    def __init__(self, flip_ratio=0.5, dataset='waymo'):
        self.flip_ratio = flip_ratio
        self.seed = 0
        self.dataset = dataset

    def forward(
        self,
        imgs: List[np.ndarray],
        bboxes_3d: np.ndarray,
        cam_intrinsics: List[np.ndarray],
        cam_extrinsics: List[np.ndarray],
        lidar2imgs: List[np.ndarray],
        canbus: np.ndarray,
        seed=None
    ) -> Tuple[bool, List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Args:
        imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                img shape: [H, W, 3].
        bboxes_3d (np.ndarray): bounding boxes of shape [N * 7], N is the number of objects.
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras. Transformations from camera 
                to image. len(cam_intrinsics) is the number of camera. For each camera, shape is 4 * 4.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras. Transformations from
                lidar to cameras. len(cam_extrinsics) is the number of camera. For each camera, shape is 4 * 4.
        lidar2img (list of numpy.array): Transformations from lidar to images. len(lidar2img) is the number
                of camera. For each camera, shape is 4 * 4.
        canbus (numpy.array): 
        seed (int): Seed for generating random number.
        Returns:
            imgs_new (list of numpy.array): Updated multiple-view images
            cam_intrinsics_new (list of numpy.array): Updated intrinsic parameters of different cameras.
            lidar2img_new (list of numpy.array): Updated Transformations from lidar to images.
        """
        if seed is not None:
            np.random.seed(int(seed))
        if np.random.rand() >= self.flip_ratio:
            flip_flag = False
            return flip_flag, imgs, bboxes_3d, cam_intrinsics, cam_extrinsics, lidar2imgs, canbus,
        else:
            flip_flag = True
            imgs_flip = horizontal_flip_image_multiview(imgs)
            bboxes_3d_flip = horizontal_flip_bbox(bboxes_3d, self.dataset)
            img_shape = imgs[0].shape
            cam_intrinsics_flip, cam_extrinsics_flip, lidar2imgs_flip = horizontal_flip_cam_params(
                img_shape, cam_intrinsics, cam_extrinsics, lidar2imgs, self.dataset)
            canbus_flip = horizontal_flip_canbus(canbus, self.dataset)
        return flip_flag, imgs_flip, bboxes_3d_flip, cam_intrinsics_flip, cam_extrinsics_flip, lidar2imgs_flip, canbus_flip

    def __call__(
        self,
        imgs: List[np.ndarray],
        bboxes_3d: np.ndarray,
        cam_intrinsics: List[np.ndarray],
        cam_extrinsics: List[np.ndarray],
        lidar2imgs: List[np.ndarray],
        canbus: np.ndarray,
        seed=None
    ) -> Tuple[bool, List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        return self.forward(imgs, bboxes_3d, cam_intrinsics, cam_extrinsics, lidar2imgs, canbus, seed)
