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

import numpy as np
import cv2
from typing import List, Tuple

#  Available interpolation modes (opencv)
cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def scale_image_multiple_view(imgs: List[np.ndarray],
                              cam_intrinsics: List[np.ndarray],
                              cam_extrinsics: List[np.ndarray],
                              lidar2img: List[np.ndarray],
                              rand_scale: float,
                              interpolation='bilinear') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Resize the multiple-view images with the same scale selected randomly.
    Notably used in :class:`.transforms.RandomScaleImageMultiViewImage_naive
    Args:
        img (list of numpy.array): Multiple-view images to be resized.
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras that transform from lidar to cameras.
        lidar2img (list of numpy.array): Transformations from lidar to images.
        rand_scale (float): resize ratio
        interpolation (string): mode for interpolation in opencv.
    Returns:
        imgs_new (list of numpy.array): Updated multiple-view images
        cam_intrinsics_new (list of numpy.array): Updated intrinsic parameters of different cameras.
        lidar2img_new (list of numpy.array): Updated Transformations from lidar to images.
    """

    y_size = [int(img.shape[0] * rand_scale) for img in imgs]
    x_size = [int(img.shape[1] * rand_scale) for img in imgs]

    scale_factor = np.eye(4)
    scale_factor[0, 0] *= rand_scale
    scale_factor[1, 1] *= rand_scale
    imgs_new = [
        cv2.resize(img, (x_size[idx], y_size[idx]), interpolation=cv2_interp_codes[interpolation])
        for idx, img in enumerate(imgs)
    ]
    cam_intrinsics_new = [scale_factor @ cam_intrinsic for cam_intrinsic in cam_intrinsics]
    lidar2img = [intr @ extr for (intr, extr) in zip(cam_intrinsics_new, cam_extrinsics)]

    return imgs_new, cam_intrinsics_new, lidar2img
