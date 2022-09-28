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
from .functional import horizontal_flip_image_multiview, flip_canbus


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


class RandomHorizontalFlipMultiViewImage_naive(object):

    def __init__(self, flip_ratio=0.5, dataset='waymo'):
        self.flip_ratio = flip_ratio
        self.seed = 0
        self.dataset = dataset

    def forward(self, imgs, canbus, seed=None):
        if seed is not None:
            np.random.seed(int(seed))
        if np.random.rand() >= self.flip_ratio:
            flip_flag = False
            return imgs,
        else:
            flip_flag = True
            results = self.flip_bbox(results)
            results = self.flip_cam_params(results)
            results = self.flip_img(results)
            imgs_flip = horizontal_flip_image_multiview(imgs)
            canbus = flip_canbus(canbus)

    def __call__(self, results, seed=None):
        return self.forward()

    def flip_cam_params(self, results):
        flip_factor = np.eye(4)

        # print(results['img_shape'])
        # print(results.keys())
        lidar2img = []

        w = results['img_shape'][1]
        # print(w)
        if self.dataset == 'nuScenes':
            flip_factor[0, 0] = -1
            lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
            for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
                cam_intrinsic[0, 0] = -cam_intrinsic[0, 0]
                cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
                lidar2img.append(cam_intrinsic @ l2c)
        elif self.dataset == 'waymo':
            # flip_factor[0, 0] = -1
            flip_factor[1, 1] = -1
            lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
            for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
                cam_intrinsic[0, 0] = -cam_intrinsic[0, 0]

                cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
                lidar2img.append(cam_intrinsic @ l2c)
        else:
            assert False
        results['lidar2cam'] = lidar2cam
        results['lidar2img'] = lidar2img

        return results

    def flip_bbox(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                assert False
                input_dict['points'] = input_dict[key].flip(direction, points=input_dict['points'])
            else:
                if direction == 'horizontal':
                    if self.dataset == 'nuScenes':
                        input_dict[key].tensor[:, 0::7] = -input_dict[key].tensor[:, 0::7]
                        input_dict[key].tensor[:, 6] = -input_dict[key].tensor[:, 6]  #+ np.pi
                    elif self.dataset == 'waymo':
                        input_dict[key].tensor[:, 1::7] = -input_dict[key].tensor[:, 1::7]
                        input_dict[key].tensor[:, 6] = -input_dict[key].tensor[:, 6] + np.pi

                elif bev_direction == 'vertical':
                    assert False
                    input_dict[key].tensor[:, 0::7] = -input_dict[key].tensor[:, 0::7]
                    input_dict[key].tensor[:, 6] = -input_dict[key].tensor[:, 6]
                # input_dict[key].flip(direction)
        return input_dict