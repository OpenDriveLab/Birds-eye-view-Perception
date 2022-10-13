# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.datasets.pipelines.loading import LoadAnnotations3D
import torch
import cv2 as cv
import torch.nn.functional as F
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from torchvision.transforms.functional import rotate



@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        try:
            img = [mmcv.imread(name, self.color_type).astype(np.float32) for name in filename]
        except:
            # import os
            # for each in filename:
            #     os.system('rm ' + each)
            #     os.system('wget http:10.142.40.14:5000/'+each + ' -o ' + each)
            # img = [mmcv.imread(name, self.color_type).astype(np.float32) for name in filename]
            print(filename)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img[0].shape
        results['ori_lidar2img'] = copy.deepcopy(results['lidar2img'])
        # # Set initial values for default meta_keys
        results['scale_factor'] = 1.0
        # num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        num_channels = 3
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class CustomLoadAnnotations3D(object):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self, with_bbox_mask=False):

        self.with_bbox_mask = with_bbox_mask

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox_mask:
            results = self._load_custom_mask(results)
            if results is None:
                return None
        return results

    def _load_custom_mask(self, results):

        bbox_mask = results['bbox_mask']
        map_mask = results['map_mask']
        # bev_bboxes = bboxes.bev

        bbox_mask = torch.tensor(bbox_mask)
        map = torch.tensor(map_mask)

        img = torch.cat([bbox_mask, map], 0)
        img = torch.flip(img, [1, 2])

        #bbox_mask = torch.flip(bbox_mask, [1])
        #bbox_mask = torch.flip(bbox_mask, [2])
        #map = torch.flip(map, [1])

        #map = rotate(map, 90)


        #save_tensor(map, f'{results["sample_idx"]}_map.png')
        #save_tensor(bbox_mask, f'{results["sample_idx"]}_bbox.png')

        #print('map', map.shape, 'mask', bbox_mask.shape)


        # img[:2] = torch.flip(img[:2], [2])
        # img[2:] = rotate(img[2:], 90)
        # img = torch.clamp(img, max=1)
        # img = img.permute(1, 2, 0).sum(-1, keepdim=True).permute(2, 0, 1)
        # save_tensor(img, f'{results["sample_idx"]}_.png')
        # exit()
        results['gt_map_masks'] = img
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str