import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import cv2
import os.path as osp


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size == 'same2max':
            max_shape = max([img.shape for img in results['img']])[:2]
            padded_img = [mmcv.impad(img, shape=max_shape, pad_val=self.pad_val) for img in results['img']]
        elif self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']
            ]

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.seed = 0

    def __call__(self, results, seed=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        if seed is not None:
            random.seed(int(seed))
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha
            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class CropMultiViewImage(object):
    """Crop the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, size=None):
        self.size = size

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results['img'] = [img[:self.size[0], :self.size[1], ...] for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img_fixed_size'] = self.size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales
        self.seed = 0
        #[i/10 for i in range(5, 16)]

    def __call__(self, results, seed=None):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # print(seed)
        if seed is not None:
            np.random.seed(int(seed))
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]
        # print(rand_scale)
        # print(results['img_shape'])
        # for each in results['img_shape']:
        #    print(each)
        # img_shape = results['img_shape']
        # print(img_shape, rand_scale)
        #y_size = int(img_shape[0] * rand_scale)
        #x_size = int(img_shape[1] * rand_scale)
        #scale_factor = np.eye(4)
        #scale_factor[0, 0] *= rand_scale
        #scale_factor[1, 1] *= rand_scale

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        # print(x_size, y_size)

        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [
            mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in enumerate(results['img'])
        ]
        # results['img'] = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results['img']]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        #results['gt_bboxes_3d'].tensor[:, :6] *= rand_scale
        # for i in range(len(results['img'])):
        #     # print(results['bbox3d_fields'])
        #     # print(results['gt_bboxes_3d'].tensor)
        #     # print(results['lidar2img'][i])
        #     show_multi_modality_result(
        #         results['img'][i],
        #         results['gt_bboxes_3d'],
        #         None,
        #         results['lidar2img'][i],
        #         '.',
        #         f'aug2_{i}.png',
        #         box_mode='lidar',
        #         show=True,
        #         scores=None,
        # )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


from projects.mmdet3d_plugin.models.utils.draw_bbox import show_multi_modality_result


@PIPELINES.register_module()
class HorizontalRandomFlipMultiViewImage(object):

    def __init__(self, flip_ratio=0.5, dataset='nuScenes'):
        self.flip_ratio = flip_ratio
        # dataset = 'waymo'
        self.seed = 0
        self.dataset = dataset

    def __call__(self, results, seed=None):
        if seed is not None: np.random.seed(int(seed))
        if np.random.rand() >= self.flip_ratio:
            return results
        else:
            # pass
            results['flip'] = True
            results = self.flip_bbox(results)
            results = self.flip_cam_params(results)
            results = self.flip_img(results)
            results = self.flip_can_bus(results)
        # for i in range(len(results['img'])):
        #     # print(results['bbox3d_fields'])
        #     # print(results['gt_bboxes_3d'].tensor)
        #     # print(results['lidar2img'][i])
        #     show_multi_modality_result(
        #         results['img'][i],
        #         results['gt_bboxes_3d'],
        #         None,
        #         results['lidar2img'][i],
        #         '.',
        #         f'aug2_{i}.png',
        #         box_mode='lidar',
        #         show=True,
        #         scores=None,
        # )
        # bev_img = np.zeros([1500, 1100, 3], dtype=np.float32)
        #
        # def world2bev_vis(x, y):
        #     return int((x + 35) * 10), int((-y + 75) * 10)
        #
        # for corners in results['gt_bboxes_3d'].corners[:, [4, 7, 3, 0], :2]:
        #     corners = np.array([world2bev_vis(*corner) for corner in corners])
        #
        #     _img = np.zeros([1500, 1100, 3], dtype=np.float32)
        #     cv2.circle(_img, corners[0], 5, (61, 102, 255))
        #     _img = cv2.fillPoly(_img, [corners], (61, 102, 255))
        #     bev_img = cv2.addWeighted(bev_img, 1, _img, 0.5, 0)
        #
        # bev_img = cv2.circle(bev_img, world2bev_vis(0, 0), 5, (0, 255, 0), thickness=-1)
        #
        # mmcv.imwrite(bev_img, f'aug3_bev.png',)
        # exit()
        return results

    def flip_can_bus(self, results, direction='horizontal'):
        # TODO location
        if self.dataset == 'nuScenes':
            # results['can_bus'][1] = -results['can_bus'][1]  # flip location
            # results['can_bus'][-2] = -results['can_bus'][-2]  # flip direction
            results['can_bus'][-1] = -results['can_bus'][-1]  # flip direction
        # results['can_bus']
        elif self.dataset == 'waymo':
            # print(results['can_bus'])
            # results['can_bus'][1] = -results['can_bus'][-1]  # flip location
            # results['can_bus'][-2] = -results['can_bus'][-2]  # flip direction
            results['can_bus'][-1] = -results['can_bus'][-1]  # flip direction
        return results

    def flip_img(self, results, direction='horizontal'):
        results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
        return results

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


@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            'filename',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'pcd_trans',
            'sample_idx',
            'prev_idx',
            'next_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pts_filename',
            'transformation_3d_flow',
            'scene_token',
            'can_bus',
            'ori_lidar2img',
            'lss_metas',  # for lift_splat_shoot
        )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        data = {}
        img_metas = {}

        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'