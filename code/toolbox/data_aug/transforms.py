import numpy as np
import mmcv
from .functional import scale_image_multiple_view


class RandomScaleImageMultiViewImage_naive(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales
        self.seed = 0

    def forward(self, imgs, lidar2img, seed=None):
        """Call function to apply random scale for multiple-view images
        Args:
            imgs (list [np.array, ...]): multiple-view images 
            lidar2img:
            seed:
        Returns:
            imgs_new (list [np.array, ...]): updated multiple-view images 
            lidar2img: 
        """

        if seed is not None:
            np.random.seed(int(seed))
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        imgs_new, lidar2img_new = scale_image_multiple_view(imgs, lidar2img, rand_scale)

        return imgs_new, lidar2img_new

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


# class HorizontalRandomFlipMultiViewImage(object):

#     def __init__(self, flip_ratio=0.5, dataset='nuScenes'):
#         self.flip_ratio = flip_ratio
#         self.seed = 0
#         self.dataset = dataset

#     def flip_can_bus(self, can_bus, direction='horizontal'):
#         # TODO location
#         if self.dataset == 'nuScenes':
#             can_bus[-1] = -can_bus[-1]  # flip direction
#         elif self.dataset == 'waymo':
#             can_bus[-1] = -can_bus[-1]  # flip direction
#         return can_bus

#     def flip_img(self, imgs, direction='horizontal'):
#         imgs = [mmcv.imflip(img, direction) for img in imgs]
#         return imgs

#     def flip_cam_params(self, results):
#         flip_factor = np.eye(4)
#         lidar2img = []

#         w = results['img_shape'][1]
#         if self.dataset == 'nuScenes':
#             flip_factor[0, 0] = -1
#             lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
#             for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
#                 cam_intrinsic[0, 0] = -cam_intrinsic[0, 0]
#                 cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
#                 lidar2img.append(cam_intrinsic @ l2c)
#         elif self.dataset == 'waymo':
#             flip_factor[1, 1] = -1
#             lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
#             for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
#                 cam_intrinsic[0, 0] = -cam_intrinsic[0, 0]

#                 cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
#                 lidar2img.append(cam_intrinsic @ l2c)
#         else:
#             assert False
#         results['lidar2cam'] = lidar2cam
#         results['lidar2img'] = lidar2img

#         return results

#     def flip_bbox(self, input_dict, direction='horizontal'):
#         bbox3d_fields, empty_box3d, box_type_3d
#         assert direction in ['horizontal', 'vertical']
#         if len(input_dict['bbox3d_fields']) == 0:  # test mode
#             input_dict['bbox3d_fields'].append('empty_box3d')
#             input_dict['empty_box3d'] = input_dict['box_type_3d'](np.array([], dtype=np.float32))
#         assert len(input_dict['bbox3d_fields']) == 1
#         for key in input_dict['bbox3d_fields']:
#             if 'points' in input_dict:
#                 assert False
#                 input_dict['points'] = input_dict[key].flip(direction, points=input_dict['points'])
#             else:
#                 if direction == 'horizontal':
#                     if self.dataset == 'nuScenes':
#                         input_dict[key].tensor[:, 0::7] = -input_dict[key].tensor[:, 0::7]
#                         input_dict[key].tensor[:, 6] = -input_dict[key].tensor[:, 6]  #+ np.pi
#                     elif self.dataset == 'waymo':
#                         input_dict[key].tensor[:, 1::7] = -input_dict[key].tensor[:, 1::7]
#                         input_dict[key].tensor[:, 6] = -input_dict[key].tensor[:, 6] + np.pi

#                 elif bev_direction == 'vertical':
#                     assert False
#                     input_dict[key].tensor[:, 0::7] = -input_dict[key].tensor[:, 0::7]
#                     input_dict[key].tensor[:, 6] = -input_dict[key].tensor[:, 6]
#         return input_dict