import numpy as np

from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
import sys
sys.path.append('.')
from mmdet.datasets import DATASETS
import torch
import json
from typing import Dict, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from projects.mmdet3d_plugin.datasets.nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.datasets.HDmap import HDMap
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
import cv2


def gen_dx_bx(xbound, ybound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound]])

    return dx, bx, nx


class Bbox_mask(object):

    def __init__(self, nusc, point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], bev_size=(200, 200)):

        self.nusc = nusc
        # from github.com/nv-tlabs/lift-splat-shoot/
        # 200 * 200
        ybound = [point_cloud_range[1], point_cloud_range[4], (point_cloud_range[4] - point_cloud_range[1]) / bev_size[1]]
        xbound = [point_cloud_range[0], point_cloud_range[3], (point_cloud_range[3] - point_cloud_range[0]) / bev_size[0]]
        # xbound = [-51.2, 51.2, 0.512]
        # ybound = [-51.2, 51.2, 0.512]

        grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
        }

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

    def get_binimg(self, rec):
        '''
        from github.com/nv-tlabs/lift-splat-shoot/
        '''
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((2, self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = NuScenesBox(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)
            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            if inst['category_name'].split('.')[-1] == 'car':
                cv2.fillPoly(img[0], [pts], 1.0)
            cv2.fillPoly(img[1], [pts], 1.0)

        return img>0

if __name__ == '__main__':

    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
    
    # point_cloud_range = [-30., -15.,  -5.0, 30., 15., 3.0]
    # bev_size = (400, 200)
    
    point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
    bev_size = (200, 200)

    #point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    #bev_size = (200, 200)

    Bbox_mask = Bbox_mask(nusc, point_cloud_range=point_cloud_range, bev_size=bev_size)
    tokens = []
    imgs_list = []
    for sample in nusc.sample:
        #sample = nusc.get('sample', sample)
        tokens.append(sample['token'])
        imgs = Bbox_mask.get_binimg(sample)
        imgs_list.append(imgs)

    import mmcv
    import numpy as np
    import pycocotools.mask as mask_util

    def custom_encode_mask_results(mask_results):
        """Encode bitmap mask to RLE code. Semantic Masks only
        Args:
            mask_results (list | tuple[list]): bitmap mask results.
                In mask scoring rcnn, mask_results is a tuple of (segm_results,
                segm_cls_score).
        Returns:
            list | tuple: RLE encoded mask.
        """
        cls_segms = mask_results
        num_classes = len(cls_segms)
        encoded_mask_results = []
        for i in range(len(cls_segms)):
            encode_results =mask_util.encode(
                    np.array(
                        cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0]
            encode_results['counts'] = encode_results['counts'].decode()
            encoded_mask_results.append(encode_results)  # encoded with RLE
        return encoded_mask_results
    maps = {}
    for token, img in zip(tokens, imgs_list):
        encode_imgs = custom_encode_mask_results(img)
        #print(encode_imgs)

        #encode_imgs['counts'] = encode_imgs['counts'].decode()
        maps[token] = encode_imgs
    import json

    grid_length = '%.2f' % ((point_cloud_range[4] - point_cloud_range[1]) / bev_size[1])
    mask_shape_flag = f'{bev_size[0]}_{bev_size[1]}_{grid_length}'
    json.dump(maps, open(f'data/nuscenes/bbox_mask_gt_{mask_shape_flag}.json', 'w'))
