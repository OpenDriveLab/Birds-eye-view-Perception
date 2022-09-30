# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import json
from typing import Dict, Tuple


import numpy as np
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.datasets.HDmap import HDMap
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
import cv2
import os
from .bbox_mask import Bbox_mask
import pycocotools.mask as mask_util
from mmcv.parallel import DataContainer as DC
import random
from .pipelines.compose import CustomCompose
import time


def get_static_noises(noise_level):
    noises_map = []
    for i in range(9999):
        noise_degree_x = np.random.normal(0, noise_level)
        noise_degree_y = np.random.normal(0, noise_level)
        noise_degree_z = np.random.normal(0, noise_level)
        tr_noise = [np.random.normal(0.0, noise_level * 0.05) for _ in range(4)]
        noises_map.append([noise_degree_x, noise_degree_y, noise_degree_z, tr_noise])
    return noises_map


def add_noise_s2l(r, t, noise_level, scene_idx=None, noise_map=None):
    if noise_level == 0.:
        return r, t
    # rotate each axis
    if scene_idx is not None:
        noises = noise_map[scene_idx]
        noise_degree_x, noise_degree_y, noise_degree_z, tr_noise = noises
    else:
        noise_degree_x = np.random.normal(0, noise_level)
        noise_degree_y = np.random.normal(0, noise_level)
        noise_degree_z = np.random.normal(0, noise_level)
        tr_noise = [np.random.normal(0.0, noise_level * 0.05) for _ in range(4)]
    noise_x = noise_degree_x / 180 * np.pi
    noise_y = noise_degree_y / 180 * np.pi
    noise_z = noise_degree_z / 180 * np.pi
    Rx = np.array(
        [[1, 0, 0],
         [0, np.cos(noise_x), -np.sin(noise_x)],
         [0, np.sin(noise_x), np.cos(noise_x)]
         ])
    Ry = np.array(
        [[np.cos(noise_y), 0, np.sin(noise_y)],
         [0, 1, 0],
         [-np.sin(noise_y), 0, np.cos(noise_y)]
         ])
    Rz = np.array(
        [
            [np.cos(noise_z), -np.sin(noise_z), 0],
            [np.sin(noise_z), np.cos(noise_z), 0],
            [0, 0, 1]
        ])
    # translate
    return Rx @ Ry @ Rz @ r, [tr + tr_noise[i] for i, tr in enumerate(t)]


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar',
                      noise_level=0,
                      scene_idx=None,
                      noise_map=None
                      ):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path

    l2e_r_s = cs_record['rotation']
    l2e_t_s = cs_record['translation']
    e2g_r_s = pose_record['rotation']
    e2g_t_s = pose_record['translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    l2e_r_s_mat, l2e_t_s = add_noise_s2l(l2e_r_s_mat, l2e_t_s, noise_level, scene_idx, noise_map)
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    s2l_r = R.T  # points @ R.T + T
    s2l_t = T
    return s2l_r, s2l_t


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    #def __getattr__(self, item):
    #    return item+" dose not exist"

    def prepare_train_data(self, index):
        """Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        ###
        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)
        scene_idx = int(self.nusc.get('scene', sample['scene_token'])['name'][-4:])
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        e2g_r = pose_record['rotation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        ###
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                s2l_r = cam_info['sensor2lidar_rotation']
                s2l_t = cam_info['sensor2lidar_translation']
                # if self.noise_level > 0:
                # print('origin==')
                # print(s2l_t)
                # print(s2l_r)
                sensor_token = sample['data'][cam_type]
                if self.noise_level > 0:
                    s2l_r, s2l_t = obtain_sensor2top(self.nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam_type, noise_level=self.noise_level, scene_idx=scene_idx, noise_map=self.noise_map)

                lidar2cam_r = np.linalg.inv(s2l_r)
                lidar2cam_t = s2l_t @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict


@DATASETS.register_module()
class NuScenesDataset_eval_modified(CustomNuScenesDataset):
    def __init__(self, ann_file, pipeline=None, data_root=None, point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 bev_size=(200, 200), can_bus_root=None, classes=None, load_interval=1, with_velocity=True,
                 modality=None, box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False,
                 eval_version='detection_cvpr_2019', use_valid_flag=False, overlap_test=False, nusc=None,
                 noise_level=0.,
                 nusc_can_bus=None):
        super().__init__(ann_file, pipeline=pipeline, data_root=data_root, classes=classes, load_interval=load_interval,
                         with_velocity=with_velocity, modality=modality, box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt, test_mode=test_mode, eval_version=eval_version,
                         use_valid_flag=use_valid_flag)
        self.noise_level = noise_level
        if self.noise_level != 0:
            self.noise_map = get_static_noises(self.noise_level)
        self.overlap_test = overlap_test
        self.point_cloud_range = point_cloud_range
        self.bev_size = bev_size
        self.nusc = nusc if nusc is not None else NuScenes(version=self.version, dataroot=self.data_root,
                                                           verbose=True)  # val dataset will utilize the nusc of train dataset to save memory
        # the data root of can_bus can be different from NuScene
        try:
            self.nusc_can_bus = nusc_can_bus if nusc_can_bus is not None else NuScenesCanBus(dataroot=can_bus_root)
        except:
            self.nusc_can_bus = None
            print('Not found can_bus from ', can_bus_root)

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail



@DATASETS.register_module()
class NuScenesDataset_video(NuScenesDataset_eval_modified):

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        sample_idx = input_dict['sample_idx']
        sample = self.nusc.get('sample', sample_idx)
        lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', lidar['ego_pose_token'])
        rotation = Quaternion(pose['rotation'])
        translation = pose['translation']

        try:
            can_bus = self._get_can_bus_info(sample)
        except:
            can_bus = {'can_bus': np.zeros(18)}

        can_bus['can_bus'][:3] = translation
        can_bus['can_bus'][3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus['can_bus'][-2] = patch_angle / 180 * np.pi
        can_bus['can_bus'][-1] = patch_angle
        input_dict.update(
            dict(
                scene_token=sample['scene_token'],
                prev_idx=sample['prev'],
                next_idx=sample['next'],
            )
        )
        input_dict.update(can_bus)

        return input_dict
    
    def _get_can_bus_info(self, sample):
        scene_name = self.nusc.get('scene', sample['scene_token'])['name']
        sample_timestamp = sample['timestamp']
        pose_list = self.nusc_can_bus.get_messages(scene_name, 'pose')
        can_bus = []
        ## during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
        last_pose = pose_list[0]
        for i, pose in enumerate(pose_list):
            if pose['utime'] > sample_timestamp:
                break
            last_pose = pose

        _ = last_pose.pop('utime')  # useless
        pos = last_pose.pop('pos')
        rotation = last_pose.pop('orientation')

        can_bus.extend(pos)
        can_bus.extend(rotation)
        for key in last_pose.keys():
            can_bus.extend(pose[key])  # 16 elements
        can_bus.extend([0., 0.])
        return {'can_bus': np.array(can_bus)}

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """

        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


@DATASETS.register_module()
class NuScenesDataset_videoV2(NuScenesDataset_eval_modified):

    def __init__(self, queue_length=4, pipeline=None, *args, **kwargs):
        super().__init__(pipeline=[], *args, **kwargs)
        self.pipeline = CustomCompose(pipeline)
        self.queue_length = queue_length

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        seed = time.time()
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)

            example = self.pipeline(input_dict, seed=seed)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)
        # return example

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)


        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        sample_idx = input_dict['sample_idx']
        sample = self.nusc.get('sample', sample_idx)
        lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', lidar['ego_pose_token'])
        rotation = Quaternion(pose['rotation'])
        translation = pose['translation']

        try:
            can_bus = self._get_can_bus_info(sample)
        except:
            can_bus = {'can_bus': np.zeros(18)}

        can_bus['can_bus'][:3] = translation
        can_bus['can_bus'][3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus['can_bus'][-2] = patch_angle / 180 * np.pi
        can_bus['can_bus'][-1] = patch_angle
        input_dict.update(
            dict(
                scene_token=sample['scene_token'],
                prev_idx=sample['prev'],
                next_idx=sample['next'],
            )
        )
        input_dict.update(can_bus)

        return input_dict

    def _get_can_bus_info(self, sample):
        scene_name = self.nusc.get('scene', sample['scene_token'])['name']
        sample_timestamp = sample['timestamp']
        pose_list = self.nusc_can_bus.get_messages(scene_name, 'pose')
        can_bus = []
        ## during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
        last_pose = pose_list[0]
        for i, pose in enumerate(pose_list):
            if pose['utime'] > sample_timestamp:
                break
            last_pose = pose

        _ = last_pose.pop('utime')  # useless
        pos = last_pose.pop('pos')
        rotation = last_pose.pop('orientation')

        can_bus.extend(pos)
        can_bus.extend(rotation)
        for key in last_pose.keys():
            can_bus.extend(pose[key])  # 16 elements
        can_bus.extend([0., 0.])
        return {'can_bus': np.array(can_bus)}

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """

        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

# @DATASETS.register_module()
# class NuScenesDataset_lss(NuScenesDataset_video):
#     def get_data_info(self, index):
#         input_dict = super().get_data_info(index)
#         sample_idx = input_dict['sample_idx']
#         sample = self.nusc.get('sample', sample_idx)
#         rots = []
#         trans = []
#         intrins = []
#         post_rots = []
#         post_trans = []
#
#         lidar_token = sample['data']['LIDAR_TOP']
#         sd_rec = self.nusc.get('sample_data', lidar_token)
#         cs_record = self.nusc.get('calibrated_sensor',
#                              sd_rec['calibrated_sensor_token'])
#
#         # cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
#         info = self.data_infos[index]
#
#         for cam, cam_info in info['cams'].items():
#
#             samp = self.nusc.get('sample_data', sample['data'][cam])
#
#             sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
#             intrin = torch.Tensor(sens['camera_intrinsic'])
#             rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
#             tran = torch.Tensor(sens['translation'])
#
#             post_tran = torch.Tensor(cs_record['translation'])
#             post_rot = torch.Tensor(Quaternion(cs_record['rotation']).rotation_matrix)
#
#             intrins.append(intrin)
#             rots.append(rot)
#             trans.append(tran)
#             post_rots.append(post_rot)
#             post_trans.append(post_tran)
#
#         lss = [torch.stack(rots), torch.stack(trans), torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
#         input_dict['lss_metas'] = lss
#         return input_dict


@DATASETS.register_module()
class NuScenesDataset_mask(NuScenesDataset_video):

    def __init__(self, *args, **kwargs):
        super(NuScenesDataset_mask, self).__init__(*args, **kwargs)
        self.HDMap = HDMap(self.nusc, data_root=self.data_root, thickness=2, point_cloud_range=self.point_cloud_range,
                           bev_size=self.bev_size)
        self.Bbox_mask = Bbox_mask(self.nusc, point_cloud_range=self.point_cloud_range, bev_size=self.bev_size)
        self.mask_shape_flag = f'{self.HDMap.canvas_size[0]}_{self.HDMap.canvas_size[1]}_{self.HDMap.grid_length}'
        if not self.test_mode:
            self.load_mask_gt()

    def load_mask_gt(self):
        try:
            self.bbox_mask_gt = json.load(
                open(osp.join(self.data_root, f'bbox_mask_gt_{self.mask_shape_flag}.json'), 'r'))
        except:
            self.bbox_mask_maps = None
            return
        keys = list(self.bbox_mask_gt.keys())
        bbox_mask_maps = {}
        for key in keys:
            mask_list = []
            for mask in self.bbox_mask_gt[key]:
                mask['counts'] = mask['counts'].encode()
                binary_mask = mask_util.decode(mask)
                mask_list.append(binary_mask)
            bbox_mask_maps[key] = np.stack(mask_list)
        self.bbox_mask_maps = bbox_mask_maps

        del self.bbox_mask_gt

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        if not self.test_mode:
            sample_idx = input_dict['sample_idx']
            sample = self.nusc.get('sample', sample_idx)
            map_mask = self.HDMap.get(sample)
            input_dict['map_mask'] = map_mask
            if self.bbox_mask_maps is not None:
                input_dict['bbox_mask'] = self.bbox_mask_maps[sample_idx]
            else:
                input_dict['bbox_mask'] = self.Bbox_mask.get_binimg(sample)
            # save_tensor(torch.tensor(map_mask), f'{sample_idx}.png')
        return input_dict

    def evaluate(self,
                 results,
                 **kwargs):
        if isinstance(results, dict):
            results_dict = super().evaluate(results['bbox_results'], **kwargs)
            mask_results = self.evaluate_mask(results['mask_results'])
            results_dict.update(mask_results)
        else:
            print("WARNING!! MaskDataset used but no masks results")
            results_dict = super().evaluate(results, **kwargs)
            # assert False
        return results_dict

    def evaluate_mask(self, results):
        self.HDMap.thickness = 1
        return self.nusc_eval.evaluate_mask(results, self.HDMap, self.Bbox_mask)


@DATASETS.register_module()
class NuScenesDataset_lss(NuScenesDataset_mask):
    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        sample_idx = input_dict['sample_idx']
        sample = self.nusc.get('sample', sample_idx)
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = self.nusc.get('sample_data', lidar_token)
        cs_record = self.nusc.get('calibrated_sensor',
                                  sd_rec['calibrated_sensor_token'])

        # cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        info = self.data_infos[index]

        for cam, cam_info in info['cams'].items():
            samp = self.nusc.get('sample_data', sample['data'][cam])

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = Quaternion(sens['rotation']).rotation_matrix
            tran = sens['translation']
            rot, tran = add_noise_s2l(rot, tran, noise_level=self.noise_level)
            rot = torch.Tensor(rot)
            tran = torch.Tensor(tran)

            post_tran = torch.Tensor(cs_record['translation'])
            post_rot = torch.Tensor(Quaternion(cs_record['rotation']).rotation_matrix)

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        lss = [torch.stack(rots), torch.stack(trans), torch.stack(intrins), torch.stack(post_rots),
               torch.stack(post_trans)]
        input_dict['lss_metas'] = lss
        return input_dict

