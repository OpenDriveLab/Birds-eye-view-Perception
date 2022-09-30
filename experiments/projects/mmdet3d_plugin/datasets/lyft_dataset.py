# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
import pandas as pd
import tempfile
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from os import path as osp
from pyquaternion import Quaternion

#from mmdet3d.core.evaluation.lyft_eval import lyft_eval
from projects.mmdet3d_plugin.core.evaluation.lyft_eval import lyft_eval  # debug purpose
from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.lyft_dataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import quaternion_yaw

@DATASETS.register_module()
class CustomLyftDataset(LyftDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lyft = Lyft(
             data_path=osp.join(self.data_root, self.version),
             json_path=osp.join(self.data_root, self.version, self.version),
             verbose=True)

    def get_data_info(self, index):
        input_dict = super(CustomLyftDataset, self).get_data_info(index)
        sample_idx = input_dict['sample_idx']
        sample = self.lyft.get('sample', sample_idx)
        lidar = self.lyft.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = self.lyft.get('ego_pose', lidar['ego_pose_token'])
        rotation = Quaternion(pose['rotation'])
        translation = pose['translation']
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        input_dict.update(
            dict(
                scene_token=sample['scene_token'],
                prev_idx=sample['prev'],
                next_idx=sample['next'],
            )
        )
        can_bus = {'can_bus': np.zeros(18)}
        can_bus['can_bus'][:3] = translation
        can_bus['can_bus'][3:7] = rotation
        if patch_angle < 0:
            patch_angle += 360
        can_bus['can_bus'][-2] = patch_angle / 180 * np.pi
        can_bus['can_bus'][-1] = patch_angle

        input_dict.update(can_bus)

        return input_dict

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in Lyft protocol.
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
        lyft = Lyft(
            data_path=osp.join(self.data_root, self.version),
            json_path=osp.join(self.data_root, self.version, self.version),
            verbose=True)
        eval_set_map = {
            'v1.01-train': 'val',
        }
        metrics = lyft_eval(lyft, self.data_root, result_path,
                            eval_set_map[self.version], output_dir, logger)

        # record metrics
        detail = dict()
        metric_prefix = f'{result_name}_Lyft'

        for i, name in enumerate(metrics['class_names']):
            AP = float(metrics['mAPs_cate'][i])
            detail[f'{metric_prefix}/{name}_AP'] = AP

        detail[f'{metric_prefix}/mAP'] = metrics['Final mAP']
        return detail

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        lyft_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_lyft_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_lyft_box_to_global(self.data_infos[sample_id], boxes)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                lyft_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    name=name,
                    score=box.score)
                annos.append(lyft_anno)
            lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, res_path)
        return res_path


nuscenes2LYFT = {
    0: 0,  # car -> car
    1: 1,  # truck -> truck
    2: 4,  # trailer -> other_vehicle
    3: 2,  # bus -> bus
    4: 4,  # construction_vehicle -> other_vehicle
    5: 6,  # bicycle -> bicycle
    6: 5,  # motorcycle -> motorcycle
    7: 7,  # pedestrian -> pedestrian
    8: -1,  # None
    9: -1,  # None
}


def output_to_lyft_box(detection):
    """Convert the output to the box class in the Lyft.
    Args:
        detection (dict): Detection results.
    Returns:
        list[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        if labels[i] >= 8:
            continue
        else:
            labels[i] = nuscenes2LYFT[labels[i]]
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        box = LyftBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i])
        box_list.append(box)
    return box_list


def lidar_lyft_box_to_global(info, boxes):
    """Convert the box from ego to global coordinate.
    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`LyftBox`]): List of predicted LyftBoxes.
    Returns:
        list: List of standard LyftBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list

